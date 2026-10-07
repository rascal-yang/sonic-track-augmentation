"""Offline SONIC G1-mode encoding; no robot control or physical rollout."""
from pathlib import Path
import argparse, hashlib, json, re, shutil, sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

p=argparse.ArgumentParser()
p.add_argument('--package',type=Path,required=True)
p.add_argument('--deploy',type=Path,required=True)
p.add_argument('--mjcf',type=Path,required=True)
p.add_argument('--ort-path',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.ort_path:sys.path.insert(0,str(a.ort_path.resolve()))
import onnxruntime as ort
model=a.deploy/'policy/release/model_encoder.onnx'
config=a.deploy/'policy/release/observation_config.yaml'
cpp=a.deploy/'src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp'
params=a.deploy/'src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp'
y=yaml.safe_load(config.read_text())['encoder']
registry={n:int(d) for n,d in re.findall(r'\{"([^"]+)",\s*(\d+),',cpp.read_text())}
layout={};offset=0
for entry in y['encoder_observations']:
    if not entry.get('enabled',True):continue
    name=entry['name'];size=registry[name]
    layout[name]={'offset':offset,'size':size};offset+=size
mode=next(m for m in y['encoder_modes'] if m['name']=='g1')
assert mode['mode_id']==0
assert set(mode['required_observations'])=={'encoder_mode_4','motion_joint_positions_10frame_step5','motion_joint_velocities_10frame_step5','motion_anchor_orientation_10frame_step5'}
# Derive deployment IsaacLab joint names from MJCF order and the actual C++ map.
mj_names=[j.get('name') for j in ET.parse(a.mjcf).find('worldbody').iter('joint') if j.get('type') != 'free']
assert len(mj_names)==29 and len(set(mj_names))==29
match=re.search(r'const std::array<int, 29> mujoco_to_isaaclab\s*=\s*\{([^}]+)',params.read_text())
order=[int(x) for x in re.findall(r'\d+',match.group(1))]
expected_names=[mj_names[i] for i in order]
opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
session=ort.InferenceSession(str(model),sess_options=opt,providers=['CPUExecutionProvider'])
assert session.get_inputs()[0].shape==[1,offset]
assert session.get_outputs()[0].shape==[1,64]
input_name=session.get_inputs()[0].name;output_name=session.get_outputs()[0].name
a.output.mkdir(parents=True,exist_ok=False)
shutil.copy2(config,a.output/'observation_config.yaml')
shutil.copy2(Path(__file__),a.output/Path(__file__).name)
report={'model':str(model.resolve()),'model_sha256':digest(model),'observation_config_sha256':digest(config),'cpp_sha256':digest(cpp),'joint_mapping_source_sha256':digest(params),'mjcf_sha256':digest(a.mjcf),'onnxruntime_version':ort.__version__,'provider':'CPUExecutionProvider','input_name':input_name,'input_dimension':offset,'output_name':output_name,'output_dimension':64,'mode':mode,'layout':layout,'joint_names':expected_names,'future_offsets_control_steps':list(range(0,50,5)),'dt_s':.02,'orientation':'R_actual_base(t)^T @ R_reference_root(t+offset); first two matrix columns flattened ROW-major; wxyz input; identity heading realignment because actual and reference use the same recorded world','input_semantics':'recorded actual motion reinterpreted as reference, teacher-forced recorded base orientation; NOT reconstruction of original controller encoder inputs','tail_policy':'clamp each future index to last recorded sample, including recorded velocity; flag last45 rows as padded','tokens_semantics':'ONNX encoded_tokens output; exporter source calls encode() with quantization enabled when configured. Float representation, not integer token IDs; not the pre-quantization residual latent.','physics_rollout_tested':False,'training_ready':False,'runs':{}}
all_tokens={}
for label in ['full_waypoints','simplified_arc']:
    source=a.package/label/'trajectory_named.npz'
    with np.load(source,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    names=data['g1_joint_names'].tolist();mapping=[names.index(n) for n in expected_names]
    q=data['g1_joint_position_rad'][:,mapping];dq=data['g1_joint_velocity_rad_s'][:,mapping]
    n=len(q);t=data['time_s'];assert np.allclose(np.diff(t),.02)
    future=np.minimum(np.arange(n)[:,None]+np.arange(0,50,5)[None,:],n-1)
    rotation=Rotation.from_quat(data['root_quaternion_world_wxyz'][:,[1,2,3,0]]).as_matrix()
    relative=np.einsum('nji,nkjl->nkil',rotation,rotation[future])
    ori=relative[:,:,:,:2].reshape(n,60)
    obs=np.zeros((n,offset),np.float32)
    values={'encoder_mode_4':np.zeros((n,4),np.float32),'motion_joint_positions_10frame_step5':q[future].reshape(n,290),'motion_joint_velocities_10frame_step5':dq[future].reshape(n,290),'motion_anchor_orientation_10frame_step5':ori}
    for key,val in values.items():
        spec=layout[key];assert val.shape==(n,spec['size'])
        obs[:,spec['offset']:spec['offset']+spec['size']]=val
    assert np.isfinite(obs).all()
    # Independent scalar construction checks window/order and orientation convention.
    max_error=0.
    for i in [0,60,830,1290,n-1]:
        expected=[]
        for j in future[i]:expected.extend((rotation[i].T@rotation[j])[:,:2].reshape(-1))
        error=float(np.max(np.abs(ori[i]-expected)));max_error=max(max_error,error)
        assert error<1e-12
        s=layout['motion_joint_positions_10frame_step5']['offset']
        assert np.array_equal(obs[i,s:s+29],q[i])
    assert np.allclose(ori[:,:6],np.array([1,0,0,1,0,0]),atol=1e-6)
    tokens=np.concatenate([session.run([output_name],{input_name:obs[i:i+1]})[0] for i in range(n)])
    assert tokens.shape==(n,64) and np.isfinite(tokens).all()
    replay_error=0.
    for i in [0,830,n-1]:
        result=session.run([output_name],{input_name:obs[i:i+1]})[0][0]
        replay_error=max(replay_error,float(np.max(np.abs(result-tokens[i]))))
    assert replay_error==0
    active_idx=np.concatenate([np.arange(layout[k]['offset'],layout[k]['offset']+layout[k]['size']) for k in values])
    inactive=np.setdiff1d(np.arange(offset),active_idx);assert np.count_nonzero(obs[:,inactive])==0
    arrays={'time_s':t,'control_index':data['control_index'],'sonic_encoded_tokens':tokens,'encoder_inputs':obs,'future_indices':future,'full_future_window_valid':np.arange(n)+45<n,'encoder_joint_names':np.array(expected_names)}
    for side in ['left','right']:
        ids=data[f'revo2_{side}_active_indices']
        arrays[f'revo2_{side}_active_joint_names']=data[f'revo2_{side}_joint_names'][ids]
        arrays[f'revo2_{side}_active_position_target_rad']=data[f'revo2_{side}_position_target_rad'][:,ids]
        arrays[f'revo2_{side}_active_actual_position_rad']=data[f'revo2_{side}_joint_position_rad'][:,ids]
    target=a.output/(label+'_tokens.npz');np.savez_compressed(target,**arrays)
    with np.load(target,allow_pickle=False) as z:assert np.array_equal(z['sonic_encoded_tokens'],tokens)
    all_tokens[label]=tokens
    report['runs'][label]={'source':str(source.resolve()),'source_sha256':digest(source),'output':target.name,'output_sha256':digest(target),'shape':list(tokens.shape),'dtype':str(tokens.dtype),'finite':True,'min':float(tokens.min()),'max':float(tokens.max()),'unique_rows':len(np.unique(tokens,axis=0)),'varying_dimensions':int(np.count_nonzero(np.ptp(tokens,axis=0)>1e-7)),'full_window_rows':n-45,'padded_tail_rows':45,'repeat_inference_max_error':replay_error,'orientation_scalar_check_max_error':max_error,'joint_name_mapping_verified':True}
    print(label,report['runs'][label],flush=True)
# Both trajectories share first830 input frames; complete 45-step lookahead matches through frame784.
report['common_prefix_first785_tokens_equal']=bool(np.array_equal(all_tokens['full_waypoints'][:785],all_tokens['simplified_arc'][:785]))
assert report['common_prefix_first785_tokens_equal']
(a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Encoding and consistency checks PASS')
