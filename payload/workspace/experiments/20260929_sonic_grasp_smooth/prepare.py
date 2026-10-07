"""Compress the selected25mm reference into C2 right-arm posture anchors."""
from pathlib import Path
import argparse,json,shutil,sys,hashlib
import numpy as np
from scipy.spatial.transform import Rotation as R,Slerp

HERE=Path(__file__).resolve().parent;W=HERE.parents[1];B=W/'experiments/20260925_measured_walk_grasp'
sys.path.insert(0,str(B/'code'));sys.path.insert(0,str(W/'g1_closed_loop/.deps'))
from online_goals import Kinematics
import onnxruntime as ort
p=argparse.ArgumentParser();p.add_argument('--name',default='smooth025');p.add_argument('--time-scale',type=float,default=1.);p.add_argument('--safe-arc',action='store_true');p.add_argument('--hold-seconds',type=float);p.add_argument('--flow',action='store_true');p.add_argument('--raise-extra',type=float,default=0.);a=p.parse_args()
c=HERE/'candidates'/a.name;c.mkdir(parents=True,exist_ok=False)
base=W/'experiments/20260929_sonic_palm_wrap/candidates/left025'
with np.load(base/'grasp_reference.npz') as z:src={k:z[k].copy() for k in z.files}
with np.load(base/'grasp_tokens.npz') as z:token_base={k:z[k].copy() for k in z.files}
anchor=np.array([0,170,275,375,460,560,810] if a.safe_arc else [0,170,375,460,560,810]);times=np.array([0,2.4,3.8,4.9,6.6,8.2,11.7] if a.safe_arc else [0,2.4,4.6,6.3,7.9,11.4])*a.time_scale
if a.flow:
 assert a.safe_arc
 times=np.array([0,3.4,5.5,7.5,9.2,11.2,14.3])
times[1:]+=a.raise_extra
if a.hold_seconds is not None:times[-1]=times[-2]+a.hold_seconds
times=np.round(times/.02)*.02;t=np.arange(round(times[-1]/.02)+1)*.02;n=len(t)
segment=np.clip(np.searchsorted(times,t,side='right')-1,0,len(times)-2)
duration=times[segment+1]-times[segment];u=np.clip((t-times[segment])/duration,0,1)
b=u**3*(10-15*u+6*u*u);db=30*u*u*(1-u)**2/duration
# Keep contact-finger timing linear between phase endpoints, as in source; smooth arm separately.
source_coordinate=np.interp(t,times,anchor)
def linear(values):
 arr=np.asarray(values);flat=arr.reshape(len(arr),-1)
 return np.stack([np.interp(source_coordinate,np.arange(len(arr)),flat[:,j]) for j in range(flat.shape[1])],axis=1).reshape((n,)+arr.shape[1:]).astype(arr.dtype)
def quat_interp(values):
 arr=np.asarray(values);flat=arr.reshape(len(arr),-1,4);out=[]
 for j in range(flat.shape[1]):out.append(Slerp(np.arange(len(arr)),R.from_quat(flat[:,j][:,[1,2,3,0]]))(source_coordinate).as_quat()[:,[3,0,1,2]])
 return np.stack(out,axis=1).reshape((n,)+arr.shape[1:]).astype('float32')
ref={}
for key,value in src.items():
 if value.ndim and value.shape[0]==len(src['time_s']):ref[key]=quat_interp(value) if 'quaternion' in key else linear(value)
 else:ref[key]=value.copy()
ref['time_s']=t;ref['control_index']=np.arange(n);ref['source_time_s']=(source_coordinate+830)*.02
names=src['g1_joint_names'].tolist();arm=[j for j,name in enumerate(names) if name.startswith('right_') and any(k in name for k in ['shoulder','elbow','wrist'])]
nominal_q=ref['g1_joint_position_rad'].copy()
q0=src['g1_joint_position_rad'][anchor[segment]][:,arm];q1=src['g1_joint_position_rad'][anchor[segment+1]][:,arm]
ref['g1_joint_position_rad'][:,arm]=q0+b[:,None]*(q1-q0)
ref['g1_joint_velocity_rad_s']=np.gradient(ref['g1_joint_position_rad'].astype(float),.02,axis=0,edge_order=2).astype('float32')
ref['g1_joint_velocity_rad_s'][:,arm]=db[:,None]*(q1-q0)
if a.flow:
 from scipy.interpolate import make_interp_spline
 qs=src['g1_joint_position_rad'][anchor[:4]][:,arm]
 zeros=np.zeros(len(arm));spline=make_interp_spline(times[:4],qs,k=5,bc_type=([(1,zeros),(2,zeros)],[(1,zeros),(2,zeros)]))
 mask=t<=times[3];ref['g1_joint_position_rad'][np.ix_(mask,arm)]=spline(t[mask])
 ref['g1_joint_velocity_rad_s'][np.ix_(mask,arm)]=spline(t[mask],nu=1)
kin=Kinematics('__RUNTIME__/vendor/SONIC');order=[names.index(k) for k in kin.names]
model_names={kin.model.body(k).name:k for k in range(kin.model.nbody)}
affected=[(j,model_names[str(name)]) for j,name in enumerate(ref['body_names']) if str(name).startswith('right_') and str(name) in model_names]
for i in range(n):
 root=ref['root_position_world_m'][i];quat=ref['root_quaternion_world_wxyz'][i]
 kin.fk(nominal_q[i,order],root,quat);old_p=kin.data.xpos.copy();old_r=kin.data.xmat.copy()
 kin.fk(ref['g1_joint_position_rad'][i,order],root,quat)
 for j,bid in affected:
  ref['body_position_world_m'][i,j]+=kin.data.xpos[bid]-old_p[bid]
  dr=kin.data.xmat[bid].reshape(3,3)@old_r[bid].reshape(3,3).T
  rr=R.from_quat(ref['body_quaternion_world_wxyz'][i,j][[1,2,3,0]])
  ref['body_quaternion_world_wxyz'][i,j]=(R.from_matrix(dr)*rr).as_quat()[[3,0,1,2]]
np.savez_compressed(c/'grasp_reference.npz',**ref)
layout=json.loads((W/'experiments/20260929_sonic_token_encoding/encoded_release/report.json').read_text())['layout']
obs=np.zeros((n,1762),np.float32);future=np.minimum(np.arange(n)[:,None]+np.arange(0,50,5)[None],n-1)
eo=[names.index(k) for k in token_base['encoder_joint_names']]
rot=R.from_quat(ref['root_quaternion_world_wxyz'][:,[1,2,3,0]]).as_matrix();relative=np.einsum('nji,nkjl->nkil',rot,rot[future])
values={'motion_joint_positions_10frame_step5':ref['g1_joint_position_rad'][:,eo][future].reshape(n,290),'motion_joint_velocities_10frame_step5':ref['g1_joint_velocity_rad_s'][:,eo][future].reshape(n,290),'motion_anchor_orientation_10frame_step5':relative[:,:,:,:2].reshape(n,60)}
for key,value in values.items():spec=layout[key];obs[:,spec['offset']:spec['offset']+spec['size']]=value
model='__RUNTIME__/models/release/model_encoder.onnx'
opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
net=ort.InferenceSession(model,sess_options=opt,providers=['CPUExecutionProvider'])
tokens=np.concatenate([net.run(None,{net.get_inputs()[0].name:row[None]})[0] for row in obs]);assert np.isfinite(tokens).all()
encoded=dict(time_s=t,control_index=np.arange(n),sonic_encoded_tokens=tokens,encoder_inputs=obs,future_indices=future,full_future_window_valid=np.arange(n)+45<n,encoder_joint_names=token_base['encoder_joint_names'],source_control_coordinate=source_coordinate+830)
for side in ['left','right']:
 ids=ref[f'revo2_{side}_active_indices'];encoded[f'revo2_{side}_active_joint_names']=ref[f'revo2_{side}_joint_names'][ids]
 encoded[f'revo2_{side}_active_position_target_rad']=ref[f'revo2_{side}_position_target_rad'][:,ids]
 encoded[f'revo2_{side}_active_actual_position_rad']=ref[f'revo2_{side}_joint_position_rad'][:,ids]
np.savez_compressed(c/'grasp_tokens.npz',**encoded)
for name in ['initial_state.npz','deployment_parameters.json']:shutil.copy2(base/name,c/name)
waypoints=dict(source_grasp_indices=anchor.tolist(),time_s=times.tolist(),phase_labels=(['start','raised_clearance','above_edge','pregrasp','closed','lifted','hold_end'] if a.safe_arc else ['start','raised_clearance','pregrasp','closed','lifted','hold_end']),right_arm_joint_names=[names[i] for i in arm],right_arm_q_rad=src['g1_joint_position_rad'][anchor][:,arm].tolist(),interpolation='quintic joint interpolation, zero velocity/acceleration at each anchor; finger stream retimed linearly per phase')
if a.flow:waypoints['interpolation']='C2 quintic spline through first4approach postures; zero endpoint v/a only. Closure/lift/hold piecewise quintic. Fingers retimed linearly.'
(c/'waypoints.json').write_text(json.dumps(waypoints,indent=2)+'\n')
plan=dict(candidate=a.name,left_mm=25,source='left025 fixed-reference candidate',hypothesis='Smooth right-arm anchors and phase retiming shorten execution without losing the deeper grasp.',duration_s=n*.02,baseline_duration_s=16.22,reference_arm_anchor_count=6,time_scale=a.time_scale,review_steps=[0,int(times[1]/.02),int(times[2]/.02),int(times[3]/.02),int(times[4]/.02),n-16],changes=['Replace dense right-arm samples with6quintic anchors','Retiming all reference fields and finger commands coherently','Recompute joint velocities and root-relative future orientations; rerun encoder'],unchanged=['25mm lateral-grasp baseline poses','Decoder/PD/assets','Source830 one-time initial physical state','No runtime IK'],reference_sha256=hashlib.sha256((c/'grasp_reference.npz').read_bytes()).hexdigest(),token_sha256=hashlib.sha256((c/'grasp_tokens.npz').read_bytes()).hexdigest(),training_ready=False)
plan['reference_arm_anchor_count']=len(anchor);plan['review_steps']=[0]+[int(x/.02) for x in times[1:-1]]+[n-16];plan['changes'][0]=f'Replace dense right-arm samples with{len(anchor)}quintic anchors';plan['safe_arc']=a.safe_arc;plan['hold_seconds_override']=a.hold_seconds;plan['flow_approach']=a.flow;plan['raise_extra_s']=a.raise_extra
(c/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');shutil.copy2(Path(__file__),c/'prepare_source.py');print(json.dumps(plan,indent=2))
