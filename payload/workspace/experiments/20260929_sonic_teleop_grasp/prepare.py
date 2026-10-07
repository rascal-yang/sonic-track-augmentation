"""Compile successful reference poses into release SONIC teleop input; no IK."""
from pathlib import Path
import argparse, json, shutil, sys, hashlib, os
import numpy as np
from scipy.spatial.transform import Rotation as R
H=Path(__file__).resolve().parent; W=H.parents[1]; SOURCE=Path(os.environ.get('G1_GRASP_SOURCE', W/'experiments/20260929_sonic_grasp_smooth/candidates/safe025'))
sys.path.insert(0,str(W/'g1_closed_loop/.deps'))
import onnxruntime as ort
p=argparse.ArgumentParser();p.add_argument('--name',default='reference');p.add_argument('--output',type=Path);a=p.parse_args()
c=a.output.resolve() if a.output else H/'candidates'/a.name;c.mkdir(parents=True,exist_ok=False)
for name in ['grasp_reference.npz','initial_state.npz','deployment_parameters.json','geometry_check.json','waypoints.json']:shutil.copy2(SOURCE/name,c/name)
z=dict(np.load(SOURCE/'grasp_tokens.npz'));ref=np.load(c/'grasp_reference.npz');n=len(ref['time_s']);obs=np.zeros((n,1762),np.float32);obs[:,0]=1
# C++ policy_parameters.hpp: left leg six joints, then right leg six joints.
legs=np.array([0,3,6,9,13,17,1,4,7,10,14,18]);future=z['future_indices']
obs[:,661:781]=ref['g1_joint_position_rad'][:,legs][future].reshape(n,120)
obs[:,781:901]=ref['g1_joint_velocity_rad_s'][:,legs][future].reshape(n,120)
root=R.from_quat(ref['root_quaternion_world_wxyz'][:,[1,2,3,0]])
scale=.982730067592321
bodies=['left_wrist_yaw_link','right_wrist_yaw_link','torso_link'];offsets=np.array([[.18,-.025,0],[.18,.025,0],[0,0,.35]])
positions=[];quaternions=[]
for name,offset in zip(bodies,offsets):
 j=ref['body_names'].tolist().index(name);rot=R.from_quat(ref['body_quaternion_world_wxyz'][:,j][:,[1,2,3,0]])
 # Undo the scene's uniform robot scaling before feeding nominal-model Cartesian inputs.
 local=root.inv().apply((ref['body_position_world_m'][:,j]-ref['root_position_world_m'])/scale)
 local+= (root.inv()*rot).apply(np.broadcast_to(offset,(n,3)))
 positions.append(local);quaternions.append((root.inv()*rot).as_quat()[:,[3,0,1,2]])
obs[:,901:910]=np.stack(positions,axis=1).reshape(n,9);obs[:,910:922]=np.stack(quaternions,axis=1).reshape(n,12)
obs[:,595:601]=np.eye(3)[:,:2].reshape(6)
model=Path('__RUNTIME__/models/release/model_encoder.onnx')
opt=ort.SessionOptions();opt.intra_op_num_threads=2;net=ort.InferenceSession(str(model),sess_options=opt,providers=['CPUExecutionProvider'])
z['encoder_inputs']=obs;z['sonic_encoded_tokens']=np.concatenate([net.run(None,{net.get_inputs()[0].name:x[None]})[0] for x in obs]);np.savez_compressed(c/'grasp_tokens.npz',**z)
plan=dict(candidate=a.name,source=str(SOURCE),encoder_mode=1,body_IK=False,online_planner=False,duration_s=n*.02,review_steps=[0,120,190,245,330,410,570],hypothesis='Same successful reference poses and fingers, Cartesian teleop encoder instead of joint-reference encoder.',changes=['teleop mode1','live actual-to-reference root orientation anchor'],lower_body_indices=legs.tolist(),vr_body_order=bodies,vr_offsets_nominal_m=offsets.tolist(),robot_scale=scale,position_frame='reference pelvis; nominal robot scale',orientation_format='reference-pelvis-local wxyz',training_ready=False)
(c/'plan.json').write_text(json.dumps(plan,indent=2));(c/'provenance.json').write_text(json.dumps(dict(encoder=str(model),encoder_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),source_reference_sha256=hashlib.sha256((SOURCE/'grasp_reference.npz').read_bytes()).hexdigest()),indent=2))
wp=json.loads((c/'waypoints.json').read_text());anchors=[]
body_indices=[ref['body_names'].tolist().index(name) for name in bodies]
for time,phase in zip(wp['time_s'],wp['phase_labels']):
 i=round(time/.02)
 anchors.append(dict(time_s=time,phase=phase,control_index=i,body_order=bodies,body_position_world_m=ref['body_position_world_m'][i,body_indices].tolist(),body_quaternion_world_wxyz=ref['body_quaternion_world_wxyz'][i,body_indices].tolist(),vr3_local_nominal_m=obs[i,901:910].reshape(3,3).tolist(),vr3_local_quaternion_wxyz=obs[i,910:922].reshape(3,4).tolist()))
(c/'cartesian_waypoints.json').write_text(json.dumps(dict(source=str(c/'waypoints.json'),note='Seven reference anchors; execute original continuous interpolation, not new linear Cartesian interpolation.',waypoints=anchors),indent=2))
print(c)
