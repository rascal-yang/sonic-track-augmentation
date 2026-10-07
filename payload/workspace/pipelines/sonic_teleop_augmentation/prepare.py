"""Object-relative Cartesian augmentation; preserve geometry and contact scale."""
import argparse,shutil,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from common import *

def compile_candidate(c,spec):
 c=Path(c).resolve();c.mkdir(parents=True,exist_ok=False);write(c/'sample.json',spec)
 src=arrays(SOURCE/'grasp_reference.npz');ref={k:v.copy() for k,v in src.items()};init=arrays(SOURCE/'initial_state.npz')
 translation=np.asarray(spec.get('object_delta_m',[0,0,0]),float);yaw=float(spec.get('grasp_yaw_deg',0));stretch=float(spec.get('approach_scale',1))
 assert translation.shape==(3,) and translation[2]==0 and np.linalg.norm(translation[:2])<=.04
 assert abs(yaw)<=4 and .97<=stretch<=1.03
 rotation=R.from_euler('z',yaw,degrees=True);pivot=init['object_position'].copy();init['object_position']+=translation
 # Same bottle/table geometry. Shift the whole object reference for bookkeeping.
 ref['object_position_world_m']+=translation
 t=ref['time_s'];u=np.clip(t/3.8,0,1);blend=u**3*(10-15*u+6*u*u)
 j=ref['body_names'].tolist().index('right_wrist_yaw_link');p0=src['body_position_world_m'][:,j].copy();r0=R.from_quat(src['body_quaternion_world_wxyz'][:,j][:,[1,2,3,0]])
 target=rotation.apply(p0-pivot)+pivot+translation
 # Start unchanged; the transformed grasp is fully reached before contact.
 ref['body_position_world_m'][:,j]=p0+blend[:,None]*(target-p0)
 rot_delta=R.from_rotvec(np.tile([0,0,np.deg2rad(yaw)],(len(t),1))*blend[:,None])
 ref['body_quaternion_world_wxyz'][:,j]=(rot_delta*r0).as_quat()[:,[3,0,1,2]]
 # Spatial scaling affects the noncontact approach only; endpoints/contact fixed.
 u=np.clip(t/4.9,0,1);weight=64*u**3*(1-u)**3;weight[t>=4.9]=0
 ref['body_position_world_m'][:,j]+=(stretch-1)*weight[:,None]*(ref['body_position_world_m'][:,j]-ref['body_position_world_m'][245,j])
 # Morphology is unchanged; no new upper-body IK, q only supplies standing legs.
 np.savez_compressed(c/'grasp_reference.npz',**ref);np.savez_compressed(c/'initial_state.npz',**init)
 for name in ['deployment_parameters.json','waypoints.json']:shutil.copy2(SOURCE/name,c/name)
 z=arrays(SOURCE/'grasp_tokens.npz');n=len(t);obs=np.zeros((n,1762),np.float32);obs[:,0]=1;legs=[0,3,6,9,13,17,1,4,7,10,14,18];future=z['future_indices']
 obs[:,661:781]=ref['g1_joint_position_rad'][:,legs][future].reshape(n,120);obs[:,781:901]=ref['g1_joint_velocity_rad_s'][:,legs][future].reshape(n,120);obs[:,595:601]=np.eye(3)[:,:2].reshape(6)
 root=R.from_quat(ref['root_quaternion_world_wxyz'][:,[1,2,3,0]]);positions=[];quats=[]
 for name,off in zip(['left_wrist_yaw_link','right_wrist_yaw_link','torso_link'],[[.18,-.025,0],[.18,.025,0],[0,0,.35]]):
  b=ref['body_names'].tolist().index(name);rot=R.from_quat(ref['body_quaternion_world_wxyz'][:,b][:,[1,2,3,0]]);relative=root.inv()*rot
  positions.append(root.inv().apply((ref['body_position_world_m'][:,b]-ref['root_position_world_m'])/SCALE)+relative.apply(np.broadcast_to(off,(n,3))))
  quats.append(relative.as_quat()[:,[3,0,1,2]])
 obs[:,901:910]=np.stack(positions,axis=1).reshape(n,9);obs[:,910:922]=np.stack(quats,axis=1).reshape(n,12)
 sys.path.insert(0,str(W/'g1_closed_loop/.deps'));import onnxruntime as ort
 opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
 net=ort.InferenceSession(str(MODEL/'model_encoder.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
 z['encoder_inputs']=obs;z['sonic_encoded_tokens']=np.concatenate([net.run(None,{net.get_inputs()[0].name:o[None]})[0] for o in obs]);np.savez_compressed(c/'grasp_tokens.npz',**z)
 plan=dict(candidate=c.name,source=str(SOURCE),sample=spec,encoder_mode=1,body_IK=False,online_planner=False,duration_s=n*.02,review_steps=[0,120,190,245,330,410,570],reference_semantics='Cartesian body target poses; arm q remains source bookkeeping, not an IK solution or supplied control reference',hypothesis='Bounded object-centric Cartesian transform preserves successful wrap grasp via teleop encoder.',training_ready=False)
 write(c/'plan.json',plan)
 import subprocess
 # Reuse bilateral hand collision-mesh screen on the transformed Cartesian wrist poses.
 geo=W/'pipelines/sonic_mimicgen_augmentation/geometry_check.py'
 with (c/'geometry.log').open('w') as f:subprocess.run([sys.executable,str(geo),str(c)],stdout=f,stderr=subprocess.STDOUT,check=True)
 write(c/'prepared_files.json',{name:sha(c/name) for name in ['grasp_reference.npz','initial_state.npz','grasp_tokens.npz','geometry_check.json','deployment_parameters.json']})
 write(c/'status.json',dict(status='prepared',task_success=False,training_ready=False));return c

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--dx',type=float,default=0);p.add_argument('--dy',type=float,default=0);p.add_argument('--yaw',type=float,default=0);p.add_argument('--scale',type=float,default=1);a=p.parse_args();print(compile_candidate(a.output,dict(object_delta_m=[a.dx,a.dy,0],grasp_yaw_deg=a.yaw,approach_scale=a.scale)))
