from pathlib import Path
import argparse,json
import h5py,numpy as np
from scipy.spatial.transform import Rotation as R

p=argparse.ArgumentParser();p.add_argument('run',type=Path);r=p.parse_args().run
c=Path(json.loads((r/'token_controller.json').read_text())['tokens_file']).parent
ref=np.load(c/'grasp_reference.npz');checkpoints=[]
with h5py.File(r/'episode_raw.h5') as f:
 n=len(f['observation/timestamp']);o=f['observation'];m=f['next_observation/metrics']
 b=json.loads(f.attrs['body_names']).index('right_wrist_yaw_link');sb=ref['body_names'].tolist().index('right_wrist_yaw_link');off=np.array([.075,.02,0])*.982730067592321
 for i in sorted(set([k for k in [0,150,300,375,420,450,465,480,600,795,n-1] if k<n])):
  rr=R.from_quat(o['body_quaternion_wxyz'][i,b][[1,2,3,0]]);sr=R.from_quat(ref['body_quaternion_world_wxyz'][i,sb][[1,2,3,0]])
  palm=o['body_position_world'][i,b]+rr.apply(off);expected=ref['body_position_world_m'][i,sb]+sr.apply(off);oq=o['object_quaternion_wxyz'][i]
  checkpoints.append(dict(step=i,time_s=i*.02,palm_world_m=palm.tolist(),palm_tracking_error_m=float(np.linalg.norm(palm-expected)),palm_error_xyz_m=(palm-expected).tolist(),wrist_orientation_error_deg=float((sr.inv()*rr).magnitude()*180/np.pi),bottle_tilt_deg=float(np.arccos(np.clip(1-2*(oq[1]**2+oq[2]**2),-1,1))*180/np.pi),hand_force_N=float(o['metrics/object_hand_normal_force_N'][i]),opposed=bool(o['metrics/opposed_digits_contact'][i])))
 q=f['next_observation/object_quaternion_wxyz'][:];tilt=np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1))
 result=dict(checkpoints=checkpoints,final_bottle_tilt_deg=float(tilt[-1]*180/np.pi),max_bottle_tilt_deg=float(tilt.max()*180/np.pi),max_whole_clearance_m=float(m['whole_object_clearance_m'][:].max()),opposed_contact_duration_s=float(np.sum(m['opposed_digits_contact'][:])*.02),peak_hand_force_N=float(m['object_hand_normal_force_N'][:].max()))
forces={};base_samples=[]
for p in sorted(r.glob('observation_[0-9][0-9][0-9].json')):
 d=json.loads(p.read_text())
 if d['step']==0:continue
 for name,x in d['grasp_information']['metrics']['finger_contacts'].items():
  force=x['object_normal_force_N'];forces[name]=max(forces.get(name,0),force)
  if name=='right_base_link' and force>.1:base_samples.append({'step':d['step'],'force_N':force})
result['per_link_object_force_checkpoint_peaks_N']=forces
result['palm_base_contact_checkpoints']=base_samples
result['palm_contact_sampling_note']='Checkpoint samples only; absent base force is not full-rate proof of no palm contact.'
(r/'contact_tracking_analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
