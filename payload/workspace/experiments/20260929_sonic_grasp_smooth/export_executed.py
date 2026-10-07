"""Named actual-state and actuator streams; never substitute references for labels."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,h5py
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run
q=json.loads((r/'quality.json').read_text());assert q['task_success']
with h5py.File(r/'episode_raw.h5') as f:
 n=len(f['observation/timestamp']);data=dict(control_index=np.arange(n),time_s=np.arange(n)*.02,joint_names=np.array(json.loads(f.attrs['joint_names'])),joint_position_rad=f['observation/joint_position'][:],joint_velocity_rad_s=f['observation/joint_velocity'][:],joint_pd_target_rad=f['action/joint_position_target'][:],root_position_world_m=f['observation/root_position'][:],root_quaternion_world_wxyz=f['observation/root_quaternion_wxyz'][:])
 for side in ['left','right']:
  data[f'revo2_{side}_joint_names']=np.array(json.loads(f.attrs[f'brainco_{side}_joint_names']))
  for dest,src in [('joint_position_rad','joint_position'),('joint_velocity_rad_s','joint_velocity'),('position_target_rad','position_target')]:
   data[f'revo2_{side}_{dest}']=f[f'brainco/{side}/{src}'][:]
 c=json.loads((r/'token_controller.json').read_text())
 with np.load(c['tokens_file']) as z:data['sonic_encoded_tokens']=z['sonic_encoded_tokens'].copy()
np.savez_compressed(r/'executed_trajectory_named.npz',**data)
(r/'executed_trajectory_schema.json').write_text(json.dumps(dict(rows=n,frequency_Hz=50,body_joint_count=len(data['joint_names']),notes=['Pre-action actual state paired with decoder actuator PD target and separately commanded Revo2 targets.','43 body articulation columns include14 legacy compatibility joints; use joint names, not raw indices.','SONIC release quantized latent is64float32 values; hand targets are separate.','Simulation evidence, not hardware replay certification.'],sha256=hashlib.sha256((r/'executed_trajectory_named.npz').read_bytes()).hexdigest()),indent=2)+'\n')
print(r/'executed_trajectory_named.npz')
