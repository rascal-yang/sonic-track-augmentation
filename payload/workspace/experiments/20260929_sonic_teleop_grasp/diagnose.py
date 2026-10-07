"""Compare actual Cartesian tracking with both reference and successful baseline."""
from pathlib import Path
import argparse,json
import h5py,numpy as np
from scipy.spatial.transform import Rotation as R
H=Path(__file__).resolve().parent;W=H.parents[1];p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run.resolve();c=Path(json.loads((r/'token_controller.json').read_text())['tokens_file']).parent;ref=np.load(c/'grasp_reference.npz')
base=W/'experiments/20260929_sonic_grasp_smooth/runs/safe025_001/episode_raw.h5';rows=[]
with h5py.File(r/'episode_raw.h5') as f,h5py.File(base) as b:
 n=len(f['observation/timestamp']);names=json.loads(f.attrs['body_names']);bnames=json.loads(b.attrs['body_names']);targetnames=ref['body_names'].tolist();j=names.index('right_wrist_yaw_link');bj=bnames.index('right_wrist_yaw_link');sj=targetnames.index('right_wrist_yaw_link')
 pos=f['observation/body_position_world'][:,j];bp=b['observation/body_position_world'][:n,bj];target=ref['body_position_world_m'][:n,sj];rot=R.from_quat(f['observation/body_quaternion_wxyz'][:,j][:,[1,2,3,0]]);br=R.from_quat(b['observation/body_quaternion_wxyz'][:n,bj][:,[1,2,3,0]]);tr=R.from_quat(ref['body_quaternion_world_wxyz'][:n,sj][:,[1,2,3,0]])
 for i in sorted(set([0,60,120,150,190,220,245,260,280,300,330,360,410,500,n-1])):
  if i>=n:continue
  rows.append(dict(step=i,time_s=i*.02,wrist_world_m=pos[i].tolist(),reference_error_xyz_m=(pos[i]-target[i]).tolist(),successful_actual_error_xyz_m=(pos[i]-bp[i]).tolist(),reference_orientation_error_deg=float(np.rad2deg((tr[i].inv()*rot[i]).magnitude())),baseline_orientation_error_deg=float(np.rad2deg((br[i].inv()*rot[i]).magnitude())),root_error_xyz_m=(f['observation/root_position'][i]-b['observation/root_position'][i]).tolist(),object_position=f['observation/object_position'][i].tolist(),opposed=bool(f['observation/metrics/opposed_digits_contact'][i])))
 np.savez_compressed(r/'cartesian_tracking.npz',actual_wrist_pos=pos,reference_wrist_pos=target,baseline_wrist_pos=bp,actual_wrist_quat=rot.as_quat(),reference_wrist_quat=tr.as_quat(),baseline_wrist_quat=br.as_quat())
(r/'diagnosis.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
