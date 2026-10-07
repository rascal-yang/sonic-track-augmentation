from pathlib import Path
import json,numpy as np,h5py
from scipy.signal import savgol_filter
root=Path(__file__).resolve().parent;output={}
for r in sorted((root/'runs').iterdir()):
 if not (r/'quality.json').exists():continue
 controller=json.loads((r/'token_controller.json').read_text());c=Path(controller['tokens_file']).parent
 wp=c/'waypoints.json';ts=json.loads(wp.read_text())['time_s'][:6] if wp.exists() else [0,3.4,5.5,7.5,9.2,11.2]
 with h5py.File(r/'episode_raw.h5') as f:
  body=json.loads(f.attrs['body_names']).index('right_wrist_yaw_link');pos=f['observation/body_position_world'][:round(ts[-1]/.02)+1,body];v=savgol_filter(pos,11,3,deriv=1,delta=.02,axis=0);a=savgol_filter(pos,11,3,deriv=2,delta=.02,axis=0);j=savgol_filter(pos,11,3,deriv=3,delta=.02,axis=0)
  output[r.name]=[]
  for phase,start,end in zip(['raise','above_edge','pregrasp','closure','lift'],ts[:-1],ts[1:]):
   sl=slice(round(start/.02),round(end/.02));output[r.name].append(dict(phase=phase,start_s=start,end_s=end,acceleration_rms_m_s2=float(np.sqrt(np.mean(np.sum(a[sl]**2,axis=1)))),jerk_rms_m_s3=float(np.sqrt(np.mean(np.sum(j[sl]**2,axis=1)))),speed_peak_m_s=float(np.linalg.norm(v[sl],axis=1).max())))
(root/'stage_motion_metrics.json').write_text(json.dumps(output,indent=2)+'\n')
