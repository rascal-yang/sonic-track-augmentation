"""Compare measured motions; transient residual is a jitter proxy, not causal proof."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
out={}
for arg in sys.argv[1:]:
 r=Path(arg);rows=[json.loads(x) for x in (r/'trajectory.jsonl').read_text().splitlines()]
 t=np.array([x['sim_time'] for x in rows]);p=np.array([x['palm_positions']['right'] for x in rows]);q=np.array([x['wrist_quat_wxyz']['right'] for x in rows]);root=np.array([x['robot_root_pos'] for x in rows]);rot=Rotation.from_quat(q[:,[1,2,3,0]])
 # Same 0.50 s local polynomial trend for both runs; intentional rapid motion also contributes.
 resid=p-savgol_filter(p,25,3,axis=0)
 turn=np.rad2deg((rot[:-10].inv()*rot[10:]).magnitude())
 mask=t>=1
 hold=t>=t[-1]-2
 record=dict(duration_s=float(t[-1]),palm_high_frequency_residual_RMS_mm=float(1000*np.sqrt(np.mean(np.sum(resid[mask]**2,axis=1)))),palm_speed_p95_m_s=float(np.percentile(np.linalg.norm(np.diff(p,axis=0)/.02,axis=1),95)),wrist_turn_0p2s_max_deg=float(turn.max()),wrist_turn_0p2s_p95_deg=float(np.percentile(turn,95)),final_hold_palm_range_xyz_mm=(1000*np.ptp(p[hold],axis=0)).tolist(),final_hold_root_range_xyz_mm=(1000*np.ptp(root[hold],axis=0)).tolist(),final_hold_residual_RMS_mm=float(1000*np.sqrt(np.mean(np.sum(resid[hold]**2,axis=1)))))
 for f in ['validation.json','contact_validation.json']:
  if (r/f).exists():record[f]=json.loads((r/f).read_text())
 out[str(r.resolve())]=record
print(json.dumps(out,indent=2))
