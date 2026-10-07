"""Read-only summary for the current conversation's next decision."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
run=Path(sys.argv[1]);files=sorted(run.glob('observation_[0-9][0-9][0-9].json'))
if not files: print('initializing');raise SystemExit
s=json.loads(files[-1].read_text());g=s['protrusion_geometry'];m=s['grasp_information']['metrics'];q=np.asarray(s['wrist_quat_wxyz']['right'])
r={k:s.get(k) for k in ['observation_id','step','robot_root_pos','palm_positions','goal_error_m','navigation_error_m','phase','root_linear_velocity_m_s','feet','grail_phase_gain','planner_calls']}
r['right_euler']=Rotation.from_quat(q[[1,2,3,0]]).as_euler('xyz').tolist();r['pinch']=g['pinch_center_world_m'];r['pinch_offset']=g['pinch_offset_wrist_m']
r['parts']={n:{k:v[k] for k in ['center_world_m','pinch_to_center_m','target_minus_pinch_m']} for n,v in g['parts'].items() if n in ['body','neck','cap']}
r['metrics']={k:m[k] for k in ['whole_object_clearance_m','object_hand_normal_force_N','object_table_normal_force_N','thumb_contact_N','index_middle_contact_N']}
r['contact_watch']=s.get('contact_watch')
if r['contact_watch']:
 r['contact_watch']['current_table_forces_N']={k:v for k,v in r['contact_watch']['current_table_forces_N'].items() if v>.001}
r['closure']=s['grasp_information']['closure_fraction_target'];r['decision_exists']=(run/f'decision_{s["observation_id"]:03d}.json').exists()
print(json.dumps(r,indent=2))
