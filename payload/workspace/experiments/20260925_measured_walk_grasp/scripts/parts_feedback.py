import json,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();fs=sorted(a.run.glob('observation_[0-9][0-9][0-9].json'))
if not fs:print('initializing');raise SystemExit
s=json.loads(fs[-1].read_text());
if a.run.name.startswith('arm_pinch_'):
 import sys
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
 from protrusion_geometry import add_protrusions
 add_protrusions(s)
 (a.run/f'part_geometry_aid_{s["observation_id"]:03d}.json').write_text(json.dumps(s['protrusion_geometry'],indent=2))
g=s['protrusion_geometry'];m=s['grasp_information']['metrics']
print(json.dumps({k:s.get(k) for k in ['observation_id','step','robot_root_pos','palm_positions','active_goal','goal_error_m']},indent=2))
print('pinch',g['pinch_center_world_m'],'offset',g['pinch_offset_wrist_m'])
for n,v in g['parts'].items(): print(n, json.dumps({k:v[k] for k in ['center_world_m','palm_to_center_m','pinch_to_center_m','target_minus_pinch_m','lowest_surface_above_table_m']}))
print('contact',json.dumps({k:m[k] for k in ['whole_object_clearance_m','object_hand_normal_force_N','object_table_normal_force_N','thumb_contact_N','index_middle_contact_N','opposed_digits_contact']}))
print('decision_exists',(a.run/f'decision_{s["observation_id"]:03d}.json').exists())
