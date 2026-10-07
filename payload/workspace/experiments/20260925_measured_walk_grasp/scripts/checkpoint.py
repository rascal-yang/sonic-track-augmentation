"""Wait at most 50 s for an undecided observation, then print a compact summary."""
from pathlib import Path
import sys,json,time
r=Path(sys.argv[1])
for _ in range(50):
 fs=sorted(r.glob('observation_[0-9][0-9][0-9].json'))
 if fs:
  s=json.loads(fs[-1].read_text());i=s['observation_id']
  if not (r/f'decision_{i:03}.json').exists():break
 time.sleep(1)
else:print('RUNNING');raise SystemExit
m=s['grasp_information']['metrics'];g=s['protrusion_geometry']
print(json.dumps({k:s.get(k) for k in ['observation_id','step','phase','navigation_error_m','grail_phase_gain','planner_calls','robot_root_pos','palm_positions','wrist_quat_wxyz']},indent=2))
print('pinch',g['pinch_center_world_m'],'parts',{k:v['target_minus_pinch_m'] for k,v in g['parts'].items()});print('metrics',{k:v for k,v in m.items() if k!='finger_contacts'});w=s.get('contact_watch',{}); w['current_table_forces_N']={k:v for k,v in w.get('current_table_forces_N',{}).items() if v>.001}; print('watch',w)
