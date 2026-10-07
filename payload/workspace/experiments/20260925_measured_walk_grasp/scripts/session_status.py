from pathlib import Path
import json,sys
B=Path(__file__).resolve().parents[1];r=B/'runs'/sys.argv[1]
fs=sorted(r.glob('observation_[0-9][0-9][0-9].json'))
if not fs: print('Loading:',(r/'run.log').read_text()[-800:] if (r/'run.log').exists() else 'no log');raise SystemExit
s=json.loads(fs[-1].read_text());m=s['grasp_information']['metrics']
d={k:s.get(k) for k in ['observation_id','step','phase','robot_root_pos','object_pos','object_quat_wxyz','palm_positions','navigation_error_m','goal_error_m','root_linear_velocity_m_s','active_goal']}
d['metrics']={k:m[k] for k in ['whole_object_clearance_m','object_hand_normal_force_N','object_table_normal_force_N','thumb_contact_N','index_middle_contact_N','opposed_digits_contact','strict_lift_now']};d['contact_alert']=s['contact_watch']['alert'];d['contact_peak']=s['contact_watch']['peak_total_table_force_N'];d['gate']=None
for p in sorted(r.glob('review_request_*.json')):
 if not p.with_name(p.name.replace('review_request_','reviewed_')).exists():d['gate']=json.loads(p.read_text())
if '--brief' in sys.argv:
 d={k:d[k] for k in ['observation_id','step','phase','navigation_error_m','goal_error_m','metrics','contact_peak','gate']}
print(json.dumps(d,indent=2))
