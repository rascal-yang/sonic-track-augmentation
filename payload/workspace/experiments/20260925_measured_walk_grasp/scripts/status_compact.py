from pathlib import Path
import json,sys
B=Path(__file__).resolve().parents[1]
for r in sorted((B/'runs').iterdir()):
 if not r.is_dir() or ((r/'result.json').exists() and '--all' not in sys.argv):continue
 fs=sorted(r.glob('observation_[0-9][0-9][0-9].json'))
 if not fs:continue
 s=json.loads(fs[-1].read_text());i=s['observation_id'];lines=(r/'trajectory.jsonl').read_text().splitlines();a=json.loads(lines[-1]) if lines else s;m=a.get('grasp_metrics',s['grasp_information']['metrics'])
 print(r.name,'step',a['step'],'phase',a['phase'],'nav',round(a['navigation_error_m'],3),'palm',[round(x,3) for x in a['palm_positions']['right']],'force',round(m['object_hand_normal_force_N'],2),'opposed',m['opposed_digits_contact'],'lift',round(m['whole_object_clearance_m'],3),'obs',i,'review',(r/f'review_request_{i:03d}.json').exists() and not (r/f'reviewed_{i:03d}.json').exists())
