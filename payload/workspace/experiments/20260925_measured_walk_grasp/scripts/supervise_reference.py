"""Bounded read-only feedback with stop guards; no reference corrections."""
from pathlib import Path
import argparse,json,time,math
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run
reviewed=set()
while not (r/'result.json').exists():
 files=sorted(r.glob('observation_[0-9][0-9][0-9].json'))
 if not files:time.sleep(.5);continue
 s=json.loads(files[-1].read_text());i=s['observation_id'];dest=r/f'decision_{i:03d}.json'
 if dest.exists():time.sleep(.5);continue
 step=s['step'];reason='Follow saved reference with live SONIC feedback';action='continue'
 q=s['object_quat_wxyz'];w,x,y,z=q;tilt=math.acos(max(-1.,min(1.,1-2*(x*x+y*y))))
 if s.get('contact_watch',{}).get('alert'):action='stop';reason='Obstacle contact latch during saved-reference rollout'
 elif s['robot_root_pos'][2]<.45:action='stop';reason='Pelvis below45cm'
 elif tilt>.40:action='stop';reason='Bottle tilt exceeds0.40rad'
 elif step>250 and abs(s['robot_root_pos'][0]-.37)>.30:action='stop';reason='Lateral root drift exceeds30cm'
 gate=next((g for g in [0,825,1250,1350,1500,1700] if step>=g and g not in reviewed),None)
 if action=='stop' or gate is not None:
  marker=r/f'replay_reviewed_{i:03d}.json'
  if not marker.exists():
   (r/f'replay_review_request_{i:03d}.json').write_text(json.dumps({'step':step,'observation_id':i,'reason':reason,'proposed_action':action,'gate':gate}))
   time.sleep(.5);continue
  if gate is not None:reviewed.add(gate)
 d={'observation_id':i,'action':action,'steps':1 if action=='stop' else 25,'goal':{},'reason':reason,'author':'fixed reference supervisor; no trajectory adjustment','wall_time':time.time()}
 temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2));temp.replace(dest)
 print(step,action,reason,flush=True)
 time.sleep(.2)
