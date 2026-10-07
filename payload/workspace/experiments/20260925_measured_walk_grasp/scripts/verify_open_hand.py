"""Fixed physical acceptance sequence: walk open, approach open, fold, release."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
pose={'hand':'right','target':[.08,-.48,1.0],'wrist_euler_xyz':[-1.5707963,0,1.5707963],'duration_steps':95}
sequence=[
 ('navigate',200,{'root_xy':[-.1,-.6],'yaw':1.57079632679,'speed':.18},'Verify five-finger open pose while walking.'),
 ('reach',100,dict(pose),'Verify open approach in free space, away from toy.'),
 ('close_hand',50,dict(pose,duration_steps=45,closure=0.,closure_steps=40,finger_closures={'middle':.95,'ring':.95,'pinky':.95}),'Deliberately fold three fingers ONLY for release regression check.'),
 ('open_hand',100,dict(pose,closure_steps=70),'Verify open_hand clears all individual finger overrides.'),
 ('continue',50,{},'Verify all fingers remain open after release.')]
a.run.mkdir(exist_ok=True,parents=True)
(a.run/'acceptance_protocol.json').write_text(json.dumps({'purpose':'Hand posture verification, not object pickup or training demonstration','decision_source':'fixed automated regression sequence, no online model decisions','sequence':sequence},indent=2))
for idx,(action,steps,goal,reason) in enumerate(sequence):
 deadline=time.monotonic()+600
 path=a.run/f'observation_{idx:03d}.json'
 while not path.exists():
  if (a.run/'result.json').exists():raise RuntimeError('Simulation ended before acceptance sequence completed')
  if time.monotonic()>deadline:raise TimeoutError(path)
  time.sleep(.25)
 d=json.loads(path.read_text());h=d['brainco']['right'];q=dict(zip(h['joint_names'],h['joint_position']))
 print(json.dumps({'step':d['step'],'observation':idx,'flexion_rad':{n:round(v,5) for n,v in q.items() if 'proximal' in n},'overrides':d['grasp_information']['finger_closures_override']}),flush=True)
 out=a.run/f'decision_{idx:03d}.json';assert not out.exists()
 decision={'observation_id':idx,'action':action,'steps':steps,'goal':goal,'reason':reason,'author':'fixed physical hand-posture acceptance protocol, not a GPT grasp attempt','wall_time':time.time()}
 temp=out.with_suffix('.tmp');temp.write_text(json.dumps(decision,indent=2));temp.replace(out)
