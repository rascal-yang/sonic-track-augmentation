"""Same grip-aware stop rules for baseline and optimized fixed-token trials."""
from pathlib import Path
import argparse,json,time,math
import numpy as np
from scipy.spatial.transform import Rotation as R

p=argparse.ArgumentParser();p.add_argument('run',type=Path);r=p.parse_args().run
reviewed=set();anchor=None;carry_since=None;lost_since=None
while not (r/'result.json').exists():
 files=sorted(r.glob('observation_[0-9][0-9][0-9].json'))
 if not files:time.sleep(.5);continue
 s=json.loads(files[-1].read_text());i=s['observation_id'];dest=r/f'decision_{i:03d}.json'
 if dest.exists():time.sleep(.5);continue
 config=json.loads((r/'token_controller.json').read_text());plan=json.loads((Path(config['tokens_file']).parent/'plan.json').read_text())
 gates=plan.get('review_steps',[0,150,300,375,450,600,795])
 step=s['step'];reason='Execute fixed tokens; grip-aware evaluation';action='continue'
 q=np.array(s['object_quat_wxyz']);tilt=math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1))
 m=s['grasp_information']['metrics'];opposed=bool(m['opposed_digits_contact']);airborne=m['whole_object_clearance_m']>.02 and m['object_table_normal_force_N']<.1
 wr=R.from_quat(np.array(s['wrist_quat_wxyz']['right'])[[1,2,3,0]]);obr=R.from_quat(q[[1,2,3,0]])
 relative_p=wr.inv().apply(np.array(s['object_pos'])-s['palm_positions']['right']);relative_r=wr.inv()*obr
 if airborne and opposed:
  if carry_since is None:carry_since=step
  if step-carry_since>=25 and anchor is None:anchor=(relative_p,relative_r)
  lost_since=None
 elif anchor is not None and not opposed:
  if lost_since is None:lost_since=step
 slip=0 if anchor is None else float(np.linalg.norm(relative_p-anchor[0]))
 twist=0 if anchor is None else float((anchor[1].inv()*relative_r).magnitude())
 if s.get('contact_watch',{}).get('alert'):action='stop';reason='Monitored obstacle contact'
 elif s['robot_root_pos'][2]<.45:action='stop';reason='Pelvis below45cm'
 elif tilt>.9:action='stop';reason='Bottle tilt exceeds0.9rad'
 elif not airborne and not opposed and tilt>.55:action='stop';reason='Unsupported bottle tipping while near table'
 elif anchor is not None and (slip>.035 or twist>.35):action='stop';reason='Object slipped relative to hand beyond35mm/20deg'
 elif lost_since is not None and step-lost_since>=15:action='stop';reason='Opposed grip lost after lift for0.3s'
 elif step>250 and abs(s['robot_root_pos'][0]-.37)>.30:action='stop';reason='Lateral root drift beyond30cm'
 gate=next((g for g in gates if step>=g and g not in reviewed),None)
 info=dict(step=step,tilt_rad=tilt,airborne=airborne,opposed=opposed,relative_hand_displacement_m=slip,relative_hand_rotation_rad=twist,proposed_action=action,reason=reason)
 (r/f'grip_guard_{i:03d}.json').write_text(json.dumps(info,indent=2))
 if action=='stop' or gate is not None:
  marker=r/f'replay_reviewed_{i:03d}.json'
  if not marker.exists():
   (r/f'replay_review_request_{i:03d}.json').write_text(json.dumps(dict(info,gate=gate,observation_id=i)));time.sleep(.5);continue
  if gate is not None:reviewed.add(gate)
 d=dict(observation_id=i,action=action,steps=1 if action=='stop' else 15,goal={},reason=reason,author='Fixed-token supervisor; no action correction',wall_time=time.time())
 temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2));temp.replace(dest)
 print(step,action,reason,flush=True);time.sleep(.2)
