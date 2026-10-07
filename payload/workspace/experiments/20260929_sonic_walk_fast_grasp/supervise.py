"""Measured docking/contact gates; checkpoint review never edits latent actions."""
from pathlib import Path
import json,time,argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run
seen=set();last_phase=None;phase_start=0;anchor=None;carry=None
for previous in sorted(r.glob('guard_[0-9][0-9][0-9].json')):
 j=int(previous.stem.split('_')[-1])
 if not (r/f'decision_{j:03d}.json').exists():continue
 old=json.loads(previous.read_text())
 if old['phase']!=last_phase:last_phase=old['phase'];phase_start=old['step']
 if old.get('gate'):seen.add(old['gate'])
while not (r/'result.json').exists():
 files=sorted(r.glob('observation_[0-9][0-9][0-9].json'))
 if not files:time.sleep(.5);continue
 s=json.loads(files[-1].read_text());i=s['observation_id'];dest=r/f'decision_{i:03d}.json'
 if dest.exists():time.sleep(.3);continue
 step=s['step'];phase=s['phase'];plan=json.loads((r/'plan.json').read_text());error=s['navigation_error_m'];gi=s['grasp_token_count'];action='continue';reason='Live planner/encoder/decoder execution';steps=15;gate=None
 if phase!=last_phase:last_phase=phase;phase_start=step
 root=np.array(s['robot_root_pos']);travel=float((root-np.array(plan['start_root_m']))@plan['forward_world']);speed=np.linalg.norm(s['root_linear_velocity_m_s'][:2]);foot_speed=max(np.linalg.norm(f['linear_velocity_m_s']) for f in s['feet'].values());tilt=float(np.arccos(np.clip(R.from_quat(np.array(s['object_quat_wxyz'])[[1,2,3,0]]).as_matrix()[2,2],-1,1)));m=s['grasp_information']['metrics'];opposed=bool(m['opposed_digits_contact']);airborne=m['whole_object_clearance_m']>.02 and m['object_table_normal_force_N']<.1
 slip=twist=0.
 if airborne and opposed:
  wr=R.from_quat(np.array(s['wrist_quat_wxyz']['right'])[[1,2,3,0]]);ob=R.from_quat(np.array(s['object_quat_wxyz'])[[1,2,3,0]]);pos=wr.inv().apply(np.array(s['object_pos'])-s['palm_positions']['right']);rot=wr.inv()*ob
  if carry is None:carry=step
  if step-carry>=25 and anchor is None:anchor=(pos,rot)
  if anchor is not None:slip=float(np.linalg.norm(pos-anchor[0]));twist=float((anchor[1].inv()*rot).magnitude())
 if phase=='initial':
  if step==0:gate='initial'
  elif step>=50:action='navigate';gate='start_walk';reason='Translated rear start with planned cruise; measure actual displacement and docking'
 elif phase=='walking':
  nav_error=s.get('planner_navigation_error_m',error)
  if nav_error<.30:steps=5
  brake_distance=.025 if 'redock' in seen else float(plan.get('brake_distance_m',.075))
  if nav_error<brake_distance:action='brake';gate='arrival_brake';reason='Brake before operation point to allow measured stopping distance; verify actual settling'
  elif step-phase_start>650:action='stop';reason='Walking timeout without measured docking'
  elif step-phase_start>200 and travel<.12:action='stop';reason='Insufficient forward travel despite large-step planner reference'
  else:
   for d in [.3,.6]:
    if travel>=d and f'walk_{d}' not in seen:gate=f'walk_{d}';break
 elif phase=='braking':
  if step-phase_start>=60 and speed<.08 and foot_speed<.15:
   if error<.085:action='handoff';gate='handoff';reason='Measured braking/feet settled, blend reference into fast grasp initial posture'
   elif not plan.get('allow_redock',True):action='stop';reason='Calibrated stop missed operation point; no near-table redocking'
   elif step-phase_start<180:action='navigate';gate='redock';reason='Standing drift: replan before grasp'
  if step-phase_start>200:action='stop';reason='Braking/docking convergence timeout'
 elif phase=='handoff':pass
 elif phase=='settled':
  if step-phase_start>=40 and error<.05 and speed<.06 and foot_speed<.10 and s['stance_joint_rmse_rad']<.16 and s['stance_yaw_error_rad']<.12:
   action='start_grasp';gate='start_grasp';reason='Measured body/feet settled and within5cm; replay original fast grasp tokens'
  elif step-phase_start>175:action='stop';reason='Fast grasp initial-state convergence timeout; no blind closure'
 elif phase=='grasp':
  for g in [120,195,255,330,420,570]:
   if gi>=g and f'grasp_{g}' not in seen:gate=f'grasp_{g}';break
  reason='Original fast grasp tokens, unchanged hands and release decoder'
 if s.get('contact_watch',{}).get('alert'):action='stop';reason='Physics obstacle contact latch'
 elif root[2]<.45:action='stop';reason='Low pelvis guard'
 elif abs(root[0]-plan['goal_root_m'][0])>.35:action='stop';reason='Lateral deviation exceeds35cm'
 elif tilt>.9 or (not airborne and not opposed and tilt>.55):action='stop';reason='Bottle tipping guard'
 elif slip>.035 or twist>.35:action='stop';reason='Object slipped relative to carrying hand'
 info=dict(step=step,phase=phase,travel_m=travel,error_m=error,root_speed_m_s=float(speed),foot_speed_m_s=float(foot_speed),grasp_index=gi,tilt_rad=tilt,slip_m=slip,twist_rad=twist,action=action,reason=reason,gate=gate)
 (r/f'guard_{i:03d}.json').write_text(json.dumps(info,indent=2))
 if gate is not None or action=='stop':
  marker=r/f'reviewed_{i:03d}.json'
  if not marker.exists():(r/f'review_request_{i:03d}.json').write_text(json.dumps(info));time.sleep(.5);continue
  if gate:seen.add(gate)
 goal={'speed':.10} if gate=='redock' else {}
 d=dict(observation_id=i,action=action,steps=1 if action=='stop' else steps,goal=goal,reason=reason,author='Measured phase supervisor; current session reviews checkpoint images',wall_time=time.time());tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(dest);print(json.dumps(info),flush=True);time.sleep(.2)
