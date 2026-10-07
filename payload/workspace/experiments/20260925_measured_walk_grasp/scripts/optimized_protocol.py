"""Feedback supervisor with visible agent review gates and bounded physical retries.

No LLM API calls. Decisions between agent review gates are deterministic.
No robot/object state resets after starting the episode.
"""
from pathlib import Path
import argparse,json,time,math
import numpy as np
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--resume',action='store_true');a=p.parse_args();r=a.run
cfg=json.loads((r/'optimized_config.json').read_text())
state={'stage':'initial','stage_start':0,'retries':0,'corrections':0,'closure_boost':False,'hold_start':None}
if a.resume:state.update(json.loads((r/'supervisor_state.json').read_text()))

def norm(x):return float(np.linalg.norm(x))
def review(s,label):
 i=s['observation_id'];marker=r/f'reviewed_{i:03d}.json'
 if not marker.exists():
  (r/f'review_request_{i:03d}.json').write_text(json.dumps({'observation_id':i,'label':label,'image':str(r/f'observation_{i:03d}.png'),'head':str(r/f'observation_{i:03d}_head.png')}))
  print('VISUAL_REVIEW',r.name,i,label,flush=True)
  while not marker.exists():time.sleep(.5)

def settled_wrist(s):
 rows=[]
 for line in (r/'trajectory.jsonl').read_text().splitlines():
  try: row=json.loads(line)
  except json.JSONDecodeError: continue
  if s['step']-25<=row['step']<=s['step']: rows.append(row)
 by_step={x['step']:x for x in rows};rows=[by_step[k] for k in sorted(by_step)]
 if len(rows)<25:return False
 ps=np.array([x['palm_positions']['right'] for x in rows]);qs=np.array([x['wrist_quat_wxyz']['right'] for x in rows]);times=np.array([x['step']*.02 for x in rows]);dt=np.diff(times)
 speed=np.linalg.norm(np.diff(ps,axis=0),axis=1)/dt;rr=Rotation.from_quat(qs[:,[1,2,3,0]]);angular=(rr[:-1].inv()*rr[1:]).magnitude()/dt
 report={'step':s['step'],'window_s':float(times[-1]-times[0]),'palm_speed_max_m_s':float(speed.max()),'wrist_angular_speed_max_rad_s':float(angular.max()),'palm_position_error_m':s['goal_error_m']}
 report['passed']=bool(report['window_s']>=.48 and speed.max()<.035 and angular.max()<.35 and s['goal_error_m']<.035)
 with (r/'closure_convergence.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
 return report['passed']

def decide(s):
 step=s['step'];stage=state['stage'];m=s['grasp_information']['metrics'];obj=np.array(s['object_pos']);palm=np.array(s['palm_positions']['right'])
 q=s['object_quat_wxyz'];w,x,y,z=q;axis=np.array([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)]);tilt=math.acos(np.clip(axis[2],-1,1))
 def act(action='continue',steps=50,goal=None,reason='',next_stage=None):
  if next_stage:state.update(stage=next_stage,stage_start=step)
  return dict(action=action,steps=steps,goal=goal or {},reason=reason,selected_part='bottle_mid_body',attempt_index=state['retries']+1)
 def reach(target,duration,next_stage,action='reach',speed=.8,steps=None,profile='quintic',arc=None):
  return act(action, min(duration,100) if steps is None else steps,dict(hand='right',target=list(target),wrist_euler_xyz=[0,0,1.57],duration_steps=duration,arm_speed_limit_rad_s=speed,progress_profile=profile,clearance_arc_m=arc or [0,0,0],orientation_finish_fraction=.4 if arc else 1.),reason='Quintic task-space reference; smooth task-space timing and bounded arm velocity; live SONIC balance',next_stage=next_stage)
 def body_goal():
  goal=obj+np.array(cfg['palm_offset_from_bottle_m'])
  if state['retries']==0:goal+=np.array(cfg.get('first_attempt_target_offset_m',[0,0,0]))
  goal[2]+=cfg['retry_height_offsets_m'][state['retries']]
  return goal
 def retry(reason):
  # Never open an unsupported load or chase a fallen bottle around the desk.
  if state['retries']>=cfg['max_retries'] or tilt>.40 or m['object_table_normal_force_N']<.5:
   return act('stop',1,reason='Recovery guard ends trial: '+reason,next_stage='done')
  state['retries']+=1;state['corrections']=0;state['closure_boost']=False
  with (r/'retry_events.jsonl').open('a') as f:f.write(json.dumps({'step':step,'reason':reason,'attempt':state['retries']+1,'object_position':obj.tolist(),'object_tilt_rad':tilt,'state_reset':False})+'\n')
  return act('open_hand',60,dict(closure=0.,closure_steps=60),reason='Supported bottle: reopen all digits, then retreat and re-estimate grasp',next_stage='retry_open')
 if s.get('contact_watch',{}).get('alert'):
  review(s,'unexpected obstacle contact: stop and retain failure')
  return act('stop',1,reason='200 Hz obstacle contact latch',next_stage='done')
 if s['robot_root_pos'][2]<.45:return act('stop',1,reason='Low pelvis guard',next_stage='done')
 if stage=='initial':
  review(s,'initial asset appearance, open hands, standing clearance')
  return act(steps=60,reason='Natural stance settle',next_stage='start_nav')
 if stage=='start_nav':
  if s['phase']=='grasp':
   if cfg.get('continuous_arc'):return reach(body_goal(),cfg['approach_steps'],'contact_wait',steps=25,profile='early_deceleration',arc=cfg['clearance_arc_m'])
   return reach(cfg['side_raise_m'],cfg['raise_steps'],'raise_wait')
  return act('navigate',25,dict(root_xy=cfg['stand_root_m'][:2],yaw=math.pi/2,speed=.15),reason='Live SONIC planner navigates from measured root',next_stage='walk')
 if stage=='walk':
  if s['navigation_error_m']<=cfg.get('docking_radius_m',.025):return act('brake',75,reason='Enter station radius; request standing motion',next_stage='brake')
  if step-state['stage_start']>=1000:return act('stop',1,reason='Navigation timeout',next_stage='done')
  return act(steps=10,reason='Continue feedback walking; fine near-table observation')
 if stage=='brake':
  slow=norm(s['root_linear_velocity_m_s'][:2])<=.10 and all(norm(x['linear_velocity_m_s'])<=.18 for x in s['feet'].values())
  if slow and s['navigation_error_m']<=.09:return act('settle',150,reason='Measured root and feet settled; continuous reference handoff',next_stage='handoff')
  if step-state['stage_start']>=300:return act('stop',1,reason='Docking guard timeout',next_stage='done')
  return act(steps=50,reason='Wait for stable feet')
 if stage=='handoff':
  if s['phase']!='grasp':return act(steps=50,reason='Complete smooth standing handoff')
  review(s,'docked stance and table clearance')
  if s['navigation_error_m']>.09:return act('stop',1,reason='Docking beyond 9 cm',next_stage='done')
  if cfg.get('continuous_arc'):return reach(body_goal(),cfg['approach_steps'],'contact_wait',steps=25,profile='early_deceleration',arc=cfg['clearance_arc_m'])
  return reach(cfg['side_raise_m'],cfg['raise_steps'],'raise_wait')
 if stage=='raise_wait':
  if step-state['stage_start']<cfg['raise_steps']+20:return act(steps=50,reason='Complete safe side raise')
  review(s,'hand above desk, inspect swept approach')
  if norm(palm-np.array(cfg['side_raise_m']))>.04:return act('stop',1,reason='Side-raise tracking error >4cm; reject blind approach',next_stage='done')
  return reach(body_goal(),cfg['approach_steps'],'contact_wait',steps=25,profile='early_deceleration')
 if stage=='approach_wait':
  if step-state['stage_start']<cfg['approach_steps']+15:return act(steps=50,reason='Finish free-space approach')
  # Small terminal segment with slower arm limit and short feedback intervals.
  return reach(body_goal(),cfg['contact_steps'],'contact_wait',speed=.45,steps=25)
 if stage=='contact_wait':
  if cfg.get('continuous_arc') and step-state['stage_start']>=100 and not state.get('arc_reviewed'):
   review(s,'continuous arc mid-approach: whole hand above tabletop');state['arc_reviewed']=True
  if tilt>.40:return retry('Object tilted during approach')
  if step-state['stage_start']<cfg['approach_steps']+20:return act(steps=25,reason='Slow contact approach; 0.5 s feedback')
  if not settled_wrist(s):
   if step-state['stage_start']>=cfg['approach_steps']+300:return act('stop',1,reason='Actual wrist did not settle within bounded preclose dwell',next_stage='done')
   return act(steps=10,reason='Wait for >=0.48s measured palm speed<.035m/s and wrist angular speed<.35rad/s; preserve pose reference')
  review(s,'preclose pose, actual pinch alignment, bottle displacement')
  if norm(palm-np.array(s['active_goal']['target']))>.035:
   if state['corrections']<1:
    state['corrections']+=1
    return act(steps=50,reason='Wait for measured palm to track target before closure')
   return retry('Palm failed to track target')
  closure=cfg.get('first_closure',.35) if state['retries']==0 else .40
  return act('close_hand',10,dict(closure=closure,closure_steps=60,finger_stiffness=4),reason='Slow finger closure while preserving wrist reference',next_stage='closed')
 if stage=='closed':
  if tilt>.35:
   review(s,'closure tilt guard: stop, old upright contact target invalid')
   return act('stop',1,reason='Bottle tilted during closure; no blind closure boost/lift',next_stage='done')
  if step-state['stage_start']<60:return act(steps=10,reason='Fine feedback during slow closure; monitor object tilt and opposed contacts')
  review(s,'check opposed contacts before lifting')
  if not m['opposed_digits_contact']:
   if not state['closure_boost']:
    state['closure_boost']=True
    return act('close_hand',10,dict(closure=.45,closure_steps=60,finger_stiffness=4),reason='Bounded closure boost; unchanged wrist pose',next_stage='closed')
   return retry('Missing opposed thumb/finger grip')
  target=np.array(s['active_goal']['target']);target[2]+=.14
  return reach(target,cfg['lift_steps'],'lift_wait',action='lift',speed=.55,steps=25)
 if stage=='lift_wait':
  if step-state['stage_start']<cfg['lift_steps']+20:return act(steps=25,reason='Slow lift; monitor slip and contacts')
  review(s,'whole bottle clearance after lift')
  if not m['strict_lift_now']:return retry('Lift did not clear table')
  state['hold_start']=step
  return act(steps=50,reason='Verify stable opposed hold',next_stage='hold')
 if stage=='hold':
  if not m['strict_lift_now']:return retry('Slip during hold')
  if step-state['hold_start']<175:return act(steps=50,reason='Continue stable hold validation')
  review(s,'final successful hold or final failure')
  return act('stop',1,reason='Hold complete; classify from recorded physical data',next_stage='done')
 if stage=='retry_open':
  target=palm+np.array([.07,-.06,.025])
  return reach(target,100,'retry_retreat',speed=.45,steps=50)
 if stage=='retry_retreat':
  if step-state['stage_start']<120:return act(steps=50,reason='Finish physical retreat; object remains live')
  review(s,'recovery retreat and new bottle pose')
  if tilt>.40:return act('stop',1,reason='Bottle no longer upright',next_stage='done')
  return reach(body_goal(),cfg['approach_steps'],'contact_wait',speed=.55,steps=25,profile='early_deceleration')
 raise RuntimeError(state)

while True:
 fs=sorted(r.glob('observation_[0-9][0-9][0-9].json'))
 if fs:
  s=json.loads(fs[-1].read_text());i=s['observation_id'];out=r/f'decision_{i:03d}.json'
  if not out.exists():
   d=decide(s);d.update(observation_id=i,author='Deterministic feedback supervisor authored by GPT/Codex; visual review by session at marked gates; no model API',wall_time=time.time())
   tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(out)
   (r/'supervisor_state.json').write_text(json.dumps(state,indent=2))
   print(r.name,'step',s['step'],'->',d['action'],state['stage'],'attempt',state['retries']+1,flush=True)
   if d['action']=='stop':break
 if (r/'result.json').exists():break
 time.sleep(.3)
