"""Frozen feedback execution of GPT-designed bottle strategy; no model API calls."""
from pathlib import Path
import argparse,json,time,math
B=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run
state={'stage':'initial','stage_start':0,'corrections':0,'closure_boost':False}
def norm(x):return math.sqrt(sum(v*v for v in x))
def review(s,label):
 i=s['observation_id']; marker=r/f'reviewed_{i:03d}.json'
 if not marker.exists():
  req=r/f'review_request_{i:03d}.json'; req.write_text(json.dumps({'observation_id':i,'label':label,'image':str(r/f'observation_{i:03d}.png'),'head':str(r/f'observation_{i:03d}_head.png')}))
  print('VISUAL_REVIEW',r.name,i,label,flush=True)
  while not marker.exists():time.sleep(1)
def decide(s):
 step=s['step'];stage=state['stage'];m=s['grasp_information']['metrics']
 def act(action='continue',steps=100,goal=None,reason='',next_stage=None):
  if next_stage:state.update(stage=next_stage,stage_start=step)
  return dict(action=action,steps=steps,goal=goal or {},reason=reason,selected_part='upper_bottle_body')
 def reach(target,duration,steps,next_stage,action='reach'):
  return act(action,steps,dict(hand='right',target=target,wrist_euler_xyz=[0,0,1.57],duration_steps=duration),reason='Authorized successful side-grasp waypoint; smooth actual-state arm IK',next_stage=next_stage)
 if s.get('contact_watch',{}).get('alert'):
  review(s,'unexpected table contact; abort')
  return act('stop',1,reason='Physics contact latch; abort, count attempt as failed',next_stage='done')
 if s['robot_root_pos'][2]<.45:return act('stop',1,reason='Pelvis too low',next_stage='done')
 if stage=='initial':
  review(s,'initial open hands and scene')
  return act(steps=60,reason='Settle initial natural stance',next_stage='start_nav')
 if stage=='start_nav' and s['phase']=='grasp':
  return reach([.60,-.44,.92],180,180,'side_raise')
 if stage=='start_nav':return act('navigate',100,dict(root_xy=[.37,-.54],yaw=math.pi/2,speed=.18),reason='SONIC V2 physically navigate to fixed station',next_stage='walk')
 if stage=='walk':
  if s['navigation_error_m']<=.05:return act('brake',100,reason='Within 5cm; request standing motion',next_stage='brake')
  if step-state['stage_start']>=1000:return act('stop',1,reason='Navigation timeout',next_stage='done')
  return act(steps=100,reason='Continue measured-root feedback SONIC navigation')
 if stage=='brake':
  slow=norm(s['root_linear_velocity_m_s'][:2])<=.12 and all(norm(x['linear_velocity_m_s'])<=.2 for x in s['feet'].values())
  if slow and s['navigation_error_m']<=.09:
   return act('settle',200,reason='Root/feet slow; blend reference to standing without physical teleport',next_stage='handoff')
  if step-state['stage_start']>=250:return act('stop',1,reason='Braking or docking guard failed',next_stage='done')
  return act(steps=50,reason='Wait for measured feet and pelvis to settle')
 if stage=='handoff':
  if s['phase']!='grasp':return act(steps=50,reason='Complete 3s handoff and 1s standing')
  review(s,'docked stance before side raise')
  if s['navigation_error_m']>.09:return act('stop',1,reason='Final stance outside 9cm limit',next_stage='done')
  return reach([.60,-.44,.92],180,180,'side_raise')
 if stage=='side_raise':
  review(s,'raised hand outside front desk edge')
  return reach([.415,-.19,.86],220,140,'approach')
 if stage=='approach':return act(steps=90,reason='Finish smooth approach and settle',next_stage='preclose')
 if stage=='preclose':
  review(s,'inspect pinch alignment before closing')
  # Same small feedback correction rule for every start. Avoid changing pose in contact.
  q=s['object_quat_wxyz'];w,x,y,z=q
  axis=[2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)]
  target=[s['object_pos'][i]+.12*axis[i] for i in range(3)]
  pinch=s['protrusion_geometry']['pinch_center_world_m'];delta=[target[i]-pinch[i] for i in range(3)]
  if norm(delta)>.015 and m['object_hand_normal_force_N']<.5 and state['corrections']<2:
   state['corrections']+=1
   old=s['active_goal']['target'];new=[old[i]+max(-.01,min(.01,delta[i])) for i in range(3)]
   return reach(new,100,120,'preclose')
  return act('close_hand',80,dict(closure=.35,closure_steps=80,finger_stiffness=4),reason='Same .35 closure as successful bottle side grasp',next_stage='closed')
 if stage=='closed':
  review(s,'opposed digit contacts before lift')
  if not m['opposed_digits_contact']:
   if not state['closure_boost']:
    state['closure_boost']=True
    return act('close_hand',80,dict(closure=.45,closure_steps=80,finger_stiffness=4),reason='Single bounded closure increase for missing opposition')
   return act('stop',1,reason='No opposed grip after allowed closure increase',next_stage='done')
  target=list(s['active_goal']['target']);target[2]+=.15
  return reach(target,180,90,'lifting','lift')
 if stage=='lifting':return act(steps=120,reason='Finish lift; contact guard remains active',next_stage='lifted')
 if stage=='lifted':
  review(s,'whole bottle clearance and grip after lift')
  return act(steps=200,reason='Four-second final hold to assess stable lift',next_stage='held')
 if stage=='held':
  review(s,'final hold')
  return act('stop',1,reason='Protocol complete; classify all trials from recorded physical data',next_stage='done')
 raise RuntimeError(state)
while True:
 fs=sorted(r.glob('observation_[0-9][0-9][0-9].json'))
 if fs:
  s=json.loads(fs[-1].read_text());i=s['observation_id'];out=r/f'decision_{i:03d}.json'
  if not out.exists():
   d=decide(s);d.update(observation_id=i,author='Frozen GPT-designed feedback supervisor; no GPT inference in this decision',wall_time=time.time())
   tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(out)
   (r/'supervisor_state.json').write_text(json.dumps(state,indent=2))
   print(r.name,'step',s['step'],'phase',s['phase'],'nav',round(s['navigation_error_m'],4),'->',d['action'],state['stage'],flush=True)
   if d['action']=='stop':break
 if (r/'result.json').exists():break
 time.sleep(.5)
