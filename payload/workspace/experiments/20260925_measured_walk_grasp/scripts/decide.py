"""Write one explicit decision supplied by the current GPT session, atomically."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('action');p.add_argument('--target',nargs=3,type=float);p.add_argument('--root',nargs=2,type=float);p.add_argument('--euler',nargs=3,type=float,default=[-1.5707963,0,1.5707963]);p.add_argument('--closure',type=float);p.add_argument('--steps',type=int,default=100);p.add_argument('--duration',type=int,default=95);p.add_argument('--fold-outer',action='store_true');p.add_argument('--opposition',type=float);p.add_argument('--controller');p.add_argument('--speed',type=float,default=.18);p.add_argument('--part',default='near_leg');p.add_argument('--reason',required=True);a=p.parse_args()
if a.fold_outer and a.action!='close_hand':p.error('--fold-outer is only valid during close_hand, not initial navigation or approach')
f=sorted(a.run.glob('observation_[0-9][0-9][0-9].json'))[-1];d=json.loads(f.read_text());idx=d['observation_id'];goal={}
if a.target:goal.update(hand='right',target=a.target,wrist_euler_xyz=a.euler,duration_steps=a.duration)
if a.fold_outer:goal['finger_closures']={'middle':.95,'ring':.95,'pinky':.95}
if a.opposition is not None:goal['thumb_opposition']=a.opposition
if a.controller:goal['controller']=a.controller
if a.root:goal.update(root_xy=a.root,yaw=1.57079632679,speed=a.speed)
if a.closure is not None:goal.update(closure=a.closure,closure_steps=min(a.duration,80),finger_stiffness=4)
x={'observation_id':idx,'action':a.action,'steps':a.steps,'goal':goal,'selected_part':a.part,'reason':a.reason,'author':'current GPT/Codex session; live image + proprioception + privileged part distances and contacts','wall_time':time.time()}
out=a.run/f'decision_{idx:03d}.json';assert not out.exists();temp=out.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2));temp.replace(out);print(out)
