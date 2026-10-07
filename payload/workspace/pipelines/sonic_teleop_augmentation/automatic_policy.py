"""In-process numeric guard; no external file polling or fabricated human review."""
import os,sys,types
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from common import read,write
class AutoGraspPolicy:
 def __init__(self):self.carry=None;self.anchor=None;self.lost=None
 def decide(self,s):
  step=s['step'];m=s['grasp_information']['metrics'];root=np.array(s['robot_root_pos']);q=np.array(s['object_quat_wxyz']);tilt=float(np.arccos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
  opposed=bool(m['opposed_digits_contact']);airborne=m['whole_object_clearance_m']>.02 and m['object_table_normal_force_N']<.1
  wr=R.from_quat(np.array(s['wrist_quat_wxyz']['right'])[[1,2,3,0]]);p=wr.inv().apply(np.array(s['object_pos'])-s['palm_positions']['right']);rot=wr.inv()*R.from_quat(q[[1,2,3,0]])
  if airborne and opposed:
   if self.carry is None:self.carry=step
   if step-self.carry>=25 and self.anchor is None:self.anchor=(p,rot)
   self.lost=None
  elif self.anchor is not None and not opposed:
   if self.lost is None:self.lost=step
  slip=0 if self.anchor is None else float(np.linalg.norm(p-self.anchor[0]));twist=0 if self.anchor is None else float((self.anchor[1].inv()*rot).magnitude())
  reason='Execute transformed Cartesian reference with live teleop encoder';action='continue'
  if s.get('contact_watch',{}).get('alert'):action='stop';reason='Physical obstacle contact latch'
  elif root[2]<.45:action='stop';reason='Low pelvis'
  elif tilt>.9 or (not airborne and not opposed and tilt>.55):action='stop';reason='Bottle tipping'
  elif slip>.035 or twist>.35:action='stop';reason='Carried bottle slip'
  elif self.lost is not None and step-self.lost>=15:action='stop';reason='Opposed grip lost'
  elif abs(root[0]-.37)>.30:action='stop';reason='Lateral root drift'
  if not os.environ.get('G1_LEAN_MODE'):
   out=Path(os.environ['G1_OUTPUT_DIR']);write(out/f'guard_{s["observation_id"]:03d}.json',dict(step=step,action=action,reason=reason,tilt_rad=tilt,slip_m=slip,twist_rad=twist,opposed=opposed,airborne=airborne))
  return dict(action=action,steps=1 if action=='stop' else 5,goal={},reason=reason)
def install():
 m=types.ModuleType('auto_policy');m.AutoGraspPolicy=AutoGraspPolicy;sys.modules['auto_policy']=m
