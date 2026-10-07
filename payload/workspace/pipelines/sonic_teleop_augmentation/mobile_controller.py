"""Unified release decoder: online planner/encoder locomotion then live Cartesian teleop grasp."""
from pathlib import Path
from collections import deque
import json,hashlib,sys,shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from common import W, SCALE
H=Path(os.environ['G1_AUG_CANDIDATE']);S=H/'skill'

def install():
 sys.path.insert(0,str(W/'g1_closed_loop/.deps'))
 import onnxruntime as ort
 import walk_grasp,brainco_adapter
 from direct_bottle import DirectBottlePlanner
 from sonic_planner_goals import SonicPlannerIK
 from calibration_config import ROBOT_SCALE
 from experiment import write_json
 plan=json.loads((H/'plan.json').read_text());nav_goal=plan.get('navigation_target_root_m',plan['goal_root_m']);z=np.load(S/'grasp_tokens.npz');ref=np.load(S/'grasp_reference.npz');init=np.load(H/'initial_state.npz');names=z['encoder_joint_names'].tolist()
 opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
 model=Path('__RUNTIME__/models/release')
 enc=ort.InferenceSession(str(model/'model_encoder.onnx'),sess_options=opts,providers=['CPUExecutionProvider']);dec=ort.InferenceSession(str(model/'model_decoder.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
 params=json.loads((H/'deployment_parameters.json').read_text());order=np.array(params['mujoco_to_isaaclab']);default=np.array(params['default_angles'])[order];scale=np.array(params['action_scale'])[order];kp=np.array(params['kp'])[order];kd=np.array(params['kd'])[order]
 class HybridPlanner(DirectBottlePlanner):
  def __init__(self,env,out,sonic):
   super().__init__(env,out,sonic);self.nav_planner=SonicPlannerIK(env,out,sonic);self.nav_planner.seed=int(plan['planner_seed']);self.phase='initial';self.handoff_end=None;self.grasp_start=None;self.grasp_count=0
   self.qstart=ref['g1_joint_position_rad'][0,[ref['g1_joint_names'].tolist().index(n) for n in self.kin.names]]
   count=len(self.lib.dof_pos);ps=np.tile(init['root_position'],(count,1));qs=np.tile(self.qstart,(count,1));rs=np.tile(init['root_quaternion_wxyz'],(count,1));self.write(qs,ps,rs,0)
   raw=np.c_[ps,rs,qs];raw[:,2]/=ROBOT_SCALE;self.nav_planner.raw_motion=raw
   assert [self.kin.names[i] for i in self.lib.m_cfg.mujoco_to_isaaclab_dof]==names
  def frame(self):return int((self.cmd.motion_start_time_steps+self.cmd.time_steps)[0])
  def write(self,qs,ps,rs,start):
   pose=np.zeros((len(qs),30,3),np.float32);pose[:,0]=R.from_quat(rs[:,[1,2,3,0]]).as_rotvec();pose[:,1:]=qs[:,:,None]*self.kin.axes;self.write_reference(start,qs,ps,pose)
  def observation(self):
   s=super().observation();robot=self.env.scene['robot'];root=robot.data.root_pos_w[0].cpu().numpy();q=robot.data.joint_pos[0,[robot.joint_names.index(n) for n in self.kin.names]].cpu().numpy();rot=R.from_quat(robot.data.root_quat_w[0].cpu().numpy()[[1,2,3,0]])
   s.update(planner_navigation_error_m=float(np.linalg.norm(root[:2]-np.array(nav_goal[:2]))),planner='SONIC V2 -> release encoder -> release decoder, then live root-corrected teleop grasp',navigation_disabled=self.phase not in ['walking','braking'],navigation_goal=plan['goal_root_m'][:2],navigation_error_m=float(np.linalg.norm(root[:2]-np.array(plan['goal_root_m'][:2]))),phase=self.phase,planner_calls=self.nav_planner.serial,navigation_last_report=self.nav_planner.last_report,grasp_token_count=self.grasp_count,stance_joint_rmse_rad=float(np.sqrt(np.mean((q-self.qstart)**2))),stance_yaw_error_rad=float(abs((rot.inv()*R.from_quat(ref['root_quaternion_world_wxyz'][0,[1,2,3,0]])).as_euler('xyz')[2])),handoff_end=self.handoff_end)
   return s
  def apply(self,d):
   action=d['action'];frame=self.frame()
   if action in ['continue','stop']:return
   if action=='navigate':
    self.nav_planner.nav={'root_xy':nav_goal[:2],'yaw':plan['yaw_rad'],'speed':float(d.get('goal',{}).get('speed',plan['cruise_speed_m_s']))};self.nav_planner.action='navigate';self.nav_planner.force=True;self.phase='walking'
   elif action=='brake':self.nav_planner.action='continue';self.nav_planner.force=True;self.phase='braking'
   elif action=='handoff':
    s=self.observation();assert s['navigation_error_m']<.05 and np.linalg.norm(s['root_linear_velocity_m_s'][:2])<.15
    start=frame;count=len(self.lib.dof_pos)-start;duration=150;mapping=np.argsort(np.asarray(self.lib.m_cfg.mujoco_to_isaaclab_dof));oldq=self.lib.dof_pos[start:].cpu().numpy()[:,mapping];oldp=self.lib.body_pos_w_full[start:,0].cpu().numpy();oldr=R.from_quat(self.lib.body_quat_w_full[start:,0].cpu().numpy()[:,[1,2,3,0]])
    u=np.clip(np.arange(count)/duration,0,1);a=u**3*(10-15*u+6*u*u);qs=oldq+a[:,None]*(self.qstart-oldq);ps=oldp+a[:,None]*(np.array(plan['goal_root_m'])-oldp);endr=R.from_quat(ref['root_quaternion_world_wxyz'][0,[1,2,3,0]]);rs=(oldr*R.from_rotvec((oldr.inv()*endr).as_rotvec()*a[:,None])).as_quat()[:,[3,0,1,2]];self.write(qs,ps,rs,start);self.phase='handoff';self.handoff_end=frame+duration
    write_json(self.out/'handoff.json',dict(start_frame=frame,end_frame=self.handoff_end,actual_start=s,reference_only=True,physical_state_writes=False));np.savez_compressed(self.out/'handoff_reference.npz',q=qs[:duration+50],root=ps[:duration+50],quat=rs[:duration+50])
   elif action=='start_grasp':
    s=self.observation();assert s['navigation_error_m']<.05 and s['stance_joint_rmse_rad']<.16 and s['stance_yaw_error_rad']<.12
    assert np.linalg.norm(s['root_linear_velocity_m_s'][:2])<.06
    self.phase='grasp';self.grasp_start=frame;self.grasp_count=0;write_json(self.out/'grasp_handoff.json',dict(frame=frame,actual=s,source_tokens=str(S/'grasp_tokens.npz'),no_state_reset=True))
   else:raise ValueError(action)
  def tick(self):
   if self.phase in ['walking','braking']:self.nav_planner.tick()
   if self.phase=='handoff' and self.frame()>=self.handoff_end:self.phase='settled'
 walk_grasp.WalkGraspPlanner=HybridPlanner
 old_install=brainco_adapter.install_extension
 def extension_install(ext):
  old_install(ext);ext.controller='sonic_full'
  from initialization import install_restore
  install_restore(ext,H/'initial_state.npz')
  import torch
  robot=ext.env.scene['robot'];ids=[robot.joint_names.index(n) for n in names];planner=ext.planner;history=deque(maxlen=10);records=[];state=dict(step=0,target=None,previous=np.zeros(29),configured=False)
  lean=bool(os.environ.get('G1_LEAN_MODE'))
  ext.lean_body_names=names
  ext.lean_hand_names={side:z[f'revo2_{side}_active_joint_names'].tolist() for side in ['left','right']}
  snap=ext.out/'hybrid_source_snapshot';snap.mkdir();
  for f in Path(__file__).resolve().parent.glob('*.py'):shutil.copy2(f,snap/f.name)
  for n in ['plan.json','initial_state.npz','deployment_parameters.json']:shutil.copy2(H/n,ext.out/n)
  old_before=ext.before_step;term=ext.env.action_manager.get_term('joint_pos');old_apply=term.apply_actions
  ext.wrapped.action_transform_module.forward=lambda *a,**k:torch.zeros((1,29),device=ext.env.device)
  def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=ext.env.device)[None]
  def apply_actions():
   old_apply()
   if state['target'] is not None:
    robot.set_joint_position_target(tensor(state['target']),joint_ids=ids);robot.set_joint_velocity_target(tensor(np.zeros(29)),joint_ids=ids);robot.set_joint_effort_target(tensor(np.zeros(29)),joint_ids=ids)
  term.apply_actions=apply_actions
  def flush():
   if records:
    i=state['step']-1;np.savez_compressed(ext.out/f'decoder_chunk_{i//50:04d}.npz',**{key:np.stack([x[key] for x in records]) for key in records[0]});records.clear()
  def before():
   old_before()
   if not state['configured']:
    robot.write_joint_stiffness_to_sim(tensor(kp),joint_ids=ids);robot.write_joint_damping_to_sim(tensor(kd),joint_ids=ids);state['configured']=True
    write_json(ext.out/'token_actuator_setup.json',dict(joint_names=names,joint_ids=ids,kp=kp.tolist(),kd=kd.tolist(),default=default.tolist(),action_scale=scale.tolist()))
   i=state['step'];d=robot.data;r=R.from_quat(d.root_quat_w[0].cpu().numpy()[[1,2,3,0]]);frame=planner.frame();future=np.minimum(frame+np.arange(0,50,5),len(planner.lib.dof_pos)-1);phase=planner.phase;gi=-1
   if phase=='grasp':
    gi=planner.grasp_count;assert gi<len(z['time_s']);encoder_input=z['encoder_inputs'][gi].copy()
    reference_rotation=R.from_quat(ref['root_quaternion_world_wxyz'][gi,[1,2,3,0]])
    encoder_input[595:601]=(r.inv()*reference_rotation).as_matrix()[:,:2].reshape(6)
    # Compensate measured docking error in VR point positions, in the reference-root axes.
    correction=reference_rotation.inv().apply((d.root_pos_w[0].cpu().numpy()-ref['root_position_world_m'][gi])/SCALE)
    encoder_input[901:910]=(encoder_input[901:910].reshape(3,3)-correction).reshape(9)
    token=enc.run(None,{enc.get_inputs()[0].name:encoder_input[None]})[0][0];planner.grasp_count+=1
    if planner.grasp_count==len(z['time_s']):
     import os
     os.environ['G1_MAX_STEPS']=str(i+1)
   else:
    qs=planner.lib.dof_pos[future].cpu().numpy();dqs=planner.lib.dof_vel[future].cpu().numpy();rq=planner.lib.body_quat_w_full[future,0].cpu().numpy();relative=r.inv()*R.from_quat(rq[:,[1,2,3,0]]);encoder_input=np.zeros(1762,np.float32);encoder_input[4:294]=qs.reshape(-1);encoder_input[294:584]=dqs.reshape(-1);encoder_input[601:661]=relative.as_matrix()[:,:,:2].reshape(-1);token=enc.run(None,{enc.get_inputs()[0].name:encoder_input[None]})[0][0]
   row=dict(q=d.joint_pos[0,ids].cpu().numpy().astype(float)-default,dq=d.joint_vel[0,ids].cpu().numpy().astype(float),omega=r.inv().apply(d.root_ang_vel_w[0].cpu().numpy()),gravity=r.inv().apply([0,0,-1]),last=state['previous'].copy())
   if not history:
    for _ in range(9):history.append(row)
   history.append(row);obs=np.concatenate([token]+[np.concatenate([h[k] for h in history]) for k in ['omega','q','dq','last','gravity']]).astype('float32');assert obs.shape==(994,) and np.isfinite(obs).all();raw=dec.run(None,{dec.get_inputs()[0].name:obs[None]})[0][0];target=default+scale*raw;state.update(target=target,previous=raw.copy(),step=i+1)
   for side in ['left','right']:
    hand=ext.env.scene['brainco_'+side];hi=[hand.joint_names.index(n) for n in z[f'revo2_{side}_active_joint_names']];hand.set_joint_position_target(tensor(z[f'revo2_{side}_active_position_target_rad'][max(gi,0)]),joint_ids=hi)
   if lean:
    ext.lean_command=dict(control_index=np.array(i),token=token.copy(),phase=np.array(phase),grasp_index=np.array(gi),
      left_hand_target=z['revo2_left_active_position_target_rad'][max(gi,0)].copy(),
      right_hand_target=z['revo2_right_active_position_target_rad'][max(gi,0)].copy())
    return
   records.append(dict(control_index=np.array(i),inputs=obs,outputs=raw,targets=target,encoder_inputs=encoder_input,phase=np.array(phase),grasp_index=np.array(gi)))
   if state['step']%50==0:flush()
   with (ext.out/'token_progress.jsonl').open('a') as f:f.write(json.dumps(dict(step=i,phase=phase,grasp_index=gi,reference_frame=frame,root=d.root_pos_w[0].cpu().tolist(),navigation_error_m=float(np.linalg.norm(d.root_pos_w[0,:2].cpu().numpy()-plan['goal_root_m'][:2]))))+'\n')
  ext.before_step=before
  from training_capture import TrainingCapture
  old_close=TrainingCapture.close
  def close(c,*a,**k):flush();return old_close(c,*a,**k)
  TrainingCapture.close=close
  write_json(ext.out/'token_controller.json',dict(controller='SONIC V2 reference -> online release encoder -> release decoder; live object-centric teleop targets during grasp',planner_model='target_vel/V2/planner_sonic.onnx',encoder=str(model/'model_encoder.onnx'),decoder=str(model/'model_decoder.onnx'),decoder_sha256=hashlib.sha256((model/'model_decoder.onnx').read_bytes()).hexdigest(),grasp_tokens_sha256=hashlib.sha256((S/'grasp_tokens.npz').read_bytes()).hexdigest(),online_encoder_anchor='Walking mode0; grasp mode1 live orientation anchor and measured root-position correction',body_joints=names,frequency_Hz=50,physics_Hz=200,online_IK=False,physical_state_writes_during_episode=False,initialization='one-time sampled robot/bilateral hands/object initialization; no runtime physical state writes',hands='Source fast initial targets while walking; exact fast sequence after handoff',training_ready=False))
 brainco_adapter.install_extension=extension_install
