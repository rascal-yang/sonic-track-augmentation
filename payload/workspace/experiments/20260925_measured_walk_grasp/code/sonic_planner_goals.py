"""Real SONIC V2 ONNX locomotion planner plus world-space hand IK."""
import hashlib
from calibration_config import ROBOT_SCALE
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from experiment import ROOT, write_json
from online_goals import OnlinePlanner

MODEL = Path(os.environ.get('SONIC_PLANNER_MODEL', Path.home() / 'code/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx'))


def resample(qpos, indices):
    indices=np.clip(indices,0,len(qpos)-1)
    result=np.stack([np.interp(indices,np.arange(len(qpos)),qpos[:,i]) for i in range(36)],axis=1)
    result[:,3:7]=Slerp(np.arange(len(qpos)),Rotation.from_quat(qpos[:,[4,5,6,3]]))(indices).as_quat()[:,[3,0,1,2]]
    return result


class SonicONNX:
    def __init__(self):
        sys.path.insert(0,str(ROOT/'.deps'))
        import onnxruntime as ort
        options=ort.SessionOptions(); options.intra_op_num_threads=4
        self.session=ort.InferenceSession(str(MODEL),sess_options=options,providers=['CPUExecutionProvider'])

    def predict(self, context, direction, facing, mode, speed, seed):
        feed={'context_mujoco_qpos':context[None].astype(np.float32),
              'target_vel':np.array([speed],np.float32), 'mode':np.array([mode],np.int64),
              'movement_direction':np.array([direction],np.float32),
              'facing_direction':np.array([facing],np.float32), 'height':np.array([-1],np.float32),
              'random_seed':np.array([seed],np.int64), 'has_specific_target':np.zeros((1,1),np.int64),
              'specific_target_positions':np.zeros((1,4,3),np.float32),
              'specific_target_headings':np.zeros((1,4),np.float32),
              'allowed_pred_num_tokens':np.array([[0]*10+[1]],np.int64)}
        t=time.monotonic(); raw,n=self.session.run(None,feed)
        valid=raw[0,:int(n.item())].copy()
        if len(valid)<4 or not np.isfinite(valid).all(): raise ValueError('Invalid SONIC planner output')
        elapsed=time.monotonic()-t
        return valid,feed,elapsed


class SonicPlannerIK(OnlinePlanner):
    def __init__(self, env, out, sonic):
        super().__init__(env,out,sonic)
        self.backend=SonicONNX()
        self.raw_motion=None
        self.nav=None
        self.action='continue'
        self.serial=0
        self.last_frame=-999
        self.force=True
        self.goal_start=0
        self.hand_origin=None
        self.duration=100
        self.seed=17
        self.last_report=None
        self.waypoint_sequence=[]
        write_json(out/'planner_provenance.json',{'model_path':str(MODEL),
            'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
            'backend':'ONNX Runtime CPU','model_fps':30,'control_fps':50,
            'replan_interval_control_steps':25,'context':'previous raw planner reference, XY reanchored to measured root, +2 frames lookahead',
            'navigation':'GPT root XY goal -> measured root error -> mode/direction/speed',
            'GRAIL_residual_enabled':os.environ.get('G1_GRAIL_ACTIVE')=='1','low_level':'SONIC sonic_manipulation_base',
            'IK':'hand/waist only; planner root and leg references preserved',
            'model_library':'planner contains pretrained motion style clips; no task demonstration supplied'})

    def observation(self):
        obs=super().observation()
        root=self.env.scene['robot'].data.root_pos_w[0].cpu().numpy()
        obs.pop('stance_limit_m',None)
        obs.update(planner='SONIC V2 ONNX + IK',navigation_goal=self.nav,
                   navigation_error_m=None if self.nav is None else float(np.linalg.norm(root[:2]-self.nav['root_xy'])),
                   planner_calls=self.serial,last_planner_report=self.last_report)
        return obs

    def apply(self, decision):
        if decision['action'] in ('continue','stop'): return
        p=decision.get('goal') or {}
        action=decision['action']
        frame=int((self.cmd.motion_start_time_steps+self.cmd.time_steps)[0])
        if action=='navigate':
            xy=np.asarray(p['root_xy'],dtype=float)
            if xy.shape!=(2,) or not np.isfinite(xy).all(): raise ValueError('finite root_xy required')
            root=self.env.scene['robot'].data.root_pos_w[0].cpu().numpy()
            if np.linalg.norm(xy-root[:2])>1.: raise ValueError('first experiment navigation goal limited to 1m')
            yaw=float(p.get('yaw',np.pi/2)); speed=float(p.get('speed',.15))
            if not np.isfinite(yaw) or not .05<=speed<=.3: raise ValueError('invalid yaw/speed')
            self.nav={'root_xy':xy.tolist(),'yaw':yaw,'speed':speed}
            self.goal=None
        else:
            side=p.get('hand','left');target=np.asarray(p['target'],dtype=float)
            if side not in ('left','right') or target.shape!=(3,) or not np.isfinite(target).all(): raise ValueError('invalid hand/target')
            euler=p.get('wrist_euler_xyz')
            if euler is not None and (np.shape(euler)!=(3,) or not np.isfinite(euler).all()): raise ValueError('invalid wrist orientation')
            self.hand_origin=np.asarray(super().observation()['palm_positions'][side])
            self.goal={'action':action,'hand':side,'target':target.tolist(),'wrist_euler_xyz':euler}
            self.duration=int(p.get('duration_steps',100))
            if not 25<=self.duration<=200: raise ValueError('duration_steps must be 25..200')
            if action in ('close_hand','lift','open_hand'):
                getattr(self.lib,'_motion_hand_action_'+side)[frame:]= -1. if action=='open_hand' else 1.
        if action=='lift':
            import torch
            lift=float(p.get('lift_m',.10))
            if not 0<=lift<=.2: raise ValueError('lift_m must be 0..0.2')
            count=len(self.lib.dof_pos)-frame
            alpha=np.clip(np.arange(count)/self.duration,0,1)
            alpha=alpha*alpha*(3-2*alpha)
            origin=self.env.scene['object'].data.root_pos_w[0].clone()
            self.lib._motion_object_root_pos[frame:,0]=origin
            self.lib._motion_object_root_pos[frame:,0,2]+=torch.tensor(alpha*lift,device=origin.device)
            self.lib._motion_object_lin_vel[frame:,0]=0
            self.lib._motion_object_lin_vel[frame:,0,2]=torch.tensor(np.gradient(alpha*lift,.02),device=origin.device)
        self.action=action;self.goal_start=frame;self.force=True
        self.waypoint_sequence=decision.get('waypoint_sequence',[])
        if self.waypoint_sequence:
            assert self.goal is not None
            assert np.allclose(self.waypoint_sequence[0]['target'],self.goal['target'])
            assert int(self.waypoint_sequence[0]['duration_steps'])==self.duration
            for w in self.waypoint_sequence:
                assert np.shape(w['target'])==(3,) and np.isfinite(w['target']).all()
                assert 25<=int(w['duration_steps'])<=200
            write_json(self.out/f'waypoint_sequence_{decision["observation_id"]:03d}.json',{
                'reference_start_frame':frame,'actual_start_palm_m':self.hand_origin.tolist(),
                'waypoints':self.waypoint_sequence,'execution':'lookahead sequence used in IK reference; execute current segment then replan at next observation'})
        write_json(self.out/f'goal_application_{decision["observation_id"]:03d}.json',{
            'action':action,'goal':p,'reference_start_frame':frame,'backend':'SONIC planner then IK',
            'grail_residual_disabled':os.environ.get('G1_GRAIL_ACTIVE')!='1','policy_observation_delay_steps':1})

    def tick(self):
        frame=int((self.cmd.motion_start_time_steps+self.cmd.time_steps)[0])
        if not self.force and frame-self.last_frame<25: return
        self.force=False;self.last_frame=frame
        lib=self.lib
        if self.raw_motion is None:
            qs=lib.dof_pos.cpu().numpy()[:,np.argsort(np.asarray(lib.m_cfg.mujoco_to_isaaclab_dof))]
            self.raw_motion=np.concatenate([lib.body_pos_w_full[:,0].cpu().numpy(),lib.body_quat_w_full[:,0].cpu().numpy(),qs],axis=1).copy()
        start=frame+2
        root=self.env.scene['robot'].data.root_pos_w[0].cpu().numpy()
        # Correct accumulated reference drift using measured robot translation.
        # This transforms reference poses only, never the physical articulation.
        anchor_delta=root[:2]-self.raw_motion[frame,:2]
        self.raw_motion[frame:,:2]+=anchor_delta
        context=resample(self.raw_motion,start+np.arange(4)*50/30)
        yaw=self.nav['yaw'] if self.nav else np.pi/2
        facing=[np.cos(yaw),np.sin(yaw),0.]
        mode=0;speed=-1.;direction=facing
        if self.action=='navigate' and self.nav:
            delta=np.array(self.nav['root_xy'])-root[:2];dist=np.linalg.norm(delta)
            if dist>.015:
                direction=[*(delta/dist),0.];mode=1;speed=min(self.nav['speed'],max(.05,dist*1.5))
        raw,feed,elapsed=self.backend.predict(context,direction,facing,mode,speed,self.seed+self.serial)
        predicted=resample(raw,np.arange(int(len(raw)*50/30))*30/50)
        n=min(len(predicted),len(self.raw_motion)-start)
        predicted=predicted[:n]
        # Crossfade new planner motion into the previous planner motion over 8 frames.
        old=self.raw_motion[start:start+n].copy()
        alpha=np.clip(np.arange(n)/7.,0,1)
        blended=old+alpha[:,None]*(predicted-old)
        a=Rotation.from_quat(old[:,[4,5,6,3]]);b=Rotation.from_quat(predicted[:,[4,5,6,3]])
        blended[:,3:7]=(a*Rotation.from_rotvec((a.inv()*b).as_rotvec()*alpha[:,None])).as_quat()[:,[3,0,1,2]]
        self.raw_motion[start:start+n]=blended
        self.raw_motion[start+n:]=blended[-1]
        combined=blended.copy();errors=[]
        if self.goal:
            # Solve sparse upper-body keyframes, interpolate their corrections.
            indices=np.unique(np.r_[np.arange(0,n,5),n-1])
            corrections=[]
            for i in indices:
                progress=np.clip((start+i-self.goal_start)/self.duration,0,1)
                progress=progress*progress*(3-2*progress)
                target=self.hand_origin+progress*(np.array(self.goal['target'])-self.hand_origin)
                euler=self.goal['wrist_euler_xyz']
                if self.waypoint_sequence:
                    segment_elapsed=start+i-self.goal_start;origin=self.hand_origin.copy()
                    for w in self.waypoint_sequence:
                        duration=int(w['duration_steps']);end=np.asarray(w['target'])
                        a=float(np.clip(segment_elapsed/duration,0,1));a=a*a*(3-2*a)
                        target=origin+a*(end-origin);euler=w.get('wrist_euler_xyz',euler)
                        if segment_elapsed<=duration:break
                        segment_elapsed-=duration;origin=end
                q,report=self.kin.solve(blended[i,7:].copy(),blended[i,:3],blended[i,3:7],
                    self.goal['hand'],target,euler)
                corrections.append(q-blended[i,7:]);errors.append(report['ik_palm_error_m'])
            corrections=np.asarray(corrections)
            for j in range(29): combined[:,7+j]+=np.interp(np.arange(n),indices,corrections[:,j])
            # Preserve continuity with the previously applied IK reference as well.
            previous_q=lib.dof_pos[start:start+n].cpu().numpy()[:,np.argsort(np.asarray(lib.m_cfg.mujoco_to_isaaclab_dof))]
            upper=[i for i,name in enumerate(self.kin.names) if name.startswith(self.goal['hand']+'_') and any(key in name for key in ('shoulder','elbow','wrist')) or name=='waist_pitch_joint']
            combined[:,np.array(upper)+7]=previous_q[:,upper]+alpha[:,None]*(combined[:,np.array(upper)+7]-previous_q[:,upper])
        tail=np.repeat(combined[-1:],len(self.raw_motion)-start-n,axis=0)
        full=np.concatenate([combined,tail],axis=0)
        pose=np.zeros((len(full),30,3),np.float32)
        pose[:,0]=Rotation.from_quat(full[:,[4,5,6,3]]).as_rotvec()
        pose[:,1:]=full[:,7:,None]*self.kin.axes
        full[:,2]*=ROBOT_SCALE
        self.write_reference(start,full[:,7:],full[:,:3],pose)
        np.savez_compressed(self.out/f'planner_{self.serial:03d}.npz',**feed,raw_qpos_30hz=raw,
            planner_qpos_50hz=blended,planner_ik_qpos_50hz=combined)
        self.last_report={'call':self.serial,'frame':frame,'mode':mode,'speed':speed,
            'reference_xy_correction_m':anchor_delta.tolist(),'measured_root':root.tolist(),
            'inference_seconds':float(elapsed),'valid_frames_30hz':len(raw),'frames_50hz':n,
            'max_ik_keyframe_error_m':max(errors) if errors else None,
            'predicted_root_end':blended[-1,:3].tolist()}
        with (self.out/'planner_calls.jsonl').open('a') as f:
            import json
            f.write(json.dumps(self.last_report)+'\n')
        self.serial+=1
