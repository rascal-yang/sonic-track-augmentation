"""Navigate physically with SONIC, then blend references into the proven grasp start."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from stationary_hint import StationaryHintPlanner
from sonic_planner_goals import SonicPlannerIK
from online_goals import OnlinePlanner
from experiment import write_json


def smooth(x):
    x=np.clip(x,0,1)
    return x**3*(10+x*(-15+6*x))


class WalkGraspPlanner(StationaryHintPlanner):
    def __init__(self,env,out,sonic):
        super().__init__(env,out,sonic)
        self.nav_planner=SonicPlannerIK(env,out,sonic)
        self.phase='initial_stand'
        self.grail_gain=0.
        cfg=json.loads((Path(__file__).resolve().parents[1]/'optimized_config.json').read_text())
        self.stand_root=np.array(cfg['stand_root_m'])
        self.handoff_waist_pitch_delta=float(cfg.get('handoff_waist_pitch_delta_rad',0.))
        self.stand_quat=Rotation.from_euler('z',np.pi/2).as_quat()[[3,0,1,2]]
        self.stand_q=self.kin.default.copy()
        for i,name in enumerate(self.kin.names):
            if name.endswith('shoulder_pitch_joint'): self.stand_q[i]=.8
            if name.endswith('elbow_joint'): self.stand_q[i]=.2
        self.handoff_end=None
        write_json(out/'walk_grasp_provenance.json',{
            'walk':'SONIC V2 ONNX planner + SONIC frozen base; table-interaction latent residual zero',
            'handoff':'blend future reference into known relaxed standing pose over 3 s; no articulation/root teleport',
            'grasp':'stop ONNX replanning; stand 1 s; authorized bottle grasp hints + actual-state right-arm IK; GRAIL residual zero',
            'stand_root_m':self.stand_root.tolist(),'initial_root_m':self.reference_origin.tolist(),
            'external_GPT6_API':False,'actuator_output_filter':False})

    def frame(self):
        return int((self.cmd.motion_start_time_steps+self.cmd.time_steps)[0])

    def observation(self):
        obs=StationaryHintPlanner.observation(self)
        robot=self.env.scene['robot'];root=robot.data.root_pos_w[0].cpu().numpy()
        obs.update(planner='SONIC walk -> reference handoff -> stationary actual-arm IK + SONIC grasp',
                   phase=self.phase,navigation_disabled=self.phase in ('handoff','grasp'),
                   navigation_goal=self.stand_root[:2].tolist(),
                   navigation_error_m=float(np.linalg.norm(root[:2]-self.stand_root[:2])),
                   root_linear_velocity_m_s=robot.data.root_lin_vel_w[0].cpu().tolist(),
                   grail_phase_gain=self.grail_gain,planner_calls=self.nav_planner.serial,
                   navigation_last_report=self.nav_planner.last_report)
        obs['feet']={}
        for side in ['left','right']:
            bid=robot.body_names.index(side+'_ankle_roll_link')
            obs['feet'][side]={'position_m':robot.data.body_pos_w[0,bid].cpu().tolist(),
                               'linear_velocity_m_s':robot.data.body_lin_vel_w[0,bid].cpu().tolist()}
        return obs

    def apply(self,decision):
        action=decision['action']
        if action in ('continue','stop'): return
        if action=='navigate':
            if self.phase in ('handoff','grasp'): raise ValueError('Finish current experiment before navigating again')
            self.phase='walking';self.nav_planner.apply(decision)
        elif action=='brake':
            if self.phase!='walking': raise ValueError('Brake only from walking')
            self.phase='braking';self.nav_planner.action='continue';self.nav_planner.force=True
        elif action=='settle':
            if self.phase!='braking': raise ValueError('Brake and inspect actual feet before handoff')
            obs=self.observation()
            if obs['navigation_error_m']>.09: raise ValueError('Handoff root farther than 9 cm from proven stance')
            if np.linalg.norm(obs['root_linear_velocity_m_s'][:2])>.12: raise ValueError('Root not yet slow enough')
            if any(np.linalg.norm(x['linear_velocity_m_s'])>.2 for x in obs['feet'].values()):
                raise ValueError('Feet still moving too fast for standing handoff')
            self._handoff(decision,obs)
        else:
            if self.phase!='grasp': raise ValueError('Grasp requires completed standing handoff')
            super().apply(decision)

    def _handoff(self,decision,obs):
        start=self.frame()+2;count=len(self.lib.dof_pos)-start;duration=150
        mapping=np.argsort(np.asarray(self.lib.m_cfg.mujoco_to_isaaclab_dof))
        old_q=self.lib.dof_pos[start:].cpu().numpy()[:,mapping].astype(float)
        old_p=self.lib.body_pos_w_full[start:,0].cpu().numpy().astype(float)
        old_r=Rotation.from_quat(self.lib.body_quat_w_full[start:,0].cpu().numpy()[:,[1,2,3,0]])
        end_q=self.stand_q.copy()
        robot=self.env.scene['robot']
        actual_q=np.array([float(robot.data.joint_pos[0,robot.joint_names.index(n)]) for n in self.kin.names])
        for i,name in enumerate(self.kin.names):
            if not any(k in name for k in ('shoulder','elbow','wrist')):end_q[i]=actual_q[i]
        waist_i=self.kin.names.index('waist_pitch_joint')
        end_q[waist_i]=np.clip(end_q[waist_i]+self.handoff_waist_pitch_delta,*self.kin.model.jnt_range[1:][waist_i])
        end_root=robot.data.root_pos_w[0].cpu().numpy().copy()
        end_quat=robot.data.root_quat_w[0].cpu().numpy().copy()
        alpha=smooth(np.arange(count)/duration)
        qs=old_q+alpha[:,None]*(end_q-old_q)
        ps=old_p+alpha[:,None]*(end_root-old_p)
        end_r=Rotation.from_quat(end_quat[[1,2,3,0]])
        rs=old_r*Rotation.from_rotvec((old_r.inv()*end_r).as_rotvec()*alpha[:,None])
        pose=np.zeros((count,30,3),np.float32);pose[:,0]=rs.as_rotvec();pose[:,1:]=qs[:,:,None]*self.kin.axes
        self.write_reference(start,qs,ps,pose)
        self.reference_origin=end_root.copy();self.reference_quat=end_quat.copy()
        self.phase='handoff';self.handoff_end=start+duration
        write_json(self.out/'handoff.json',{'observation_id':decision['observation_id'],
            'reference_start_frame':start,'duration_steps':duration,'grail_ramp_steps_after_transition':50,
            'preserve_measured_leg_support':True,'waist_pitch_reference_delta_rad':self.handoff_waist_pitch_delta,'actual_start':obs,'initial_reference_root':old_p[0].tolist(),'final_reference_root':ps[-1].tolist(),
            'joint_reference_speed_max_rad_s':float(np.abs(np.diff(qs,axis=0)/.02).max()),
            'physical_state_writes':False})
        np.savez_compressed(self.out/'handoff_reference.npz',joint_pos=qs,root_pos=ps,pose_aa=pose)

    def tick(self):
        if self.phase in ('walking','braking'):
            self.nav_planner.tick()
        elif self.phase=='handoff':
            self.grail_gain=0.
            if self.frame()>=self.handoff_end+50:
                self.phase='grasp'
        elif self.phase=='grasp':
            self.grail_gain=0.
