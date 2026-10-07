"""Automatic measured-state phase gates. No fabricated visual-review approvals."""
import numpy as np
from scipy.spatial.transform import Rotation as R

class Supervisor:
    def __init__(self, plan):
        self.plan=plan;self.phase=None;self.phase_start=0;self.carry=None;self.anchor=None

    def decide(self,s):
        p=self.plan;step=int(s['step']);phase=s['phase'];error=float(s['navigation_error_m'])
        if phase!=self.phase:self.phase=phase;self.phase_start=step
        age=step-self.phase_start;root=np.array(s['robot_root_pos']);start=np.array(p['start_root_m'])
        travel=float((root-start)@p['forward_world']);speed=float(np.linalg.norm(s['root_linear_velocity_m_s'][:2]))
        foot_speed=max(float(np.linalg.norm(f['linear_velocity_m_s'])) for f in s['feet'].values())
        tilt=float(np.arccos(np.clip(R.from_quat(np.array(s['object_quat_wxyz'])[[1,2,3,0]]).as_matrix()[2,2],-1,1)))
        m=s['grasp_information']['metrics'];opposed=bool(m['opposed_digits_contact'])
        airborne=m['whole_object_clearance_m']>.02 and m['object_table_normal_force_N']<.1
        slip=twist=0.
        if airborne and opposed:
            wr=R.from_quat(np.array(s['wrist_quat_wxyz']['right'])[[1,2,3,0]])
            pos=wr.inv().apply(np.array(s['object_pos'])-s['palm_positions']['right'])
            rot=wr.inv()*R.from_quat(np.array(s['object_quat_wxyz'])[[1,2,3,0]])
            if self.carry is None:self.carry=step
            if step-self.carry>=25 and self.anchor is None:self.anchor=(pos,rot)
            if self.anchor is not None:
                slip=float(np.linalg.norm(pos-self.anchor[0]));twist=float((self.anchor[1].inv()*rot).magnitude())
        action='continue';reason='Automatic measured-state gate';steps=15
        if phase=='initial' and step>=50:action='navigate';reason='Start independently planned locomotion'
        elif phase=='walking':
            nav=s.get('planner_navigation_error_m',error)
            if nav<.30:steps=5
            if nav<p.get('brake_distance_m',.18):action='brake';reason='Brake before calibrated operation stance'
            elif age>650:action='stop';reason='Locomotion timeout'
            elif age>200 and travel<.12:action='stop';reason='Insufficient actual walking displacement'
        elif phase=='braking':
            if age>=60 and speed<.08 and foot_speed<.15:
                if error<.05:action='handoff';reason='Measured stable braking; begin reference transition'
                else:action='stop';reason='Missed docking; reject without near-table redocking'
            if age>200:action='stop';reason='Braking timeout'
        elif phase=='settled':
            if age>=40 and error<.05 and speed<.06 and foot_speed<.10 and s['stance_joint_rmse_rad']<.16 and s['stance_yaw_error_rad']<.12:
                action='start_grasp';reason='Actual base, feet, posture and yaw satisfy handoff tolerances'
            elif age>175:action='stop';reason='Grasp entry convergence timeout'
        elif phase not in ['initial','walking','braking','handoff','settled','grasp']:
            raise ValueError('Unexpected phase '+phase)
        # Check object target freshness before any skill entry. No old target after displacement.
        object_shift=float(np.linalg.norm(np.array(s['object_pos'])[:2]-p['object_initial_position_m'][:2]))
        path=np.array(p['goal_root_m'])-start;path[2]=0;path/=np.linalg.norm(path)
        lateral=float(abs(np.cross(path[:2],(root-start)[:2])))
        if s.get('contact_watch',{}).get('alert'):action='stop';reason='Physics contact latch'
        elif root[2]<.45:action='stop';reason='Low pelvis guard'
        elif lateral>.35:action='stop';reason='Lateral corridor deviation'
        elif phase!='grasp' and object_shift>.01:action='stop';reason='Object moved before grasp; stale skill target'
        elif tilt>.9 or (not airborne and not opposed and tilt>.55):action='stop';reason='Bottle tipping guard'
        elif slip>.035 or twist>.35:action='stop';reason='Carried object slip guard'
        diagnostics=dict(step=step,phase=phase,action=action,reason=reason,travel_m=travel,error_m=error,
                         foot_speed_m_s=foot_speed,root_speed_m_s=speed,tilt_rad=tilt,slip_m=slip,
                         twist_rad=twist,object_shift_m=object_shift,visual_review='pending',
                         root_position_m=root.tolist(), stance_joint_rmse_rad=s['stance_joint_rmse_rad'],
                         stance_yaw_error_rad=s['stance_yaw_error_rad'])
        decision=dict(observation_id=s['observation_id'],action=action,steps=1 if action=='stop' else steps,
                      goal={},reason=reason,author='Automatic numeric supervisor; no GPT API or human image approval')
        return decision,diagnostics


def install():
    import os,sys,types
    from pathlib import Path
    from common import read,write
    class AutoGraspPolicy:
        def __init__(self):
            self.supervisor=Supervisor(read(Path(os.environ['G1_AUG_CANDIDATE'])/'plan.json'))
        def decide(self,s):
            decision,diagnostics=self.supervisor.decide(s)
            if not os.environ.get('G1_LEAN_MODE') or decision['action']=='stop':
                write(Path(os.environ['G1_OUTPUT_DIR'])/f"guard_{s['observation_id']:03d}.json",diagnostics)
            return decision
    m=types.ModuleType('auto_policy');m.AutoGraspPolicy=AutoGraspPolicy;sys.modules['auto_policy']=m
