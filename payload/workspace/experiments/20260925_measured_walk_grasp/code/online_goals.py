"""Synthetic reference generation and bounded online IK; never reads a demo motion."""
import json
from calibration_config import ROBOT_SCALE
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import joblib
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from experiment import KEY, write_json


class Kinematics:
    def __init__(self, sonic):
        source = Path(sonic) / 'gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml'
        tree = ET.parse(source)
        # Kinematics only: preserve joint/body/inertial transforms, remove render meshes.
        root = tree.getroot()
        for node in list(root):
            if node.tag == 'asset': root.remove(node)
        for parent in root.iter():
            for child in list(parent):
                if child.tag == 'geom': parent.remove(child)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.model.body_pos[:] *= ROBOT_SCALE
        self.model.body_ipos[:] *= ROBOT_SCALE
        self.model.jnt_pos[:] *= ROBOT_SCALE
        self.model.body_inertia[:] *= ROBOT_SCALE**2
        self.data = mujoco.MjData(self.model)
        mujoco.mj_setConst(self.model,self.data)
        self.names = [self.model.joint(i).name for i in range(1, self.model.njnt)]
        assert len(self.names) == 29 and self.model.nq == 36
        self.axes = self.model.jnt_axis[1:].copy()
        self.default = np.zeros(29)
        for i, name in enumerate(self.names):
            for suffix, value in [('hip_pitch_joint', -.312), ('knee_joint', .669),
                                  ('ankle_pitch_joint', -.363), ('elbow_joint', .6),
                                  ('shoulder_pitch_joint', .2)]:
                if name.endswith(suffix): self.default[i] = value
            if name == 'left_shoulder_roll_joint': self.default[i] = .2
            if name == 'right_shoulder_roll_joint': self.default[i] = -.2

    def fk(self, q, pos, quat):
        self.data.qpos[:] = np.r_[pos, quat, q]
        mujoco.mj_forward(self.model, self.data)

    def palm(self, side):
        body = self.model.body(side + '_wrist_yaw_link').id
        return self.data.xpos[body] + self.data.xmat[body].reshape(3,3) @ np.array([.07,0,0])*ROBOT_SCALE

    def solve(self, q, pos, quat, side, target, euler=None, feet=None, allow_waist=True):
        q = np.asarray(q, dtype=np.float64).copy()
        pos = np.asarray(pos, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        initial = q.copy()
        ids = [i for i,n in enumerate(self.names) if (n.startswith(side+'_') and
               any(x in n for x in ('shoulder','elbow','wrist'))) or (allow_waist and n == 'waist_pitch_joint') or
               (feet is not None and any(x in n for x in ('hip','knee','ankle')))]
        def residual(x):
            trial = initial.copy(); trial[ids] = x
            self.fk(trial, pos, quat)
            parts = [10*(self.palm(side)-target), .02*(x-initial[ids])]
            if euler is not None:
                mat = self.data.xmat[self.model.body(side+'_wrist_yaw_link').id].reshape(3,3)
                parts += [.5*Rotation.from_matrix(Rotation.from_euler('xyz',euler).as_matrix().T @ mat).as_rotvec()]
            if feet is not None:
                for name, (p, r) in feet.items():
                    bid = self.model.body(name).id
                    parts += [30*(self.data.xpos[bid]-p), 3*Rotation.from_matrix(r.T @ self.data.xmat[bid].reshape(3,3)).as_rotvec()]
            return np.concatenate(parts)
        bounds = self.model.jnt_range[1:][ids]
        lower, upper = bounds[:,0]+1e-5, bounds[:,1]-1e-5
        x0 = np.clip(initial[ids], lower+1e-7, upper-1e-7)
        result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=120)
        q[ids] = result.x
        self.fk(q,pos,quat)
        return q, {'ik_palm_error_m': float(np.linalg.norm(self.palm(side)-target)),
                   'optimizer_success': bool(result.success), 'joint_limits_satisfied': bool(np.all(q>=self.model.jnt_range[1:,0]-1e-6) and np.all(q<=self.model.jnt_range[1:,1]+1e-6))}


def prepare_online(out, sonic, assets, seed):
    kin = Kinematics(sonic)
    rng = np.random.default_rng(seed)
    pos = np.array([rng.uniform(-.10,.10), rng.uniform(-.58,-.53), .76])
    yaw = float(np.pi/2 + rng.uniform(-.12,.12))
    quat = Rotation.from_euler('z',yaw).as_quat()[[3,0,1,2]]
    n=4000
    data=out/'synthetic'
    for folder in ('robot','objects','meta','bps'): (data/folder).mkdir(parents=True)
    pose=np.zeros((n,30,3),np.float32)
    pose[:,0]=[0,0,yaw]; pose[:,1:]=kin.default[:,None]*kin.axes
    robot={'root_trans_offset':np.tile(pos,(n,1)).astype('float32'), 'root_rot':np.tile(quat[[1,2,3,0]],(n,1)).astype('float32'),
           'pose_aa':pose,'dof':np.tile(np.r_[kin.default,np.zeros(14)],(n,1)).astype('float32'), 'fps':50,
           'hand_action_left':-np.ones(n,np.float32),'hand_action_right':-np.ones(n,np.float32)}
    objpos=np.array([0.,0.,.708],np.float32)
    objects={'root_pos':np.tile(objpos,(n,1,1)), 'root_quat':np.tile([1.,0.,0.,0.],(n,1,1)).astype('float32'),
             'fps':50, 'contact_points_left_hand':{},'contact_points_right_hand':{}}
    joblib.dump({'online_static':robot},data/'robot'/f'{KEY}.pkl')
    joblib.dump({'online_static':objects},data/'objects'/f'{KEY}.pkl')
    joblib.dump({'object_name':'object','table_pos':np.array([0,.12,.64]),'table_quat':np.array([1.,0,0,0]),'table_size':np.array([1.,.6,.04])},data/'meta'/f'{KEY}.pkl')
    for name in (f'{KEY}.npy','_basis.npy'): shutil.copy2(assets/'bps'/name,data/'bps'/name)
    write_json(out/'online_setup.json',{'seed':seed,'robot_root_pos':pos.tolist(),'robot_root_quat_wxyz':quat.tolist(),
        'object_pos':objpos.tolist(),'source':'MJCF default stand + seeded root pose, no demonstration motion loaded',
        'asset_key_only':KEY,'palm_offset_wrist_local':[.07,0,0],
        'limitations':['IK has no collision planner','stance is a small shift with fixed foot targets, not a walking planner',
                      'GRAIL weights were trained on demonstrations','contact references are plans, not measured contacts']})
    return data


class OnlinePlanner:
    def __init__(self, env, out, sonic):
        self.env,self.out=env,out
        self.kin=Kinematics(sonic)
        self.cmd=env.command_manager.get_term('motion')
        self.lib=self.cmd.motion_lib
        self.goal=None

    def observation(self):
        robot=self.env.scene['robot']
        palms={}
        wrists={}
        for side in ('left','right'):
            bid=robot.body_names.index(side+'_wrist_yaw_link')
            p=robot.data.body_pos_w[0,bid].cpu().numpy()
            quat=robot.data.body_quat_w[0,bid].cpu().numpy()
            wrists[side]=quat.tolist()
            palms[side]=(p+Rotation.from_quat(quat[[1,2,3,0]]).apply(np.array([.07,0,0])*ROBOT_SCALE)).tolist()
        return {'palm_positions':palms,'wrist_quat_wxyz':wrists,'active_goal':self.goal,
                'goal_error_m':None if not self.goal else float(np.linalg.norm(np.array(palms[self.goal['hand']])-self.goal['target'])),
                'reference_frame':int((self.cmd.motion_start_time_steps+self.cmd.time_steps)[0]),
                'stance_limit_m':.08,'coordinate_frame':'world; positions m, wrist Euler xyz radians'}

    def apply(self, decision):
        import torch
        action=decision['action']
        if action in ('continue','stop'): return
        params=decision.get('goal') or {}
        side=params.get('hand','right')
        if side not in ('left','right'): raise ValueError('hand must be left/right')
        lib=self.lib
        start=int((self.cmd.motion_start_time_steps+self.cmd.time_steps+lib.length_starts[self.cmd.motion_ids])[0])
        count=len(lib.dof_pos)-start
        assert count>100
        # Take the current reference as boundary to avoid a discontinuity; actual tracking error is observed separately.
        q=lib.dof_pos[start].cpu().numpy()[np.argsort(np.asarray(lib.m_cfg.mujoco_to_isaaclab_dof))].copy()
        pos=lib.body_pos_w_full[start,0].cpu().numpy().copy()
        quat=lib.body_quat_w_full[start,0].cpu().numpy().copy()
        self.kin.fk(q,pos,quat)
        target=np.asarray(params.get('target',self.kin.palm(side)),dtype=float)
        if target.shape!=(3,) or not np.isfinite(target).all(): raise ValueError('finite xyz target required')
        if np.linalg.norm(target-pos)>1.2: raise ValueError('target outside bounded workspace')
        endpos=pos.copy(); feet=None
        if action=='shift_stance':
            delta=np.asarray(params.get('delta',[0,0,0]),dtype=float)
            if delta.shape!=(3,) or not np.isfinite(delta).all() or np.linalg.norm(delta)>.080001: raise ValueError('stance shift maximum 8cm')
            feet={name:(self.kin.data.xpos[self.kin.model.body(name).id].copy(),self.kin.data.xmat[self.kin.model.body(name).id].reshape(3,3).copy()) for name in ('left_ankle_roll_link','right_ankle_roll_link')}
            endpos+=delta
        qend,report=self.kin.solve(q.copy(),endpos,quat,side,target,params.get('wrist_euler_xyz'),feet)
        duration=int(params.get('duration_steps',100))
        if not 25<=duration<=200: raise ValueError('duration_steps must be 25..200')
        lift=float(params.get('lift_m',.10))
        if not np.isfinite(lift) or not 0<=lift<=.2: raise ValueError('lift_m must be 0..0.2')
        alpha=np.clip(np.arange(count)/duration,0,1); alpha=alpha*alpha*(3-2*alpha)
        qs=q[None]+alpha[:,None]*(qend-q)
        ps=pos[None]+alpha[:,None]*(endpos-pos)
        pose=np.zeros((count,30,3),np.float32)
        pose[:,0]=Rotation.from_quat(quat[[1,2,3,0]]).as_rotvec()
        pose[:,1:]=qs[:,:,None]*self.kin.axes
        self.write_reference(start, qs, ps, pose)
        # Object goal and contact label explicitly describe intent; physical object is never written.
        if action=='lift':
            current=lib._motion_object_root_pos[start,0].clone()
            lib._motion_object_root_pos[start:,0,2]=current[2]+torch.tensor(alpha*lift,device=current.device)
            lib._motion_object_lin_vel[start:,0,2]=torch.tensor(np.gradient(alpha*lift,.02),device=current.device)
        if action in ('close_hand','lift','open_hand'):
            closed=action!='open_hand'
            getattr(lib,'_motion_hand_action_'+side)[start:]=1. if closed else -1.
            getattr(lib,'_motion_object_in_contact_'+side)[start:]=float(closed)
            getattr(lib,'_motion_object_contact_center_'+side)[start:]=0. # intended object-local center
        self.goal={'action':action,'hand':side,'target':target.tolist(), 'wrist_euler_xyz':params.get('wrist_euler_xyz')}
        report.update(self.goal,ik_feasible=report['ik_palm_error_m'] <= .02,reference_start_frame=start,duration_steps=duration,
                      max_joint_reference_change_rad=float(np.max(np.abs(qend-q))),root_delta=(endpos-pos).tolist(),
                      contact_label_semantics='planned, not measured',policy_observation_delay_steps=1)
        np.savez_compressed(self.out/f'reference_{decision["observation_id"]:03d}.npz',joint_pos=qs,root_pos=ps,pose_aa=pose)
        write_json(self.out/f'goal_application_{decision["observation_id"]:03d}.json',report)

    def write_reference(self, start, qs, ps, pose):
        import torch
        lib=self.lib
        fk=lib.mesh_parsers.fk_batch(torch.tensor(pose)[None],torch.tensor(ps,dtype=torch.float32)[None],return_full=True,fps=50,target_fps=50)
        for key in ['global_translation','global_velocity']:
            values=fk[key]
            fk[key]=values[:,:,:1]+ROBOT_SCALE*(values-values[:,:,:1])
        mapping=lib.m_cfg.mujoco_to_isaaclab_body
        for dest,src in [('body_pos_w','global_translation'),('body_quat_w','global_rotation'),('body_lin_vel_w','global_velocity'),('body_ang_vel_w','global_angular_velocity')]:
            value=fk[src][0][:,mapping]
            if dest=='body_quat_w': value=value[:,:,[3,0,1,2]]
            getattr(lib,dest+'_full')[start:]=value.to(getattr(lib,dest+'_full'))
            getattr(lib,dest)[start:]=value[:,lib.body_indexes].to(getattr(lib,dest))
        dofmap=lib.m_cfg.mujoco_to_isaaclab_dof
        lib.dof_pos[start:]=torch.tensor(qs[:,dofmap],device=lib.dof_pos.device,dtype=lib.dof_pos.dtype)
        lib.dof_vel[start:]=torch.tensor(np.gradient(qs,.02,axis=0)[:,dofmap],device=lib.dof_vel.device,dtype=lib.dof_vel.dtype)
        lib.root_linv_vel_w[start:]=lib.body_lin_vel_w_full[start:,0]
        lib.root_ang_vel_w[start:]=lib.body_ang_vel_w_full[start:,0]
