"""Additional grasp geometry, contact sensing, and explicit control variants."""
import json
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from experiment import ROOT, write_json

FINGER_LINKS=['right_hand_'+name+'_link' for name in ['palm','index_0','index_1','middle_0','middle_1','thumb_0','thumb_1','thumb_2']]


def configure_sensors(scene):
    from isaaclab.sensors import ContactSensorCfg
    for i,name in enumerate(FINGER_LINKS):
        setattr(scene,f'grasp_finger_{i}',ContactSensorCfg(
            prim_path='{ENV_REGEX_NS}/Robot/'+name,
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Object','{ENV_REGEX_NS}/Table'],
            track_pose=True,track_contact_points=True,track_friction_forces=True,
            max_contact_data_count_per_prim=64,history_length=2))


class Extension:
    def __init__(self,wrapped,planner,out):
        self.wrapped=wrapped;self.env=wrapped.env;self.planner=planner;self.out=out
        self.method=os.environ.get('GRASP_METHOD','candidates')
        self.controller=os.environ.get('GRASP_CONTROLLER','sonic_full')
        self.env._grasp_extension=self
        self.vertices=np.load(Path(__file__).resolve().parents[1]/'assets/water_bottle_v1/collision_source.npz')['vertices']
        hulls=json.loads((Path(__file__).resolve().parents[1]/'assets/water_bottle_v1/collision_hulls.json').read_text())
        self.hull_vertices=np.vstack([h['vertices'] for h in hulls])
        self.finger_targets=None;self.grasp_target=None;self.last_actual_ik=None
        self.hand_offset=np.array([.06,.035,0.])
        self.count=0;self.held=0;self.maximum_held=0
        self.close_from=0.;self.close_to=0.;self.close_start=0;self.close_duration=50
        # Fractional closures use the same existing joint limits/PD actuators.
        for primitive in wrapped._finger_primitive_map: primitive['mode']='linear'
        action_term=self.env.action_manager.get_term('joint_pos')
        original_apply=action_term.apply_actions
        def apply_actions():
            original_apply()
            robot=self.env.scene['robot']
            if self.controller in ('ik_upper','ik_upper_gc') and self.last_actual_ik is not None and self.planner.phase == 'grasp':
                q=self.last_actual_ik
                ids=[robot.joint_names.index(n) for n in planner.kin.names if n.startswith('right_') and any(k in n for k in ('shoulder','elbow','wrist'))]
                indexes=[planner.kin.names.index(robot.joint_names[i]) for i in ids]
                import torch
                robot.set_joint_position_target(torch.tensor(q[indexes],device=self.env.device,dtype=torch.float32)[None],joint_ids=ids)
        action_term.apply_actions=apply_actions
        write_json(out/'variant.json',{'method':self.method,'controller':self.controller,
            'finger_control':'fractional interpolation of existing open/closed primitive; original actuator limits',
            'asset_and_physics_changes':False,'sensors':'8 right-hand links filtered against toy and table; contact points and friction enabled',
            'success_definition':'all toy convex vertices >table+2cm for .5s; hand contact >.2N; table contact <.1N; pelvis>.45m',
            'agent':'current conversation; not an external GPT6 API'})

    def frame(self):
        return int((self.planner.cmd.motion_start_time_steps+self.planner.cmd.time_steps)[0])

    def object_points(self):
        obj=self.env.scene['object'].data
        p=obj.root_pos_w[0].cpu().numpy();q=obj.root_quat_w[0].cpu().numpy()
        r=Rotation.from_quat(q[[1,2,3,0]])
        return r,p

    def contacts(self):
        result={}
        for i,name in enumerate(FINGER_LINKS):
            data=self.env.scene[f'grasp_finger_{i}'].data
            def arr(value):
                if value is None:return None
                a=value[0].cpu().numpy()
                return np.where(np.isfinite(a),a,0).tolist()
            force=data.force_matrix_w[0].cpu().numpy().reshape(-1,3)
            result[name]={'object_force_vector_N':force[0].tolist(),'object_normal_force_N':float(np.linalg.norm(force[0])),
                'table_normal_force_N':float(np.linalg.norm(force[1])),
                'contact_positions_m':arr(data.contact_pos_w),'friction_force_N':arr(data.friction_forces_w)}
        return result

    def metrics(self):
        r,p=self.object_points();bottom=float((r.apply(self.hull_vertices)+p)[:,2].min())
        robot=self.env.scene['robot'].data
        contact=self.contacts();force=sum(v['object_normal_force_N'] for v in contact.values())
        table=float(self.env.scene['object_to_table_contact_sensor'].data.force_matrix_w[0].norm())
        thumb=sum(v['object_normal_force_N'] for k,v in contact.items() if 'thumb' in k)
        fingers=sum(v['object_normal_force_N'] for k,v in contact.items() if 'index' in k or 'middle' in k)
        return {'minimum_object_z_m':bottom,'whole_object_clearance_m':bottom-.75,
            'object_hand_normal_force_N':force,'object_table_normal_force_N':table,
            'thumb_contact_N':thumb,'index_middle_contact_N':fingers,
            'opposed_digits_contact':thumb>.1 and fingers>.1,'finger_contacts':contact,
            'strict_lift_now':bottom>.77 and force>.2 and table<.1 and float(robot.root_pos_w[0,2])>.45}

    def candidates(self,obs):
        kin=self.planner.kin;r,p=self.object_points();candidates=[]
        q=np.array([obs['joint_pos'][obs['privileged']['joint_names'].index(n)] for n in kin.names])
        robotquat=obs['robot_root_quat_wxyz'];root=np.array([-.05,-.50,.785])
        # Thin cross sections of the near arm, excluding central torso and large glove.
        open_points=np.array([[.041,-.003,0],[.119,-.005,.029],[.165,-.003,.029],
                              [.119,-.005,-.028],[.165,-.003,-.028],[.064,.016,0],[.064,.062,0],[.095,.078,0]])
        for z in [.20,.22,.24,.255]:
            section=self.vertices[(abs(self.vertices[:,2]-z)<.006)&(self.vertices[:,0]<-.055)]
            if len(section)<10:continue
            center=(np.percentile(section,5,axis=0)+np.percentile(section,95,axis=0))/2
            center[2]=z;g=r.apply(center)+p
            import itertools
            for roll,yaw in itertools.product([-1.57,-.7,0.,.5],[1.57,1.85,2.1]):
                wrist=Rotation.from_euler('xyz',[roll,0,yaw])
                palm=g-wrist.apply(self.hand_offset)
                wristpos=palm-wrist.apply([.07,0,0])
                posed=wrist.apply(open_points)+wristpos
                clearance=float(posed[:,2].min()-.009-.75)
                closing_axis=wrist.apply([0,1,0]);world=r.apply(section)
                width=float(np.ptp(world@closing_axis))
                solution,report=kin.solve(q,root,robotquat,'right',palm,[roll,0,yaw])
                kin.fk(solution,root,robotquat)
                actual=Rotation.from_matrix(kin.data.xmat[kin.model.body('right_wrist_yaw_link').id].reshape(3,3))
                error=float((wrist.inv()*actual).magnitude())
                candidates.append({'id':f'arm_z{z:.3f}_roll{roll:.2f}_yaw{yaw:.2f}','grasp_center_world_m':g.tolist(),
                    'grasp_center_local_m':center.tolist(),'palm_target_m':palm.tolist(),
                    'wrist_euler_xyz':[roll,0,yaw],'section_thickness_m':width,
                    'open_hand_table_clearance_proxy_m':clearance,'ik_position_error_m':report['ik_palm_error_m'],
                    'ik_rotation_error_rad':error,'stance_evaluated_m':root.tolist(),
                    'geometry_pass':width<.065 and clearance>-.002 and report['ik_palm_error_m']<.015 and error<.25,
                    'score':clearance-abs(width-.03)-report['ik_palm_error_m']-error*.03})
        return sorted(candidates,key=lambda c:(c['geometry_pass'],c['score']),reverse=True)

    def observe(self,data):
        palm=np.array(data['palm_positions']['right']);quat=np.array(data['wrist_quat_wxyz']['right'])
        pads={}
        bodies=data['privileged']['robot_bodies']
        for name,offset in [('right_hand_index_1_link',[.04,0,0]),('right_hand_middle_1_link',[.04,0,0]),('right_hand_thumb_2_link',[0,.04,0])]:
            b=bodies[name];q=np.array(b['quaternion_wxyz']);rr=Rotation.from_quat(q[[1,2,3,0]])
            pads[name]=(np.array(b['position_m'])+rr.apply(offset)).tolist()
        measured_center=(np.array(pads['right_hand_thumb_2_link'])+(np.array(pads['right_hand_index_1_link'])+pads['right_hand_middle_1_link'])/2)/2

        r=Rotation.from_quat(quat[[1,2,3,0]])
        details={'method':self.method,'controller':self.controller,'metrics':self.metrics(),
            'actual_grasp_center_m':(palm+r.apply(self.hand_offset)).tolist(),
            'grasp_center_offset_wrist_relative_to_palm_m':self.hand_offset.tolist(),
            'closure_fraction_target':self.close_to,'note':'Center is a calibrated geometric estimate; per-finger contacts are measured.'}
        details['finger_pad_centers_m']=pads
        details['pad_based_grasp_center_m']=measured_center.tolist()
        details['pad_based_offset_in_wrist_axes_m']=r.inv().apply(measured_center-palm).tolist()
        details['pad_to_object_surface_m']=float(__import__('scipy').spatial.cKDTree(self.object_points()[0].apply(self.vertices)+self.object_points()[1]).query(measured_center)[0])
        if self.planner.goal and self.planner.goal.get('wrist_euler_xyz') is not None:
            goal=Rotation.from_euler('xyz',self.planner.goal['wrist_euler_xyz'])
            details['actual_wrist_rotation_error_rad']=float((goal.inv()*r).magnitude())
        if self.grasp_target is not None:details['grasp_center_error_m']=float(np.linalg.norm(palm+r.apply(self.hand_offset)-self.grasp_target))
        if self.method in ('candidates','rollouts','combined'):
            details['candidates']=self.candidates(data)
        data['grasp_information']=details
        write_json(self.out/'latest_grasp_information.json',details)

    def apply(self,decision):
        p=decision.get('goal') or {};action=decision['action'];frame=self.frame()
        if action in ('close_hand','open_hand') or 'closure' in p:
            self.close_from=self.close_to
            self.close_to=float(p.get('closure',0. if action=='open_hand' else 1.))
            self.close_start=frame;self.close_duration=int(p.get('closure_steps',50))
        if 'target' in p and self.controller in ('ik_upper','ik_upper_gc'):
            obs=self.planner.observation();quat=np.array(obs['wrist_quat_wxyz']['right'])
            self.direct_origin=getattr(self,'direct_target_current',np.array(obs['palm_positions']['right'])).copy()
            self.direct_rotation=getattr(self,'direct_rotation_current',Rotation.from_quat(quat[[1,2,3,0]]))
            self.direct_goal_rotation=Rotation.from_euler('xyz',self.planner.goal['wrist_euler_xyz'])
            robot=self.env.scene['robot']
            if self.last_actual_ik is None:
                self.last_actual_ik=np.array([float(robot.data.joint_pos_target[0,robot.joint_names.index(n)]) for n in self.planner.kin.names])
        if 'grasp_center' in p:self.grasp_target=np.array(p['grasp_center'])
        if 'finger_stiffness' in p:
            import torch
            robot=self.env.scene['robot']
            ids=[i for i,n in enumerate(robot.joint_names) if n.startswith('right_hand_') and n!='right_hand_thumb_0_joint']
            gain=float(p['finger_stiffness']);assert .5<=gain<=4
            limits_before=robot.root_physx_view.get_dof_max_forces()[0,ids].cpu().tolist()
            stiffness_before=robot.root_physx_view.get_dof_stiffnesses()[0,ids].cpu().tolist()
            for actuator in robot.actuators.values():
                for j,name in enumerate(actuator.joint_names):
                    if name.startswith('right_hand_') and name!='right_hand_thumb_0_joint':
                        actuator.stiffness[:,j]=gain
            robot.write_joint_stiffness_to_sim(torch.full((1,len(ids)),gain,device=self.env.device),joint_ids=ids)
            limits_after=robot.root_physx_view.get_dof_max_forces()[0,ids].cpu().tolist()
            stiffness_after=robot.root_physx_view.get_dof_stiffnesses()[0,ids].cpu().tolist()
            assert limits_before==limits_after
            with (self.out/'finger_gain_changes.jsonl').open('a') as f:f.write(json.dumps({'frame':frame,'stiffness_Nm_per_rad':gain,'effort_limit_changed':False,
                'joint_names':[robot.joint_names[i] for i in ids], 'effort_limits_before_Nm':limits_before,'effort_limits_after_Nm':limits_after,
                'stiffness_before_Nm_per_rad':stiffness_before,'stiffness_after_Nm_per_rad':stiffness_after})+'\n')
        if 'target' in p:
            self.arm_progress_profile=p.get('progress_profile','quintic')
            self.arm_clearance_arc=np.asarray(p.get('clearance_arc_m',[0,0,0]),dtype=float)
            self.arm_orientation_finish=float(p.get('orientation_finish_fraction',1.))
        if 'arm_speed_limit_rad_s' in p:
            self.arm_speed_limit=float(np.clip(p['arm_speed_limit_rad_s'],.2,1.))
        if p.get('controller'):
            self.controller=p['controller']
            with (self.out/'controller_changes.jsonl').open('a') as f:f.write(json.dumps({'frame':frame,'controller':self.controller,'decision':decision['observation_id']})+'\n')

    def before_step(self):
        import torch
        frame=self.frame();progress=np.clip((frame-self.close_start)/max(self.close_duration,1),0,1)
        fraction=self.close_from+(self.close_to-self.close_from)*progress
        getattr(self.planner.lib,'_motion_hand_action_right')[frame:]=fraction*2-1
        if self.controller in ('ik_upper','ik_upper_gc') and self.planner.goal and self.planner.phase == 'grasp':
            robot=self.env.scene['robot'];data=robot.data
            q=np.array([float(data.joint_pos[0,robot.joint_names.index(n)]) for n in self.planner.kin.names])
            root=data.root_pos_w[0].cpu().numpy();quat=data.root_quat_w[0].cpu().numpy()
            u=np.clip((frame-self.planner.goal_start)/self.planner.duration,0,1);progress=u**3*(10+u*(-15+6*u))
            if getattr(self,'arm_progress_profile','quintic')=='early_deceleration':
                progress=35*u**3-105*u**4+126*u**5-70*u**6+15*u**7
            target=self.direct_origin+progress*(np.array(self.planner.goal['target'])-self.direct_origin)
            target+=getattr(self,'arm_clearance_arc',np.zeros(3))*np.sin(np.pi*progress)**2
            rotation_progress=progress
            if getattr(self,'arm_orientation_finish',1.)<1:
                v=np.clip(progress/self.arm_orientation_finish,0,1);rotation_progress=v**3*(10-15*v+6*v*v)
            rot=self.direct_rotation*Rotation.from_rotvec((self.direct_rotation.inv()*self.direct_goal_rotation).as_rotvec()*rotation_progress)
            self.direct_target_current=target.copy();self.direct_rotation_current=rot
            solved,report=self.planner.kin.solve(q,root,quat,'right',target,rot.as_euler('xyz'),allow_waist=False)
            if self.controller=='ik_upper_gc':
                # Gravity compensation as a bounded PD equilibrium offset. The
                # physical actuator's existing drive-force limit is unchanged.
                arm_ids=[i for i,n in enumerate(self.planner.kin.names) if n.startswith('right_') and any(k in n for k in ('shoulder','elbow','wrist'))]
                robot_ids=[robot.joint_names.index(self.planner.kin.names[i]) for i in arm_ids]
                gains=robot.root_physx_view.get_dof_stiffnesses()[0,robot_ids].cpu().numpy()
                gravity=self.planner.kin.data.qfrc_bias[6:][arm_ids].copy()
                offset=np.clip(gravity/np.maximum(gains,1e-3),-.4,.4)
                solved[arm_ids]+=offset
                bounds=self.planner.kin.model.jnt_range[1:][arm_ids]
                solved[arm_ids]=np.clip(solved[arm_ids],bounds[:,0]+1e-5,bounds[:,1]-1e-5)
                self.last_ik_debug={'gravity_compensation_Nm':gravity.tolist(),'PD_equilibrium_offset_rad':offset.tolist(),
                    'ik_position_error_m':report['ik_palm_error_m'],'motor_force_limits_changed':False}
            # Cap the right-arm command delta at 0.02 rad per 50 Hz step.
            # This is the separate IK arm controller, not a lag filter on SONIC output.
            arm_ids=[i for i,n in enumerate(self.planner.kin.names) if n.startswith('right_') and any(k in n for k in ('shoulder','elbow','wrist'))]
            previous=q if self.last_actual_ik is None else self.last_actual_ik
            desired_delta=np.clip(solved[arm_ids]-previous[arm_ids],-.02,.02)
            optimized=os.environ.get('G1_OPTIMIZED')=='1'
            if optimized:
                # Bound velocity and acceleration of only the separately controlled arm.
                # Task-space quintic progress still determines approach/contact speed.
                max_velocity=float(getattr(self,'arm_speed_limit',.8))
                if getattr(self,'arm_progress_profile','quintic')=='early_deceleration':
                    remaining=np.linalg.norm(np.array(self.planner.goal['target'])-target)
                    max_velocity=min(max_velocity,.45+.35*float(np.clip((remaining-.035)/.04,0,1)))
                desired_delta=np.clip(desired_delta,-max_velocity*.02,max_velocity*.02)
                # Preserve responsive feedback correction; no acceleration saturation.

            solved[arm_ids]=previous[arm_ids]+desired_delta
            self.last_actual_ik=solved
            with (self.out/'direct_arm_execution.jsonl').open('a') as stream:
                stream.write(json.dumps({'frame':frame,'target_palm_m':target.tolist(),'reference_euler':rot.as_euler('xyz').tolist(),'ik_error_m':report['ik_palm_error_m'],'max_arm_target_delta_rad':float(np.max(abs(solved[arm_ids]-previous[arm_ids]))),'controller':'actual-state IK + bounded gravity compensation, SONIC other joints'})+'\n')
        self.count+=1
