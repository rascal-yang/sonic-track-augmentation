"""Separate physical Revo2 articulations fixed to G1 wrists.

Legacy fingers remain hidden, noncolliding, negligible-mass compatibility joints.
Revo2 masses, colliders, drives and mimic constraints take part in PhysX.
"""
import json
import os
from calibration_config import ROBOT_SCALE
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

BASE=Path(__file__).resolve().parents[1]
MOUNT_R=Rotation.from_matrix([[0,0,1],[1,0,0],[0,1,0]])
MOUNT_ROTATIONS={'right': MOUNT_R, 'left': Rotation.from_euler('x',np.pi)*MOUNT_R}
MOUNT_P=np.array([.025,0,0])
SPECS={}
for side in ['left','right']:
    root=ET.parse(BASE/'assets/brainco_candidate'/f'revo2_{side}_clean.urdf').getroot()
    SPECS[side]={j.get('name'):{'limit':{k:float(v) for k,v in j.find('limit').attrib.items()},'mimic':None if j.find('mimic') is None else j.find('mimic').attrib} for j in root.findall('joint') if j.get('type')!='fixed'}


def patch_robot_spawn(scene):
    import isaaclab.sim as sim
    from isaaclab.sim.utils import clone
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    original_robot_spawn=scene.robot.spawn.func
    @clone
    def robot_spawn(path,cfg,translation=None,orientation=None,**kwargs):
        result=original_robot_spawn(path,cfg,translation,orientation,**kwargs)
        stage=sim.get_current_stage()
        for side in ['left','right']:
            for prim in list(Usd.PrimRange(stage.GetPrimAtPath(path))):
                if prim.GetName().startswith(side+'_hand_') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
                    mass=UsdPhysics.MassAPI.Apply(prim);mass.CreateMassAttr(.0001);mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-7))
                    for child in Usd.PrimRange(prim):
                        if child.IsInstance():child.SetInstanceable(False)
                        if child.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
        return result
    scene.robot.spawn.func=robot_spawn



def configure_scene(scene):
    import isaaclab.sim as sim
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.sim.utils import clone
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,Gf
    for side in ['left','right']:
        @clone
        def hand_spawn(path,cfg,translation=None,orientation=None,_side=side,**kwargs):
            result=sim.spawn_from_usd(path,cfg,translation,orientation,**kwargs)
            stage=sim.get_current_stage();robot_path=str(Path(path).parent/'Robot');wrist=robot_path+'/'+_side+'_wrist_yaw_link'
            handbase=path+'/'+_side+'_base_link'
            joint=UsdPhysics.FixedJoint.Define(stage,path+'/WristMount')
            joint.CreateBody0Rel().SetTargets([wrist]);joint.CreateBody1Rel().SetTargets([handbase]);joint.CreateExcludeFromArticulationAttr(True)
            xyzw=MOUNT_ROTATIONS[_side].as_quat();joint.CreateLocalPos0Attr(Gf.Vec3f(*MOUNT_P));joint.CreateLocalRot0Attr(Gf.Quatf(float(xyzw[3]),Gf.Vec3f(*xyzw[:3])))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0));joint.CreateLocalRot1Attr(Gf.Quatf(1))
            # Wrist/palm adjacency may overlap at the adapter; filter only that pair.
            for neighbor in [_side+'_wrist_yaw_link',_side+'_wrist_pitch_link',_side+'_wrist_roll_link']:
                prim=stage.GetPrimAtPath(robot_path+'/'+neighbor)
                if prim:UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(handbase)
            for name,spec in SPECS[_side].items():
                if spec['mimic']:
                    j=stage.GetPrimAtPath(path+'/joints/'+name);reference=path+'/joints/'+spec['mimic']['joint']
                    axis='rot'+UsdPhysics.RevoluteJoint(j).GetAxisAttr().Get()
                    api=PhysxSchema.PhysxMimicJointAPI.Apply(j,axis);api.CreateReferenceJointRel().SetTargets([reference]);api.CreateGearingAttr(-float(spec['mimic'].get('multiplier',1)));api.CreateOffsetAttr(0.)
                    drive=UsdPhysics.DriveAPI(j,'angular');drive.GetStiffnessAttr().Set(0);drive.GetDampingAttr().Set(0)
            return result
        active={n:s for n,s in SPECS[side].items() if not s['mimic']};passive={n:s for n,s in SPECS[side].items() if s['mimic']}
        cfg=ArticulationCfg(prim_path='{ENV_REGEX_NS}/BrainCo_'+side,
            spawn=sim.UsdFileCfg(usd_path=str(BASE/'assets/brainco_usd'/side/'hand.usd'),func=hand_spawn,activate_contact_sensors=True,rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False,max_depenetration_velocity=1.),articulation_props=sim.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=32,solver_velocity_iteration_count=8)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,.8)),
            actuators={'active':ImplicitActuatorCfg(joint_names_expr=list(active),stiffness=4.,damping=.12,armature=.0002,effort_limit_sim={n:s['limit']['effort'] for n,s in active.items()},velocity_limit_sim={n:s['limit']['velocity'] for n,s in active.items()}),
                       'mimic':ImplicitActuatorCfg(joint_names_expr=list(passive),stiffness=0.,damping=0.,armature=.0002,effort_limit_sim=0.,velocity_limit_sim=4.)})
        setattr(scene,'brainco_'+side,cfg)
    import grasp_extensions
    links=['right_base_link','right_thumb_metacarpal_link','right_thumb_proximal_link','right_thumb_distal_link']+[f'right_{finger}_{part}_link' for finger in ['index','middle','ring','pinky'] for part in ['proximal','distal']]
    grasp_extensions.FINGER_LINKS=links
    for i,name in enumerate(links):
        setattr(scene,f'grasp_finger_{i}',ContactSensorCfg(prim_path='{ENV_REGEX_NS}/BrainCo_right/'+name,filter_prim_paths_expr=['{ENV_REGEX_NS}/Object','{ENV_REGEX_NS}/Table'],track_pose=True,track_contact_points=True,track_friction_forces=True,max_contact_data_count_per_prim=64,history_length=2))


    # Separate per-body sensors preserve Isaac Lab's one-body filtered contact rule.
    left_tree=ET.parse(BASE/'assets/brainco_candidate/revo2_left_clean.urdf').getroot()
    left_links=['left_base_link']+[j.find('child').get('link') for j in left_tree.findall('joint') if j.get('type')!='fixed']
    for i,name in enumerate(left_links):
        setattr(scene,f'table_watch_left_{i}',ContactSensorCfg(prim_path='{ENV_REGEX_NS}/BrainCo_left/'+name,filter_prim_paths_expr=['{ENV_REGEX_NS}/Table'],history_length=2))
    for side in ['left','right']:
        for part in ['elbow_link','wrist_roll_link','wrist_pitch_link','wrist_yaw_link']:
            name=side+'_'+part
            setattr(scene,'table_watch_'+name,ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+name,filter_prim_paths_expr=['{ENV_REGEX_NS}/Table'],history_length=2))


    for name in list(vars(scene)):
        if name.startswith('table_watch_') or name.startswith('grasp_finger_'):
            original=getattr(scene,name)
            setattr(scene,'obstacle_watch_'+name,ContactSensorCfg(prim_path=original.prim_path,filter_prim_paths_expr=['{ENV_REGEX_NS}/Room'],history_length=2))


def install_extension(extension):
    import torch
    from experiment import write_json
    env=extension.env;robot=env.scene['robot'];extension.hand_offset=np.array([.075,.02,0])*ROBOT_SCALE
    original_before=extension.before_step;original_apply=extension.apply
    grasp_started=False
    finger_overrides={}
    opposition=.75
    command_from={}
    command_start=0
    command_duration=80
    def align_on_reset():
        for side in ['left','right']:
            hand=env.scene['brainco_'+side];i=robot.body_names.index(side+'_wrist_yaw_link');p=robot.data.body_pos_w[0,i].cpu().numpy();q=robot.data.body_quat_w[0,i].cpu().numpy();r=Rotation.from_quat(q[[1,2,3,0]]);r2=r*MOUNT_ROTATIONS[side];q2=r2.as_quat();state=hand.data.default_root_state.clone();state[0,:3]=torch.tensor(p+r.apply(MOUNT_P*ROBOT_SCALE),device=env.device);state[0,3:7]=torch.tensor(q2[[3,0,1,2]],device=env.device);state[0,7:]=0;hand.write_root_state_to_sim(state);hand.write_joint_state_to_sim(torch.zeros_like(hand.data.joint_pos),torch.zeros_like(hand.data.joint_vel));hand.reset()
    extension.align_brainco=align_on_reset
    def before():
        original_before()
        frame=extension.frame();a=np.clip((frame-extension.close_start)/max(extension.close_duration,1),0,1);fraction=extension.close_from+(extension.close_to-extension.close_from)*a
        for side in ['left','right']:
            hand=env.scene['brainco_'+side];target=torch.zeros_like(hand.data.joint_pos)
            for i,n in enumerate(hand.joint_names):
                spec=SPECS[side][n]
                if not spec['mimic']:
                    f=fraction if side=='right' else 0.
                    if side=='right':
                        for digit,value in finger_overrides.items():
                            if ('_'+digit+'_') in n:f=value
                    if 'thumb_metacarpal' in n:f=(opposition if side=='right' else .15)*min(frame/100.,1.)
                    closed=max([spec['limit']['lower'],spec['limit']['upper']],key=abs)
                    target[0,i]=f*closed
                    if side=='right' and n in command_from:
                        blend=float(np.clip((frame-command_start)/max(command_duration,1),0,1))
                        target[0,i]=command_from[n]+blend*(float(target[0,i])-command_from[n])
            hand.set_joint_position_target(target)
    def apply(decision):
        nonlocal finger_overrides, opposition, command_from, command_start, command_duration, grasp_started
        goal=dict(decision.get('goal') or {});gain=goal.pop('finger_stiffness',None)
        action=decision['action']
        if action=='close_hand':
            grasp_started=True
        elif action=='open_hand':
            grasp_started=False
        # Open means all five flexion drives. Old per-finger overrides must not
        # survive release, or turn initial locomotion into a three-finger fist.
        # Once grasping, preserve the grip during reach/lift/navigation updates.
        if action=='open_hand' or (not grasp_started and action in ('navigate','reach')):
            goal['closure']=0.
            goal['finger_closures']={}
        overrides=goal.pop('finger_closures',None);opp=goal.pop('thumb_opposition',None)
        if overrides is not None or opp is not None:
            hand=env.scene['brainco_right']
            command_from={n:float(hand.data.joint_pos[0,i]) for i,n in enumerate(hand.joint_names) if not SPECS['right'][n]['mimic']}
            command_start=extension.frame();command_duration=int(goal.get('closure_steps',80))
        if overrides is not None:
            assert all(k in ['thumb','index','middle','ring','pinky'] and 0<=float(v)<=1 for k,v in overrides.items())
            finger_overrides={k:float(v) for k,v in overrides.items()}
        if opp is not None:
            assert 0<=float(opp)<=1
            opposition=float(opp)
        copy=dict(decision,goal=goal);original_apply(copy)
        if not os.environ.get('G1_LEAN_MODE'):
            with (extension.out/'hand_command_log.jsonl').open('a') as stream:
                stream.write(json.dumps({'observation_id':decision['observation_id'],
                    'frame':extension.frame(),'action':action,'grasp_started':grasp_started,
                    'requested_goal':decision.get('goal',{}),'applied_goal':goal,
                    'effective_finger_overrides':dict(finger_overrides)})+'\n')
        if gain is not None:
            assert .5<=float(gain)<=4
            hand=env.scene['brainco_right'];ids=[i for i,n in enumerate(hand.joint_names) if not SPECS['right'][n]['mimic']]
            hand.actuators['active'].stiffness[:]=float(gain);hand.write_joint_stiffness_to_sim(torch.full((1,len(ids)),float(gain),device=env.device),joint_ids=ids)
    def observe(data):
        hand=env.scene['brainco_right'];pads={};states={}
        for side in ['left','right']:
            h=env.scene['brainco_'+side];states[side]={'joint_names':h.joint_names,'joint_position':h.data.joint_pos[0].cpu().tolist(),'joint_velocity':h.data.joint_vel[0].cpu().tolist(),'root_position':h.data.root_pos_w[0].cpu().tolist()}
        for finger in ['thumb','index','middle','ring','pinky']:
            name=f'right_{finger}_distal_link';i=hand.body_names.index(name);p=hand.data.body_pos_w[0,i].cpu().numpy();q=hand.data.body_quat_w[0,i].cpu().numpy();r=Rotation.from_quat(q[[1,2,3,0]]);offset=[0,.024,0] if finger=='thumb' else [0,0,.025];pads[name]=(p+r.apply(np.asarray(offset)*ROBOT_SCALE)).tolist()
        center=(np.array(pads['right_thumb_distal_link'])+(np.array(pads['right_index_distal_link'])+pads['right_middle_distal_link'])/2)/2
        palm=np.array(data['palm_positions']['right']);q=np.array(data['wrist_quat_wxyz']['right']);r=Rotation.from_quat(q[[1,2,3,0]])
        details={'method':'Revo2 measured feedback','controller':extension.controller,'metrics':extension.metrics(),'actual_grasp_center_m':center.tolist(),'closure_fraction_target':extension.close_to,'finger_pad_centers_m':pads,'pad_based_grasp_center_m':center.tolist(),'pad_based_offset_in_wrist_axes_m':r.inv().apply(center-palm).tolist(),'finger_closures_override':dict(finger_overrides),'thumb_opposition_target':opposition,'note':'Revo2 physical links, six active drives and five passive PhysX mimic constraints per hand.'}
        data['grasp_information']=details;data['brainco']=states;write_json(extension.out/'latest_grasp_information.json',details)
        data['grasp_information']['grasp_started']=grasp_started
    extension.before_step=before;extension.apply=apply;extension.observe=observe
    write_json(extension.out/'brainco_integration.json',{'model':'BrainCo Revo2 local V2 URDF','attachment':'physical fixed joint excluded from articulation between G1 wrist and independent Revo2 articulation; no per-step pose teleport','legacy_fingers':'hidden, collision disabled, .0001kg per link compatibility placeholders','mount_p':MOUNT_P.tolist(),'mount_R_by_side':{k:v.as_matrix().tolist() for k,v in MOUNT_ROTATIONS.items()},'active_joints_per_hand':6,'passive_mimic_joints_per_hand':5,'joint_effort_limits':'local URDF limits; not measured hardware calibration','stabilization':'estimated rotor armature .0002 kg*m2 per hand joint; solver32/8; zero-angle open pose, signed joint range for closure','mimic':'q_distal = multiplier * q_proximal via PhysX mimic constraint'})
