"""Reconstructed desk/plush scene and complete privileged state for planner trials."""
import json
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from experiment import ROOT, KEY, SONIC, write_json

ASSET = Path(__file__).resolve().parent.parent / 'assets/water_bottle_v1'
DESK = ROOT / 'data/office_desk_v1'
ROOM=json.loads((Path(__file__).resolve().parent.parent/'rooms.json').read_text())[os.environ['G1_ROOM']]
SHIFT=np.array(ROOM['table_center'])


def prepare_reference(data, out):
    import joblib
    obj_file = data/'objects'/f'{KEY}.pkl'
    objects = joblib.load(obj_file)
    pose = json.loads((ASSET/'validation_isaac.json').read_text())
    pose['settled_position_m']=(np.array(pose['settled_position_m'])+SHIFT+np.array(ROOM.get('object_shift',[0,0,0]))).tolist()
    for value in objects.values():
        value['root_pos'][:] = pose['settled_position_m']
        value['root_quat'][:] = pose['settled_quaternion_wxyz']
    joblib.dump(objects, obj_file)
    robot_file = data/'robot'/f'{KEY}.pkl'
    robot = joblib.load(robot_file)
    # Relaxed symmetric arms: little elbow flexion, palms alongside the hips.
    # Joint convention is from this G1 MJCF; zero shoulder pitch points forward.
    from online_goals import Kinematics
    kin=Kinematics(SONIC)
    initial_root=np.array(json.loads(os.environ['G1_START_ROOT']))
    initial_quat=Rotation.from_euler('z',np.pi/2).as_quat()[[3,0,1,2]]
    parked=kin.default.copy()
    for i,name in enumerate(kin.names):
        if name.endswith('shoulder_pitch_joint'): parked[i]=.8
        if name.endswith('elbow_joint'): parked[i]=.2
    kin.fk(parked,initial_root,initial_quat)
    for side,filename in [('left','inactive_arm_setup.json'),('right','active_arm_setup.json')]:
        bid=kin.model.body(side+'_wrist_yaw_link').id
        write_json(out/filename,dict(target_palm_m=kin.palm(side).tolist(),
            wrist_euler=Rotation.from_matrix(kin.data.xmat[bid].reshape(3,3)).as_euler('xyz').tolist(),
            method='symmetric relaxed joint pose, shoulder pitch .8 rad, elbow .2 rad',initial_only=True))
    for value in robot.values():
        value['dof'][:,:29]=parked
        value['pose_aa'][:,1:]=parked[:,None]*kin.axes
        value['root_trans_offset'][:] = initial_root
        value['root_rot'][:] = Rotation.from_euler('z', np.pi/2).as_quat()
        value['pose_aa'][:,0] = [0,0,np.pi/2]
    joblib.dump(robot, robot_file)
    meta = {'object_name':'water_bottle', 'table_pos':np.zeros(3),
            'table_quat':np.array([1.,0,0,0]), 'table_size':np.array([*ROOM['table_size'][:2],.05])}
    joblib.dump(meta, data/'meta'/f'{KEY}.pkl')
    # Replace the old apple descriptor with distances to the current toy surface.
    vertices = np.load(ASSET/'collision_source.npz')['vertices']
    basis = np.load(data/'bps/_basis.npy')
    template = np.load(data/'bps'/f'{KEY}.npy')
    distances = cKDTree(vertices).query(basis.reshape(-1,3))[0]
    if distances.size == template.size:
        np.save(data/'bps'/f'{KEY}.npy',distances.reshape(template.shape).astype(template.dtype))
    else:
        raise ValueError(('Unexpected BPS shape', basis.shape, template.shape))
    write_json(out/'reconstructed_setup.json', {'robot_root_m':initial_root.tolist(), 'robot_yaw_rad':np.pi/2,
        'object_position_m':pose['settled_position_m'], 'object_quaternion_wxyz':pose['settled_quaternion_wxyz'],
        'desk_asset':ROOM['table_asset'],'room':ROOM, 'object_asset':str(ASSET/'bottle.usda'),
        'desk_top_m':.75, 'bottle_height_measured_m':.23, 'bottle_mass_assumed_kg':.25,
        'BPS':'nearest vertex distances to water bottle using existing basis',
        'physical_model':'rigid water bottle proxy; no deformation', 'demonstration_used':False})


def make_table_wrapper(path):
    from pxr import Usd, UsdGeom, UsdPhysics
    stage=Usd.Stage.CreateNew(str(path))
    root=UsdGeom.Xform.Define(stage,'/Table')
    root.GetPrim().GetReferences().AddReference(ROOM['table_asset'], '/Table')
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1)
    api=UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    api.CreateRigidBodyEnabledAttr(True);api.CreateKinematicEnabledAttr(True)
    stage.GetRootLayer().Save()


def add_privileged(env, observation):
    robot=env.scene['robot']; obj=env.scene['object']
    def array(value): return value.detach().cpu().numpy()
    p=array(obj.data.root_pos_w[0]);q=array(obj.data.root_quat_w[0])
    rotation=Rotation.from_quat(q[[1,2,3,0]])
    local=np.load(ASSET/'collision_source.npz')['vertices']
    points=rotation.apply(local)+p
    tree=cKDTree(points)
    body_pos=array(robot.data.body_pos_w[0])
    palms=observation['palm_positions']
    table_min=SHIFT+np.array([-ROOM['table_size'][0]/2,-ROOM['table_size'][1]/2,.70]);table_max=SHIFT+np.array([ROOM['table_size'][0]/2,ROOM['table_size'][1]/2,.75])
    def table_distance(point): return float(np.linalg.norm(np.maximum(np.maximum(table_min-point, point-table_max),0)))
    bodies={}
    for i,name in enumerate(robot.body_names):
        bodies[name]={'position_m':body_pos[i].tolist(),'quaternion_wxyz':array(robot.data.body_quat_w[0,i]).tolist(),
            'linear_velocity_m_s':array(robot.data.body_lin_vel_w[0,i]).tolist(),
            'angular_velocity_rad_s':array(robot.data.body_ang_vel_w[0,i]).tolist(),
            'to_plush_surface_vertex_m':float(tree.query(body_pos[i])[0]),
            'to_tabletop_aabb_m':table_distance(body_pos[i]),'height_over_table_m':float(body_pos[i,2]-.75)}
    pairs={}
    for name,point in dict(palms,root=observation['robot_root_pos']).items():
        point=np.asarray(point);dist,index=tree.query(point)
        pairs[name]={'to_object_origin_m':float(np.linalg.norm(point-p)),
            'object_origin_delta_m':(p-point).tolist(),'to_surface_vertex_m':float(dist),
            'nearest_surface_world_m':points[index].tolist(),'to_tabletop_aabb_m':table_distance(point),
            'height_over_table_m':float(point[2]-.75),'front_edge_y_delta_m':float(table_min[1]-point[1])}
    sensors={}
    for name,sensor in env.scene.sensors.items():
        data=sensor.data;entry={}
        for attribute in ('net_forces_w','force_matrix_w'):
            value=getattr(data,attribute,None)
            if value is not None:
                a=array(value[0]);entry[attribute]=a.tolist();entry[attribute+'_norm_N']=float(np.linalg.norm(a))
        if entry:sensors[name]=entry
    candidates={}
    # Semantic region labels are approximate, inferred from canonical upright mesh.
    for name,lo,hi in [('body',[-.04,-.04,.04],[.04,.04,.16]), ('neck',[-.04,-.04,.18],[.04,.04,.21]), ('cap',[-.04,-.04,.21],[.04,.04,.23])]:
        selected=local[np.all((local>=lo)&(local<=hi),axis=1)]
        if len(selected):
            center=(selected.min(0)+selected.max(0))/2
            candidates[name]={'local_center_m':center.tolist(),'world_center_m':(rotation.apply(center)+p).tolist(),
                'local_extent_m':np.ptp(selected,axis=0).tolist(),'label_source':'approximate geometry region, not measured anatomy'}
    observation['privileged']={'source':'Isaac Sim ground truth; dimensions/material parameters remain reconstruction estimates',
        'units':'meters, seconds, Newtons; world Z up; quaternion wxyz',
        'object_origin_note':'canonical upright bottom, not center of mass',
        'object_aabb_world_m':[points.min(0).tolist(),points.max(0).tolist()],
        'object_linear_velocity_m_s':array(obj.data.root_lin_vel_w[0]).tolist(),
        'object_angular_velocity_rad_s':array(obj.data.root_ang_vel_w[0]).tolist(),
        'mass_kg':.25,'rigid_body':True,'tabletop_aabb_m':[table_min.tolist(),table_max.tolist()],
        'robot_body_names':robot.body_names,'robot_bodies':bodies,
        'joint_names':robot.joint_names,'joint_velocity_rad_s':array(robot.data.joint_vel[0]).tolist(),
        'joint_limits_rad':array(robot.data.soft_joint_pos_limits[0]).tolist(),
        'applied_joint_torque_Nm':array(robot.data.applied_torque[0]).tolist(),
        'distances':pairs,'contact_sensors':sensors,'candidate_regions':candidates,
        'distance_note':'surface distances are to mesh vertices; tabletop distance is to its AABB, not whole-body collision clearance'}
    return observation


if __name__=='__main__':
    import argparse, os
    parser=argparse.ArgumentParser();parser.add_argument('name');args=parser.parse_args()
    os.environ['G1_DESK_PLUSH']='1'
    from experiment import run
    run(args.name,sonic_planner=True)
