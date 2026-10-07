"""Join a stationary Cartesian skill to an independently replanned mobile start."""
import argparse
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
from common import *


def compile_mobile(skill, destination, dx=0., dy=0., yaw=0.):
    c=Path(destination).resolve(); c.mkdir(parents=True,exist_ok=False)
    skill=Path(skill).resolve()
    assert abs(dx)<=.05 and abs(dy)<=.05 and abs(yaw)<=4
    shutil.copytree(skill,c/'skill')
    source=W/'experiments/20260929_sonic_mimicgen_augmentation/smoke_runs_v2/candidate_000100'
    initial=arrays(source/'initial_state.npz')
    operation=arrays(skill/'initial_state.npz')
    helper=module('old_aug_transforms',W/'pipelines/sonic_mimicgen_augmentation/common.py')
    initial=helper.transform_initial(initial,initial['root_position'].copy(),np.array([dx,dy,0]),R.from_euler('z',yaw,degrees=True),include_object=False)
    # Only bottle changes for the common-origin series. Robot root/hands share one initial state.
    for key in ['object_position','object_quaternion_wxyz','object_linear_velocity','object_angular_velocity']:
        initial[key]=operation[key].copy()
    np.savez_compressed(c/'initial_state.npz',**initial)
    shutil.copy2(skill/'deployment_parameters.json',c/'deployment_parameters.json')
    plan=read(source/'plan.json')
    plan.update(source_fast=str(c/'skill'),start_root_m=initial['root_position'].tolist(),
        object_initial_position_m=initial['object_position'].tolist(),
        object_initial_quaternion_wxyz=initial['object_quaternion_wxyz'].tolist(),
        hypothesis='Measured stable walking handoff followed by live root-compensated teleop grasp maintains the stationary wrap grip.',
        architecture='SONIC planner -> mode0 encoder -> shared decoder; continuous handoff -> mode1 teleop encoder -> same decoder/history',
        initial_variant=dict(dx=dx,dy=dy,yaw_deg=yaw),common_origin=(dx==0 and dy==0 and yaw==0),
        training_ready=False)
    plan['augmentation']=dict(grasp=read(skill/'sample.json'),robot_start=plan['initial_variant'])
    plan['translation_world_m']=(initial['root_position']-np.asarray(plan['goal_root_m'])).tolist()
    plan['translation_distance_m']=float(np.linalg.norm(initial['root_position'][:2]-np.asarray(plan['goal_root_m'])[:2]))
    if 'planner_screen' in plan:plan['source_planner_screen']=plan.pop('planner_screen')
    write(c/'plan.json',plan)
    write(c/'sample.json',dict(grasp=read(skill/'sample.json'),robot_start=plan['initial_variant']))
    write(c/'prepared_files.json',{str(p.relative_to(c)):sha(p) for p in c.rglob('*') if p.is_file() and p.name!='prepared_files.json'})
    write(c/'status.json',dict(status='prepared',task_success=False))
    return c


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--skill',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--dx',type=float,default=0);p.add_argument('--dy',type=float,default=0);p.add_argument('--yaw',type=float,default=0)
    a=p.parse_args();print(compile_mobile(a.skill,a.output,a.dx,a.dy,a.yaw))
