"""Compile bounded randomized candidates. Prepared is never a physical-success label."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R, Slerp
from common import (HERE, WORKSPACE, BASE, WALK, SOURCE, MODEL_ROOT, arrays, read, write,
                    sha, transform_initial, points_transform, quat_rotate,
                    contact_preserving_scale, encode_reference)


def sample(seed, profile):
    rng = np.random.default_rng(seed)
    if profile == 'identity':
        return dict(seed=seed, profile=profile, start_lateral_m=0., start_longitudinal_m=0.,
                    start_yaw_deg=0., task_translation_m=[0., 0., 0.], task_yaw_deg=0.,
                    approach_scale=1., time_scale=1., planner_seed=17)
    # Smoke isolates a small initial robot displacement; full profiles also adapt the skill.
    smoke = profile == 'smoke'
    return dict(seed=seed, profile=profile,
                start_lateral_m=float(rng.uniform(-.005 if smoke else -.05, .005 if smoke else .05)),
                start_longitudinal_m=float(rng.uniform(-.005 if smoke else -.06, .005 if smoke else .06)),
                start_yaw_deg=0. if smoke else float(rng.uniform(-4, 4)),
                task_translation_m=([float(rng.uniform(-.015,.015)), float(rng.uniform(-.01,.01)), 0.]
                                    if profile == 'object_se2' else [0.,0.,0.]),
                task_yaw_deg=float(rng.uniform(-2,2)) if profile == 'object_se2' else 0.,
                approach_scale=1. if smoke else float(rng.uniform(.98,1.02)),
                time_scale=1. if smoke else float(rng.uniform(1.,1.06)),
                planner_seed=17)


def retime(ref, factor):
    if factor == 1:
        return ref
    old = ref['time_s'].copy()
    # Scale pre-hold duration; preserve the entire source hold duration.
    lift = 8.2
    end_lift = round(lift * factor / .02) * .02
    t = np.arange(round((end_lift + old[-1] - lift)/.02)+1) * .02
    source_t = np.where(t <= end_lift, t * lift/end_lift, lift + t-end_lift)
    source_t = np.clip(source_t, old[0], old[-1])
    rate = np.where(t <= end_lift, lift/end_lift, 1.)
    out = {}
    for key, value in ref.items():
        if not value.ndim or value.shape[0] != len(old):
            out[key] = value.copy()
        elif 'quaternion' in key:
            flat = value.reshape(len(old), -1, 4)
            out[key] = np.stack([Slerp(old, R.from_quat(flat[:,j][:,[1,2,3,0]]))(source_t).as_quat()[:,[3,0,1,2]]
                                 for j in range(flat.shape[1])], axis=1).reshape((len(t),)+value.shape[1:])
        else:
            flat = value.reshape(len(old), -1)
            out[key] = np.stack([np.interp(source_t, old, flat[:,j]) for j in range(flat.shape[1])], axis=1).reshape((len(t),)+value.shape[1:])
    spline = CubicHermiteSpline(old, ref['g1_joint_position_rad'], ref['g1_joint_velocity_rad_s'])
    out['g1_joint_position_rad'] = spline(source_t)
    out['g1_joint_velocity_rad_s'] = spline(source_t, 1) * rate[:,None]
    out['time_s'] = t
    out['control_index'] = np.arange(len(t))
    out['source_time_s'] = source_t
    for side in ['left','right']:
        out[f'revo2_{side}_joint_velocity_rad_s'] = np.gradient(out[f'revo2_{side}_joint_position_rad'], .02, axis=0)
    return out


def compile_candidate(destination, spec):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    write(destination/'sample.json', spec)
    write(destination/'status.json', dict(status='preparing', task_success=False, training_ready=False))
    try:
        return _compile(destination, spec)
    except Exception as e:
        write(destination/'status.json', dict(status='preparation_failed', error=repr(e), task_success=False, training_ready=False))
        raise


def _compile(c, spec):
    sys.path.insert(0,str(BASE/'code'))
    sys.path.insert(0,str(WORKSPACE/'g1_closed_loop/.deps'))
    import onnxruntime as ort
    from online_goals import Kinematics
    from sonic_planner_goals import SonicONNX
    from calibration_config import ROBOT_SCALE
    original = arrays(SOURCE/'grasp_reference.npz')
    ref = {k:v.copy() for k,v in original.items()}
    source_init = arrays(SOURCE/'initial_state.npz')
    pivot = source_init['object_position'].astype(float)
    delta = np.array(spec['task_translation_m'])
    rotation = R.from_euler('z', spec['task_yaw_deg'], degrees=True)
    if abs(delta[2])>1e-9 or np.linalg.norm(delta[:2])>.04 or abs(spec['task_yaw_deg'])>5:
        raise ValueError('Task SE2 range exceeds validated geometric screening envelope')
    for key in ['root_position_world_m','body_position_world_m','object_position_world_m']:
        ref[key] = points_transform(ref[key],pivot,delta,rotation)
    for key in ['root_quaternion_world_wxyz','body_quaternion_world_wxyz','object_quaternion_world_wxyz']:
        ref[key] = quat_rotate(ref[key],rotation)
    if not .95 <= spec['approach_scale'] <= 1.05 or not .9 <= spec['time_scale'] <= 1.15:
        raise ValueError('Unsupported scaling envelope')
    kin = Kinematics('__RUNTIME__/vendor/SONIC')
    names = ref['g1_joint_names'].tolist()
    order = [names.index(k) for k in kin.names]
    q0 = ref['g1_joint_position_rad'].astype(float).copy()
    right = [i for i,n in enumerate(kin.names) if n.startswith('right_') and any(k in n for k in ['shoulder','elbow','wrist'])]
    palms, orientations = [], []
    bid = kin.model.body('right_wrist_yaw_link').id
    for i,q in enumerate(q0):
        kin.fk(q[order],ref['root_position_world_m'][i],ref['root_quaternion_world_wxyz'][i])
        palms.append(kin.palm('right').copy())
        orientations.append(kin.data.xmat[bid].reshape(3,3).copy())
    palms = np.array(palms)
    pregrasp = int(round(4.9/.02))
    targets = contact_preserving_scale(palms,palms[pregrasp],ref['time_s'],spec['approach_scale'])
    errors, angles = [], []
    if spec['approach_scale'] != 1:
        for i in range(1,pregrasp):
            q,report = kin.solve(q0[i,order],ref['root_position_world_m'][i],ref['root_quaternion_world_wxyz'][i],
                                 'right',targets[i],R.from_matrix(orientations[i]).as_euler('xyz'),allow_waist=False)
            angle = (R.from_matrix(orientations[i]).inv()*R.from_matrix(kin.data.xmat[bid].reshape(3,3))).magnitude()
            if report['ik_palm_error_m']>.003 or angle>.025 or not report['joint_limits_satisfied']:
                raise ValueError(f'IK residual rejected at frame {i}: {report}, orientation={angle}')
            ref['g1_joint_position_rad'][i,order] = q
            errors.append(report['ik_palm_error_m']);angles.append(angle)
        # Preserve source analytical velocity; add only the derivative of the correction.
        correction = ref['g1_joint_position_rad'].astype(float)-q0
        ref['g1_joint_velocity_rad_s'] = ref['g1_joint_velocity_rad_s'] + np.gradient(correction,.02,axis=0)
    # Update FK-supported body poses by correction; keep source geometric calibration offsets.
    model_bodies = {kin.model.body(i).name:i for i in range(kin.model.nbody)}
    for i in range(len(q0)):
        kin.fk(q0[i,order],ref['root_position_world_m'][i],ref['root_quaternion_world_wxyz'][i])
        oldp,oldr = kin.data.xpos.copy(), kin.data.xmat.copy()
        kin.fk(ref['g1_joint_position_rad'][i,order],ref['root_position_world_m'][i],ref['root_quaternion_world_wxyz'][i])
        for j,name in enumerate(ref['body_names']):
            if name not in model_bodies:continue
            k=model_bodies[name]
            ref['body_position_world_m'][i,j] += kin.data.xpos[k]-oldp[k]
            dr=R.from_matrix(kin.data.xmat[k].reshape(3,3)@oldr[k].reshape(3,3).T)
            ref['body_quaternion_world_wxyz'][i,j] = quat_rotate(ref['body_quaternion_world_wxyz'][i,j],dr)
    # Scaling is zero at start and throughout contact: init joint state is preserved.
    assert np.array_equal(ref['g1_joint_position_rad'][pregrasp:],q0[pregrasp:])
    ref = retime(ref,spec['time_scale'])
    q = ref['g1_joint_position_rad'][:,order]
    lower,upper = kin.model.jnt_range[1:].T
    if np.max(np.maximum(lower-q,q-upper))>1e-6 or np.abs(np.diff(q,axis=0)).max()>.12:
        raise ValueError('Reference joint limits or inter-frame jump rejected')
    skill = c/'skill';skill.mkdir()
    np.savez_compressed(skill/'grasp_reference.npz',**ref)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    model=MODEL_ROOT/'policy/release/model_encoder.onnx'
    encoder=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    source_tokens=arrays(SOURCE/'grasp_tokens.npz')
    if spec['approach_scale']==1 and spec['time_scale']==1:
        # Global rigid-transform equivariance: retain the verified quantized stream exactly.
        encoded=source_tokens
        encoding='Reuse exact verified tokens: q/dq and root-relative rotations invariant under global SE2'
    else:
        encoded=encode_reference(ref,source_tokens['encoder_joint_names'].tolist(),encoder)
        encoding='Reencoded adapted reference, never reencoded measured rollout states'
    np.savez_compressed(skill/'grasp_tokens.npz',**encoded)
    op_init=transform_initial(source_init,pivot,delta,rotation)
    np.savez_compressed(skill/'initial_state.npz',**op_init)
    source_plan=read(WALK/'plan.json')
    initial=transform_initial(arrays(WALK/'initial_state.npz'),pivot,delta,rotation)
    facing=rotation.apply(source_plan['forward_world'])
    lateral=np.array([facing[1],-facing[0],0.])
    start_delta=spec['start_lateral_m']*lateral+spec['start_longitudinal_m']*facing
    initial=transform_initial(initial,initial['root_position'].copy(),start_delta,
                              R.from_euler('z',spec['start_yaw_deg'],degrees=True),include_object=False)
    if abs(spec['start_yaw_deg'])>8 or abs(spec['start_lateral_m'])>.12 or abs(spec['start_longitudinal_m'])>.15:
        raise ValueError('Start sampling exceeds clear approach corridor')
    if not (-.50<initial['object_position'][0]<.50 and -.24<initial['object_position'][1]<.24):
        raise ValueError('Bottle support footprint outside tabletop margin')
    np.savez_compressed(c/'initial_state.npz',**initial)
    for dest in [c,skill]:shutil.copy2(SOURCE/'deployment_parameters.json',dest/'deployment_parameters.json')
    goal=points_transform(np.array(source_plan['goal_root_m']),pivot,delta,rotation)
    navigation=points_transform(np.array(source_plan['navigation_target_root_m']),pivot,delta,rotation)
    plan=dict(source_plan,source_fast=str(skill),start_root_m=initial['root_position'].tolist(),goal_root_m=goal.tolist(),
              navigation_target_root_m=navigation.tolist(),yaw_rad=float(np.arctan2(facing[1],facing[0])),forward_world=facing.tolist(),
              translation_world_m=(initial['root_position']-goal).tolist(),translation_distance_m=float(np.linalg.norm(initial['root_position'][:2]-goal[:2])),
              augmentation=spec,planner_seed=spec['planner_seed'],minimum_travel_m=.65,grasp_frames=len(ref['time_s']),
              object_initial_position_m=initial['object_position'].tolist(),object_initial_quaternion_wxyz=initial['object_quaternion_wxyz'].tolist(),
              success='Complete adapted grasp; travel>=0.65m; dock<=0.05m; strict final hold>=3s; opposed>=95%; monitored table<=0.05N; final3s slip<=15mm/rotation<=10deg; capture and mounts pass.',
              hypothesis='Object-centric skill reuse and independently replanned locomotion survive bounded initialization/path changes.',
              architecture='SONIC V2 online walking reference -> encoder -> decoder, then compiled skill tokens; separate Revo2 targets',training_ready=False)
    write(c/'plan.json',plan)
    # Reuse actual mesh geometry checker, passing absolute candidate path.
    geo=HERE/'geometry_check.py'
    with (c/'geometry.log').open('w') as f:
        subprocess.run([sys.executable,str(geo),str(skill)],stdout=f,stderr=subprocess.STDOUT,check=True)
    # Independent SONIC planner screening for every sampled start, not warped walking tokens.
    actualq=initial['joint_position'][[initial['joint_names'].tolist().index(k) for k in kin.names]]
    root=initial['root_position'].copy();root[2]/=ROBOT_SCALE
    context=np.tile(np.r_[root,initial['root_quaternion_wxyz'],actualq],(4,1))
    direction=navigation-initial['root_position'];direction[2]=0;direction/=np.linalg.norm(direction)
    raw,feed,elapsed=SonicONNX().predict(context,direction,facing,1,plan['cruise_speed_m_s'],spec['planner_seed'])
    if np.max(np.maximum(lower-raw[:,7:],raw[:,7:]-upper))>1e-5:raise ValueError('Planner joint limit rejection')
    np.savez_compressed(c/'planner_screen.npz',raw_qpos_30hz=raw,**feed)
    write(c/'preflight.json',dict(geometry=read(skill/'geometry_check.json'),encoding=encoding,
          ik_max_position_error_m=max(errors,default=0),ik_max_orientation_error_rad=max(angles,default=0),
          max_adjacent_joint_change_rad=float(abs(np.diff(q,axis=0)).max()),planner_frames=len(raw),planner_seconds=elapsed,
          contact_q_unchanged_before_retime=True,task_success=False,limitations=['Mesh-AABB screen is not a swept-volume or full-room collision proof','No object/robot geometry scaling','Static scene and small straight approach corridor only']))
    write(c/'provenance.json',dict(source=str(SOURCE),source_tokens_sha256=sha(SOURCE/'grasp_tokens.npz'),
          encoder_sha256=sha(model),decoder_sha256=sha(MODEL_ROOT/'policy/release/model_decoder.onnx'),
          planner_sha256=sha(MODEL_ROOT/'planner/target_vel/V2/planner_sonic.onnx'),sample=spec,
          papers=['https://arxiv.org/abs/2605.27724','https://proceedings.mlr.press/v229/mandlekar23a.html']))
    write(c/'prepared_files.json',{str(p.relative_to(c)):sha(p) for p in c.rglob('*') if p.is_file() and p.name not in ['status.json','prepared_files.json']})
    write(c/'status.json',dict(status='prepared',task_success=False,training_ready=False))
    return c


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--count',type=int,default=3)
    p.add_argument('--seed',type=int,default=100);p.add_argument('--profile',choices=['identity','smoke','robot','object_se2'],default='smoke')
    p.add_argument('--randomize-planner-seed',action='store_true',help='Experimental: vary planner seed as well as initial state')
    args=p.parse_args()
    if args.count<1 or args.count>1000:p.error('count must be 1..1000')
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in range(args.seed,args.seed+args.count):
        c=args.output/f'candidate_{seed:06d}'
        try:
            spec=sample(seed,args.profile)
            if args.randomize_planner_seed:spec['planner_seed']=int(np.random.default_rng(seed+1000000).integers(1,1000000))
            compile_candidate(c,spec);results.append(dict(candidate=str(c.resolve()),status='prepared'))
        except Exception as e:results.append(dict(candidate=str(c.resolve()),status='preparation_failed',error=repr(e)))
    write(args.output/'preparation_summary.json',results)
    print(json.dumps(results,indent=2))
    if any(x['status']!='prepared' for x in results):raise SystemExit(1)

if __name__=='__main__':main()
