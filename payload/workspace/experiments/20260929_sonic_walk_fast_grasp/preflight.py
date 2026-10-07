from pathlib import Path
import sys,json,shutil
import numpy as np
from scipy.spatial.transform import Rotation
H=Path(__file__).resolve().parent;W=H.parents[1];B=W/'experiments/20260925_measured_walk_grasp';S=W/'experiments/20260929_sonic_grasp_smooth/candidates/safe025'
sys.path.insert(0,str(B/'code'));sys.path.insert(0,str(W/'g1_closed_loop/.deps'))
from sonic_planner_goals import SonicONNX
from online_goals import Kinematics
from calibration_config import ROBOT_SCALE
kin=Kinematics('__RUNTIME__/vendor/SONIC');net=SonicONNX()
with np.load(S/'initial_state.npz') as z:init={k:z[k].copy() for k in z.files}
yaw=Rotation.from_quat(init['root_quaternion_wxyz'][[1,2,3,0]]).as_euler('xyz')[2];forward=np.array([np.cos(yaw),np.sin(yaw),0]);delta=-.9*forward;goal=init['root_position'].copy()
for key in ['root_position','body_position_world','brainco_left_root_position','brainco_right_root_position']:init[key]+=delta
np.savez_compressed(H/'initial_state.npz',**init)
q=init['joint_position'][[init['joint_names'].tolist().index(n) for n in kin.names]];root=init['root_position'].copy();root[2]/=ROBOT_SCALE;context=np.tile(np.r_[root,init['root_quaternion_wxyz'],q],(4,1));results={}
for speed in [.25,.55]:
 raw,feed,elapsed=net.predict(context,forward,forward,1,speed,17)
 feet=[]
 for row in raw:
  p=row[:3].copy();p[2]*=ROBOT_SCALE;kin.fk(row[7:],p,row[3:7]);feet.append([kin.data.xpos[kin.model.body(side+'_ankle_roll_link').id].copy() for side in ['left','right']])
 feet=np.array(feet);local_progress=np.einsum('nsj,j->ns',feet-raw[:,None,:3],forward);amplitude=np.ptp(local_progress,axis=0)
 results[str(speed)]={'frames_30hz':len(raw),'planned_forward_displacement_m':float((raw[-1,:3]-raw[0,:3])@forward),'foot_relative_forward_excursion_m':amplitude.tolist(),'hip_pitch_range_rad':[float(np.ptp(raw[:,7+kin.names.index(side+'_hip_pitch_joint')])) for side in ['left','right']],'joint_limit_max_excess_rad':float(np.maximum(raw[:,7:]-kin.model.jnt_range[1:,1],kin.model.jnt_range[1:,0]-raw[:,7:]).max()),'inference_s':elapsed}
 np.savez_compressed(H/f'planner_screen_{speed:.2f}.npz',raw_qpos=raw,feet=feet,**feed)
plan=dict(source_fast=str(S),translation_world_m=delta.tolist(),translation_distance_m=.9,start_root_m=init['root_position'].tolist(),goal_root_m=goal.tolist(),yaw_rad=float(yaw),forward_world=forward.tolist(),cruise_speed_m_s=.55,terminal_speed_rule='min(cruise,max0.05,1.5*distance), then measured braking/settling',planner_screen=results,architecture='Online SONIC V2 planner -> future q/dq/orientation -> release encoder -> same release decoder as fast grasp; fixed original fast tokens after measured handoff. Save every actual latent and target.',success='Travel>=0.80m; docking error<=0.05m before grasp; complete586fast token rows; final strict lift>=3s,opposed,monitored table force<=0.05N; final3s relative slip<=15mm/10deg.',matched_lessons=['Measured feet/root arrival and handoff must be verified','No physical state reset during navigation/grasp','Larger planner command is not proof of larger actual stride','Previous offline actual-state encoder lacked live anchor orientation correction'],hypothesis='0.9m rearward start and0.55m/s cruising create clearly visible steps that survive encoder/decoder; decelerate near desk, retain successful fast grasp.',alternatives=['Selected: straight forward walk then grip-aware settle and stored fast grasp','If decoder locomotion fails: discriminate encoder/reference conditioning before changing contact geometry'],training_ready=False)
(H/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');shutil.copy2(S/'deployment_parameters.json',H/'deployment_parameters.json');print(json.dumps(plan,indent=2))
