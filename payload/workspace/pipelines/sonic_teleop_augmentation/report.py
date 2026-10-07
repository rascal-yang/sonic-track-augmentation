from pathlib import Path
import argparse,json,sys
import numpy as np,h5py
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run.resolve();W=Path(__file__).resolve().parents[2]
candidate=Path(json.loads((r/'token_controller.json').read_text())['tokens_file']).parent
z=np.load(candidate/'grasp_tokens.npz')
src=np.load(candidate/'grasp_reference.npz')
chunks=[np.load(p) for p in sorted(r.glob('decoder_chunk_*.npz'))]
data={k:np.concatenate([c[k] for c in chunks]) for k in ['control_index','inputs','outputs','targets']}
setup=json.loads((r/'token_actuator_setup.json').read_text());names=setup['joint_names']
with h5py.File(r/'episode_raw.h5') as f:
 n=len(f['observation/timestamp']);assert np.array_equal(data['control_index'],np.arange(n))
 # Teleop tokens are computed online with the measured orientation anchor.
 ids=[json.loads(f.attrs['joint_names']).index(x) for x in names];ci=f['physics/control_index'][:]
 physical=f['physics/joint_position_target'][:,ids]
 target_error=float(abs(physical-data['targets'][ci]).max());assert target_error<2e-6
 kp_error=float(abs(f['physics/joint_stiffness'][:,ids]-np.array(setup['kp'])).max());kd_error=float(abs(f['physics/joint_damping'][:,ids]-np.array(setup['kd'])).max());assert kp_error<1e-5 and kd_error<1e-5
 source_order=[src['g1_joint_names'].tolist().index(name) for name in names]
 actual_q=f['observation/joint_position'][:,ids];joint_error=actual_q-src['g1_joint_position_rad'][:n,source_order]
 root=f['observation/root_position'][:];root_error=np.linalg.norm(root-src['root_position_world_m'][:n],axis=1)
 bn=json.loads(f.attrs['body_names']);wb=bn.index('right_wrist_yaw_link');sw=src['body_names'].tolist().index('right_wrist_yaw_link');wrist_error=np.linalg.norm(f['observation/body_position_world'][:,wb]-src['body_position_world_m'][:n,sw],axis=1)
 m=f['next_observation/metrics'];hold=0
 for flag in m['strict_lift_now'][:][::-1]:
  if not flag:break
  hold+=1
 opposed=bool(m['opposed_digits_contact'][-1]);success=hold*.02>=3 and opposed
 contacts=[json.loads(x) for x in (r/'table_contacts_physics.jsonl').read_text().splitlines()]
 peak=max(x['total_table_force_N'] for x in contacts)
 result={'run':r.name,'success':bool(success),'clean_success':bool(success and peak<=.05),'control_steps':n,'duration_s':n*.02,'source_token_rows':len(z['time_s']),'complete_sequence':n==len(z['time_s']),'strict_final_hold_s':hold*.02,'final_whole_bottle_clearance_m':float(m['whole_object_clearance_m'][-1]),'final_opposed_contact':opposed,'obstacle_contact_peak_N':peak,'minimum_pelvis_z_m':float(f['next_observation/root_position'][:,2].min()),'final_root':f['next_observation/root_position'][-1].tolist(),'root_tracking_error_final_m':float(root_error[-1]),'root_tracking_error_max_m':float(root_error.max()),'right_wrist_tracking_error_final_m':float(wrist_error[-1]),'right_wrist_tracking_error_max_m':float(wrist_error.max()),'body_joint_deviation_from_source_posture_rmse_rad':float(np.sqrt(np.mean(joint_error**2))),'every_input_token_matches_file':False,'physical_target_max_difference_rad':target_error,'kp_error':kp_error,'kd_error':kd_error,'decoder_history':'live state, oldest to newest','online_encoder':True,'IK_override':False,'reference_or_robot_teleport':False,'training_ready':False}
 # Reconstruct all994 inputs independently from captured live state and previous raw actions.
 from scipy.spatial.transform import Rotation
 rot=Rotation.from_quat(f['observation/root_quaternion_wxyz'][:][:,[1,2,3,0]])
 omega=rot.inv().apply(f['observation/root_angular_velocity_world'][:]);gravity=rot.inv().apply(np.tile([0,0,-1],(n,1)))
 state_fields=[omega,actual_q-np.array(setup['default']),f['observation/joint_velocity'][:,ids],np.vstack([np.zeros((1,29)),data['outputs'][:-1]]),gravity]
 history_index=np.maximum(np.arange(n)[:,None]+np.arange(-9,1)[None,:],0)
 rebuilt=np.concatenate([data['inputs'][:,:64]]+[v[history_index].reshape(n,-1) for v in state_fields],axis=1).astype('float32')
 result['decoder_input_reconstruction_max_error']=float(abs(rebuilt-data['inputs']).max());assert result['decoder_input_reconstruction_max_error']<2e-6
 # Recompute encoder from captured live orientation; all other inputs are prepared Cartesian targets.
 sys.path.insert(0,str(W/'g1_closed_loop/.deps'))
 import onnxruntime as ort
 enc_actual=np.load(r/'teleop_encoder_execution.npz')['encoder_inputs']
 enc_expected=z['encoder_inputs'][:n].copy()
 refrot=Rotation.from_quat(src['root_quaternion_world_wxyz'][:n,[1,2,3,0]])
 enc_expected[:,595:601]=(rot.inv()*refrot).as_matrix()[:,:,:2].reshape(n,6)
 result['encoder_input_reconstruction_error']=float(abs(enc_expected-enc_actual).max());assert result['encoder_input_reconstruction_error']<2e-6
 opt=ort.SessionOptions();opt.intra_op_num_threads=2
 model=json.loads((r/'token_controller.json').read_text())['encoder_file']
 net=ort.InferenceSession(model,sess_options=opt,providers=['CPUExecutionProvider'])
 rebuilt_tokens=np.concatenate([net.run(None,{net.get_inputs()[0].name:x[None]})[0] for x in enc_actual])
 result['encoder_token_recompute_error']=float(abs(rebuilt_tokens-data['inputs'][:,:64]).max());assert result['encoder_token_recompute_error']==0
 result['encoder_mode']=1;assert np.all(enc_actual[:,0]==1)
 result['upper_body_joint_reference_input_max']=float(abs(enc_actual[:,4:584]).max());assert result['upper_body_joint_reference_input_max']==0
 np.savez_compressed(r/'executed_teleop_tokens.npz',time_s=np.arange(n)*.02,sonic_encoded_tokens=data['inputs'][:,:64],encoder_inputs=enc_actual,decoder_inputs=data['inputs'],decoder_outputs=data['outputs'],pd_targets=data['targets'],**{k:z[k][:n] if z[k].ndim>1 else z[k] for k in z.files if k.startswith('revo2_')})
 result['hand_target_max_difference_rad']={}
 for side in ['left','right']:
  hn=json.loads(f.attrs[f'brainco_{side}_joint_names']);hi=[hn.index(x) for x in z[f'revo2_{side}_active_joint_names']]
  he=float(abs(f[f'brainco/{side}/position_target'][:][:,hi]-z[f'revo2_{side}_active_position_target_rad'][:n]).max());assert he==0
  result['hand_target_max_difference_rad'][side]=he
 result['body_velocity_target_max_abs']=float(abs(f['physics/joint_velocity_target'][:,ids]).max())
 result['body_effort_command_max_abs']=float(abs(f['physics/joint_effort_command'][:,ids]).max())
 assert result['body_velocity_target_max_abs']==0 and result['body_effort_command_max_abs']==0
 result['forward_displacement_m']=float(f['next_observation/root_position'][-1,1]-root[0,1])
 result['source_forward_displacement_m']=float(src['root_position_world_m'][n-1,1]-src['root_position_world_m'][0,1])
 result['final_hand_bottle_force_N']=float(m['object_hand_normal_force_N'][-1])
 result['final_bottle_table_force_N']=float(m['object_table_normal_force_N'][-1])
 result['source_start_step']=830
 result['initial_state_restore']=json.loads((r/'initial_state_validation.json').read_text())
 result['one_time_initialization_before_control']=True
 result['decoder_startup_previous_action']='zero; repeat first measured state for missing9historyframes'
 arm=[i for i,name in enumerate(names) if name.startswith('right_') and any(part in name for part in ['shoulder','elbow','wrist'])]
 result['right_arm_deviation_from_source_posture_rmse_rad']=float(np.sqrt(np.mean(joint_error[:,arm]**2)))
 actual_wq=Rotation.from_quat(f['observation/body_quaternion_wxyz'][:,wb][:,[1,2,3,0]])
 source_wq=Rotation.from_quat(src['body_quaternion_world_wxyz'][:n,sw][:,[1,2,3,0]])
 wrist_rot_error=(source_wq.inv()*actual_wq).magnitude()
 result['right_wrist_orientation_final_error_rad']=float(wrist_rot_error[-1])
 result['right_wrist_orientation_max_error_rad']=float(wrist_rot_error.max())
 np.savez_compressed(r/'tracking_errors.npz',time_s=f['observation/timestamp'][:],root_error_m=root_error,wrist_error_m=wrist_error,body_joint_error_rad=joint_error)
np.savez_compressed(r/'decoder_execution.npz',**data)
(r/'token_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
