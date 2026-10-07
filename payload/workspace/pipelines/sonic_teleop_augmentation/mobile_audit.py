"""Audit actual decoded travel, full physics commands, continuity and grasp quality."""
from pathlib import Path
import argparse,json,sys
import numpy as np,h5py
from scipy.spatial.transform import Rotation as R
p=argparse.ArgumentParser();p.add_argument('run',type=Path);r=p.parse_args().run
plan=json.loads((r/'plan.json').read_text());setup=json.loads((r/'token_actuator_setup.json').read_text())
z=np.load(Path(plan['source_fast'])/'grasp_tokens.npz')
chunks=[np.load(p) for p in sorted(r.glob('decoder_chunk_*.npz'))]
d={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0].files}
with h5py.File(r/'episode_raw.h5') as f:
 n=len(f['observation/timestamp']);assert np.array_equal(d['control_index'],np.arange(n))
 ids=[json.loads(f.attrs['joint_names']).index(x) for x in setup['joint_names']];ci=f['physics/control_index'][:]
 err=float(abs(f['physics/joint_position_target'][:,ids]-d['targets'][ci]).max());assert err<2e-6
 kp_error=float(abs(f['physics/joint_stiffness'][:,ids]-setup['kp']).max());kd_error=float(abs(f['physics/joint_damping'][:,ids]-setup['kd']).max());assert max(kp_error,kd_error)<1e-6
 assert float(abs(f['physics/joint_velocity_target'][:,ids]).max())<1e-8
 assert float(abs(f['physics/joint_effort_command'][:,ids]).max())<1e-8
 rot=R.from_quat(f['observation/root_quaternion_wxyz'][:][:,[1,2,3,0]])
 fields=[rot.inv().apply(f['observation/root_angular_velocity_world'][:]),f['observation/joint_position'][:,ids]-setup['default'],f['observation/joint_velocity'][:,ids],np.vstack([np.zeros((1,29)),d['outputs'][:-1]]),rot.inv().apply(np.tile([0,0,-1],(n,1)))]
 hi=np.maximum(np.arange(n)[:,None]+np.arange(-9,1)[None,:],0)
 rebuilt=np.concatenate([d['inputs'][:,:64]]+[v[hi].reshape(n,-1) for v in fields],axis=1).astype('float32');inputerr=float(abs(rebuilt-d['inputs']).max());assert inputerr<2e-6
 g=d['grasp_index'];mask=g>=0;complete=np.array_equal(g[mask],np.arange(len(z['time_s'])))
 enc_error=None
 if mask.any():
  ref=np.load(Path(plan['source_fast'])/'grasp_reference.npz')
  enc_expected=z['encoder_inputs'][g[mask]].copy()
  reference_rotation=R.from_quat(ref['root_quaternion_world_wxyz'][g[mask]][:,[1,2,3,0]])
  enc_expected[:,595:601]=(rot[mask].inv()*reference_rotation).as_matrix()[:,:,:2].reshape(-1,6)
  correction=reference_rotation.inv().apply((f['observation/root_position'][:][mask]-ref['root_position_world_m'][g[mask]])/.982730067592321)
  enc_expected[:,901:910]=(enc_expected[:,901:910].reshape(-1,3,3)-correction[:,None,:]).reshape(-1,9)
  enc_error=float(abs(enc_expected-d['encoder_inputs'][mask]).max()) if mask.any() else None
  assert enc_error is None or enc_error<2e-6
 tokenmatch=False
 handerrors={}
 for side in ['left','right']:
  names=json.loads(f.attrs[f'brainco_{side}_joint_names']);idx=[names.index(x) for x in z[f'revo2_{side}_active_joint_names']]
  handerrors[side]=float(abs(f[f'brainco/{side}/position_target'][:][:,idx]-z[f'revo2_{side}_active_position_target_rad'][np.maximum(g,0)]).max());assert handerrors[side]<1e-6
 end=f['next_observation'];root=end['root_position'][:];forward=np.array(plan['forward_world']);travel=(root-np.array(plan['start_root_m']))@forward
 m=end['metrics'];hold=0
 for v in m['strict_lift_now'][:][::-1]:
  if not v:break
  hold+=1
 names=json.loads(f.attrs['body_names']);wb=names.index('right_wrist_yaw_link');wr=R.from_quat(end['body_quaternion_wxyz'][:,wb][:,[1,2,3,0]]);ob=R.from_quat(end['object_quaternion_wxyz'][:][:,[1,2,3,0]])
 rel=wr.inv().apply(end['object_position'][:]-end['body_position_world'][:,wb]);rr=wr.inv()*ob;window=min(150,n);slip=float(np.linalg.norm(rel[-window:]-rel[-window],axis=1).max());twist=float(np.rad2deg((rr[-window].inv()*rr[-window:]).magnitude().max()))
 contacts=[json.loads(x) for x in (r/'table_contacts_physics.jsonl').read_text().splitlines()];peak=max(x['total_table_force_N'] for x in contacts)
 phases={str(k):int(sum(d['phase']==k)) for k in np.unique(d['phase'])};walk=np.isin(d['phase'],['walking','braking']);feet={}
 if walk.any():
  for side in ['left','right']:
   b=names.index(side+'_ankle_roll_link');xyz=end['body_position_world'][:,b];feet[side]={'forward_excursion_relative_root_m':float(np.ptp(((xyz-root)@forward)[walk])),'world_height_range_m':float(np.ptp(xyz[walk,2]))}
 dock=None
 if mask.any():dock=float(np.linalg.norm(f['observation/root_position'][int(np.flatnonzero(mask)[0]),:2]-plan['goal_root_m'][:2]))
 root_step=float(np.linalg.norm(np.diff(root,axis=0),axis=1).max()) if n>1 else 0.
 opposed=float(np.mean(m['opposed_digits_contact'][-window:]));success=bool(root[:,2].min()>.45 and root_step<.05 and complete and travel.max()>=float(plan.get('minimum_travel_m',.8)) and dock is not None and dock<=.05 and hold*.02>=3 and opposed>=.95 and peak<=.05 and slip<=.015 and twist<=10)
 result=dict(task_success=success,control_steps=n,duration_s=n*.02,phase_frames=phases,actual_forward_displacement_m=float(travel[-1]),max_forward_displacement_m=float(travel.max()),docking_error_at_grasp_m=dock,final_navigation_error_m=float(np.linalg.norm(root[-1,:2]-plan['goal_root_m'][:2])),actual_feet=feet,complete_fast_grasp_sequence=bool(complete),strict_final_hold_s=hold*.02,final_clearance_m=float(m['whole_object_clearance_m'][-1]),final3s_opposed_fraction=opposed,final3s_relative_slip_m=slip,final3s_relative_rotation_deg=twist,monitored_table_force_peak_N=peak,minimum_pelvis_z_m=float(root[:,2].min()),maximum_root_step_m=root_step,physical_PD_target_max_error_rad=err,kp_max_error=kp_error,kd_max_error=kd_error,decoder_live_input_reconstruction_error=inputerr,fixed_preview_tokens_used=False,grasp_encoder_input_reconstruction_error=enc_error,hand_target_max_error_rad=handerrors,training_ready=False)
np.savez_compressed(r/'decoder_execution.npz',**d)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1_closed_loop/.deps'))
import onnxruntime as ort
opts=ort.SessionOptions();opts.intra_op_num_threads=2
enc=ort.InferenceSession(json.loads((r/'token_controller.json').read_text())['encoder'],sess_options=opts,providers=['CPUExecutionProvider'])
error=0.
for i in range(n):
 out=enc.run(None,{enc.get_inputs()[0].name:d['encoder_inputs'][i:i+1]})[0][0];error=max(error,float(abs(out-d['inputs'][i,:64]).max()))
assert error<2e-6;result['online_token_reencoding_max_error']=error
(r/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

hand_data={}
for side in ['left','right']:
 hand_data[f'revo2_{side}_active_joint_names']=z[f'revo2_{side}_active_joint_names']
 hand_data[f'revo2_{side}_active_position_target_rad']=z[f'revo2_{side}_active_position_target_rad'][np.maximum(g,0)]
np.savez_compressed(r/'executed_teleop_tokens.npz',time_s=np.arange(n)*.02,sonic_encoded_tokens=d['inputs'][:,:64],encoder_inputs=d['encoder_inputs'],decoder_inputs=d['inputs'],decoder_outputs=d['outputs'],pd_targets=d['targets'],phase=d['phase'],grasp_index=g,**hand_data)
