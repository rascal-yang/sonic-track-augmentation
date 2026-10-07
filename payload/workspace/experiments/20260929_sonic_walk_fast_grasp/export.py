"""Export executed latent stream and separate Revo2 targets without claiming replay success."""
from pathlib import Path
import argparse,json,h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);r=p.parse_args().run
z=np.load(r/'decoder_execution.npz');c=json.loads((r/'token_controller.json').read_text());a=json.loads((r/'audit.json').read_text());setup=json.loads((r/'token_actuator_setup.json').read_text())
with h5py.File(r/'episode_raw.h5') as f:
 data=dict(time_s=f['observation/timestamp'][:],sonic_encoded_tokens=z['inputs'][:,:64],phase=z['phase'],fast_grasp_index=z['grasp_index'],body_joint_names=np.array(c['body_joints']),body_PD_target_rad=z['targets'],root_position_m=f['observation/root_position'][:],root_quaternion_wxyz=f['observation/root_quaternion_wxyz'][:])
 ids=[json.loads(f.attrs['joint_names']).index(x) for x in c['body_joints']];data['body_joint_position_rad']=f['observation/joint_position'][:,ids];data['body_joint_velocity_rad_s']=f['observation/joint_velocity'][:,ids]
 source=np.load(Path(json.loads((r/'plan.json').read_text())['source_fast'])/'grasp_tokens.npz')
 for side in ['left','right']:
  names=source[f'revo2_{side}_active_joint_names'];ids=[json.loads(f.attrs[f'brainco_{side}_joint_names']).index(x) for x in names];data[f'revo2_{side}_active_joint_names']=names;data[f'revo2_{side}_active_position_target_rad']=f[f'brainco/{side}/position_target'][:][:,ids]
np.savez_compressed(r/'executed_sonic_tokens.npz',**data)
(r/'executed_tokens_metadata.json').write_text(json.dumps(dict(source_run=str(r.resolve()),control_Hz=50,latent_dimension=64,latent_format='float32 release encoder quantized values; not integer vocabulary IDs',hands='Two separate6joint Revo2 target vectors, outside SONIC latent',task_success=a['task_success'],fixed_token_replay_tested=False,notes='This stream was executed with online planner/encoder during locomotion and stored fast grasp tokens. Replaying it elsewhere requires the matching release decoder, deployment PD parameters and compatible initial state; it is not a hardware-certified trajectory.',training_ready=False),indent=2))
# Augment the capture with the independently logged, audited controller stream.
with h5py.File(r/'episode_raw.h5','r+') as f:
 group=f.require_group('sonic');phase_names=sorted(set(z['phase'].tolist()))
 values={'encoded_tokens':z['inputs'][:,:64],'encoder_inputs':z['encoder_inputs'],'decoder_inputs':z['inputs'],'decoder_raw_actions':z['outputs'],'body_position_target':z['targets'],'control_index':z['control_index'],'fast_grasp_index':z['grasp_index'],'phase_index':np.array([phase_names.index(x) for x in z['phase']],np.int16)}
 for key,value in values.items():
  if key in group:assert np.array_equal(group[key][:],value)
  else:group.create_dataset(key,data=value,compression='gzip')
 group.attrs['phase_names']=json.dumps(phase_names);group.attrs['body_joint_names']=json.dumps(c['body_joints']);group.attrs['provenance']='Added after rollout from audited decoder_chunk logs; original observation/action/physics datasets retained.'
 group.attrs['frequency_Hz']=50;f.attrs['processed_env_action_note']='Legacy wrapper placeholder, not the applied SONIC latent or decoder output. Use sonic/encoded_tokens and actual action/joint_position_target; Revo2 targets stored separately under brainco.'
