"""Independent replay repeatability against each token stream's first execution."""
from pathlib import Path
import argparse,json
import numpy as np,h5py
from scipy.spatial.transform import Rotation as R
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve();config=json.loads((run/'token_controller.json').read_text());c=Path(config['tokens_file']).parent;plan=json.loads((c/'plan.json').read_text());first=Path(plan['first_execution'])
assert first.resolve()!=run.resolve(), 'Source and replay must be different runs'
replay_capture=run/('episode_raw.h5' if (run/'episode_raw.h5').exists() else 'episode_state.h5')
assert (first/'episode_raw.h5').stat().st_ino != replay_capture.stat().st_ino, 'Independent capture must not alias the source file'
def magnitude_stats(v,scale=1):
 v=np.asarray(v,dtype=np.float64)*scale
 return dict(mean=float(v.mean()),rmse=float(np.sqrt(np.mean(v*v))),p95=float(np.percentile(v,95)),maximum=float(v.max()),final=float(v[-1]))
result=dict(first_execution=str(first),independent_replay=str(run),kind=plan['kind'],comparison='Replay minus FIRST PHYSICAL EXECUTION, equal timestamps, no temporal alignment/warping. Not error versus ideal waypoints.',encoder_rerun=False,planner_rerun=False,IK=False)
with h5py.File(first/'episode_raw.h5') as f,h5py.File(replay_capture) as g:
 n=len(g['observation/timestamp']);m=len(f['observation/timestamp']);result.update(source_control_steps=m,replay_control_steps=n,complete_sequence=n==m)
 assert n<=m;assert np.array_equal(g['observation/timestamp'][:],f['observation/timestamp'][:n])
 bnames=json.loads(f.attrs['body_names']);jnames=json.loads(f.attrs['joint_names']);assert bnames==json.loads(g.attrs['body_names']);assert jnames==json.loads(g.attrs['joint_names'])
 body_names=np.load(c/'grasp_tokens.npz')['encoder_joint_names'].tolist();ids=[jnames.index(k) for k in body_names];arm=[jnames.index(k) for k in body_names if k.startswith('right_') and any(s in k for s in ['shoulder','elbow','wrist'])]
 result['matched_state_errors']={}
 for group in ['observation','next_observation']:
  out={}
  for side in ['right','left']:
   i=bnames.index(side+'_wrist_yaw_link');dp=g[group+'/body_position_world'][:,i]-f[group+'/body_position_world'][:n,i]
   qa=f[group+'/body_quaternion_wxyz'][:n,i][:,[1,2,3,0]];qb=g[group+'/body_quaternion_wxyz'][:,i][:,[1,2,3,0]]
   out[side+'_wrist_position_mm']=magnitude_stats(np.linalg.norm(dp,axis=1),1000)
   out[side+'_wrist_orientation_deg']=magnitude_stats((R.from_quat(qa).inv()*R.from_quat(qb)).magnitude(),180/np.pi)
  for name in ['root_position','object_position']:
   out[name+'_mm']=magnitude_stats(np.linalg.norm(g[group+'/'+name][:]-f[group+'/'+name][:n],axis=1),1000)
  dq=g[group+'/joint_position'][:]-f[group+'/joint_position'][:n]
  for name,indices in [('body29',ids),('right_arm7',arm)]:
   out[name+'_joint_angle']=dict(rmse_rad=float(np.sqrt(np.mean(dq[:,indices].astype(float)**2))),max_abs_rad=float(abs(dq[:,indices]).max()))
  for field in ['joint_position','joint_velocity','root_position','root_quaternion_wxyz','root_linear_velocity_world','root_angular_velocity_world','body_position_world','body_quaternion_wxyz','object_position','object_quaternion_wxyz']:
   x=g[group+'/'+field][:];y=f[group+'/'+field][:n];out[field+'_bitwise_equal']=bool(np.array_equal(x,y))
  result['matched_state_errors'][group]=out
 result['physics_max_abs_difference']={};result['physics_bitwise_equal']={}
 for key in ['joint_position_target','joint_velocity_target','joint_effort_command','joint_stiffness','joint_damping']:
  x=g['physics/'+key][:];y=f['physics/'+key][:len(x)];result['physics_max_abs_difference'][key]=float(abs(x-y).max());result['physics_bitwise_equal'][key]=bool(np.array_equal(x,y))
 result['hands']={}
 for side in ['left','right']:
  out={}
  for key in ['joint_position','position_target']:
   x=g[f'brainco/{side}/{key}'][:];y=f[f'brainco/{side}/{key}'][:n];out[key+'_max_abs_rad']=float(abs(x-y).max());out[key+'_bitwise_equal']=bool(np.array_equal(x,y))
  result['hands'][side]=out
 result['all_next_metrics_equal']=all(np.array_equal(g['next_observation/metrics/'+key][:],f['next_observation/metrics/'+key][:n]) for key in g['next_observation/metrics'] if isinstance(g['next_observation/metrics/'+key],h5py.Dataset))
a=np.load(first/'decoder_execution.npz');b=np.load(run/'decoder_execution.npz');result['decoder']={}
for k in ['inputs','outputs','targets']:
 result['decoder'][k+'_max_abs_difference']=float(abs(b[k]-a[k][:n]).max());result['decoder'][k+'_bitwise_equal']=bool(np.array_equal(b[k],a[k][:n]))
result['decoder']['saved_token_max_abs_difference']=float(abs(b['inputs'][:,:64]-a['inputs'][:n,:64]).max())
result['quality_first']=json.loads((first/'audit.json').read_text());result['quality_replay']=json.loads((run/'audit.json').read_text())
result['scope']='One matched-environment deterministic simulation replay per method; no real hardware, perturbation or cross-machine reproducibility claim.'
# Predeclared repeatability limits; report exact errors even when below threshold.
e=result['matched_state_errors']['next_observation']
result['repeatability_limits']=dict(root_position_max_mm=10,right_wrist_position_max_mm=10,object_position_max_mm=10,body_joint_rmse_rad=.02)
result['repeatability_passed']=bool(result['complete_sequence'] and result['quality_replay']['task_success'] and
    e['root_position_mm']['maximum']<=10 and e['right_wrist_position_mm']['maximum']<=10 and
    e['object_position_mm']['maximum']<=10 and e['body29_joint_angle']['rmse_rad']<=.02 and
    result['decoder']['saved_token_max_abs_difference']==0)
(run/'replay_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
