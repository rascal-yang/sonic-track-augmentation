"""Matched-time physical tracking errors, not independent fixed-token replay repeatability."""
from pathlib import Path
import json
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
H=Path(__file__).resolve().parent;W=H.parents[1]
ref=np.load(H/'candidates/reference/grasp_reference.npz');names=ref['g1_joint_names'].tolist();bodies=ref['body_names'].tolist()
runs={'ik_reference_g1_encoder':W/'experiments/20260929_sonic_grasp_smooth/runs/safe025_001','cartesian_teleop_encoder':H/'runs/teleop_001'}
def stats(x):
 x=np.asarray(x,dtype=float)
 return dict(mean=float(x.mean()),rmse=float(np.sqrt(np.mean(x*x))),p95=float(np.percentile(x,95)),maximum=float(x.max()),final=float(x[-1]))
result=dict(definition='Actual pre-action simulated states after SONIC decoder closed-loop execution versus same-time compiled source reference; no time warping; 586 samples at50Hz. Includes encoding, controller and physics effects, not decoder-only error.',independent_fixed_token_repeat_replay_performed_this_comparison=False,runs={})
for label,path in runs.items():
 with h5py.File(path/'episode_raw.h5') as f:
  o=f['observation'];n=len(o['timestamp']);assert n==len(ref['time_s'])==586
  assert np.allclose(o['timestamp'][:],ref['time_s'],atol=1e-6)
  bn=json.loads(f.attrs['body_names']);jn=json.loads(f.attrs['joint_names']);q=o['joint_position'][:,[jn.index(k) for k in names]];dq=q-ref['g1_joint_position_rad'];entry={}
  for side in ['right','left']:
   name=side+'_wrist_yaw_link';i=bn.index(name);j=bodies.index(name)
   pos=np.linalg.norm(o['body_position_world'][:,i]-ref['body_position_world_m'][:,j],axis=1)*100
   actual=R.from_quat(o['body_quaternion_wxyz'][:,i][:,[1,2,3,0]]);target=R.from_quat(ref['body_quaternion_world_wxyz'][:,j][:,[1,2,3,0]])
   angle=np.rad2deg((target.inv()*actual).magnitude())
   entry[side+'_wrist_position_cm']=stats(pos);entry[side+'_wrist_orientation_deg']=stats(angle)
  entry['root_position_cm']=stats(np.linalg.norm(o['root_position'][:]-ref['root_position_world_m'],axis=1)*100)
  groups={'body29':list(range(29)),'right_arm7':[i for i,k in enumerate(names) if k.startswith('right_') and any(x in k for x in ['shoulder','elbow','wrist'])],'legs12':[i for i,k in enumerate(names) if any(x in k for x in ['hip','knee','ankle'])]}
  entry['joint_reference_deviation']={key:dict(rmse_rad=float(np.sqrt(np.mean(dq[:,ids]**2))),rmse_deg=float(np.rad2deg(np.sqrt(np.mean(dq[:,ids]**2)))),max_abs_deg=float(np.rad2deg(np.abs(dq[:,ids]).max()))) for key,ids in groups.items()}
  entry['joint_metric_caveat']='Teleop has no upper-body joint-reference constraint; joint differences reflect alternative posture as well as tracking, not solely tracking error.'
  entry['physical_run']=str(path);result['runs'][label]=entry
(H/'decoder_rollout_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
a,b=[result['runs'][k] for k in runs]
rows=[('右腕位置平均误差（cm）',a['right_wrist_position_cm']['mean'],b['right_wrist_position_cm']['mean']),('右腕位置 RMSE（cm）',a['right_wrist_position_cm']['rmse'],b['right_wrist_position_cm']['rmse']),('右腕位置 P95（cm）',a['right_wrist_position_cm']['p95'],b['right_wrist_position_cm']['p95']),('右腕位置最大误差（cm）',a['right_wrist_position_cm']['maximum'],b['right_wrist_position_cm']['maximum']),('右腕朝向平均误差（°）',a['right_wrist_orientation_deg']['mean'],b['right_wrist_orientation_deg']['mean']),('右腕朝向最大误差（°）',a['right_wrist_orientation_deg']['maximum'],b['right_wrist_orientation_deg']['maximum']),('骨盆位置平均误差（cm）',a['root_position_cm']['mean'],b['root_position_cm']['mean']),('身体29关节 RMSE（°）',a['joint_reference_deviation']['body29']['rmse_deg'],b['joint_reference_deviation']['body29']['rmse_deg']),('右臂7关节 RMSE（°）',a['joint_reference_deviation']['right_arm7']['rmse_deg'],b['joint_reference_deviation']['right_arm7']['rmse_deg'])]
md='# SONIC decoder 执行后的偏差统计\n\n统计已有两次成功抓取的586帧物理执行记录，对齐同一参考和同一时刻。之前的路点误差同样是经过decoder实际执行后的误差，不是编码前IK的求解残差。\n\n|指标|IK参考 → G1 encoder → decoder|Cartesian参考 → Teleop encoder → decoder|\n|---|---:|---:|\n'+''.join(f'|{k}|{x:.2f}|{y:.2f}|\n' for k,x,y in rows)+'\n误差包含编码、控制器和物理跟踪的共同影响，不能单独归因于decoder。Teleop没有要求复现上半身关节角，因此关节差异也包含不同手臂姿态，不等同于任务失败。原版使用离线token，本次Teleop使用实时根部朝向编码；不是只改变单个网络的消融。\n\n本次未新增独立固定token物理回放，不能据此宣称再次replay的复现误差为零。详细数值：[decoder_rollout_comparison.json](decoder_rollout_comparison.json)。\n'
(H/'DECODER_COMPARISON.md').write_text(md)
print(md)
