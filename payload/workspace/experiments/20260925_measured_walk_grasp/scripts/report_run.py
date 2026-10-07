"""Independent episode report and honest task/quality labels from executed physics."""
from pathlib import Path
import json,argparse
import numpy as np,h5py
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run
v=json.loads((r/'validation.json').read_text());h=json.loads((r/'hand_validation.json').read_text())
contacts=[json.loads(x) for x in (r/'table_contacts_physics.jsonl').read_text().splitlines()]
table=np.array([sum(v for k,v in s['links_N'].items() if not k.startswith('room:')) for s in contacts]);room=np.array([sum(v for k,v in s['links_N'].items() if k.startswith('room:')) for s in contacts])
obs=[json.loads(p.read_text()) for p in sorted(r.glob('observation_[0-9][0-9][0-9].json'))];dec=[json.loads(p.read_text()) for p in sorted(r.glob('decision_[0-9][0-9][0-9].json'))]
calls=[json.loads(x) for x in (r/'planner_calls.jsonl').read_text().splitlines()] if (r/'planner_calls.jsonl').exists() else []
arm=[json.loads(x) for x in (r/'direct_arm_execution.jsonl').read_text().splitlines()] if (r/'direct_arm_execution.jsonl').exists() else []
retries=[json.loads(x) for x in (r/'retry_events.jsonl').read_text().splitlines()] if (r/'retry_events.jsonl').exists() else []
with h5py.File(r/'episode_raw.h5','r+') as f:
 m=f['next_observation/metrics'];success=v['strict_hold_final_s']>=3 and bool(m['opposed_digits_contact'][-1]);clean=success and max(table.max(),room.max())<=.05
 names=json.loads(f.attrs['joint_names']);ids=[i for i,n in enumerate(names) if n.startswith('right_') and any(k in n for k in ['shoulder','elbow','wrist'])];q=f['action/joint_position_target'][:,ids];first=arm[0]['frame'] if arm else len(q);dq=np.diff(q[max(first,1)-1:],axis=0)/.02;acc=np.diff(dq,axis=0)/.02
 bn=json.loads(f.attrs['body_names']);bid=bn.index('right_wrist_yaw_link');pos=f['observation/body_position_world'][:,bid];vel=np.linalg.norm(np.diff(pos,axis=0)/.02,axis=1)
 f.attrs['controller']='Live SONIC balance; ONNX walking when navigating; actual-state right-arm IK with gravity compensation, bounded target velocity and optional arm-only acceleration limits; GRAIL residual zero. Session image review gates and deterministic feedback between gates.'
 f.attrs['asset_manifest']=(r.parents[1]/'asset_manifest.json').read_text()
 if (r/'optimized_config.json').exists():f.attrs['strategy_config']=(r/'optimized_config.json').read_text()
 visual=json.loads((r/'visual_realism_review.json').read_text()) if (r/'visual_realism_review.json').exists() else {}
 visually_accepted=bool(visual.get('accepted') and visual.get('training_ready'))
 f.attrs['training_ready']=bool(clean and visually_accepted)
 f.attrs['episode_success']=bool(success);f.attrs['positive_behavior_cloning_eligible']=bool(clean and visually_accepted);f.attrs['demonstration_quality']='closed_loop_clean_success' if clean else 'closed_loop_failure_or_contact';f.attrs['external_GPT_API']=False
 decision_ids=np.full(len(q),-1,dtype=np.int32);attempt_ids=np.ones(len(q),dtype=np.int16)
 for d in dec:
  o=next((o for o in obs if o['observation_id']==d['observation_id']),None)
  if o is not None:
   decision_ids[o['step']:]=d['observation_id'];attempt_ids[o['step']:]=d.get('attempt_index',1)
 for key,values in [('decision_id',decision_ids),('attempt_index',attempt_ids)]:
  path='annotation/'+key
  if path in f:del f[path]
  f.create_dataset(path,data=values)
 f.attrs['contains_recovery']=bool(retries)
 f.attrs['nominal_success_without_retry']=bool(clean and not retries)
 if (r/'optimized_config.json').exists():
  stress=bool(json.loads((r/'optimized_config.json').read_text()).get('stress_test'))
  f.attrs['deliberate_retry_stress_test']=stress
  if stress:f.attrs['positive_behavior_cloning_eligible']=False
  f.attrs['recovery_training_eligible']=bool(clean and retries and visually_accepted)
 finalclear=float(m['whole_object_clearance_m'][-1]);root=f['observation/root_position'][:];minimumroot=float(root[:,2].min())
firstclose=next((i for i,d in enumerate(dec) if d['action']=='close_hand'),len(dec));goals=[d['goal']['target'] for d in dec[:firstclose] if d['action']=='reach']
summary={'run':r.name,'success':bool(success),'clean_success':bool(clean),'control_steps':v['control_steps'],'duration_s':v['simulation_duration_s'],'strict_final_hold_s':v['strict_hold_final_s'],'first_lift_confirmation_s':v['first_success_confirmation_s'],'final_whole_bottle_clearance_m':finalclear,'table_contact_peak_N':float(table.max()),'room_contact_peak_N':float(room.max()),'minimum_pelvis_z_m':minimumroot,'navigation_planner_calls':len(calls),'navigation_reached_grasp':any(s['phase']=='grasp' for s in obs),'initial_root':root[0].tolist(),'final_root':root[-1].tolist(),'preclosure_spatial_waypoints':goals,'preclosure_waypoint_count':len(goals),'retries':len(retries),'attempts':(1+len(retries)) if firstclose<len(dec) else 0,'max_arm_target_velocity_rad_s':float(np.abs(dq).max()) if dq.size else None,'max_arm_target_acceleration_rad_s2':float(np.abs(acc).max()) if acc.size else None,'wrist_speed_p95_m_s_during_arm_control':float(np.percentile(vel[first:],95)) if len(vel)>first else None,'stop_reason':dec[-1]['reason'] if dec else '', 'runtime_result':json.loads((r/'result.json').read_text()), 'external_GPT_API':False,'GRAIL_interaction_enabled':False}
(r/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
(r/'REPORT.md').write_text(f'''# {r.name} 独立实验报告

- 成功：**{success}**；无监测障碍接触的成功：**{clean}**。
- 仿真时长 {summary['duration_s']:.2f} 秒；整瓶最终离桌 {finalclear*100:.2f} 厘米，末段连续严格保持 {v['strict_hold_final_s']:.2f} 秒。
- 桌子接触峰值 {table.max():.4f} N；房间障碍接触峰值 {room.max():.4f} N。监测覆盖双手、肘部和腕部，不能据此证明全身无任何碰撞。
- 闭指前空间路点 {len(goals)} 个；物理重试 {len(retries)} 次，共 {summary['attempts']} 次含闭指动作的抓取尝试。
- SONIC ONNX 规划调用 {len(calls)} 次；最小骨盆高度 {minimumroot:.3f} m。
- 结束原因：{summary['stop_reason']}。

## 控制和判据

SONIC 根据实时状态控制平衡；走近阶段使用 SONIC V2 ONNX 规划。站定抓取阶段，右臂使用当前身体状态的 IK 与受限重力补偿，其他关节由 SONIC 执行。GRAIL 交互残差为零。没有逐步回放历史电机指令，也没有运行中重置机器人或瓶子位置。

GPT/Codex 本会话在标记的暂停检查点看图、查看本体状态和特权信息；其间由固定反馈程序执行，未调用外部 GPT API。每条自动决策与人工智能会话检查记录均保留。桌子112×60×75 cm、水瓶高23 cm、G1直立高130 cm采用用户尺寸；IK/双手/安装偏移同步缩放，质量与摩擦仍为估计。背景提篮和玩偶固定在校准摆放；目标瓶子保持动态。

成功要求：整瓶最低点离桌 >2cm、有手瓶接触、瓶桌接触 <0.1N、骨盆 >45cm、末段连续保持至少3秒，末帧有拇指与食/中指对向接触。碰桌和房间障碍质量单独统计。

## 文件

[第一视角](head.mp4) · [第三视角](000000.mp4) · [腕部视角](right_wrist.mp4) · [原始训练数据](episode_raw.h5)

[完整指标](summary.json) · [采集验证](validation.json) · [双手安装验证](hand_validation.json)

优化配置与代码以本目录快照为准。目标指令加速度更低不自动意味着真实手部抖动更小，应结合视频与实际速度指标评价。
''')
print(json.dumps(summary,indent=2))
