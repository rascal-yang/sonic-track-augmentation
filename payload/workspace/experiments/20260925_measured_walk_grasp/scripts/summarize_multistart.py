"""Audit and label all planned attempts, preserving failures in denominator."""
from pathlib import Path
import json,csv
import h5py,numpy as np
B=Path(__file__).resolve().parents[1];protocol=json.loads((B/'protocol.json').read_text());rows=[]
for name,start in protocol['starts'].items():
 r=B/'runs'/name
 if not (r/'validation.json').exists():continue
 v=json.loads((r/'validation.json').read_text());c=json.loads((r/'contact_validation.json').read_text())
 obs=[json.loads(p.read_text()) for p in sorted(r.glob('observation_[0-9][0-9][0-9].json'))]
 final=obs[-1];m=final['grasp_information']['metrics']
 docking=next((s for s in obs if s['phase']=='grasp'),None)
 residual=[json.loads(l) for l in (r/'grail_execution.jsonl').read_text().splitlines()]
 assert residual and all(x['maximum_absolute_scaled_residual']==0 for x in residual)
 calls=[json.loads(l) for l in (r/'planner_calls.jsonl').read_text().splitlines()] if (r/'planner_calls.jsonl').exists() else []
 arm=[json.loads(l) for l in (r/'direct_arm_execution.jsonl').read_text().splitlines()] if (r/'direct_arm_execution.jsonl').exists() else []
 audit={'GRAIL_applied_residual_max_abs':0,'SONIC_ONNX_calls':len(calls),'walking_calls':sum(x['mode']==1 for x in calls),'standing_calls':sum(x['mode']==0 for x in calls),'external_GPT6_API':False,'authorized_hint_source':protocol['hint_source'],'arm_IK_steps':len(arm)}
 success=bool(v['strict_hold_final_s']>=3 and m['opposed_digits_contact'])
 with h5py.File(r/'episode_raw.h5','r+') as f:
  for key in ['whole_object_clearance_m','opposed_digits_contact']:
   if key in f['next_observation/metrics']:m[key]=f['next_observation/metrics/'+key][-1].item()
  success=bool(v['strict_hold_final_s']>=3 and m['opposed_digits_contact'])
  if arm:
   names=json.loads(f.attrs['joint_names']);ids=[i for i,n in enumerate(names) if n.startswith('right_') and any(k in n for k in ['shoulder','elbow','wrist'])]
   commands=f['action/joint_position_target'][:][:,ids];first=arm[0]['frame']
   audit['arm_activation_frame']=first;audit['actual_arm_target_delta_max_rad']=float(np.max(np.abs(np.diff(commands[first-1:],axis=0))))
   assert audit['actual_arm_target_delta_max_rad']<.02001
   assert docking and first>=docking['step']
   assert max(x['frame'] for x in calls)<first
  f.attrs['episode_success']=success;f.attrs['positive_behavior_cloning_eligible']=success and c['table_force_max_N']<=.05
  f.attrs['demonstration_quality']='fixed_protocol_multistart_success' if success else 'fixed_protocol_multistart_failure'
  f.attrs['test_protocol']=json.dumps(protocol);f.attrs['controller']=protocol['controller']
  f.attrs['success_definition']=protocol['success']
 (r/'controller_audit.json').write_text(json.dumps(audit,indent=2))
 dec=json.loads(sorted(r.glob('decision_[0-9][0-9][0-9].json'))[-1].read_text())
 row=dict(run=name,start_x_m=start[0],start_y_m=start[1],distance_m=float(np.linalg.norm(np.array(start[:2])-protocol['target_root_m'][:2])),navigation_success=docking is not None,docking_error_m=None if docking is None else docking['navigation_error_m'],end_to_end_success=success,clean_success=success and c['table_force_max_N']<=.05,final_clearance_m=m['whole_object_clearance_m'],final_hold_s=v['strict_hold_final_s'],table_peak_N=c['table_force_max_N'],duration_s=v['simulation_duration_s'],stop_reason=dec['reason'],planner_calls=len(calls))
 rows.append(row)
 report=f"# {name}\n\n- Initial root: {start}; target: {protocol['target_root_m']}.\n- Navigation reached guarded standing phase: {row['navigation_success']}; docking error: {row['docking_error_m']} m.\n- End-to-end success: **{success}**; clean success: **{row['clean_success']}**.\n- Whole-bottle final clearance: {row['final_clearance_m']*100:.2f} cm; final continuous strict hold: {row['final_hold_s']:.2f} s.\n- Monitored bilateral hand/forearm table peak: {row['table_peak_N']:.4f} N. Coverage does not certify all room contacts.\n- Stop reason: {row['stop_reason']}.\n- Simulation duration: {row['duration_s']:.2f} s.\n\n## Control and feedback\n\n{protocol['controller']}\n\n{protocol['author']}\n\nOne attempt per start, at most two bounded preclosure position corrections and one closure increase, no replayed joint trajectory or physical teleport. Exact decisions and visual reviews preserved.\n\n## Artifacts\n\n- [First person](head.mp4), [third person](000000.mp4), [wrist](right_wrist.mp4).\n- [Training data](episode_raw.h5), [capture validation](validation.json), [hand validation](hand_validation.json), [contacts](contact_validation.json), [controller audit](controller_audit.json).\n"
 (r/'REPORT.md').write_text(report)
summary={'planned':len(protocol['starts']),'completed':len(rows),'successes':sum(x['end_to_end_success'] for x in rows),'navigation_successes':sum(x['navigation_success'] for x in rows),'clean_successes':sum(x['clean_success'] for x in rows),'trials':rows}
if len(rows)==len(protocol['starts']):summary['end_to_end_success_rate']=summary['successes']/len(rows)
(B/'summary.json').write_text(json.dumps(summary,indent=2))
if rows:
 with (B/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
report='# 水瓶不同起点走近与抓取测试\n\n'+f"完成 {len(rows)}/{len(protocol['starts'])} 个预先固定起点；成功 {summary['successes']}/{len(rows)}。"+'\n\n|起点|距离(m)|到站|到站误差(cm)|抓取成功|末段保持(s)|最终整瓶离桌(cm)|碰桌峰值(N)|视频|\n|---|---:|---|---:|---|---:|---:|---:|---|\n'
for x in rows:
 err='—' if x['docking_error_m'] is None else f"{x['docking_error_m']*100:.2f}"
 report+=f"|{x['run']}|{x['distance_m']:.3f}|{x['navigation_success']}|{err}|{x['end_to_end_success']}|{x['final_hold_s']:.2f}|{x['final_clearance_m']*100:.2f}|{x['table_peak_N']:.3f}|[第一视角](runs/{x['run']}/head.mp4) / [第三视角](runs/{x['run']}/000000.mp4)|\n"
report+='\n## 实验范围\n\n五个固定位置各一次，相同朝向、随机种子、物体属性和控制参数。全部尝试计入，未经随机抽样，不能解释为真实环境总体成功率。\n\n'+protocol['controller']+'\n\n'+protocol['author']+'\n\n成功判据：整瓶最低点离桌超过2cm、有手部接触、瓶与桌接触力小于0.1N、骨盆高度大于45cm，末段连续保持至少3秒，末帧拇指与食/中指有对向接触。碰桌质量单独统计。\n\n[预先固定配置](protocol.json) · [CSV](summary.csv) · [详细结果](summary.json)\n'
(B/'REPORT.md').write_text(report);print(json.dumps(summary,indent=2))
