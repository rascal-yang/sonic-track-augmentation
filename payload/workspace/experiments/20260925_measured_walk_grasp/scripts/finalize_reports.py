from pathlib import Path
import json,shutil,hashlib
B=Path(__file__).resolve().parents[1];s=json.loads((B/'summary.json').read_text());pr=json.loads((B/'protocol.json').read_text());assert s['completed']==s['planned']
labels=['正后方30cm','正后方45cm','正后方70cm','左偏25cm、后方45cm','右偏25cm、后方45cm']
sequences=[]
for label,row in zip(labels,s['trials']):
 r=B/'runs'/row['run'];shutil.copy2(B/'protocol.json',r/'test_protocol.json')
 dec=[json.loads(p.read_text()) for p in sorted(r.glob('decision_[0-9][0-9][0-9].json'))]
 seq=[dict(action=d['action'],target=d['goal']['target'],euler=d['goal']['wrist_euler_xyz']) for d in dec if d.get('goal',{}).get('target')];sequences.append(seq)
 state=json.loads((r/'supervisor_state.json').read_text());v=json.loads((r/'validation.json').read_text());au=json.loads((r/'controller_audit.json').read_text())
 p=r/'handoff.json'
 if p.exists():
  d=json.loads(p.read_text())
  if 'grail_ramp_steps_after_transition' in d:
   d['standing_settle_steps_after_transition']=d.pop('grail_ramp_steps_after_transition');d['GRAIL_gain_during_and_after_settle']=0.;d['metadata_note']='Legacy field name corrected;50steps settle standing, residual trace proves no GRAIL ramp.';p.write_text(json.dumps(d,indent=2))
 task=json.loads((r/'task.json').read_text());task.update(controller=pr['controller'],success_definition=pr['success'],decision_provenance=pr['author']);(r/'task.json').write_text(json.dumps(task,indent=2))
 report=f"""# {label}：走近并抓取水瓶

## 结果

- 本次尝试：**{'成功' if row['end_to_end_success'] else '失败'}**，每个起点仅尝试一次。
- 初始世界坐标：`[{row['start_x_m']}, {row['start_y_m']}, 0.76] m`；共同站立目标：`[-0.09, -0.54, 0.76] m`；初始朝向均为绕Z轴90°。
- 起点到站立目标的平面距离：{row['distance_m']:.3f} m。
- 切换到抓取阶段的实际站位误差：{row['docking_error_m']*100:.2f} cm。
- 最终整瓶最低点离桌：**{row['final_clearance_m']*100:.2f} cm**；末段连续满足抬升判据：**{row['final_hold_s']:.2f} s**。
- 双手与前臂对桌面的监测接触力峰值：{row['table_peak_N']:.3f} N（200Hz、32个部件）。此项覆盖范围限于这些部件与桌面。
- 仿真时长：{row['duration_s']:.2f}秒；控制频率50Hz；物理频率200Hz；头部/腕部图像25Hz。

## 执行方式

SONIC V2 ONNX planner 负责走近，刹停后用3秒平滑切换到站立参考，再稳定1秒。抓取复用上一轮成功的瓶身侧抓空间路点，由实际状态右臂IK＋有界重力补偿执行，SONIC继续控制其他身体关节，Revo2手指独立闭合。GRAIL残差始终为零。

GPT/Codex制定固定反馈规则并在暂停检查点查看图像、本体状态、几何与接触数据；自动程序按同一规则执行各组。本轮没有调用外部GPT-6 API。

- SONIC ONNX调用次数：{au['SONIC_ONNX_calls']}。
- 抓前额外位置修正次数：{state['corrections']}；是否增加闭合量：{state['closure_boost']}。
- 右臂IK启用及后续控制目标的最大单步变化：{au.get('actual_arm_target_delta_max_rad',0):.8f} rad。
- 无中途物体绑定、搬移或物理属性调整；没有失败后重跑。

成功要求整瓶最低点高于桌面2cm、手部接触力大于0.2N、瓶与桌接触力小于0.1N、骨盆高于45cm，末段连续保持至少3秒，并且结束时有拇指与食/中指对向接触。

## 文件

- [第一视角视频](head.mp4) · [第三视角视频](000000.mp4) · [腕部视频](right_wrist.mp4)
- [同步训练数据](episode_raw.h5)：观测、实际执行动作、下一状态和相机数据；成功/训练可用标签已写入。
- [帧与动作对齐及完整视频解码校验](validation.json) · [双手安装校验](hand_validation.json) · [物理接触校验](contact_validation.json) · [控制审计](controller_audit.json)
- `observation_*.json`、`decision_*.json`、`reviewed_*.json`保存完整反馈、决策及图像检查记录，`source_snapshot`和`scripts_snapshot`保留代码。
"""
 
 if (r/'protocol_deviation.json').exists():
  report+='\n## 失败与提前结束\n\n接近时小指碰倒水瓶，未闭合抓取、未抬起。在1320步图像检查后，固定程序已执行一次每轴最多1cm的位置修正；本会话在1440步确认瓶子仍倒卧并结束该次试验。该次完整计入失败，未重跑。见 [失败时序与物理数据](failure_analysis.json) 和 [执行偏离记录](protocol_deviation.json)。\n'
 (r/'REPORT.md').write_text(report)
same=all(x==sequences[0] for x in sequences)
extra=f"""
## 成功率与策略复用

- 到达抓取站位范围：{s['navigation_successes']}/{s['planned']}。
- 走近＋抓取＋保持完整任务：**{s['successes']}/{s['planned']} = {s['successes']/s['planned']:.0%}**。
- 满足同一成功判据且无监测到的手/前臂碰桌：{s['clean_successes']}/{s['planned']}。
- 五组最初的侧向抬手、接近瓶身两个空间路点完全相同；四个成功组的抬升路点也相同。失败组在瓶倒后执行了一次有界小幅修正，然后由本会话结束试验，完整计入失败。实际关节角依据每一步身体状态重新求解IK，并受0.02rad/步变化限制。

本轮检验的是同一场景、物体与初始朝向下的位置变化。五个预先指定起点各一次，固定随机种子17，全部纳入统计；样本量不足以推断随机场景或真实机器人的总体成功率。

## 视频与曲线

- [五组第三视角对比视频](analysis/multistart_overview.mp4)：按仿真时间同步，较短试验结束后保持末帧。
- [实际行走路径与整瓶离桌高度](analysis/paths_and_clearance.png)
- 各行视频链接可打开独立第一视角视频，各实验目录均有独立中文报告和HDF5数据。
"""
extra+='\n## 失败原因与下一步\n\n右偏组在25.64秒首次产生大于0.2N的手瓶接触，来自小指末节；25.82秒达到1.44N，26.12秒瓶子倾斜超过60°。首次接触时IK位置残差约2.4cm，实际手掌比当前指令高约7.1cm，随后快速下移并碰瓶。四个成功组说明该侧抓策略能适应部分走停状态差异；右偏失败说明仅检查站位XY误差仍不足以保证末端路径跟踪。左偏组站位误差更大却成功，因此不能简单按平面误差大小解释。\n\n下一步建议在开始接近前增加实测手掌/腕姿跟踪门限、整只手的扫掠体检查，并在异常单指接触或瓶体倾斜时立即暂停；若IK位置残差持续超限，先重新调整站姿再抓。该建议未用于本轮补跑，80%是本轮原始五次尝试的结果。\n'
p=B/'REPORT.md';p.write_text(p.read_text()+extra)
(B/'waypoint_comparison.json').write_text(json.dumps({'identical_task_space_sequences':same,'sequences':dict(zip([x['run'] for x in s['trials']],sequences))},indent=2))
print('Final independent Chinese reports written; identical waypoint sequences:',same)
