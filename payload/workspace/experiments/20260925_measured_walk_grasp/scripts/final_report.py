from pathlib import Path
import json,csv
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];p=json.loads((B/'protocol.json').read_text());names=p['evaluation_runs'];rows=[]
for n in names:
 f=B/'runs'/n/'summary.json'
 if f.exists():rows.append(json.loads(f.read_text()))
stress=B/'runs'/p.get('retry_stress_run','stationary_retry_006')/'summary.json';recovery=json.loads(stress.read_text()) if stress.exists() else None
supp_path=B/'runs/walk_04_left_repeat/summary.json';supp=json.loads(supp_path.read_text()) if supp_path.exists() else None
summary={'planned':len(names),'completed':len(rows),'successes':sum(r['success'] for r in rows),'clean_successes':sum(r['clean_success'] for r in rows),'trials':rows,'retry_stress_test_separate':recovery,'supplementary_left_repeat':supp,'infrastructure_interruptions':sum(bool(r.get('infrastructure_interrupted')) for r in rows),'complete':len(rows)==len(names) and recovery is not None and supp is not None}
if len(rows)==len(names):summary['success_rate']=summary['successes']/len(rows)
manifest=[]
for sr in sorted((B/'runs').glob('*/summary.json')):
 d=json.loads(sr.read_text());rr=sr.parent;stress_run=rr.name==p.get('retry_stress_run');incomplete=bool(d.get('infrastructure_interrupted'));manifest.append({'run':rr.name,'raw_hdf5':str(rr/'episode_raw.h5'),'head_video':str(rr/'head.mp4'),'success':d['success'],'complete_capture':not incomplete,'default_positive_BC':bool(d['clean_success'] and not stress_run and not incomplete),'recovery_episode':bool(d['retries']),'deliberate_stress_test':stress_run,'report':str(rr/'REPORT.md')})
(B/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
(B/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
if rows:
 keys=['run','success','clean_success','attempts','retries','duration_s','strict_final_hold_s','final_whole_bottle_clearance_m','table_contact_peak_N','room_contact_peak_N','navigation_planner_calls','stop_reason']
 with (B/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in rows])
report=f'''# 新场景、新材质：G1 / Revo2 闭环走近抓瓶

{'全部计划实验与独立重试测试已完成。' if summary['complete'] else '正在运行；以下只列已验证结果。'}

## 站定抓取与优化

|方案|成功|总仿真时长|初次确认抬起|最终整瓶离桌|末段严格保持|监测障碍接触|
|---|---|---:|---:|---:|---:|---:|
|基线 stationary_002|是|19.22s|13.28s|10.45cm|6.44s|0N|
|采用的两点优化 stationary_004|是|17.42s|11.96s|9.42cm|5.96s|0N|

表中“初次确认抬起”是连续满足抬升条件0.5秒的首次时刻；最终成功另要求末段连续保持至少3秒。

全过程缩短约9.4%，初次确认抬起提前约1.32秒。采用更短的桌沿外抬手、提前减速的接近时间曲线、接触阶段更低的手臂目标速度，以及缓慢闭指和抬升。依照成功数据选择瓶身中段的对向抓握，目标以当前瓶子位置为基准，未回放历史电机指令。抬升目标由15cm调整为14cm，多个参数同时改变，不能将改善归因于单一组件。

闭指前保留两个空间目标：桌沿外抬手、瓶身侧面。减少为一个目标的连续弧线在实际运行中碰桌，故未采用。名义IK与包围盒通过并不能代替物理跟踪与碰撞检查。

平滑性指标并非全面改善：闭指前两秒，实际手腕速度P95由0.11585降至0.10776m/s，滤波后的jerk RMS由3.447降至3.100m/s³，但加速度RMS由0.204升至0.328m/s²。使用同一11样本、三阶Savitzky–Golay导数估计，仅比较这两次实验。

[站定第一视角前后对比](reports/stationary_before_after.mp4) · [动作曲线](reports/stationary_comparison.png) · [基线报告](runs/stationary_002/REPORT.md) · [采用版本报告](runs/stationary_004/REPORT.md)

## 五个固定起点的完整任务

已有记录 {len(rows)}/{len(names)} 组（含进程中断），成功 {summary['successes']}/{len(rows)}。初始朝向均为朝向桌子；位置在站位后方30、45、70cm，以及左右偏25cm、后方45cm。每个起点一个连续物理episode，最多允许两次受限重试，所有正常测试失败计入分母。

|组|成功|抓取尝试/重试|时长(s)|末段保持(s)|离桌(cm)|桌/房间接触峰值(N)|视频/报告|
|---|---|---:|---:|---:|---:|---:|---|
'''
for r in rows:
 n=r['run'];report+=f"|{n}|{'是' if r['success'] else '否'}|{r['attempts']}/{r['retries']}|{r['duration_s']:.2f}|{r['strict_final_hold_s']:.2f}|{r['final_whole_bottle_clearance_m']*100:.2f}|{r['table_contact_peak_N']:.3f}/{r['room_contact_peak_N']:.3f}|[第一视角](runs/{n}/head.mp4) · [第三视角](runs/{n}/000000.mp4) · [独立报告](runs/{n}/REPORT.md)|\n"
if len(rows)==5:report+='\n[五组第一视角对比](reports/multistart_head.mp4) · [实际行走路径](reports/actual_routes.png)\n'
report+='\n左侧首轮在第1184步进程异常退出，保守记为未成功，未从五组分母中删除。其HDF5仅恢复1150步，标为不完整，不能作为完整成功训练样本。退出原因未确定。原视频保留。\n'
if supp:
 report+=f"\n同参数左侧独立复测：{'成功' if supp['success'] else '未成功'}，时长{supp['duration_s']:.2f}s，离桌{supp['final_whole_bottle_clearance_m']*100:.2f}cm，末段保持{supp['strict_final_hold_s']:.2f}s。计入复测的累计完整任务启动为6次，成功{summary['successes']+int(supp['success'])}/6。复测没有替换原结果。 [复测视频](runs/walk_04_left_repeat/head.mp4) · [复测报告](runs/walk_04_left_repeat/REPORT.md)\n"
report+='''
## 独立的自动重试测试

第一次抓取目标刻意向X正方向偏移8cm，以检验夹空后的恢复。该实验不混入五个正常起点的成功率。保持同一个物理场景：没有重置或瞬移瓶子、机器人；受限恢复顺序是张手、退回、读取当前瓶子位置、改变候选高度后再接近并闭合。
'''
if recovery:
 report+=f"\n结果：{'成功' if recovery['success'] else '失败'}；实际重试 {recovery['retries']} 次；整瓶末段离桌 {recovery['final_whole_bottle_clearance_m']*100:.2f}cm，严格保持 {recovery['strict_final_hold_s']:.2f}s。\n\n[重试第一视角](runs/{recovery['run']}/head.mp4) · [独立报告](runs/{recovery['run']}/REPORT.md) · [重试事件](runs/{recovery['run']}/retry_events.jsonl)\n"
else:report+='\n测试尚未完成。\n'
report+='''
## 右侧起点失败与恢复边界

右侧组闭指后有约18N对向接触，但瓶底未离桌；抬升初段瓶子先沿桌面移动约4cm，然后倾倒并脱手。腕部跟踪和接触几何的耦合是待验证原因，不能只凭接触力判断抓稳。当前恢复仅支持直立且受桌面支撑的瓶子，倾倒后触发结束，因此此组没有自动重抓。后续应在离桌前增加横向拖移/倾角检查，并验证接触中心和夹持裕量；本轮没有把未验证改动包装成已解决。

## 开发失败与未采用方案

- stationary_001：缺少几何区域描述文件，0个控制步；修复启动依赖后另建实验，原错误保留。
- stationary_003：硬限制反馈关节目标加速度后，接近末段实际手掌上下摆幅约4.66cm，基线同长度窗口约0.65cm。闭指前主动结束，拒绝该方案。指令更平滑不等于真实手部更稳。
- stationary_arc_005：单目标连续弧线，名义整手包围盒净间隙约6.8cm；实际执行后半段跟踪滞后，小指在5.355s碰桌，约4.14N触发停止。没有把这一轮计作成功。
- walk_01_near、walk_03_far、walk_04_left：走近开发验证均到站，误差约2.0/3.7/7.2cm；由于加载的抓取优化已被拒绝，在抓取前结束。它们是导航开发记录，既不记作完整抓取成功，也不混入正式五组分母。

正式near/far目录的高层配置在首次抓取前修订，保留旧配置、修订原因和脚本；机器人和物体状态没有重置。目录v3是运行编号，实际采用各目录的final_guarded_two_target_reference配置。

## 实际控制、反馈和成功定义

- 走近：SONIC V2 ONNX根据实测根节点位置生成参考，SONIC低层策略依据实时本体状态执行。站位为[0.37,-0.54,0.76]m，朝向+Y。
- 抓取：右臂当前状态IK与受限重力补偿，SONIC控制其余关节。GRAIL交互残差为零；不将它标记为GRAIL控制实验。
- GPT/Codex本会话在暂停检查点读取第三/第一视角图像、本体状态、物体真值、距离和接触；其间由已记录的确定性反馈程序执行。未调用外部GPT API，不是每个物理步都进行模型推理。
- 物理200Hz、控制50Hz、第一视角25FPS。对双手、肘部、腕部与桌子/房间的接触持续监测，超过0.5N触发下个控制检查点，响应最多约20ms。覆盖不等于全身所有碰撞的认证。
- 成功：整瓶最低点离桌>2cm、有手瓶接触、瓶桌接触<0.1N、骨盆>45cm，末段连续满足至少3秒，末帧有拇指与食/中指对向接触。任务成功与碰撞质量分开统计。
- 恢复只在瓶子仍由桌面支撑、姿态合适时执行；倒瓶、失稳、障碍接触结束实验。未声称能恢复所有失败。

## 资产与数据

使用20260920_photo_scene的新办公场景和水瓶，以及20260921_robot_materials的新G1/Revo2外观。保留原资产，未为了成功修改质量、摩擦、碰撞或关节力矩上限；这些值仍是此前重建估计值，未经过真机标定。

正常结束的每组独立保存视频、episode_raw.h5、图像/状态/决策、200Hz接触记录、源码快照、配置和报告。验证包括状态—动作连续对齐、相机外参/时间戳、双手安装旋转与位置、mimic关节、有限数值、视频全量解码。失败episode保留并标注；刻意偏移目标的恢复测试不作为默认正常行为克隆正样本。

五个固定起点不是随机抽样；本报告的成功率仅针对这五次测试，不能推断任意场景或真机成功率。

[配置](protocol.json) · [资产清单](asset_manifest.json) · [完整汇总](summary.json) · [CSV](summary.csv) · [复运行说明](README.md) · [数据清单与训练筛选标记](dataset_manifest.json)
'''
(B/'REPORT.md').write_text(report)
if rows:
 fig,ax=plt.subplots(figsize=(8,7));ax.add_patch(plt.Rectangle((-.56158,-.29423),1.12316,.58846,facecolor='tan',alpha=.4,label='Desk'))
 for r in rows:
  with h5py.File(B/'runs'/r['run']/'episode_raw.h5') as f:xy=f['observation/root_position'][:,:2]
  ax.plot(xy[:,0],xy[:,1],label=r['run']);ax.plot(*xy[0],'o')
 ax.plot(.37,-.54,'k*',ms=13,label='Station');ax.plot(.36,-.15,'rx',ms=10,label='Bottle');ax.set(xlabel='World X (m)',ylabel='World Y (m)',title='Measured pelvis XY trajectories');ax.axis('equal');ax.grid(alpha=.3);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(B/'reports/actual_routes.png',dpi=160);plt.close(fig)
print({k:v for k,v in summary.items() if k not in ['trials','retry_stress_test_separate','supplementary_left_repeat']})
