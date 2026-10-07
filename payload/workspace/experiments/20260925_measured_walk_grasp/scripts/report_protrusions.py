"""Build per-run reports and a video index using measured results only."""
import csv,json
from pathlib import Path
B=Path(__file__).resolve().parents[1]
rows=[]
for run in sorted((B/'runs').iterdir()):
 if not run.is_dir() or not (run/'validation.json').exists() or (run/'acceptance_protocol.json').exists():continue
 v=json.loads((run/'validation.json').read_text());hv=json.loads((run/'hand_validation.json').read_text());obs=[]
 for f in sorted(run.glob('observation_[0-9][0-9][0-9].json')):
  s=json.loads(f.read_text());idx=s['observation_id'];aid=run/f'part_geometry_aid_{idx:03d}.json';g=json.loads(aid.read_text()) if aid.exists() else s['protrusion_geometry'];dfile=run/f'decision_{idx:03d}.json';d=json.loads(dfile.read_text()) if dfile.exists() else {};m=s['grasp_information']['metrics']
  for part,geo in g['parts'].items():
   obs.append({'observation_id':idx,'time_s':s['step']*.02,'part':part,'selected_part':d.get('selected_part'),'action':d.get('action'),'palm_to_part_center_m':geo['palm_to_center_m'],'pinch_to_part_center_m':geo['pinch_to_center_m'],'thumb_to_part_surface_vertex_m':geo['fingertip_to_surface_vertex_m']['thumb'],'index_to_part_surface_vertex_m':geo['fingertip_to_surface_vertex_m']['index'],'whole_object_clearance_m':m['whole_object_clearance_m'],'hand_force_N':m['object_hand_normal_force_N'],'table_force_N':m['object_table_normal_force_N'],'thumb_force_N':m['thumb_contact_N'],'index_middle_force_N':m['index_middle_contact_N']})
 with (run/'part_distances.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(obs[0]));w.writeheader();w.writerows(obs)
 target='near_leg' if run.name.startswith('leg') else 'near_forearm';tr=[r for r in obs if r['part']==target];best=min(tr,key=lambda r:r['pinch_to_part_center_m']);maxforce=max(r['hand_force_N'] for r in tr)
 result=json.loads((run/'result.json').read_text())
 outcome='成功' if v['success_final'] else ('曾短暂满足，但结尾未保持' if v['success_ever'] else '未达到整只玩偶离桌并稳定保持的标准')
 specific=('腿部尝试：先从上方闭合，再在滑脱后侧转手腕重试。后者受到相邻鞋子与外侧手指接触的干扰。850步起右臂改用IK物理驱动和有界重力补偿。' if target=='near_leg' else '手臂尝试：中指、无名指、小指独立预收拢，使用拇指与食指夹持。手臂区域使用根据网格补充的near_forearm标注，逐帧位置由实际物体位姿变换。控制器切换详见controller_changes.jsonl。')
 if run.name=='arm_pinch_003':
  specific+=' 先走近再抬手；转腕时接触推移并旋转了玩偶，因此重新计算前臂位置。最终闭合时夹持中心仍离前臂约4.7厘米，手部目标跟踪误差约6厘米，手指无接触，为空握。随后退出接触结束。'
 if run.name=='arm_pinch_002':
  outcome='提前终止，未完成闭合抓取'
  specific+=' 本段在15.56秒切回行走时触发ee_body_pos参考跟踪终止；尚未执行闭合抓取。没有摔倒，不作为一次完整抓取比较。'
 text=f'''# {run.name} 独立实验报告

## 结果

- **{outcome}**。
- 仿真时长：{v['simulation_duration_s']:.2f}秒；全物体离桌间隙最大值：{v['max_whole_clearance_m']*100:.3f}厘米。
- 严格成功保持最长：{v['strict_hold_max_s']:.2f}秒；结尾保持：{v['strict_hold_final_s']:.2f}秒。
- 所选部位夹持中心距离最小的反馈点：{best['time_s']:.2f}秒，{best['pinch_to_part_center_m']*100:.2f}厘米。此距离不代表夹持成功。
- 行走前6秒前进：{hv['forward_displacement_at_6s_m']*100:.1f}厘米。

## 过程与输入

{specific}

当前GPT/Codex会话读取第一视角/外部图像、机器人本体状态、物体与手部位姿、逐指接触力，写入decision_NNN.json。Isaac Sim在决策点暂停等待；不是实时速度运行，也没有额外GPT API调用。SONIC V2负责行走规划，SONIC负责下肢/身体执行；右臂是否采用直接IK执行按controller_changes.jsonl记录。GRAIL交互残差关闭。

特权信息的部位位置=玩偶实际世界位姿×网格局部标注；机器人指尖=实测Revo2关节角的URDF正运动学。原始反馈及补充part_geometry_aid文件保留每个部位到手掌、五指尖和夹持中心的距离。部位标签由网格形状估计；表面距离是到采样网格顶点的距离。仿真位姿可精确读取，但玩偶尺寸、重建形状和物理参数仍是估计值。

## 视频与记录

- [第一视角](head.mp4)
- [外部视角](000000.mp4)
- [腕部视角](right_wrist.mp4)
- [部位距离表](part_distances.csv)
- [距离与接触反馈曲线](feedback_curves.png)
- 原始数据：episode_raw.h5；决策：decision_*.json；图像：observation_*.png。
- 视频逐帧解码、数据时间对齐与Revo2安装/联动约束验证均通过，见validation.json、hand_validation.json。

严格成功定义：所有玩偶碰撞顶点高于桌面2厘米、手接触力>0.2N、桌面支撑<0.1N、机器人稳定站立，连续至少0.5秒。未成功数据应标为失败/纠错轨迹；此次没有导出新的LeRobot成功训练集。
'''
 text+='\n## 视频校验规格\n\n| 文件 | 编码 | 分辨率 | 帧率 | 帧数 | 时长（秒） | 字节数 |\n|---|---|---|---|---|---|---|\n'
 for name,info in v['videos'].items():
  text+=f"| {name} | {info['codec_name']} | {info['width']}×{info['height']} | {info['r_frame_rate']} | {info['nb_frames']} | {info['duration']} | {info['bytes']} |\n"
 text+='\n[实际评估与控制契约](evaluation_contract.json)优先于旧task.json中的物体原点抬升代理定义。\n'
 if run.name in ('arm_pinch_002','arm_pinch_003'):
  text+='\n## 后续纠正：三指预闭合设置\n\n初始导航主动指定三指95%闭合，旧open_hand未清除逐指覆盖。不能视作五指正常张开的Revo2基线。已修复，见同级revo2_open_hand_check_004姿态验收。\n'
 (run/'REPORT.md').write_text(text)
 rows.append((run.name,outcome,v,best,target))
html=['<!doctype html><meta charset="utf-8"><title>Revo2 突起部位抓取</title><style>body{background:#17202b;color:#edf0f3;font:17px system-ui;max-width:1200px;margin:32px auto;padding:20px}video{width:49%;border-radius:8px}a{color:#86cfff}img{max-width:480px}section{padding:20px;background:#223040;margin:20px 0;border-radius:12px}</style><h1>Revo 2 · 玩偶突起部位抓取</h1><p>图像 + 本体状态 + 特权几何/接触反馈。每个实验保存独立报告和距离记录。</p>']
for n,o,v,b,t in rows:
 html.append(f'<section><h2>{n}</h2><p>{o}；{v["simulation_duration_s"]:.0f}秒</p><video controls src="runs/{n}/head.mp4"></video> <video controls src="runs/{n}/000000.mp4"></video><p>左：第一视角，右：外部视角。<a href="runs/{n}/REPORT.md">独立报告</a> · <a href="runs/{n}/part_distances.csv">距离CSV</a></p></section>')
html.append('<h2>局部几何标注</h2><img src="assets/parts_annotated.png"><p>近/远标签以初始站位为准，不表示解剖学左右。尺寸和隐藏面为单照片重建估计。</p>')
(B/'index.html').write_text('\n'.join(html))
(B/'REPORT.md').write_text('# Revo2 突起部位抓取实验\n\n[视频总览](index.html)\n\n'+ '\n'.join(f'- [{n}](runs/{n}/REPORT.md)：{o}。' for n,o,*_ in rows)+'\n\n原资产、原实验均保留。新代码及每次运行的source_snapshot保存修改记录。这些试验包含不同手指与手臂控制设置，仅是探索性尝试，不能当作受控成功率对照。\n')
print([(n,o) for n,o,*_ in rows])
