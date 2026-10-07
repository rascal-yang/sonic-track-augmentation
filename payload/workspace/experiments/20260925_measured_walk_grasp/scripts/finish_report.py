"""Write result reports from completed validations, without inferring success."""
import json
from pathlib import Path
B=Path(__file__).resolve().parents[1]
for name in ['gpt_walk_dex3_001','gpt_walk_revo2_001']:
    p=B/'runs'/name;v=json.loads((p/'validation.json').read_text());hand='BrainCo Revo 2' if 'revo2' in name else '原Dex3三指手'
    decisions=sorted(p.glob('decision_[0-9][0-9][0-9].json'))
    lines=[f'# {name} 实验报告',f'\n构型：{hand}。控制：当前GPT/Codex会话在线读取图像、本体和特权信息，更新路点；SONIC planner＋IK＋冻结SONIC执行，GRAIL残差关闭。等待决策时物理时间暂停，没有外部GPT API调用，也没有自动几何策略代替本次会话决策。',f'\n## 结果\n\n- 实际仿真时长：{v["simulation_duration_s"]}秒。\n- 控制步：{v["control_steps"]}；物理子步：{v["physics_substeps"]}；GPT决策：{len(decisions)}次。\n- 严格成功曾发生：{v["success_ever"]}；结束成功：{v["success_final"]}。\n- 最大整物最低碰撞点离桌高度：{100*v["max_whole_clearance_m"]:.3f}cm；最长严格保持：{v["strict_hold_max_s"]}秒。\n- 显式close_hand命令：{v["close_attempts"]}次；reach中的开合微调另外保留在日志中。\n- 已通过：相机安装变换、图像/状态/动作时间戳、状态连续性、数值有限性、三路视频完整解码。', '\n## 视频\n\n| 文件 | 编码 | 分辨率 | fps | 帧数 | 秒 | 字节 |\n|---|---|---|---|---|---|---|']
    for n,x in v['videos'].items():lines.append(f'| {n} | {x["codec_name"]} | {x["width"]}×{x["height"]} | {x["r_frame_rate"]} | {x["nb_frames"]} | {x["duration"]} | {x["bytes"]} |')
    lines.append('\n## 在线决策时间线\n\n| 仿真秒 | 动作 | 理由 |\n|---|---|---|')
    for file in decisions:
        d=json.loads(file.read_text());o=json.loads((p/f'observation_{d["observation_id"]:03d}.json').read_text());lines.append(f'| {o["step"]*.02:.1f} | {d["action"]} | {d["reason"]} |')
    lines.append('\n每个decision均对应先生成的observation JSON/图像。首段4秒，此后2秒反馈。运行中的单次路点并非预先录好的未来动作；历史握姿与几何知识作为先验使用。result.json中的task_success_proxy是旧原点抬升诊断，最终标签以严格全碰撞体离桌＋接触判据的validation.json为准。')
    if 'revo2' in name:
        h=json.loads((p/'hand_validation.json').read_text());lines.append(f'\n## 换手验证\n\n6秒时实际前进{h["forward_displacement_at_6s_m"]*100:.1f}cm；骨盆最低{h["minimum_pelvis_z_m"]:.3f}m。左右手固定关节最大位置误差分别为{h["left"]["mount_max_error_m"]*1000:.4f}/{h["right"]["mount_max_error_m"]*1000:.4f}mm。联动误差详见hand_validation.json。asset_integration_validation.json确认16个旧手刚体不可见且无活动碰撞，左右Revo2均有实体物理连接。')
        lines.append('\n旧手14个关节仅作SONIC兼容占位，每个旧手刚体质量0.0001kg。新手每侧6主动＋5被动联动自由度，原URDF质量约0.385kg；保留模型力矩上限。数值稳定用电机等效转动惯量0.0002kg·m²/关节与手部求解器32/8迭代属于估计；手内自碰撞关闭。未进行真实手动力学标定，不能把这次结果直接当实机预测。')
        lines.append('\n## 失败现象\n\n第一次闭合曾测得拇指9.2N、对侧手指10.3N接触，但试抬时接触丢失；增加闭合量使指尖中心沿弯曲轨迹移位。后续修正一度只有无名指和小指接触，出现推移物体。最后张手撤回，未伪装成成功示范。观察支持“当前抓取姿态与闭合控制尚未适配Revo2”，但单次试验不能判定手型优劣；手臂目标跟踪误差也曾达到约8cm。')
    (p/'REPORT.md').write_text('\n'.join(lines))
revo=json.loads((B/'runs/gpt_walk_revo2_001/validation.json').read_text());hv=json.loads((B/'runs/gpt_walk_revo2_001/hand_validation.json').read_text())
(B/'REPORT.md').write_text(f'''# 缩窗、先走近、GPT在线修正与Revo 2换手

已完成场景修改、左右Revo2的物理接入，并保存一条{revo['simulation_duration_s']:.0f}秒正式试验视频。**本次没有抓稳玩偶**，数据如实标为失败，原成功实验保持不变。

## 本次变化

| 项目 | 原设置 | 新设置 |
|---|---|---|
| 窗洞宽×高 | 1.84×1.20m | 1.30×0.90m |
| 窗台高度 | 1.00m | 1.10m |
| 内开窗扇角度 | 67° | 45° |
| 机器人初始根部y | -0.54m | -0.94m |
| 高层路点 | 固定几何反馈代码 | 当前GPT/Codex会话根据逐次图像＋状态在线提交 |
| 灵巧手 | 原三指手 | 左右BrainCo Revo 2物理模型 |

正式Revo2运行6秒时前进{hv['forward_displacement_at_6s_m']*100:.1f}cm，随后伸手、闭合、试抬、根据接触失败重新调整，最后张开撤回。GPT共13次在线决策，第一段4秒，此后每2秒仿真时间反馈；等待决策时物理暂停，因此不是墙钟实时控制，也不应拿这次等待时间评价纯模型推理延迟。

## 视频入口

- 第一视角：runs/gpt_walk_revo2_001/head.mp4
- 第三视角：runs/gpt_walk_revo2_001/000000.mp4
- 手腕视角：runs/gpt_walk_revo2_001/right_wrist.mp4
- 浏览器打开index.html可查看视频及对应GPT决策。

## 分开的实验记录

- gpt_walk_dex3_001：原手基线，32秒，先走近后抓取；3次显式闭合，未持续抓稳。
- revo2_mount_debug_001：初始化时配置顺序错误，无有效完整视频。
- revo2_mount_debug_002：初始几步出现数值发散，不作为有效抓取或训练样本。
- revo2_mount_debug_003：修正后的2秒稳定安装测试，固定连接及联动验证通过。
- gpt_walk_revo2_001：正式28秒Revo2在线反馈试验，完整视频与传感器、物理约束验证通过，抓取失败。详细数值及完整决策时间线在该目录REPORT.md。

## 实现与边界

新手使用本机已有V2 URDF/STL，来源说明保留于assets/brainco_candidate；官方参考：<https://github.com/BrainCoTech/revo2_description>。本地模型不声称与最新官方版本逐字一致。

Revo2通过物理固定关节接G1手腕，有质量、真实网格碰撞、主动驱动及PhysX联动；运行中不传送手或物体。原手16个刚体关闭显示与碰撞，14个旧关节以极小质量占位保留，以兼容既有SONIC输入。新手的数值稳定惯量、驱动与手内自碰撞设置需看分报告，不代表实机标定。GRAIL交互残差仍关闭，SONIC没有重新训练。

场景尺寸和未拍摄部分仍是照片约束下的估计。此次交付是有效的换手仿真与失败试验视频，不是成功抓取示范或已验证的实机策略。Revo2多出的控制通道已保存到原始HDF5，但旧43DOF LeRobot导出器对新手构型禁用，避免漏动作；此前成功训练数据不受影响。
''')
print(B/'REPORT.md')
