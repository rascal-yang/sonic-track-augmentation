# SONIC Teleop encoder：抓取阶段成功

本次 **1 次物理尝试，1 次成功，0 次失败修正**。仅比较抓取阶段，没有运行行走规划。原快速版是已保存的成功运行，本轮没有重新计为一次测试。

|指标|原快速版 G1 encoder|本次 Teleop encoder|
|---|---:|---:|
|执行时间|11.72 s|11.72 s|
|连续严格持瓶|4.68 s|4.42 s|
|最终整瓶离桌|16.32 cm|12.85 cm|
|最后 3 秒相对手位移|0.55 mm|0.67 mm|
|最后 3 秒对向接触比例|100%|100%|
|监测桌面接触峰值|0 N|0 N|
|手腕加速度 RMS|0.692 m/s²|0.809 m/s²|
|手腕 jerk RMS|12.638 m/s³|15.282 m/s³|

结论：这条成功抓取参考可以通过 Cartesian teleop mode1 执行，不需要重新求解上半身 IK。本次并未改善整体平滑度；提起高度比原版低约 3.48cm，但完整满足当前抓握验收。

## 视频

- [并排第一视角：左原版、右 Teleop](comparison_head.mp4)
- [Teleop 第一视角](runs/teleop_001/head.mp4)
- [Teleop 第三视角](runs/teleop_001/000000.mp4)
- [原版第一视角](../20260929_sonic_grasp_smooth/runs/safe025_001/head.mp4)
- [关键阶段截图](runs/teleop_001/contact_review_montage.jpg)

视频为 RTX realtime / DLAA 预览，不是 RT2。全视频解码和帧/状态对齐通过；原生播放速度下的时间质量仍待验收，腕部相机存在自身遮挡，training_ready=false。

## 实际执行与验证

沿用源 7 路点所编译出的连续参考，提取左右手腕和躯干派生头部目标。未重新用笛卡尔线性插值替代原连续曲线，也未测试任意 GPT 新路点。下半身沿用站立参考，Revo2 手指目标逐帧保持原值。由 mode1 在线 encoder 产生 64 维 token，release decoder 加实时本体历史生成全部 29 个身体关节 PD targets。

编码输入全身关节参考区为零；逐帧 encoder 重算误差 0，encoder 输入和 decoder 历史重建误差 0，物理 PD target 最大误差约 5.94e-8 rad。586 控制步、2344 物理子步，双手安装位置/旋转与 mimic、接触监测、视频对齐均通过。初始化仅在第一物理步前恢复一次，此后没有物理状态写入。

末帧右腕相对参考位置误差 1.83cm、朝向误差 23.32°，所以“任务成功”不表示精确复制目标位姿。不能从单次成功推断任意物体位置、任意 waypoint、完整移动抓取或真机成功率。原 G1 mode0 也有跟踪误差；本次同时采用 mode1 所需的实时根部朝向锚定，因此不是孤立的网络精度消融。

## 保存的数据

- [七个笛卡尔参考锚点](candidates/reference/cartesian_waypoints.json)
- [实际执行 token、encoder/decoder 输入输出与 PD targets](runs/teleop_001/executed_teleop_tokens.npz)
- [完整物理、相机和 sonic_teleop 数据](runs/teleop_001/episode_raw.h5)
- [物理验收](runs/teleop_001/quality.json)、[控制审计](runs/teleop_001/token_summary.json)、[逐阶段偏差](runs/teleop_001/diagnosis.json)、[反思](runs/teleop_001/reflection.json)

GPT 指当前会话进行方案编写与结果分析，执行中使用数值监督器，没有外部 GPT API。因为首次完整成功，本轮没有增加修正实验。
