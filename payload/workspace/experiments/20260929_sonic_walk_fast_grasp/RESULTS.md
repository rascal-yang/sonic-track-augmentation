# 大步行走 + 原快速抓取：成功

成功运行：`runs/walk_fast_007`。从原操作位后方约81cm出发，以SONIC planner生成明显行走，在线经过release encoder和decoder，到位并平滑过渡后执行原快速抓取。全程没有在线IK覆盖、运动中重置机器人状态或移动瓶子。

| 指标 | 实测 |
|---|---:|
| 总时间 | 21.42 s |
| 最终前进距离（沿初始朝向） | 81.81 cm |
| 左/右脚相对身体前后运动范围 | 36.65 / 37.06 cm |
| 左/右脚世界高度变化范围 | 9.60 / 10.20 cm |
| 抓取开始时操作位误差 | 3.29 cm |
| 最终操作位误差 | 0.73 cm |
| 原快速抓取序列 | 586/586帧，11.72 s，token与手指目标未改 |
| 最终整瓶离桌高度 | 16.73 cm |
| 最终连续严格悬空保持 | 4.72 s |
| 最后3s对向接触占比 | 100% |
| 最后3s瓶子相对手腕移动 / 转动 | 1.25 mm / 0.68° |
| 双手、前臂监测到的桌面接触峰值 | 0 N |

脚的前后运动范围不是单步落脚距离。0.55m/s是巡航指令，不是全程平均速度。阶段耗时：初始化1.20s，行走3.00s，刹车1.30s，姿态过渡3.00s，稳定等待1.20s，快速抓取11.72s。

视频：[第一视角](runs/walk_fast_007/head.mp4)、[第三视角](runs/walk_fast_007/000000.mp4)、[右腕视角](runs/walk_fast_007/right_wrist.mp4)、[抽帧检查](runs/walk_fast_007/review_montage.jpg)。均为Isaac Sim RTX实时DLAA预览，**不是RT2**。第三视角录像含额外边界帧，不能直接当作逐帧对齐训练RGB；头部/腕部画面与HDF5记录状态通过对齐检查。三段视频均完整解码通过，但原速时序画质尚未认证，因此training_ready=false。

可导出文件：[executed_sonic_tokens.npz](runs/walk_fast_007/executed_sonic_tokens.npz)，包含1071×64的实际执行latent、50Hz时间戳、阶段、29个身体关节目标/状态，以及左右Revo2各6个主动关节目标。token是release encoder的量化浮点输出，不是整数词表ID。完整状态、物理子步、相机数据在`episode_raw.h5`；新增`sonic/`组从已审计的控制日志导入。`action/processed_env_action`是旧框架占位值，不能当作实际SONIC动作标签。

本次直接验证了 **planner参考 → 在线encoder → decoder → 仿真物理** 的行走，没有被过滤为原地踏步。它不等同于“把实际测得轨迹离线重新编码后再回放”；导出的固定token流尚未做单独整段回放。行走同时改变了参考来源、实时根朝向锚点和幅度，不能把改善全部归因于步幅。

站位策略：先根据干净停步的实测脚中心与快速抓取初始脚中心计算平移，再共同移动初始机器人与导航目标[1.175cm,8.908cm,0]。最终抓取目标仍是原操作位。该校准避免了桌边反复挪步；实际到位与姿态门槛仍须通过，不靠状态重置“对齐”。详见`plan.json`与`calibrate_stance.py`。

已保留失败证据：001记录目录冲突、002初始等待超时、005资产相对路径错误均未执行物理；003刹车过晚，右腕桌面碰撞20.59N；004干净行走/停步后因旧指令白名单中止；006后退修正时右指桌面碰撞11.73N。007为校准后的单次成功，不代表扰动成功率或真机验证。

检查证据：[audit.json](runs/walk_fast_007/audit.json)、[接触审计](runs/walk_fast_007/contact_validation.json)、[采集和视频校验](runs/walk_fast_007/validation.json)、[双手安装校验](runs/walk_fast_007/hand_validation.json)、[视觉审查](runs/walk_fast_007/visual_realism_review.json)。4284个200Hz物理子步全部纳入检查；实际PD目标与decoder输出换算误差小于6e-8rad，实时decoder输入及导航token重算一致。原成功快速版和保守版资产未修改。

用户于本次独立回放任务授权节省空间：失败运行001–006的原始目录已删除，简短原因与指标保存在`cleanup_manifest.json`/`attempts.jsonl`；成功007完整保留。上述“失败证据保留”历史描述由本清理说明更新。

后续完整独立回放已完成：原始执行 token 可成功复现；实际运动重新编码版在接近阶段因位置偏差停止。见 [独立回放报告](../20260929_sonic_walk_independent_replay/RESULTS.md)。
