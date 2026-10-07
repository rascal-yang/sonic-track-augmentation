# 实测场景走近抓瓶与轨迹简化

桌子112×60×75 cm，水瓶高23 cm，G1默认直立高130 cm。“水杯”按场景中的水瓶执行。

已完成走近、抓瓶与稳定持握。成功基线为 `walk_009`；简化连续弧线已通过独立物理验证。

| 运行 | 结果 | 仿真时长 | 整瓶最终离桌 | 严格末段保持 | 桌/房间接触峰值 |
|---|---|---:|---:|---:|---:|
| walk_001 | 初始化拒绝 | 0.02 s | — | — | — |
| [walk_002](runs/walk_002/REPORT.md) | 未完成抓起 | 12.62 s | -0.00 cm | 0.00 s | 0.000/0.000 N |
| [walk_003](runs/walk_003/REPORT.md) | 程序中断 | 6.10 s | -0.00 cm | 0.00 s | 0.000/0.000 N |
| [walk_004](runs/walk_004/REPORT.md) | 未完成抓起 | 27.92 s | -0.01 cm | 0.00 s | 0.000/0.000 N |
| [walk_005](runs/walk_005/REPORT.md) | 程序中断 | 26.70 s | -0.00 cm | 0.00 s | 0.000/0.000 N |
| [walk_006](runs/walk_006/REPORT.md) | 未完成抓起 | 32.32 s | -0.00 cm | 0.00 s | 0.000/0.000 N |
| [walk_007](runs/walk_007/REPORT.md) | 未完成抓起 | 21.32 s | -0.00 cm | 0.00 s | 0.000/0.000 N |
| [walk_008](runs/walk_008/REPORT.md) | 未完成抓起 | 16.62 s | -0.00 cm | 0.00 s | 0.000/0.000 N |
| [walk_009](runs/walk_009/REPORT.md) | 成功 | 32.82 s | 9.41 cm | 5.86 s | 0.000/0.000 N |
| [walk_010_arc](runs/walk_010_arc/REPORT.md) | 成功 | 34.52 s | 14.98 cm | 6.10 s | 0.000/0.000 N |

walk_005只有部分轨迹：视频/观测到26.70秒，接触日志到26.40秒，缺失最终HDF5与末段子步。不能按完整物理验收解释该行。所有失败保留，不把逐轮调参结果当作泛化成功率。

## 主要发现

- 位置误差足够小不等于手已稳住：早期闭指时手腕仍下移，瓶子倾倒。后来加入实际速度与位置共同门限。
- 单把手心目标向瓶身推进1 cm仍未解决到位误差；较近站位提供了必要的伸展余量。
- 给站立交接增加腰部前倾导致身体和双脚后移；撤掉该改动，保留实测腿部/腰部姿态后恢复稳定支撑。
- `walk_009`采用起点Y=-0.97 m、目标站位Y=-0.485 m、4 cm刹停范围、实测支撑交接和推进1 cm的掌心目标，完成对向夹持及持瓶。成功不单独归因于某一项改动。

## 证据与复用

- [逐轮失败分析与计划](FAILURE_ANALYSIS.md)，各运行 `reflection.json` 与 `reports/attempt*` 保留预测、改变和验证结果。
- 历史成功参考：`../20260921_new_scene_closed_loop/runs/walk_02_middle_v3`。只复用动作阶段/路点先验，实际状态重新计算IK，不回放旧电机命令。
- [本次成功配置](reference/successful_walk009_config.json)及[实际轨迹先验说明](reference/successful_walk009_actual_hint.json)。`successful_walk009_actual_hint.npz`为实测状态先验，不是可无条件执行的控制器。
- 每轮目录包含源代码/配置快照、实际轨迹、决策、视频、HDF5及验证报告；新运行不得覆盖旧目录。

## 控制与验收边界

SONIC V2 ONNX负责行走，SONIC负责实时平衡，右臂使用实际状态IK和受限重力补偿；GRAIL残差为零。会话在暂停图像检查点审查，其间运行确定性监督程序，没有外部GPT API。执行中不重置机器人或瓶子。提篮与玩偶固定，瓶子动态；质量、摩擦仍是估计。

成功要求：整瓶最低点离桌>2 cm，有手瓶接触，瓶桌接触<0.1 N，骨盆>45 cm，末段保持≥3秒，末帧拇指与食/中指有对向接触。碰撞独立统计，覆盖双手、腕、肘对桌/房间接触，不能扩展为全身碰撞认证。

在线视频使用旧实时RTX + DLAA；RT2成品由g1-realtosim恢复保存状态重渲染，不推进物理。任务成功、动作数据校验与视觉验收独立；未完成原速时序验收的数据保持training_ready=false。

运行入口：本机sonic Python执行 `scripts/launch_optimized.py <新运行名>`，起点在protocol.json注册，控制参数见optimized_config.json。当前配置已由walk_010_arc验证；复用成功方案仍应读取成功运行快照。

## 最终交付

- [结果与简化对比](RESULTS.md) · [RT2成品视频](renders/walk_010_arc_rt2/video.mp4)
- [已验证的简化配置](reference/successful_walk010_arc_config.json) · [实际轨迹先验](reference/successful_walk010_arc_actual_hint.json)
当前optimized_config.json为已成功验证的walk_010_arc配置；详细限制见结果报告。
