# 25mm抓取：保留的成功版本

按用户要求，本轮仅保留快速版safe025与保守版smooth_final025的候选、完整运行数据及视频；其他候选和回放记录已删除。

|版本|总用时(s)|连续持瓶(s)|整瓶离桌(cm)|最后3s相对手位移(mm)|手腕加速度RMS(m/s²)|jerk RMS(m/s³)|
|---|---:|---:|---:|---:|---:|---:|
|快速版|11.72|4.68|16.32|0.55|0.692|12.638|
|保守版|15.42|4.58|15.93|1.27|0.572|10.838|

两版均完整通过SONIC encoder/decoder物理回放，7个右臂关键参考姿态，未监测到手或前臂碰桌。保守版的实际加速度和jerk低于快速版；不能宣称全段平滑度已全面优于原始轨迹。未验证位置扰动鲁棒性或真机回放。

- 快速版：[路点](candidates/safe025/waypoints.json)、[稠密参考](candidates/safe025/grasp_reference.npz)、[token](candidates/safe025/grasp_tokens.npz)、[实际state与PD target](runs/safe025_001/executed_trajectory_named.npz)、[第一视角](runs/safe025_001/head.mp4)、[第三视角](runs/safe025_001/000000.mp4)、[验收](runs/safe025_001/quality.json)。
- 保守版：[路点](candidates/smooth_final025/waypoints.json)、[稠密参考](candidates/smooth_final025/grasp_reference.npz)、[token](candidates/smooth_final025/grasp_tokens.npz)、[实际state与PD target](runs/smooth_final025_001/executed_trajectory_named.npz)、[第一视角](runs/smooth_final025_001/head.mp4)、[第三视角](runs/smooth_final025_001/000000.mp4)、[验收](runs/smooth_final025_001/quality.json)。

控制50Hz，物理200Hz；64维SONIC latent与Revo2手指目标分开记录。一次恢复原轨迹830步初态，之后由实时本体历史和固定token经decoder驱动29身体关节，没有在线IK。完整HDF5、接触记录、安装/对齐验证及源代码快照保留在各run内。视频为RTX realtime/DLAA预览，不是RT2；训练视频时间质量未认证，training_ready=false。
