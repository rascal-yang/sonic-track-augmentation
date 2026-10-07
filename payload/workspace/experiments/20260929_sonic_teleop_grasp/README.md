# 抓取阶段：SONIC Teleop encoder 对照

使用成功快速版 `../20260929_sonic_grasp_smooth/candidates/safe025` 的 7 路点所编译出的连续参考；只测试原地抓取，不重新规划行走。源资产、源轨迹与原成功运行不修改。

本实验把已有参考中的左右手腕、躯干位姿转换为官方 VR 三点接口，通过 release teleop encoder（mode 1）产生 64 维 token，再由 release decoder 与真实本体历史执行所有 29 个身体关节。没有新解 IK，也没有上半身关节驱动覆盖；Revo2 仍执行原来的独立手指目标。

## 坐标和输入

- 三点顺序为左腕、右腕、躯干派生头部点；不是相机位置。
- 采用官方局部偏移 `[.18,-.025,0]`、`[.18,.025,0]`、`[0,0,.35]` 米；三点位于参考骨盆坐标系，四元数为 wxyz。
- 场景 G1 缩放为 0.982730067592321，将身体相对位置除以该比例后编码；固定偏移使用模型原尺寸。世界坐标目标和 nominal-model 接口尺度分开。
- 下半身参考沿用原站立序列，按左腿六关节、右腿六关节排序，取 10 帧、步长 5。没有在线行走 planner。
- 根部朝向项在线计算为实际骨盆逆旋转乘当前参考骨盆旋转。
- 1762 维输入中全身关节参考区保持零；只有 mode1 所需字段启用。
- 输入来自成功轨迹的控制参考，不是把实测 rollout 误当成 encoder 的逆映射。复用了原关节路点派生的笛卡尔参考；尚未测试任意 GPT 新路点的泛化。

## 执行

使用 `/home/kinetra-4090-01/miniconda3/envs/sonic/bin/python`：

```bash
python experiments/20260929_sonic_teleop_grasp/prepare.py --name reference
python experiments/20260929_sonic_teleop_grasp/run.py --name teleop_001 --candidate reference
# 查看初始 observation_000.png 后，在另一个进程启动自动数值监督：
python experiments/20260929_sonic_teleop_grasp/supervise.py experiments/20260929_sonic_teleop_grasp/runs/teleop_001
python experiments/20260929_sonic_teleop_grasp/finalize.py experiments/20260929_sonic_teleop_grasp/runs/teleop_001
```

目录不可覆盖；重试须用新名称。Isaac Sim 串行运行。继承物理频率 200Hz 的双手/前臂桌面及房间接触监测和报警锁存，5 个控制步检查一次瓶子倾斜、支撑、对向接触和持物滑移。GPT 指本会话分析结果并为下一轮设计修正，不是外部模型 API 实时控制。自动监督不会创建伪造人工批准文件。

验收包括：完整执行、最后连续严格悬空至少 3 秒、最后 3 秒对向接触比例至少 95%、相对手位移不超过 15mm、相对转动不超过 10°、监测桌面接触峰值不超过 0.05N；另独立验证 encoder 重算、decoder 历史、实际 PD targets、手安装与 mimic、图像/动作对齐和视频全解码。

每次保存实际 encoder 输入、执行 token、decoder 输入输出、PD targets、完整物理状态与接触、第一/第三/腕部视角。视频为 RTX realtime/DLAA 预览，不是 RT2；视觉时间质量未验收前 `training_ready=false`。

结果：[RESULTS.md](RESULTS.md)。`candidates/reference/grasp_tokens.npz` 中的 token 是参考骨盆姿态下的离线预览；真正执行的 token 位于 `runs/teleop_001/executed_teleop_tokens.npz`，在线朝向锚定可能使两者不同。
