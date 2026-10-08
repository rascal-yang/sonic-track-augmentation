# RT2 渲染设置、速度与画质对比

更新日期：2026-10-08。当前默认采用 **D 的 RT2＋原生分辨率 DLAA，普通状态 1 update**；可切换为 2/4 updates。最新固定 token 实测中，相机区间由 D4 的 46.51s 降为 D1 的 13.46s，含物理的执行区间由 198.87s 降为 158.03s。每配置只有一次运行，画质抽查通过不代表完整时序验收。

## 如何运行

在 `g1-realtosim` 仓库，使用已有 `g1_capture_v1` 原始状态、对应场景及 episode JSON：

```bash
python -m g1_realtosim --runtime configs/local.runtime.json render \
  --episode /path/to/episode.json --output /path/to/new_render \
  --profile rt2 --rt2-updates 1
# 对照：把 --rt2-updates 改成 2 或 4。
# 保留 PT128：--profile pt128；RT2 updates 参数不改变 PT 的累积算法。
```

`g1_realtosim.render` 保存并恢复机器人、双手、操作物体及声明的所有动态场景刚体，保留缩放与相机变换，不推进物理；缺少状态/资产时不应拿别的轨迹填补。默认输出无损 HDF5 RGB、观看用 H.264 MP4、相机/帧索引、实际设置和计时。`integrations/training_capture.py` 是历史参考快照，不是该入口；`recipes/mobile60_scene/render_episode.py` 委托外部 mobile60 场景包，其内部渲染不会因为本仓库保存状态入口更新而自动替换。使用新默认设置请走以上入口，或使用下述更新后的 SONIC 管线。

在 `sonic-grasp-augmentation` 仓库，使用与新资源版本匹配的完整资源包，选择一个新的 runtime：

```bash
python launch.py prepare --runtime /path/to/new_runtime
python launch.py candidate --runtime /path/to/new_runtime \
  --output /path/to/candidate
python launch.py collect --runtime /path/to/new_runtime \
  --candidate /path/to/candidate --output /path/to/new_pair \
  --render-updates 1
```

`collect` 包括首次物理执行＋RGB/标签采集、独立冻结 token replay、LeRobot 导出。`trial`、`collect` 均支持 `--render-updates 1|2|4`，默认1；批量 collect 同样生效；批次续跑绑定更新次数，改为2/4时使用新输出目录，防止混用旧D1样本。独立 replay 沿用原无相机设计，`replay --render-updates` 不会新增 RGB。直接调用 payload 的 lean_run 时可用 `SONIC_RENDER_UPDATES=1|2|4`。启动适配器在 Kit 启动前选择 RT2；原相机 pose、Replicator CPU RGB 读回、视频写入及控制时序继续复用。

两个仓库的 `render_settings.py` 核心参数与辅助实现逐字相同，只保留各自的入口适配。无需安装另一仓库；控制器、模型、PD、接触门限、双手、物理200Hz/控制50Hz/原生相机25Hz均未调整。SONIC 更新 `resources.json` 后旧 runtime 会被拒绝，使用新目录；Git 源码不包含模型、USD和成功种子等大资源，仍需完整资源交付包。

## 跳过独立 replay 以提速

**独立物理 replay 是额外复验步骤，可以选择不运行，以缩短首次采集时间。** 10月8日 RT2/DLAA、1 update 的完整配对实测为348.37s（5分48秒），其中首次生成＋RGB采集进程186.02s（3分6秒），无相机 replay 进程162.02s（2分42秒）。首次生成的执行/采集本身161.86s，其余为启动收尾等开销。只保留首次采集按该 generate 进程实测约需3分钟；这不是另一次独立 trial 测速，也不包含资源准备、GPT种子搜索、候选编码/几何筛选或GR00T读取检查。

当前使用已有的 `trial` 入口跳过 replay，不需要修改控制源码。在 **sonic-grasp-augmentation** 仓库运行：

```bash
# 只做首次生成和RGB/实际token采集，不启动独立replay。
python launch.py trial --runtime /path/to/new_runtime \
  --candidate /path/to/candidate --output /path/to/new_trial \
  --render-updates 1

# 如需后续补做复验，可使用这次首次采集的实际token。
python launch.py replay --runtime /path/to/new_runtime \
  --first /path/to/new_trial --output /path/to/new_replay
```

`trial` 保留首次采集产物及 `lean_result.json` 的任务成功检查，**不会自动导出 LeRobot 数据集**。`collect`（单条及批量）仍默认逐条执行 replay＋导出，目前没有 `--no-replay` 参数；`--no-export` 仅关闭导出，仍会执行 replay。不要将“约3分钟首次采集”写成已经验证的“约3分钟完整采集＋LeRobot导出”。

跳过 replay 不改变首次执行的物理步、控制频率、接触保护或RGB设置，但缺少“冻结实际token后在新进程独立复现成功”的证据。未运行的 replay 应记为未执行（`not_sampled`），不能标为 `replay_accepted=true`；首次抓取成功、导出可读、视觉验收仍分别记录，`training_ready` 不因跳过复验而改为true。速度数字是同机单次结果，不保证60条批次或其他机器的耗时。

这里可跳过的是 SONIC 的独立物理复验。g1 的保存状态重渲染不推进物理，用于生成RGB，与该 replay 不是同一步骤。

## 参数与作用

| 项目 | 当前值/设置键 | 含义与取舍 |
|---|---|---|
| 渲染器 | `RealTimePathTracing`；`/persistent/rtx/modes/rt2/enabled=true` | RT2，区别于旧 `RayTracedLighting` 与离线 PathTracing |
| 分辨率 | 640×480 | 原生输出及 DLAA；不是低分辨率 DLSS 放大 |
| 抗锯齿 | `/rtx/post/aa/op=4` | 原生 DLAA。DLSS 模式的 `execMode=0` 即使仍在 settings 中，也不能单独据此认定当前使用 DLSS Performance |
| 普通更新次数 | 1；可选2/4 | 每个状态/相机 pose 设置后渲染若干次，再读一次 RGB。其间不执行动作、不推进物理，不输出多帧，也不等于 samples/pixel |
| 首帧/预热 | SONIC 1/2档为4次；4档为8次 | 给初始化、相机同步和时序历史预热；普通帧仍按1/2/4更新 |
| 保存状态入口预热 | g1 保留原20次首次预热＋9个预热状态各4次（共56次） | 此入口的预热流程与在线采集不同；正式输出的每姿态更新由 `--rt2-updates` 指定，预热计时单独记录 |
| 一般反弹 | `/rtx/rtpt/maxBounces=4` | 与透射/反射外观和开销相关，不是总像素采样数 |
| 镜面/透射反弹 | `/rtx/rtpt/maxSpecularAndTransmissionBounces=6` | 给玻璃、瓶身和反射更多路径预算 |
| 透射虚拟运动 | `/rtx/rtpt/translucency/virtualMotion/enabled=true` | D 质量适配参数之一；没有单独因果归因实验 |
| 反射、透射 | `/rtx/reflections/enabled=true`、`/rtx/translucency/enabled=true` | 保留原场景视觉特性，不用删除透明性来掩盖手指缺损 |
| 插值帧 | `/rtx-transient/dlssg/enabled=false` | 没有用生成帧代替真实对应状态 RGB |
| 同步 | app/Replicator `asyncRendering=false`；`waitIdle=true` | 沿用本机已验证读回方式；元数据一致不独立证明零传感器延迟 |
| GPU | GPU0，关闭 multiGPU/autoEnable | 本轮单4090配置；未进行多GPU吞吐测试 |
| 曝光 | `/rtx/post/tonemap/exposureKey=0.6` | 保留原曝光设置；实际值以输出设置记录为准 |
| PT128（独立配置） | 32采样/更新、128上限、OptiX降噪、反弹4/6 | g1 PT路径继续验证每姿态累积稳定，不能只看 `totalSpp` 字段 |

这里没有选择 Isaac Lab `performance` 预设，也没有启用 DLSS 降分辨率 Performance。无窗口、减少普通更新和精简数据管线属于另外的加速措施。处于 RT2 时，日志里的 `/rtx/pathtracing/spp` 等离线 PT 参数不代表 RT2 的有效采样预算。

## 9月21日 A/B/C/D：保存状态渲染对照

[历史四路视频](../media/rendering_comparisons/abcd_20260921.mp4)：A/B在上，C/D在下；1280×1024，H.264，25FPS/250帧/10s。沿用原文件，无重新编码，SHA与尺寸见 [manifest.json](../media/rendering_comparisons/manifest.json)。

同一 walk_03_far 轨迹，控制索引600..1098、步长2，原生640×480/25FPS。只恢复保存状态，不推进物理。正式输出计时含恢复、渲染、读回、HDF5/MP4写入和截图，不含启动、预热、退出和后期拼接。

| 方案 | 配置 | 当时正式输出 | 代码与证据状态 |
|---|---|---:|---|
| A | 优化离线PT128，反弹4/6，OptiX | 277.758s | 原优化代码和实验目录已缺失。视频/历史计时保留；同类PT源码存在但不冒充原A |
| B | RT2，默认DLSS Performance，1 update | 6.077s | 原 render_route.py 保留 |
| C | 同B，4 updates | 9.118s | 原共享入口通过参数选择 |
| D | RT2原生DLAA，反弹4/6，virtualMotion，4 updates | 9.707s | 原共享入口保留；是本项目质量适配，不是Genie官方预设 |

历史B/C瓶身发黑；D更接近保留PT图像。原报告相对A的全图PSNR：B25.84/C27.62/D31.32dB；手瓶ROI：20.81/21.70/29.41dB。D全图SSIM0.9445低于C0.9544；不能只看PSNR宣称全面更好。参考PT也不是现实世界真值，时间差分指标不是纯闪烁指标。

10月8日复跑原D，正式输出10.733s，但相同历史保存场景的白标签发黑，没有完全复现旧图像；保留设置、输入根场景和原始轨迹摘要一致，原因未确定。不能把9月报告的质量或速度结论直接套到当前环境，更不能把277.758s与最新执行时间作严格加速比。

## 10月8日最新管线：D4 / D2 / D1

RTX4090、IsaacSim5.1.0、驱动580.178.04，同初态seed17及1071个实际冻结SONIC token，21.42仿真秒；相机原生25Hz、存储50Hz因果保持，没有插值。每种配置n=1，顺序运行，未随机顺序重复测速。

[左D4/右D2](../media/rendering_comparisons/d4_vs_d2_20261008.mp4) · [左D2/右D1](../media/rendering_comparisons/d2_vs_d1_20261008.mp4)：均1280×512、H.264、50FPS/1071帧/21.42s。

| 区间 | 旧realtime/DLAA，2次 | D4 | D2 | D1 |
|---|---:|---:|---:|---:|
| 相机初始化＋渲染/读回＋视频提交 | 25.526s | 46.514s | 24.394s | 13.465s |
| 物理/接触/观测/相机执行区间 | 173.546s | 198.869s | 176.001s | 158.031s |
| 整个子进程 | 223.789s | 216.122s | 193.222s | 175.150s |
| 实际render调用 | 1074 | 2148 | 1074 | 539 |

D1相较D2相机区间少44.8%，执行区间少10.2%；相较D4执行少约20.5%。物理、接触检查占主要时间，不能把渲染更新减半理解成整条管线减半。执行差值还包含其他区间运行波动，不能全部归因于少一次render。旧基线进程启动开销较大，不应因D4总进程短而说D4渲染更快。

各组1072份初始＋积分后/自动reset前状态、decoder输入输出逐值一致，task_success=true，严格持瓶4.40s。旧基线抽查有手指/瓶盖/标签缺损或不自然透明，D4/D2/D1抽查保留完整外观；D1篮子细纹及反射存在小差异，部分细节偏软。

D1图像时间诊断：532个有效原生帧中527个最接近同索引D2；148个明显运动帧全部最接近同索引。没有一致的整帧滞后模式。这是包含累积/编码影响的图像相似度诊断，不能排除亚帧延迟或两组共享延迟。

上述是冻结token、decoder-only物理执行加诊断相机，不包括在线encoder/planner/GPT、候选准备或最终LeRobot导出。不是完整数据增广配对的“158秒总耗时”，也不能直接推算60条批次吞吐。原六分钟记录的计时范围、完整可移植运行耗时见主README。

## 能力边界与验收

- 当前默认是基于这一场景的速度候选；不保证任意透明物体、快速运动、不同驱动/材质都适用1次更新。需要更稳定历史可显式选2/4。
- 抽查图像与连续截图不证明无闪烁。未完成原生帧率感知播放、双手全部遮挡区域、外视角、深度/分割或最终next_observation RGB验收，保持 `accepted=false`、`training_ready=false`。
- 任务成功、控制/状态对齐、完整视频解码、视觉接受是不同检查。不能用更好看的重渲染冒充新的物理抓取成功。
- g1 HDF5中的原轨迹外链不是自包含归档；移动输出时连同源数据处理，或按数据契约导出自包含数据。
- 视频是观看证据，H.264有损；历史视频包含原始问题，不是训练样本。仅这三份获用户授权的对比样例纳入Git，大模型/场景/原始数据仍留在资源包。
- Git直接保存这些小于50MiB的MP4，不要求Git LFS；资源SHA与源码版本绑定。克隆Git仍不等于下载完整仿真资产。

## 维护与复现依据

`render_settings.py` 为两个入口共用的最小参数实现。更新参数时同步两份；SONIC修改payload后重算resources摘要并在新runtime验证。输出 `renderer_settings.json` 或 g1 `settings.json` 为实际读回值，`protocol.json` / `complete.json` / `launcher_timing.json` 分别记录输入、执行范围与过程计时。

文档使用已有实验记录与新入口验证，不重写控制器。样例原来源、文件字节、SHA256及ffprobe在media目录manifest中；本轮仓库入口实际验证见 [RENDERING_VALIDATION.md](RENDERING_VALIDATION.md)。
