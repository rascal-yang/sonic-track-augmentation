# 新渲染入口验证记录（2026-10-08）

本轮将 D 的 RT2＋原生 DLAA 配置同步到两个仓库，默认普通帧1次更新，可显式选择2/4。复用原状态恢复、相机、控制器、物理、接触检查和导出实现。参数与历史对照见 [RENDERING.md](RENDERING.md)，本轮数值证据见 [rendering_sync_20261008.json](evidence/rendering_sync_20261008.json)。所有测速均单次运行，不承诺其他机器或60条批次达到相同速度。

## 环境与范围

RTX4090，驱动580.178.04，IsaacSim5.1.0、IsaacLab0.54.2，Python3.11.15、torch2.7.0+cu128。单GPU、同步读回，AA设置4（DLAA），反弹4/6，没有使用 Isaac Lab performance 预设或 DLSS Performance。

实验根目录为 `experiments/20261008_render_repository_sync`。Git保留本页和机器可读摘要；完整HDF5、原始日志及新采集视频留在本地实验目录，没有额外放入Git。三份历史/updates比较视频单独在media目录保留，逐字节复制、完整解码检查通过，来源及摘要见 [manifest.json](../media/rendering_comparisons/manifest.json)。

## g1-realtosim：恢复保存状态，不推进物理

最终运行 `g1_rt2_1update_final` 取 walk_03_far 的控制索引600..1098、步长2，250帧/25FPS/10s；正式状态各1次更新。原预热仍为20＋9×4＝56次，与普通帧参数独立。

| 区间 | 实测 |
|---|---:|
| Kit启动 | 9.130s |
| 预热 | 0.646s |
| 正式输出，包含恢复/渲染/读回/HDF5与MP4写入 | 5.397s |
| 其中姿态恢复 | 1.376s |
| 其中渲染 | 1.309s |
| 其中RGB读回 | 0.363s |
| 其中输出写入 | 1.238s |

这些组件不是整个进程的完整互斥分桶；退出另计。实际设置确认RT2、DLAA、反弹4/6、GPU0、multiGPU=false及waitIdle=true。HDF5 RGB形状250×480×640×3，控制索引逐值检查通过；首状态所有声明刚体恢复误差小于1e-6。MP4为H.264、640×480，250帧，完整解码通过。原状态恢复校验不等同于每帧RGB零传感器延迟证明。

画质抽查0/25/50/75/100/150/200/249帧：可见手指保留，但历史场景的白色瓶身标签发黑，未复现9月D视频外观；原因仍未确定。隔离分支基线还显示视口网格；原工作目录已有的关闭网格修改将在合入时保留，它没有进入本轮新提交，也没有拿它覆盖此次验证结果。`visual_realism_accepted=false`，不能认定这份历史场景输出已适合训练。

## sonic-grasp-augmentation：完整物理配对与导出

实际通过 `launch.py collect --render-updates 1` 生成新的 `sonic_pair`：首次执行1071个控制步、21.42仿真秒、4284个接触样本；随后使用实际冻结token独立无RGB replay，最后导出 `lerobot_all_train`。

| 区间 | 执行/采集 | 含启动退出的进程 |
|---|---:|---:|
| 首次生成，有RGB | 161.862s | 186.017s |
| 独立replay，无RGB | 140.807s | 162.016s |
| 配对总计，含导出 | — | **348.369s，5分48秒** |

总计不含资源准备、candidate编码/几何筛选、GPT获取种子或GR00T读取检查。单次结果比之前迁移时的8分8秒短，但跨运行的GPU竞争、启动和执行差异不能全部归因于这次render设置修改。固定token D4/D2/D1的可比区间另见渲染文档。

两次均 `task_success=true`、`complete_sequence=true`，严格最终持瓶4.44s，离桌0.124m，末3秒对向接触比例1.0，滑移0.203mm，旋转0.070度；本次只做终段指标及在线聚合门限检查，没有宣称新生成与replay的全轨迹逐值等价。原相机25Hz，存储视频50FPS因果保持，不采用插帧。生成的H.264视频640×480/1071帧/21.42s，完整解码通过；replay没有head.mp4，符合原无相机流程。通过仓库内官方GR00T loader读取1个导出episode。

抽查0/200/400/600/750/850/950/1070帧：标签为白色，瓶盖和可见手指不透明、完整。仍未完成原生帧率感知播放、所有遮挡区域、外视角、深度/分割及最终next_observation RGB验收，保持 `training_ready=false`。成功抓取及loader通过不能代替视觉训练验收。

## 资源、测试及改动范围

最终资源清单含2376个payload文件，修改了两份lean入口并增加参数helper，其余资源沿用原始源码/数据。完整物理配对使用v1准备目录；评审后添加的 `offline_update_count` 仅供g1离线预热使用，SONIC不调用。AST比对证明最终v2除这个新增函数外的全部helper代码与实测v1一致，最终v2重新prepare、doctor及GR00T check-export通过。未把旧runtime冒充最终资源版本。

本机ONNX运行库通过已有 `--ort-path` 提供，av/cv2经已有导出依赖路径提供，均沿用README依赖说明；没有把依赖缺失当作仿真失败，也没有下载替换控制器。两个仓库的render_settings.py逐字节相同。g1单元测试9项通过；SONIC14项通过，包含预热次数独立和批量续跑拒绝混用更新次数的检查。

评审修正了降低普通更新时意外减少g1预热的问题，保留原56次预热；批量collection_request记录render_updates，改参数不能继续旧批次。PT128累积、物理频率、PD、接触门限及双手安装未修改。g1已有动画/数据导出等工作目录改动单独保留，没有收进这次提交。

用户要求先不推送。本轮只更新本地代码、文档与Git提交，SONIC完整资源交付包另行生成；Git克隆仍需该资源包及Isaac运行环境。
