# 2026-10-08 迁移与执行验证

复用本机既有 Isaac Sim 环境，将仓库资源物化到 `/tmp/sonic_grasp_portable_20261008_v2`，不从原实验目录加载控制源码。首次生成、独立物理 replay、LeRobot 导出和包内真实 GR00T loader 检查均成功。没有在另一台干净机器安装、训练模型或重跑60条物理采集。

环境：Python3.11.15、Isaac Sim5.1.0、IsaacLab0.54.2、torch2.7.0+cu128、RTX4090、驱动580.178.04。现有环境的 ORT 与 av 分别通过本机 `.deps` 和 `.export_deps` 提供；这两个目录只提供第三方 Python 包。新机器由 `setup_environment.sh` 安装这些包。安装脚本的干净环境安装尚未验证。安装文件固定 OpenCV4.11 以兼容 numpy1.26；本机现有 OpenCV4.13 导入仍可用，这不是干净安装的推荐组合。

## 已执行检查

|检查|结果|
|---|---|
|8项迁移回归测试|全部通过：路径带空格、二进制复制、资源损坏、无关旧目录、上游robot路径限制、视频工具缺失、runtime篡改、默认种子不受继承环境影响、批次续跑来源绑定；其中有些检查合并在同一测试中|
|源码与资源|346个Python文件AST解析通过；2375项资源SHA256一致；setup脚本bash语法通过|
|新runtime依赖检查|`v3` 中 doctor 返回0，所需Python包、ffmpeg/ffprobe和GPU均可用；doctor本身不证明物理成功|
|成功种子重新编码|`seed-reference` 实际 mode1 encoder 执行成功|
|离线增广|新位置dx=5mm单候选、2条批次及完整60条候选准备完成；60份清单/840项文件哈希一致，各候选table_screen_passed=true；prepared不等于物理成功|
|完整首次生成|1071控制步、21.42s仿真时长、4284接触采样、task_success=true|
|独立replay|新进程完整1071步、task_success=true，未录制视频|
|数据导出/读取|LeRobot v2.1、1071帧；真实GR00T episode loader读取成功，RGB480×640×3、64维实际token、双手各6维目标逐项与执行标签一致|
|批次续跑/统一导出|同一候选复用本次已验证成功的pair，批次completed=1/requested=1并完成统一导出；未再做一次物理采集|
|离线包搬迁|完整预检包SHA256通过；解压到另一新目录，重新物化并验证2375项资源，包内8项测试、doctor、真实encoder与几何筛选再次通过|
|依赖安装预检查|现有仿真环境中pip install --dry-run解析requirements成功；只解析/下载，无安装动作，不代表干净机器验证|
|代码复审|仅审查迁移/启动层；3项Important已修复复审通过，无新的阻塞|

## 物理与时间结果

首次与 replay：末端连续严格持瓶4.46s、末3s对向接触比例1.0、相对滑移0.673mm、旋转0.308°、提起净空12.69cm、最低骨盆高度75.34cm、监测桌面力峰值0N。

首次执行/采集272.806s，含进程启动/退出298.036s；replay执行166.829s，含进程启动/退出190.023s；配对包含导出488.434s。资源准备和GR00T读取检查另计。生成前段观测到另一项独立Isaac实验使用同一GPU；本测量不能保证独占资源下或其他机器每条的固定用时。

成功参考mode1重新编码进程1.142s；2条候选准备进程30.899s。60条候选准备约717.957s，按schedule/status文件时戳估计；工具服务重启中断了启动器的完整进程计时，准备子进程完成输出并已逐文件复核，未记录完整进程退出码。见 [60条准备证据](../evidence/batch_preparation_20261008.json)。

机器可读结果见 [consolidated_smoke_20261008.json](../evidence/consolidated_smoke_20261008.json)。原始输出保留在本机 `experiments/20261008_consolidated_pipeline/portable_pair_001`；不把这次训练视频加入交付包。

`training_ready=false`：完整视觉时序验收pending。独立replay验收的是完整物理执行与末端保持、运行保护汇总，未证明逐帧轨迹相等或现实世界可靠性。

保留的第三方源码中有上游空白格式告警和一条无效转义SyntaxWarning；新增入口与本次适配文件的diff whitespace检查通过。为保持源码原样，没有做批量格式化。
