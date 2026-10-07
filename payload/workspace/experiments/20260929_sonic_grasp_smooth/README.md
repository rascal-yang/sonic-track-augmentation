# 25mm抓取：快速版与保守版

本轮仅保留两个成功版本，其余候选、视频、回放记录和旧对照图已按用户要求删除。

- `safe025` / `runs/safe025_001`：快速版，11.72秒，持瓶4.68秒。
- `smooth_final025` / `runs/smooth_final025_001`：保守版，15.42秒，持瓶4.58秒。

[结果、数据与视频](RESULTS.md)。两版各有7个右臂关键参考姿态；参考、编码token和实际执行state/PD target分别保存。

使用 `/home/kinetra-4090-01/miniconda3/envs/sonic/bin/python`：

```bash
python experiments/20260929_sonic_grasp_smooth/run.py --name NEW_RUN --candidate safe025
python experiments/20260929_sonic_grasp_smooth/supervise.py experiments/20260929_sonic_grasp_smooth/runs/NEW_RUN
```

关键阶段需检查实际图像和接触后写review marker。GPU任务串行运行。保守版将candidate换为smooth_final025。模型和场景依赖原工作区，导出包不包含模型或资产。

重新生成参考使用上一轮未改动的 `experiments/20260929_sonic_palm_wrap/candidates/left025`：快速版 `prepare.py --name NEW --safe-arc`；保守版 `prepare.py --name NEW --safe-arc --flow --raise-extra 1.1`，生成后执行 `check_geometry.py NEW`。保留原运行源代码快照供溯源；快照中的历史路径不代表对应数据仍然保留。

本轮历史试次已清理，无法再重做其原数据对照。两版均未经过扰动鲁棒性或真机验证。视频是RTX realtime/DLAA预览，不是RT2，training_ready=false。
