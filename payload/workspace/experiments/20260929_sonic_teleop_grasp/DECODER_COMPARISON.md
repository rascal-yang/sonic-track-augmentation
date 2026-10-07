# SONIC decoder 执行后的偏差统计

统计已有两次成功抓取的586帧物理执行记录，对齐同一参考和同一时刻。之前的路点误差同样是经过decoder实际执行后的误差，不是编码前IK的求解残差。

|指标|IK参考 → G1 encoder → decoder|Cartesian参考 → Teleop encoder → decoder|
|---|---:|---:|
|右腕位置平均误差（cm）|4.05|2.52|
|右腕位置 RMSE（cm）|4.13|2.68|
|右腕位置 P95（cm）|4.83|4.33|
|右腕位置最大误差（cm）|5.15|4.68|
|右腕朝向平均误差（°）|14.87|13.49|
|右腕朝向最大误差（°）|28.66|28.29|
|骨盆位置平均误差（cm）|2.48|1.78|
|身体29关节 RMSE（°）|5.57|12.46|
|右臂7关节 RMSE（°）|9.49|17.84|

误差包含编码、控制器和物理跟踪的共同影响，不能单独归因于decoder。Teleop没有要求复现上半身关节角，因此关节差异也包含不同手臂姿态，不等同于任务失败。原版使用离线token，本次Teleop使用实时根部朝向编码；不是只改变单个网络的消融。

本次未新增独立固定token物理回放，不能据此宣称再次replay的复现误差为零。详细数值：[decoder_rollout_comparison.json](decoder_rollout_comparison.json)。

后续验证：已另做两组固定token的独立物理回放，[结果见此](../20260930_sonic_grasp_token_replay/RESULTS.md)。
