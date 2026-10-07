# 抓握先验与经验库

用户于 2026-09-25 授权维护提示词和抓握 skill，并在每次仿真尝试后反思改进。

- 当前提示词：[human_prior_v7.md](prompts/human_prior_v7.md)；[v6](prompts/human_prior_v6.md)；[v5](prompts/human_prior_v5.md)；[v4](prompts/human_prior_v4.md)；[v3](prompts/human_prior_v3.md)；[v2](prompts/human_prior_v2.md)；[v1](prompts/human_prior_v1.md) 保留。
- 自动发现 skill：`/home/kinetra-4090-01/.codex/skills/grasp-prior-refinement/SKILL.md`。
- [经验索引](lessons.json)：可更新摘要，保留状态与原始证据。
- `attempts.jsonl`：逐次追加记录；`reflections/`：详细反思。
- `revisions.jsonl`：提示词/skill 变更原因和证据。

修改提示词时新增 v2、v3 等文件，更新本入口，不覆盖既有版本。经验分 hypothesis、observed、replicated、rejected、superseded，仅重复物理验证后可升级为 replicated。当前没有经过验证的箱子抓起方案。

新任务按物体几何、手模型/安装、支撑关系、控制架构检索记录。瓶子任务还需读取 `../g1_revo2_rigid_bottle.json`。自动反思发生在实际任务会话中，不表示后台常驻运行或模型权重自更新。

## 最新实验

最新结果：[实测场景成功抓瓶与简化](../../experiments/20260925_measured_walk_grasp/RESULTS.md)。walk_009和walk_010_arc成功；简化为一个连续接近目标，稳定持瓶6.10秒，整瓶离桌14.98 cm。总时间增加1.70秒，不宣称路径或速度优化。RT2成品已保存。下文未验证候选属于历史记录，已由此次结果更新。

[2026-09-25 人类先验抓握迭代](../../experiments/20260925_human_prior_grasp/README.md)：三次物理尝试，均未抓起；区别记录手指策略、参考轨迹修改和局部驱动修改。

[支撑与反馈迭代](../../experiments/20260925_grasp_support_feedback/README.md)：5次有效物理尝试，另1次启动中止；仍无成功抓起。当前优先验证无接触预抓握的全身执行能力，再接入抓握阶段。

[右臂预测控制诊断](../../experiments/20260925_mpc_pregrasp/README.md)：5组无接触物理对照；发现位置达标时可能累计转腕。完整位姿12步版本单次边缘通过保持门槛，位置余量不足、残差频繁饱和，无抓握成功。

[箱子恢复到交互场景](../../experiments/20260925_mpc_box_in_scene/README.md)：用户指出前轮箱子在房间外后撤掉1.5m平移，两轮重跑真实接触。MPC推移箱子31.52mm后停止，仍无抓握成功。

[实测场景走近抓瓶失败分析](../../experiments/20260925_measured_walk_grasp/FAILURE_ANALYSIS.md)：走近与站立交接完成，闭指时瓶子倾倒；新增速度门限的完整效果尚未物理验证。下一轮先验证软件收尾，再做实际手腕稳定后的闭指对照。

用户于2026-09-25明确授权自主记录失败原因并用于下一次优化。后续每次尝试结束后，先追加事实、假设、证据和下一轮单项变化，再继续选择实验；保留失败，不把未验证修复写成成功。

抓瓶下一轮用户指定调整：让手心更接近瓶子。已准备[向瓶轴水平推进1 cm的候选配置](../../experiments/20260925_measured_walk_grasp/next_trial_config.json)，保持高度/朝向并保留实际稳定门限；尚未物理验证。执行前读取[更新后的分析](../../experiments/20260925_measured_walk_grasp/FAILURE_ANALYSIS.md)。

[办公室固定双Panda搬箱闭环](../../experiments/20260925_locomanip_baseline/REPORT.md)：1条最终种子与3条扩增成功；不是G1平衡或持物移动验证。

迁移后验证更新：[轨迹独立rollout失败](../../experiments/20260925_measured_walk_grasp/ROLLOUT_CHECK.md)。保存的实际状态作为SONIC参考并未复现走近，抓取前站位误差46.4 cm而停止；不能将原闭环成功标为轨迹回放成功。

[2026-09-26 G1篮子完整抓取、搬移、放置](../../experiments/20260926_g1_basket_place/REPORT.md)：3次种子尝试中1次完整成功，随后小范围独立初始/目标扩增2/2成功。固定站位右手、位置目标且偏航自由；双手、持物行走和视频入口仍未验证。失败均保留。

[2026-09-29 25mm SONIC抓取简化与调时](../../experiments/20260929_sonic_grasp_smooth/RESULTS.md)：7个右臂参考姿态；快速版11.72s完成抓取，最终保守版15.42s。全段加速度仍未低于原版，保留指标和失败假设；未做扰动鲁棒性或真机验证。

[2026-09-29 大步行走接快速抓取](../../experiments/20260929_sonic_walk_fast_grasp/RESULTS.md)：在线planner→encoder→decoder实际前进约81.8cm，接原586帧快速抓取，21.42s完成、4.72s连续悬空、0监测桌面接触。单次校准成功；固定token独立回放与扰动验证未做。

[2026-09-29 SONIC自动增广](../../experiments/20260929_sonic_mimicgen_augmentation/RESULTS.md)：3次物理测试，1次小扰动成功、2次复杂变换失败并自动拒收；默认采用小扰动，复杂变换尚未验证可靠性。

[2026-09-30 SONIC Teleop encoder 抓取](../../experiments/20260929_sonic_teleop_grasp/RESULTS.md)：单次抓取阶段成功，无新增上半身 IK；连续持瓶4.42秒，整瓶离桌12.85cm，0监测碰桌。尚未验证任意路点或扰动泛化。

[2026-09-30 G1/Teleop 抓取 token 独立回放](../../experiments/20260930_sonic_grasp_token_replay/RESULTS.md)：各1次独立decoder回放，对比各自首次执行，完整数值审计见报告；不代表真实硬件或扰动可靠性。

[2026-10-02 观察优先的GPT-6抓握讨论与实验](../../experiments/20261002_g1_observation_first/RESULTS.txt)：7次API调用，3次物理运行；确认逐指覆盖与接触顺序/路径余量问题，新四指获取未成功。提示词更新为v7。
