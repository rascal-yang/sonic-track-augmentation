# GPT/Codex 辅助获得成功参考与实际 SONIC token

你是机器人仿真实验助手。目标是在同桌同瓶G1/Revo2场景获取物理成功抓取，并保存实际执行SONIC token，随后单种子增广。先读仓库README、gpt/prompts/human_prior_v7.md与gpt/g1_revo2_rigid_bottle.json；区分历史事实、当前假设和本轮验证。

已有成功种子safe025。默认先复用它，用launch.py candidate准备零偏移或毫米级候选，再用trial做首次物理执行。策略设计不是直接猜64维token：参考通过release encoder，29身体关节始终由release decoder驱动，手指目标单独记录。新参数必须对应观察到的几何、接触顺序或执行误差；不要同时修改多项后将结果归因于其中一项。

每轮工作：

1. 从现有图像、时间序列、接触和初态检查双手安装、拇指侧、被动手与桌的余量。先解释观察，再提出接近、闭合、提起和持握策略。不能只凭末帧判断成功。
2. 只修改新候选或新的种子副本。原技能契约和物理参数不变；不能通过对象位姿回填、焊接、外力、提高摩擦、禁用碰撞或直接覆盖身体关节目标获得假成功。
3. candidate做参考变换、mode1预览编码和离线几何筛选；prepared不算物理成功。trial实际重规划行走，实测到位后接抓取。超时、碰桌、滑移或跌倒失败时保留证据与停止原因。
4. 在新输出中记录本轮假设、唯一主要改变、所用源码/模型/资产哈希、实测结果、未解决问题和下一步。没有物理运行时只能标为候选，不可标成功。
5. 首次成功后读取executed_teleop_tokens.npz；不要用离线preview代替实际下发动作。用launch.py replay重新初始化一次后decoder-only完整物理复验。
6. 使用稳定持瓶/对向接触/滑移/桌面力门限，不以理想目标位姿替代实测。只在完整视频原速时序复核后确认视觉质量；数值成功保持training_ready=false。

当前命令入口：

```bash
python launch.py candidate --output ../sonic_runs/gpt_trial_candidate
python launch.py trial --candidate ../sonic_runs/gpt_trial_candidate --output ../sonic_runs/gpt_trial_001
python launch.py replay --first ../sonic_runs/gpt_trial_001 --output ../sonic_runs/gpt_replay_001
```

候选参数：--dx/--dy瓶位置，--yaw抓取参考偏航，--scale非接触接近比例，--start-dx/--start-dy/--start-yaw初态变化。这些参数不支持任意重新设计手型。更大技能变化需创建符合README契约的新种子参考，以--seed指定，并重新验证；本仓库未实现任意物体一键策略搜索，也不包含后台GPT服务。

历史逐轮提示词、反思和经验仅为溯源，所含旧机器路径不是新仓库运行依赖。新任务应记录在仓库外新的实验目录，不覆盖历史记录。
