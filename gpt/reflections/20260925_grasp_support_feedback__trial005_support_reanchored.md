# trial005_support_reanchored

策略：withdraw_rotate_descend_side_pinch_support_reanchored

观察：交接后重建起点消除了初始位姿误差，阶段从175推进到了247；仍在根部漂移12.03cm时停止，末段手掌误差3.43cm、IK残差2.60cm，没有达到预抓握或闭合。

不确定性：到达中间阶段不代表最终窄面夹持可行；支撑参考、局部上身控制与身体实际位移仍未协调。

下一步：停止继续微调手指和局部时间参数；联合求解脚部约束、骨盆/膝踝、重心与手部目标的全身预抓握，再做跟踪可行性验证。

证据：/home/kinetra-4090-01/code/zijian/experiments/20260925_grasp_support_feedback/runs/trial005_support_reanchored/audit.json
