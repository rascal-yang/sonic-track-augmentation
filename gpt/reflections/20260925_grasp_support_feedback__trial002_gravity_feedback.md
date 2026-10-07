# trial002_gravity_feedback

策略：withdraw_rotate_descend_side_pinch_gravity_feedback

观察：重力偏置补偿后末段腰部实际0.195rad、目标0.231rad，明显没有冲到0.52rad；但手掌误差8.1cm，根部水平漂移12.05cm，触发预设停止条件，尚未进入闭合。

不确定性：计划推进速度与目标限速造成的跟踪滞后、全身支撑反应尚未分别验证；不能仅凭腰部改善认定抓握可行。

下一步：保留相同控制补偿，改为达到当前位姿容差后才推进下一小段；延长允许停留时间，验证先站稳再接近。

证据：/home/kinetra-4090-01/code/zijian/experiments/20260925_grasp_support_feedback/runs/trial002_gravity_feedback/audit.json
