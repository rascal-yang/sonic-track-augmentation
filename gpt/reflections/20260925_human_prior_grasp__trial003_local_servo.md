# trial003_local_servo

策略：withdraw_rotate_descend_side_pinch_local_servo

观察：局部伺服的物理子步目标限速为0.004rad；抓握窗口后根部水平位移32.44cm，340帧腰部实测0.520rad而下发0.231rad；拇指接触为0，未抓起，无跌倒。

不确定性：局部驱动与全身支撑相互影响的机制尚未隔离；不能把本次失败推广为侧夹不可行。

下一步：先以实际根部/足部状态验证稳定支撑和腰部执行能力，再用物体坐标目标闭环更新手臂；达到位置与接触容差后才闭合。

证据：/home/kinetra-4090-01/code/zijian/experiments/20260925_human_prior_grasp/runs/trial003_local_servo/audit.json
