# trial006_coupled_wholebody

策略：withdraw_rotate_descend_side_pinch_coupled_wholebody

观察：联合全身几何求解使手掌/脚部/模型重心残差接近零，但物理执行在阶段172就因根部漂移12.06cm停止；手掌实测误差11.44cm，箱子倾倒并平移约20cm，没有闭合和抓起。

不确定性：几何可行性未覆盖力矩、接触力、实际支撑和SONIC跟踪能力；不能把本次失败当作全身方法不可行，也不能将几何解当成可执行轨迹。

下一步：先在无物体接触的预抓握保持任务上识别并验证全身控制与局部驱动的兼容性；物体发生明显位姿变化时应终止旧抓握方案并重规划。

证据：/home/kinetra-4090-01/code/zijian/experiments/20260925_grasp_support_feedback/runs/trial006_coupled_wholebody/audit.json
