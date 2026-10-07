# trial004_support_reference

策略：withdraw_rotate_descend_side_pinch_measured_support_reference

观察：对齐实际支撑参考后，完整17.4s没有触发12cm漂移阈值，但阶段停在175；最后手掌误差3.33cm、IK残差3.36cm，未进入闭合，没有抓起。

不确定性：重新锚定的支撑姿态改变了手部可达域；当前初始接触点仍来自交接前，严格阶段门限是否造成阻塞需要验证。

下一步：交接完成后用实测手部位姿重新建立退开路径起点，保持最终物体抓握目标和其余控制参数。

证据：/home/kinetra-4090-01/code/zijian/experiments/20260925_grasp_support_feedback/runs/trial004_support_reference/audit.json
