# trial002_side_approach

策略：withdraw_rotate_descend_side_pinch

观察：9.4秒无摔倒但拇指仍无接触；下降阶段实测掌心比参考高约5cm，腰部参考没有被当前执行器跟踪到。

不确定性：终态几何误差小并未证明实际控制下可达；完整参考在未执行的470帧存在重复叠加局部修正的拼接错误，不能认可其接回。

下一步：修正未执行拼接点，并单独测试抓握窗口的限速局部关节辅助，以区分规划与跟踪问题。

证据：/home/kinetra-4090-01/code/zijian/experiments/20260925_human_prior_grasp/runs/trial002_side_approach/audit.json
