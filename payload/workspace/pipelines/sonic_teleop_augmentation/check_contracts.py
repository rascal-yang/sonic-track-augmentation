"""Regression check for the actual handoff interface and physically gated transitions."""
import ast
import numpy as np
from common import W,read
from mobile_supervisor import Supervisor
runtime=W/'pipelines/sonic_mimicgen_augmentation/runtime_adapter.py'
choices=[]
for node in ast.walk(ast.parse(runtime.read_text())):
    if isinstance(node,ast.Assign) and isinstance(node.value,ast.List):
        for target in node.targets:
            if isinstance(target,ast.Subscript) and isinstance(target.slice,ast.Constant) and target.slice.value=='available_actions':
                choices.append(set(ast.literal_eval(node.value)))
allowed=next(x for x in choices if {'handoff','start_grasp'}<=x)
plan=read(W/'experiments/20260929_sonic_mimicgen_augmentation/smoke_runs_v2/candidate_000100/plan.json')
s=dict(step=275,phase='braking',navigation_error_m=.02,robot_root_pos=plan['goal_root_m'],
       root_linear_velocity_m_s=[0,0,0],feet={'left':{'linear_velocity_m_s':[0,0,0]},'right':{'linear_velocity_m_s':[0,0,0]}},
       object_quat_wxyz=[1,0,0,0],object_pos=plan['object_initial_position_m'],observation_id=21,
       grasp_information={'metrics':dict(opposed_digits_contact=False,whole_object_clearance_m=0,object_table_normal_force_N=1)})
supervisor=Supervisor(plan);supervisor.phase='braking';supervisor.phase_start=210
assert supervisor.decide(s)[0]['action']=='handoff'
assert supervisor.decide(s)[0]['action'] in allowed
s['root_linear_velocity_m_s']=[.2,0,0]
assert supervisor.decide(s)[0]['action']=='continue'
s['root_linear_velocity_m_s']=[0,0,0];s['phase']='settled';s['step']=500
s.update(stance_joint_rmse_rad=.1,stance_yaw_error_rad=.01)
supervisor.phase='settled';supervisor.phase_start=450
assert supervisor.decide(s)[0]['action']=='start_grasp'
s['contact_watch']={'alert':True}
assert supervisor.decide(s)[0]['action']=='stop'
print('Mobile runtime accepts gated handoff/grasp transitions; moving-root and contact cases reject entry.')
