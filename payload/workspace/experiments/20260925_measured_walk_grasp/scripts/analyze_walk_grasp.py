"""Audit the executed walk/handoff/grasp phases from physical telemetry."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
rows=[json.loads(s) for s in (r/'trajectory.jsonl').read_text().splitlines()]
calls=[json.loads(s) for s in (r/'planner_calls.jsonl').read_text().splitlines()]
grail=[json.loads(s) for s in (r/'grail_execution.jsonl').read_text().splitlines()]
phases={}
for name in dict.fromkeys(x['phase'] for x in rows):
 xs=[x for x in rows if x['phase']==name]
 phases[name]={'first_time_s':xs[0]['sim_time'],'last_time_s':xs[-1]['sim_time'],'control_steps':len(xs),'initial_root_m':xs[0]['robot_root_pos'],'final_root_m':xs[-1]['robot_root_pos'],'end_distance_to_stand_m':xs[-1]['navigation_error_m'],'end_root_velocity_m_s':xs[-1]['root_linear_velocity_m_s']}
feet={}
walking=[x for x in rows if x['phase'] in ('walking','braking')]
for side in ['left','right']:
 pts=np.array([x['feet'][side]['position_m'] for x in walking]);high=pts[:,2]>.05
 feet[side]={'xy_net_displacement_m':float(np.linalg.norm(pts[-1,:2]-pts[0,:2])),'ankle_max_z_m':float(pts[:,2].max()),'raised_ankle_intervals':int(np.sum(high & ~np.r_[False,high[:-1]])),'note':'Ankle-height threshold is a foot-swing proxy, not a force-sensor contact count.'}
handoff=json.loads((r/'handoff.json').read_text());start=handoff['reference_start_frame']-2
assert all(x['frame']<start for x in calls),'ONNX unexpectedly replanned after handoff'
assert all(x['maximum_absolute_scaled_residual']==0 for x in grail if x['phase'] in ('initial_stand','walking','braking'))
assert all(abs(x['phase_gain']-1)<1e-7 for x in grail if x['phase']=='grasp')
assert any(x['maximum_absolute_scaled_residual']>1e-6 for x in grail if x['phase']=='grasp')
out={'phases':phases,'feet_walk_proxy':feet,'ONNX_calls':len(calls),'ONNX_walk_calls':sum(x['mode']==1 for x in calls),'ONNX_stand_calls':sum(x['mode']==0 for x in calls),'no_ONNX_calls_after_handoff':True,'walking_GRAIL_residual_exactly_zero':True,'grasp_GRAIL_gain_one_and_nonzero_residual':True,'handoff_reference_duration_s':handoff['duration_steps']/50,'root_xy_initial_to_final_displacement_m':float(np.linalg.norm(np.array(rows[-1]['robot_root_pos'])[:2]-np.array(rows[0]['robot_root_pos'])[:2])),'minimum_pelvis_z_m':min(x['robot_root_pos'][2] for x in rows),'external_GPT6_API':False,'agent':'Current GPT/Codex session at paused checkpoints; commands and observations preserved.'}
(r/'walk_grasp_analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
