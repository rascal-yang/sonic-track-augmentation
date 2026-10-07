import json,sys
from pathlib import Path
r=Path(sys.argv[1]);rows=[json.loads(x) for x in (r/'grail_execution.jsonl').read_text().splitlines()]
assert rows and all(x['maximum_absolute_scaled_residual']==0 for x in rows)
assert not (r/'trajectory_hint.json').exists()
assert not (r/'planner_calls.jsonl').exists()
obs=[json.loads(p.read_text()) for p in sorted(r.glob('observation_[0-9][0-9][0-9].json'))];assert all(not x.get('historical_trajectories_loaded',True) for x in obs)
a={'GRAIL_applied_residual_max_abs':0,'history_hint_file_loaded':False,'SONIC_ONNX_calls':0,'external_GPT6_API':False,'decisions':len(list(r.glob('decision_[0-9][0-9][0-9].json'))),'source':'current GPT/Codex session paused-checkpoint images, proprioception and privileged feedback'}
f=r/'direct_arm_execution.jsonl'
if f.exists():
 arm=[json.loads(x) for x in f.read_text().splitlines()];a.update(arm_IK_steps=len(arm),maximum_logged_command_delta_rad=max(x['max_arm_target_delta_rad'] for x in arm));assert a['maximum_logged_command_delta_rad']<.020001
(r/'direct_audit.json').write_text(json.dumps(a,indent=2));print(json.dumps(a,indent=2))
