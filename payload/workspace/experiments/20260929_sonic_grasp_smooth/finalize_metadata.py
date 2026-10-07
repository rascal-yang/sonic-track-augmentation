"""Correct inherited collector descriptions after rollout and audit completion."""
from pathlib import Path
import argparse, json
import h5py

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
r = p.parse_args().run
audit = json.loads((r / 'quality.json').read_text())
controller_audit = json.loads((r / 'token_summary.json').read_text())
controller = json.loads((r / 'token_controller.json').read_text())
legacy = r / 'inherited_metadata'
legacy.mkdir(exist_ok=True)
for name in ['control_definition.json', 'task.json', 'variant.json', 'direct_provenance.json', 'planner_provenance.json']:
    path = r / name
    if not (legacy / name).exists():
        (legacy / name).write_bytes(path.read_bytes())
control = dict(controller, grail_interaction_enabled=False,
               latent_dimension=64, residual_mode='none',
               token_representation='release encoder quantized float32 output',
               reference_source='walk_009 measured-state seed ->25mm offline IK -> optional seven-posture smoothing/retiming -> offline encoder; separately retimed Revo2 targets',
               authoritative_audit='quality.json')
task = dict(instruction='Replay25mm grasp-only reference encoded SONIC tokens with live-state decoder; evaluate grasp smoothing and retiming.',
            control=controller['controller'], max_control_steps=controller_audit['source_token_rows'],
            success_definition='Complete sequence; final strict lift>=3s; opposition>=95%; final3s relative slip<=15mm and rotation<=10deg; monitored table force<=0.05N. quality.json authoritative.',
            privileged_information=True, training_ready=False,
            task_success=audit['task_success'], online_planner=False, IK_override=False)
variant = dict(controller=controller['controller'], finger_control=controller['hands'],
               asset_and_physics_changes='Deployment body Kp/Kd; existing measured scene and force limits retained.',
               reference=controller['tokens_file'], training_ready=False)
for name, value in [('control_definition.json', control), ('task.json', task), ('variant.json', variant)]:
    (r / name).write_text(json.dumps(value, indent=2) + '\n')
direct = dict(historical_trajectory_loaded=True, reference_tokens_file=controller['tokens_file'], SONIC_decoder_calls=controller_audit['control_steps'], runtime_IK=False, online_encoder=False, online_planner=False, GRAIL_applied_residual=0, initialization=controller['initialization'], external_GPT6_API=False, decisions='Current session reviews paused stage checkpoints; supervisor only continues/stops, never changes fixed tokens.')
planner = dict(planner='none during rollout', low_level=controller['controller'], body_joint_count=29, joint_target='deployment default+scale*release decoder output at50Hz; reapplied each200Hz physics step', actuator_output_filter=False, reference_plan=str(Path(controller['tokens_file']).parent/'plan.json'))
for name,value in [('direct_provenance.json',direct),('planner_provenance.json',planner)]:
    (r/name).write_text(json.dumps(value,indent=2)+'\n')
with h5py.File(r / 'episode_raw.h5', 'r+') as f:
    old = {k: str(f.attrs[k]) for k in ['controller', 'episode_success', 'training_ready'] if k in f.attrs}
    if 'inherited_controller_metadata' not in f.attrs:
        f.attrs['inherited_controller_metadata'] = json.dumps(old)
    f.attrs['controller'] = controller['controller']
    f.attrs['episode_success'] = audit['task_success']
    f.attrs['training_ready'] = False
    f.attrs['token_controller_metadata'] = json.dumps(controller)
    f.attrs['token_audit'] = 'token_summary.json'
print('Controller metadata corrected; inherited descriptions preserved.')
