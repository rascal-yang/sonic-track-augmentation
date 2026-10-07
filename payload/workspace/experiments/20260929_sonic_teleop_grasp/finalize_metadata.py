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
               reference_source='Previously successful safe025 seven-posture continuous reference -> Cartesian VR3 targets plus standing leg references -> online teleop encoder mode1; source finger targets unchanged',
               authoritative_audit='quality.json')
task = dict(instruction='Execute successful grasp reference via SONIC Cartesian teleop encoder mode1 and live-state decoder; stationary grasp only.',
            control=controller['controller'], max_control_steps=controller_audit['source_token_rows'],
            success_definition='Complete sequence; final strict lift>=3s; opposition>=95%; final3s relative slip<=15mm and rotation<=10deg; monitored table force<=0.05N. quality.json authoritative.',
            privileged_information=True, training_ready=False,
            task_success=audit['task_success'], online_planner=False, IK_override=False)
variant = dict(controller=controller['controller'], finger_control=controller['hands'],
               asset_and_physics_changes='No asset/physics/PD change relative to successful safe025 source.',
               reference=controller['tokens_file'], training_ready=False)
for name, value in [('control_definition.json', control), ('task.json', task), ('variant.json', variant)]:
    (r / name).write_text(json.dumps(value, indent=2) + '\n')
direct = dict(historical_trajectory_loaded=True, reference_tokens_file=controller['tokens_file'], SONIC_decoder_calls=controller_audit['control_steps'], runtime_IK=False, online_encoder=True, online_planner=False, GRAIL_applied_residual=0, initialization=controller['initialization'], external_GPT6_API=False, decisions='Automatic numeric guard continues/stops; this GPT session inspects evidence and designs any subsequent reference correction; no external API.')
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
    import numpy as np
    z=np.load(r/'executed_teleop_tokens.npz')
    g=f.require_group('sonic_teleop')
    for key in z.files:
        if z[key].dtype.kind not in 'US' and key not in g:g.create_dataset(key,data=z[key],compression='gzip')
    g.attrs['encoder_mode']=1
    g.attrs['upper_body_joint_reference_supplied']=False
print('Controller metadata corrected; inherited descriptions preserved.')
