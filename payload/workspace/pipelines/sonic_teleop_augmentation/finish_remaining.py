"""Finish the ten prescribed start variants, then replay a fixed random sample."""
import argparse
import fcntl
import math
import random
import shutil
import subprocess
import sys
import time

import numpy as np

from common import HERE, W, Path, arrays, read, sha, write
from collect import mobile_schedule
from lean_pair import execute
from lean_prepare_replay import prepare
from mobile_prepare import compile_mobile


def check_capture(run):
    """Check training labels and decode the complete video, without re-simulating."""
    import cv2
    z = arrays(run/'training_capture.npz')
    tokens = arrays(run/'executed_teleop_tokens.npz')
    n = len(z['control_index'])
    assert n == read(run/'lean_result.json')['control_steps']
    assert np.array_equal(z['control_index'], np.arange(n))
    assert z['state'].shape == (n, 41) and z['token'].shape == (n, 64)
    assert z['projected_gravity'].shape == (n, 3)
    for key, other in [('token', 'sonic_encoded_tokens'),
                       ('left_hand_target', 'revo2_left_active_position_target_rad'),
                       ('right_hand_target', 'revo2_right_active_position_target_rad')]:
        assert np.array_equal(z[key], tokens[other]), key
        assert np.isfinite(z[key]).all(), key
    assert np.isfinite(z['state']).all() and np.isfinite(z['projected_gravity']).all()
    assert np.allclose(tokens['time_s'], np.arange(n)*.02)
    cap = cv2.VideoCapture(str(run/'head.mp4'))
    assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS)-50) < .001
    selected = np.linspace(0, n-1, 9, dtype=int).tolist()
    count, images = 0, []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.shape == (480, 640, 3)
            if count in selected:
                small = cv2.resize(frame, (320, 240))
                cv2.putText(small, f'{count/50:.2f}s', (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
                images.append(small)
            count += 1
    finally:
        cap.release()
    assert count == n and len(images) == 9
    sheet = np.concatenate([np.concatenate(images[i:i+3], axis=1) for i in [0, 3, 6]], axis=0)
    assert cv2.imwrite(str(run/'visual_contact_sheet.jpg'), sheet)
    write(run/'capture_check.json', dict(passed=True, frames=n, labels_match=True,
          all_frames_decoded=True, training_ready=False, visual_temporal_review='pending'))
    write(run/'visual_realism_review.json', dict(accepted=False, reviewed_frame_indices=[],
          prepared_frame_indices=selected, temporal_review_status='pending',
          evidence='visual_contact_sheet.jpg', note='Full decode is not temporal visual approval.'))


def near_boundary(q):
    return (q['strict_final_hold_s'] < 3.4 or q['final3s_opposed_fraction'] < .98 or
            q['final3s_slip_m'] > .0075 or q['final3s_rotation_deg'] > 5 or
            q['monitored_table_force_peak_N'] > .025 or q['minimum_pelvis_z_m'] < .60)


def retry_docking(record, root):
    """One bounded earlier-brake correction; never relax docking or grasp gates."""
    original_run = Path(record['run'])
    quality = read(original_run/'lean_result.json')
    assert quality['stop_reason'] in ['Grasp entry convergence timeout',
            'Missed docking; reject without near-table redocking'], quality['stop_reason']
    source = root/'candidates'/record['name']
    plan = read(source/'plan.json')
    guard_files = sorted(original_run.glob('guard_*.json'))
    diagnostics = read(guard_files[-1]) if guard_files else None
    correction = .02
    if diagnostics is not None:
        direction = np.asarray(plan['forward_world'])[:2]
        direction = direction/np.linalg.norm(direction)
        residual = (np.asarray(diagnostics['root_position_m'])[:2]-
                    np.asarray(plan['goal_root_m'])[:2])@direction
        assert residual > .025, 'Earlier braking is not justified for this failure'
        correction = round(float(np.clip(residual-.025, .02, .12)), 2)
    brake = round(plan.get('brake_distance_m', .18)+correction, 2)
    override_file = root/'refinement_overrides.json'
    override = read(override_file).get(record['name']) if override_file.exists() else None
    if override is not None:
        brake = float(override['brake_distance_m'])
        assert .18 <= brake <= .35
        correction = round(brake-plan.get('brake_distance_m', .18), 2)
    tag = f'{record["name"]}_brake{round(brake*100):03d}'
    candidate, run = root/'retry_candidates'/tag, root/'runs'/tag
    refinement = dict(original_run=str(original_run), original_stop_reason=quality['stop_reason'],
        diagnostics=diagnostics, earlier_braking_m=correction, brake_distance_m=brake,
        hypothesis='Earlier braking compensates measured forward docking overshoot; preserve all gates.',
        sampled_start_unchanged=True, grasp_skill_unchanged=True, reviewed_override=override)
    if not candidate.exists():
        shutil.copytree(source, candidate)
        plan.update(brake_distance_m=brake, source_fast=str(candidate/'skill'), refinement=refinement)
        write(candidate/'plan.json', plan)
        write(candidate/'prepared_files.json', {str(f.relative_to(candidate)):sha(f)
              for f in candidate.rglob('*') if f.is_file() and f.name!='prepared_files.json'})
    assert sha(source/'initial_state.npz') == sha(candidate/'initial_state.npz')
    write(root/'attempt_records'/f'{record["name"]}_original.json', record)
    write(root/'refinement_plans'/f'{record["name"]}.json', refinement)
    status = read(root/'status.json')
    status.update(current_run=str(run), retry_candidate=str(candidate))
    write(root/'status.json', status)
    if run.exists():
        assert read(run/'lean_result.json')['task_success'], 'Failed correction retained; inspect before another change'
        seconds = None
    else:
        seconds = execute('generate', candidate, run, 1800)
    check_capture(run)
    updated = {k:v for k,v in record.items() if k not in ['error', 'quality']}
    updated.update(task_success=True, run=str(run), first_wall_s=seconds,
                   quality=read(run/'lean_result.json'), refinement=refinement,
                   attempts=[str(x) for x in sorted((root/'runs').glob(record['name']+'*'))
                             if (x/'lean_result.json').exists()])
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--retry-failed', action='store_true',
                        help='Retry diagnosed docking failures once with measured earlier braking')
    args = parser.parse_args()
    source, root = args.source_root.resolve(), args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'collection.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Upstream motion metadata lookup globally replaces 'robot' with 'meta'.
    # Use neutral runtime paths, preserving all old failed artifacts.
    assert 'robot' not in str(root), 'Use a neutral output path for upstream metadata resolution'
    assert len(read(source/'mobile/success_index.json')) == 50
    jobs = mobile_schedule(read(source/'stationary/success_index.json'), extras=True)
    assert len(jobs) == 10
    plan_path = root/'collection_plan.json'
    plan = dict(source=str(source), target=10, existing_complete=50,
        per_episode_replay=False, replay_after_all_generation=True, sample_fraction=.10,
        random_seed=20261001, extra_sampling='all successful episodes near quality boundary',
        offsets=[dict(dx=x[2][0], dy=x[2][1], yaw_deg=x[2][2]) for x in jobs],
        matched_lessons=['Keep continuous decoder history and physics/control frequencies',
                        'Live root compensation after measured walking handoff',
                        'Keep bilateral contact guards and terminal stable opposed grasp'],
        rejected_changes=['Change dimensions/friction', 'Replace prescribed starts with easier ones',
                          'Replay every episode before generating the next'],
        split='train', training_ready=False)
    if plan_path.exists():
        assert read(plan_path) == plan, 'Collection plan changed; use a new output directory'
    else:
        write(plan_path, plan)
    records, failures, consecutive = [], [], 0
    def progress(phase, **extra):
        known = [read(x) for x in (root/'records').glob('*.json')]
        accepted = sum(x['task_success'] for x in known)
        attempts = [read(x) for x in (root/'runs').glob('*/lean_result.json')]
        status = dict(phase=phase, successful=accepted, target=10,
                      total_successful=50+accepted, per_episode_replay=False,
                      failed=sum(not x['task_success'] for x in known),
                      unresolved_failed_samples=sum(not x['task_success'] for x in known),
                      generation_attempts_completed=len(attempts),
                      failed_generation_attempts=sum(not x['task_success'] for x in attempts),
                      training_ready=False, **extra)
        write(root/'status.json', status)
        write(source/'phase_status.json', dict(stationary=50, mobile=50,
              robot_starts=accepted, continuation=str(root), **status))
    for i, (parent, logical_name, offset) in enumerate(jobs):
        name = f'start_{i+1:02d}_{parent["geometry_id"]}'
        run, candidate = root/'runs'/name, root/'candidates'/name
        record_path = root/'records'/f'{name}.json'
        progress('generation', current=name, current_run=str(run))
        if record_path.exists():
            record = read(record_path)
            if not record['task_success']:
                if not args.retry_failed:
                    raise RuntimeError(f'Failed attempt retained at {run}; diagnose before resume')
                progress('retry_generation', current=name, original_failed_run=record['run'])
                try:
                    record = retry_docking(record, root)
                except Exception as exc:
                    record['latest_retry_error'] = repr(exc)
                write(record_path, record)
        else:
            record = dict(name=name, logical_name=logical_name, geometry_id=parent['geometry_id'],
                split='train', run=str(run), offset=dict(dx=offset[0], dy=offset[1], yaw_deg=offset[2]),
                task_success=False, replay_accepted=False, replay_status='not_sampled', training_ready=False)
            try:
                assert shutil.disk_usage(root).free > 10*1024**3, 'Disk reserve reached'
                if not candidate.exists():
                    skill = Path(read(Path(parent['run'])/'collection_result.json')['candidate'])
                    compile_mobile(skill, candidate, *offset)
                if run.exists():
                    # Recover a completed run only; never silently overwrite an interrupted run.
                    assert (run/'lean_result.json').exists() and read(run/'lean_result.json')['task_success']
                    record['first_wall_s'] = None
                else:
                    record['first_wall_s'] = execute('generate', candidate, run, 1800)
                check_capture(run)
                record.update(task_success=True, quality=read(run/'lean_result.json'))
            except Exception as exc:
                record['error'] = repr(exc)
            write(record_path, record)
        if record['task_success']:
            records.append(record)
            consecutive = 0
        else:
            failures.append(record)
            consecutive += 1
        write(root/'success_index.json', records)
        write(root/'failures.json', failures)
        progress('generation', just_finished=name)
        print(f'GENERATION {len(records)}/10; {name}: {record["task_success"]}', flush=True)
        if consecutive >= 3:
            raise RuntimeError('Three consecutive failures; retained diagnostics')
    if failures:
        progress('generation_incomplete')
        raise RuntimeError('Incomplete prescribed starts; no replacement or complete-dataset claim')
    rng = random.Random(20261001)
    random_names = rng.sample([r['name'] for r in records], math.ceil(.10*len(records)))
    boundary_names = [r['name'] for r in records if near_boundary(r['quality'])]
    entry_names = [r['name'] for r in records
                   if read(Path(r['run'])/'grasp_handoff.json')['actual']['navigation_error_m'] > .045]
    refined_names = [r['name'] for r in records if r.get('refinement')]
    selected = sorted(set(random_names+boundary_names+entry_names+refined_names))
    write(root/'spotcheck_plan.json', dict(seed=20261001, fraction=.10,
          random_sample=random_names, boundary_sample=boundary_names, entry_boundary_sample=entry_names,
          entry_boundary_threshold_m=.045, refined_sample=refined_names, selected=selected,
          timing='All first executions complete before first spot replay'))
    checks = []
    for name in selected:
        record = next(r for r in records if r['name'] == name)
        progress('spotcheck', current=name, selected=selected, checks_completed=len(checks))
        candidate, run = root/'frozen_replays'/name, root/'replays'/name
        if not candidate.exists():
            prepare(Path(record['run']), candidate)
        if run.exists():
            assert read(run/'lean_result.json')['task_success'], 'Prior failed replay requires review'
            seconds = None
        else:
            seconds = execute('replay', candidate, run, 1800)
        checks.append(dict(name=name, run=str(run), passed=True, wall_s=seconds,
                           quality=read(run/'lean_result.json')))
        record.update(replay_accepted=True, replay_status='sampled_passed', replay=str(run))
        write(root/'records'/f'{name}.json', record)
        write(root/'success_index.json', records)
        write(root/'spotcheck_report.json', dict(passed=len(checks)==len(selected),
              first_executions=10, selected=selected, checks=checks, unchecked=10-len(selected)))
    progress('export', checks_completed=len(checks))
    from lean_export import export
    dataset = root/'lerobot_all_train'
    if not dataset.exists():
        export(root/'success_index.json', dataset, allow_unreplayed=True)
    subprocess.run([sys.executable, str(HERE/'check_export.py'), str(dataset)], cwd=W, check=True)
    progress('complete', checks_completed=len(checks), dataset=str(dataset),
             visual_review='pending', unchecked_replays=10-len(selected))
    print('COMPLETE 10/10; sampled independent replay and GR00T loader passed', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if '--output-root' in sys.argv:
            destination = Path(sys.argv[sys.argv.index('--output-root')+1])
            write(destination/'collector_error.json', dict(error=repr(error), time=time.time()))
        raise
