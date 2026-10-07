"""Resumable stationary grasp collection; physical validation, never preview success."""
import argparse
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from common import HERE, W, read, write
from compact import compact


def invoke(args, log, timeout=2400):
    with log.open('w') as stream:
        child = subprocess.Popen([sys.executable, *map(str, args)], stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + timeout
            failed_at = None
            output = None
            if '--output' in args and Path(str(args[0])).name in ['run.py', 'mobile_run.py', 'replay_run.py']:
                output = Path(args[args.index('--output') + 1])
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(child.args, timeout)
                try:
                    return child.wait(timeout=min(5, remaining))
                except subprocess.TimeoutExpired:
                    pass
                if output is not None and (output/'episode_raw.h5').exists() and (output/'run.log').exists():
                    failed = 'Error executing job with overrides:' in (output/'run.log').read_text(errors='replace')
                    if failed and failed_at is None:
                        failed_at = time.monotonic()
                    if failed_at is not None and time.monotonic()-failed_at > 15:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            return child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            return child.wait()
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise


def schedule(root, seed):
    path = root / 'schedule.json'
    if path.exists():
        return read(path)
    rng = np.random.default_rng(seed)
    # Hold out spatial bands, keeping every derivative episode in its geometry group.
    entries = []
    xs = np.linspace(-.015, .015, 10)
    ys = np.linspace(-.010, .010, 5)
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            entries.append(dict(geometry_id=f'pose_{xi:02d}_{yi:02d}',
                                split='test' if xi == 9 else 'validation' if xi == 0 else 'train',
                                object_delta_m=[float(x), float(y), 0],
                                grasp_yaw_deg=float(rng.uniform(-1.5, 1.5)),
                                approach_scale=float(rng.uniform(.995, 1.005))))
    rng.shuffle(entries)
    write(path, dict(seed=seed, target=50, samples=entries,
                     note='50 geometry groups; independent physics required. No time or asset scaling.'))
    return read(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20260930)
    p.add_argument('--limit', type=int, default=50)
    p.add_argument('--efficient-capture', action='store_true')
    p.add_argument('--reserve-gib', type=float, default=10)
    a = p.parse_args()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ['candidates', 'runs', 'logs']:
        (root / name).mkdir(exist_ok=True)
    lock = (W / 'experiments/.sonic_augmentation_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = schedule(root, a.seed)
    consecutive_failures = 0
    successes = []
    attempts = []
    for sample in manifest['samples']:
        group = sample['geometry_id']
        # At most two candidates per geometry: proposed transform, then translation only.
        for revision in range(2):
            name = f'{group}_r{revision}'
            candidate, run = root / 'candidates' / name, root / 'runs' / name
            status_file = root / 'logs' / f'{name}.json'
            if status_file.exists():
                status = read(status_file)
            else:
                free = shutil.disk_usage(root).free / 2**30
                if free < a.reserve_gib:
                    write(root / 'status.json', dict(status='disk_reserve_stop', free_gib=free,
                          successes=successes, attempts=attempts, training_ready=False))
                    return
                if run.exists() or candidate.exists():
                    raise RuntimeError(f'Incomplete prior attempt {name}; inspect before retry, no overwrite')
                spec = dict(sample)
                if revision:
                    spec.update(grasp_yaw_deg=0, approach_scale=1)
                start = time.time()
                status = dict(name=name, geometry_id=group, split=sample['split'],
                              sample=spec, run=str(run), task_success=False)
                try:
                    args = [HERE/'prepare.py', '--output', candidate, '--dx', spec['object_delta_m'][0],
                            '--dy', spec['object_delta_m'][1], '--yaw', spec['grasp_yaw_deg'],
                            '--scale', spec['approach_scale']]
                    code = invoke(args, root/'logs'/f'{name}_prepare.log', 300)
                    if code:
                        raise RuntimeError(f'preparation exit {code}')
                    args = [HERE/'run.py', '--name', name, '--candidate', candidate, '--output', run]
                    if a.efficient_capture:
                        args.append('--efficient-capture')
                    code = invoke(args, root/'logs'/f'{name}_physics.log')
                    status['physics_exit'] = code
                    if (run/'episode_raw.h5').exists():
                        status['validation_exit'] = invoke([HERE/'validate.py', run],
                                                          root/'logs'/f'{name}_validation.log', 1200)
                    if (run/'collection_result.json').exists():
                        result = read(run/'collection_result.json')
                        status.update(task_success=bool(result['task_success'] and code == 0), quality=result['quality'])
                        if result['validation_passed']:
                            # Keep lossless RGB and depth for later export; remove only duplicate caches.
                            status['freed_duplicate_bytes'] = compact(run, keep_raw=True)
                except Exception as error:
                    status['error'] = repr(error)
                status['elapsed_s'] = time.time() - start
                status['reflection'] = ('Successful perturbation; continue another geometry.' if status['task_success']
                    else 'Failed trial excluded. Translation-only fallback removes orientation/scale confounds; stop after three consecutive failures.')
                status['training_ready'] = False
                write(status_file, status)
                with (root/'attempts.jsonl').open('a') as stream:
                    import json
                    stream.write(json.dumps(status) + '\n')
            attempts.append(status)
            if status['task_success']:
                successes.append(status)
                consecutive_failures = 0
                break
            consecutive_failures += 1
            if consecutive_failures >= 3:
                write(root/'status.json', dict(status='review_required_after_failures', successes=successes,
                                               attempts=attempts, training_ready=False))
                write(root/'success_index.json', successes)
                return
        write(root/'success_index.json', successes)
        write(root/'status.json', dict(status='collecting', success_count=len(successes),
                                      attempt_count=len(attempts), training_ready=False))
        print(f'Successes {len(successes)}/{a.limit}; attempts {len(attempts)}', flush=True)
        if len(successes) >= a.limit:
            break
    write(root/'status.json', dict(status='requested_batch_complete' if len(successes) >= a.limit
          else 'schedule_exhausted', success_count=len(successes), attempt_count=len(attempts), requested_batch_successes=a.limit, dataset_target_successes=50,
          training_ready=False, visual_review='pending'))


if __name__ == '__main__':
    main()
