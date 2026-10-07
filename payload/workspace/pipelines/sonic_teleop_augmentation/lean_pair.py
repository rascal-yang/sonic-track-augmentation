"""Generate training essentials, independently replay tokens, then export all-train."""
import argparse
import os
import signal
import subprocess
import sys
import time

from common import HERE, W, Path, read, write
from lean_prepare_replay import prepare


def execute(mode, candidate, output, timeout):
    started = time.perf_counter()
    process = subprocess.Popen([sys.executable, str(HERE/'lean_run.py'),
        '--mode', mode, '--candidate', str(candidate), '--output', str(output)],
        cwd=W, start_new_session=True)
    try:
        while process.poll() is None:
            if time.perf_counter()-started > timeout:
                raise subprocess.TimeoutExpired(process.args, timeout)
            log = output/'run.log'
            if log.exists() and 'Error executing job with overrides:' in log.read_text(errors='replace'):
                raise RuntimeError(f'{mode} simulator error; see {log}')
            time.sleep(2)
        code = process.returncode
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    elapsed = time.perf_counter()-started
    assert code == 0, f'{mode} exited {code}; retained {output}'
    result = read(output/'lean_result.json')
    assert result['task_success'], f'{mode} failed terminal checks; retained {output}'
    return elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--no-export', action='store_true')
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    first, replay = root/'generate', root/'replay'
    timing = {}
    try:
        timing['first_wall_s'] = execute('generate', args.candidate.resolve(), first, args.timeout)
        candidate = prepare(first, root/'frozen_replay')
        timing['replay_wall_s'] = execute('replay', candidate, replay, args.timeout)
        write(root/'success_index.json', [dict(run=str(first), replay=str(replay),
              replay_accepted=True, geometry_id=args.candidate.name)])
        if not args.no_export:
            from lean_export import export
            export(root/'success_index.json', root/'lerobot_all_train')
        write(root/'status.json', dict(success=True, **timing))
    except BaseException as exc:
        write(root/'status.json', dict(success=False, error=repr(exc), **timing))
        raise


if __name__ == '__main__':
    main()
