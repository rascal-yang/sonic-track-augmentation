"""Reconstruct completed episode stages from filesystem timestamps (approximate)."""
import argparse
import datetime
import json
import statistics
import subprocess
from pathlib import Path


def birth(path):
    value = subprocess.check_output(['stat', '-c', '%w', str(path)], text=True).strip()
    if value == '-':
        raise RuntimeError(f'Filesystem birth time unavailable: {path}')
    return datetime.datetime.fromisoformat(value).timestamp()


def stage(folder, name):
    logs, run = folder/'logs', folder/'runs'/name
    prepare = birth(logs/f'{name}_prepare.log')
    physics = birth(logs/f'{name}_physics.log')
    validation = birth(logs/f'{name}_validation.log')
    validation_end = (run/'collection_result.json').stat().st_mtime
    decisions = sorted(p for p in run.glob('decision_*.json') if p.stem[9:].isdigit())
    first = birth(decisions[0])
    last = decisions[-1].stat().st_mtime
    return dict(prepare_s=physics-prepare, startup_to_first_decision_s=first-physics,
                execution_between_decisions_s=last-first,
                final_step_pack_exit_s=validation-last,
                validation_s=validation_end-validation,
                prepare_start=prepare, validation_end=validation_end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    entries = json.loads((root/'replay_accepted_index.json').read_text())
    rows = []
    for entry in entries:
        name = entry['name']
        folder = Path(entry['run']).parent.parent
        first, replay = stage(folder, name), stage(root/'replays', name)
        finished = (root/'replays/logs'/f'{name}.json').stat().st_mtime
        first_end = first.pop('validation_end')
        replay_end = replay.pop('validation_end')
        first_start = first.pop('prepare_start')
        replay_start = replay.pop('prepare_start')
        rows.append(dict(name=name, first=first, replay=replay,
                         first_cleanup_lossless_pack_s=replay_start-first_end,
                         replay_cleanup_s=finished-replay_end,
                         total_s=finished-first_start))
    means = {}
    if rows:
        for prefix in ['first', 'replay']:
            for key in rows[0][prefix]:
                means[prefix+'_'+key] = statistics.mean(x[prefix][key] for x in rows)
        for key in ['first_cleanup_lossless_pack_s', 'replay_cleanup_s', 'total_s']:
            means[key] = statistics.mean(x[key] for x in rows)
    result = dict(generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  completed_pairs=len(rows), episodes=rows, mean_s=means,
                  median_total_s=statistics.median(x['total_s'] for x in rows) if rows else None,
                  method='Approximate elapsed wall time from log birth times, decision '
                  'file timestamps and validation outputs. Execution includes physics, '
                  'inference, render/capture, checkpoints and temporary compression. '
                  'Final step/pack/exit is a combined bucket; no GPU-only attribution.')
    target = root/'timing_report.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(completed_pairs=len(rows), mean_s=means), indent=2))


if __name__ == '__main__':
    main()
