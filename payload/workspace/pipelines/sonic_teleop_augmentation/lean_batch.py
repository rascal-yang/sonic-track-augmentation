"""Sequential lean collection from prepared mobile candidates; stop on failure."""
import argparse
import shutil
import subprocess
import sys
from common import HERE, W, Path, read, write
from lean_export import export


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    candidates = sorted(x.parent for x in args.candidates.resolve().glob('*/prepared_files.json'))
    if args.limit is not None:
        assert args.limit > 0
        candidates = candidates[:args.limit]
    assert candidates, 'No prepared candidates'
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    index = []
    for candidate in candidates:
        pair = root/'pairs'/candidate.name
        if pair.exists():
            assert (pair/'status.json').exists() and read(pair/'status.json')['success'], \
                f'Incomplete/failed run retained at {pair}; diagnose before continuing'
        else:
            assert shutil.disk_usage(root).free > 10*1024**3, 'Less than 10 GiB free'
            process = subprocess.run([sys.executable, str(HERE/'lean_pair.py'),
                '--candidate', str(candidate), '--output', str(pair), '--no-export'], cwd=W)
            if process.returncode:
                write(root/'status.json', dict(success=False, completed=len(index), failed=str(pair)))
                raise RuntimeError(f'Pair failed; retained {pair}')
        index.extend(read(pair/'success_index.json'))
        write(root/'success_index.json', index)
        write(root/'status.json', dict(success=True, completed=len(index), requested=len(candidates)))
    dataset = root/'lerobot_all_train'
    assert not dataset.exists(), f'Export exists; retained {dataset}'
    export(root/'success_index.json', dataset)


if __name__ == '__main__':
    main()
