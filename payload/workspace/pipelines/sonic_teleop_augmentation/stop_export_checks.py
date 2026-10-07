"""Stop only obsolete loader checks for this specific mobile50 export."""
import os
import signal
from pathlib import Path

root = Path('__RUNTIME__/workspace/experiments/20260930_teleop_aug50/lerobot')
script = '__RUNTIME__/workspace/pipelines/sonic_teleop_augmentation/check_export.py'
targets = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = proc.joinpath('cmdline').read_bytes().decode().split('\0')
        if not any(a == script or a == 'pipelines/sonic_teleop_augmentation/check_export.py' for a in args):
            continue
        if not any(a in [str(root/f'mobile_{s}'), str(root.relative_to(root.parents[2])/f'mobile_{s}')]
                   for a in args for s in ['train', 'validation', 'test']):
            # Relative command paths used by this session.
            if not any(a == f'experiments/20260930_teleop_aug50/lerobot/mobile_{s}'
                       for a in args for s in ['train', 'validation', 'test']):
                continue
        targets.append(int(proc.name))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
for pid in targets:
    try:
        children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        for child in children:
            args = Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0')
            if args and Path(os.fsdecode(args[0])).name in ['ffmpeg', 'ffprobe']:
                os.kill(int(child), signal.SIGTERM)
        os.kill(pid, signal.SIGTERM)
        print('Stopped obsolete export loader check', pid, flush=True)
    except ProcessLookupError:
        pass
