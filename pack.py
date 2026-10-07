"""Create a self-contained offline source/resource delivery with SHA256."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import shutil

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
out = a.output.resolve()
if out == ROOT or ROOT in out.parents:
    p.error('Archive must be outside the repository to avoid including itself')
if out.exists():
    p.error('Output exists; choose a new archive name')
out.parent.mkdir(parents=True, exist_ok=True)
partial = out.with_name(out.name+'.partial')
if partial.exists():
    p.error('Partial archive exists; inspect it before retrying')
compression = ['--use-compress-program=pigz -p 2', '-cf'] if shutil.which('pigz') else ['-czf']
subprocess.run(['tar', *compression, str(partial),
    '--exclude='+ROOT.name+'/.git', '--exclude='+ROOT.name+'/runtime',
    '--exclude='+ROOT.name+'/.venv', '--exclude='+ROOT.name+'/runs',
    '--exclude=__pycache__', '--exclude=*.pyc',
    '-C', str(ROOT.parent), ROOT.name], check=True)
partial.replace(out)
with out.open('rb') as stream:
    checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
out.with_name(out.name+'.sha256').write_text(checksum+'  '+out.name+'\n')
print(out, out.stat().st_size, 'bytes', checksum)
