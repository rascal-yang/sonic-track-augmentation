"""Create a uniquely named run without overwriting prior experiments."""
from pathlib import Path
import argparse,json,subprocess,sys
B=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--start',nargs=3,type=float,required=True,metavar=('X','Y','Z'));p.add_argument('--stationary',action='store_true');a=p.parse_args()
name=('stationary_' if a.stationary and not a.name.startswith('stationary') else '')+a.name
if not name.replace('_','').isalnum():raise ValueError('Use letters, numbers and underscores for run name')
if (B/'runs'/name).exists():raise FileExistsError('Existing experiments are preserved; choose a new name')
d=json.loads((B/'protocol.json').read_text());d['starts'][name]=a.start;tmp=B/'protocol.tmp';tmp.write_text(json.dumps(d,indent=2));tmp.replace(B/'protocol.json')
print('A visual review gate will pause at the initial state and selected manipulation checkpoints. Use GPT/Codex to inspect the images and observations; no external GPT API is called.',flush=True)
subprocess.run([sys.executable,str(B/'scripts/launch_optimized.py'),name],cwd=B,check=True)
