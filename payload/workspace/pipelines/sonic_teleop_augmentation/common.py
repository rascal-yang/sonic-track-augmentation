from pathlib import Path
import importlib.util,json,hashlib,os
import numpy as np
HERE=Path(__file__).resolve().parent;W=HERE.parents[1];BASE=W/'experiments/20260925_measured_walk_grasp'
SOURCE=Path(os.environ.get('G1_GRASP_SOURCE', W/'experiments/20260929_sonic_grasp_smooth/candidates/safe025'))
TELEOP=W/'experiments/20260929_sonic_teleop_grasp'
MODEL=Path('__RUNTIME__/models/release')
SCALE=.982730067592321

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def arrays(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
