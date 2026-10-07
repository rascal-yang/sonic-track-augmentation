"""Three independent simulator processes; frozen protocol, explicit visual checkpoints."""
import subprocess,time,json,shutil,sys,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
B=Path(__file__).resolve().parents[1];py='/home/kinetra-4090-01/miniconda3/envs/sonic/bin/python'
protocol=json.loads((B/'protocol.json').read_text());lock=threading.Lock()
def execute(item):
 i,name=item;r=B/'runs'/name
 if i==0:
  while not (r/'result.json').exists():time.sleep(2)
  time.sleep(5)
 else:
  time.sleep(15*(i%3))
  with (B/'runs'/f'{name}_launch.log').open('w') as log:
   proc=subprocess.Popen([str(B/'pipeline.sh'),'collect','--name',name,'--brainco','--steps','2600'],cwd=B,stdout=log,stderr=subprocess.STDOUT)
   with (B/'runs'/f'{name}_supervisor.log').open('w') as sup:
    supervisor=subprocess.Popen([sys.executable,str(B/'scripts/run_protocol.py'),str(r)],cwd=B,stdout=sup,stderr=subprocess.STDOUT)
    status=proc.wait();print(name,'SIM_EXIT',status,flush=True)
    if supervisor.poll() is None:supervisor.terminate()
   if status!=0 and not (r/'result.json').exists():raise RuntimeError(f'{name} exited before capture completion')
 if not (r/'scripts_snapshot').exists():shutil.copytree(B/'scripts',r/'scripts_snapshot')
 with (r/'validation.log').open('w') as log:
  for script in ['validate_capture.py','validate_brainco.py','validate_contacts.py']:
   subprocess.run([py,str(B/'scripts'/script),'--run',str(r)],cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(name,'VALIDATED',flush=True)
 with lock:subprocess.run([py,str(B/'scripts/summarize_multistart.py')],cwd=B,check=True,stdout=subprocess.DEVNULL)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(execute,enumerate(protocol['starts'])))
print('ALL_COMPLETE',flush=True)
