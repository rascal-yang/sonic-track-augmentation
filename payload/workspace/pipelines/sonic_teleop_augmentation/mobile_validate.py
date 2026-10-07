import argparse,subprocess,sys
from pathlib import Path
from common import *
p=argparse.ArgumentParser();p.add_argument('run',type=Path);r=p.parse_args().run.resolve()
commands=[(HERE/'mobile_audit.py',[str(r)]),(W/'experiments/20260929_sonic_walk_fast_grasp/audit_capture.py',['--run',str(r)]),(BASE/'scripts/validate_brainco.py',['--run',str(r)]),(BASE/'scripts/validate_contacts.py',['--run',str(r)]),(HERE/'review_artifacts.py',[str(r)])]
checks=[]
for script,args in commands:
 with (r/(script.stem+'_validation.log')).open('w') as f:
  rc=subprocess.run([sys.executable,str(script),*args],stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
 checks.append(dict(script=script.name,exit_code=rc));print(script.name,rc,flush=True)
write(r/'validation_commands.json',checks)
audit=read(r/'audit.json') if (r/'audit.json').exists() else {}
passed=all(x['exit_code']==0 for x in checks)
result=dict(task_success=bool(passed and audit.get('task_success')),validation_passed=passed,quality=audit,training_ready=False,visual_review='pending',run=str(r))
write(r/'collection_result.json',result)
write(r/'reflection.json',dict(outcome=result['task_success'],integration_failure=read(r/'integration_failure.json') if (r/'integration_failure.json').exists() else None,evidence=audit,next_step='Inspect handoff tracking and contact diagnostics before broadening start distribution.',hypothesis=read(r/'plan.json')['hypothesis']))
write(r/'visual_realism_review.json',dict(accepted=False,training_ready=False,temporal_review_status='pending',renderer='RTX realtime DLAA preview; not RT2'))
import h5py
with h5py.File(r/'episode_raw.h5','r+') as f:
 f.attrs['episode_success']=result['task_success'];f.attrs['training_ready']=False
 f.attrs['controller']=read(r/'token_controller.json')['controller']
print(result)
from memory import record
record(r)
if not passed:raise SystemExit(2)
