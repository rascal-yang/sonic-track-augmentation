"""Only physical+data-audited successful episodes enter the collection index."""
import argparse,subprocess,sys
from pathlib import Path
from common import *
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run.resolve()
commands=[(HERE/'report.py',[str(r)]),(HERE/'quality.py',[str(r)]),(W/'experiments/20260929_sonic_walk_fast_grasp/audit_capture.py',['--run',str(r)]),(BASE/'scripts/validate_brainco.py',['--run',str(r)]),(BASE/'scripts/validate_contacts.py',['--run',str(r)]),(HERE/'review_artifacts.py',[str(r)]),(HERE/'finalize_metadata.py',[str(r)])]
checks=[]
for script,args in commands:
 with (r/(script.stem+'_validation.log')).open('w') as log:
  result=subprocess.run([sys.executable,str(script),*args],stdout=log,stderr=subprocess.STDOUT,timeout=240)
 checks.append(dict(script=script.name,exit_code=result.returncode));print(script.name,result.returncode,flush=True)
 if result.returncode and script.name in ['report.py','quality.py']:break
write(r/'validation_commands.json',checks)
quality=read(r/'quality.json') if (r/'quality.json').exists() else {}
passed=len(checks)==len(commands) and all(x['exit_code']==0 for x in checks)
c=Path(read(r/'token_controller.json')['tokens_file']).parent
result=dict(task_success=bool(quality.get('task_success') and passed),validation_passed=passed,quality=quality,sample=read(c/'sample.json'),candidate=str(c),run=str(r),training_ready=False,visual_review='pending')
write(r/'collection_result.json',result)
write(r/'reflection.json',dict(prediction=read(c/'plan.json')['hypothesis'],changed_parameters=result['sample'],outcome=result['task_success'],quality=quality,guard=read(sorted(r.glob('guard_*.json'))[-1]) if list(r.glob('guard_*.json')) else {},uncertainty='Task success does not prove arbitrary Cartesian generalization; review failure before resampling.',external_GPT_API=False))
write(r/'visual_realism_review.json',dict(accepted=False,training_ready=False,temporal_review_status='pending',full_decode_passed=passed,renderer='RTX realtime/DLAA preview; not RT2',note='Batch numeric validation; sampled/session visual review tracked separately.'))
print(result)
from memory import record
record(r)
if not passed:raise SystemExit(2)
