"""Validate a completed physical run and preserve transparent metadata."""
from pathlib import Path
import argparse,json,subprocess,sys
H=Path(__file__).resolve().parent;W=H.parents[1]
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run.resolve()
commands=[(H/'report.py',[str(r)]),(H/'quality.py',[str(r)]),(W/'experiments/20260929_sonic_walk_fast_grasp/audit_capture.py',['--run',str(r)]),(W/'experiments/20260925_measured_walk_grasp/scripts/validate_brainco.py',['--run',str(r)]),(W/'experiments/20260925_measured_walk_grasp/scripts/validate_contacts.py',['--run',str(r)]),(W/'experiments/20260929_sonic_grasp_smooth/analyze_contact.py',[str(r)]),(H/'review_artifacts.py',[str(r)])]
checks=[]
for script,args in commands:
 result=subprocess.run([sys.executable,str(script),*args],capture_output=True,text=True)
 (r/(script.stem+'_check.log')).write_text(result.stdout+result.stderr)
 checks.append(dict(script=str(script),exit_code=result.returncode));print(script.name,result.returncode,flush=True)
 if result.returncode and script.name in ['report.py','quality.py']:raise SystemExit(result.stderr)
if any(item['exit_code'] for item in checks):raise SystemExit('Validation failed; inspect logs')
(r/'checks.json').write_text(json.dumps(checks,indent=2))
# Preserve inherited task wording; replace misleading locomotion/IK descriptions.
for name in ['task.json']:
 old=json.loads((r/name).read_text());(r/('inherited_'+name)).write_text(json.dumps(old,indent=2))
 old.update(instruction='Stationary grasp-only comparison using successful fast Cartesian references through SONIC teleop encoder mode1 and release decoder.',agent='Automatic numeric safety supervision; session GPT analyzes completed attempts and designs candidate corrections; no external API',external_model_api=False,max_control_steps=586,training_ready=False)
 (r/name).write_text(json.dumps(old,indent=2))
print(json.dumps(json.loads((r/'quality.json').read_text()),indent=2))

subprocess.run([sys.executable,str(H/'finalize_metadata.py'),str(r)],check=True)
subprocess.run([sys.executable,str(H/'diagnose.py'),str(r)],check=True,stdout=(r/'diagnose_check.log').open('w'))
