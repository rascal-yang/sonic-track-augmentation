"""Launch an independent run and snapshot exact inputs before starting."""
import sys,subprocess,shutil,json,time
from pathlib import Path
B=Path(__file__).resolve().parents[1];name=sys.argv[1];r=B/'runs'/name
with (B/'runs'/f'{name}_launch.log').open('w') as f:
 sim=subprocess.Popen([str(B/'pipeline.sh'),'collect','--name',name,'--brainco','--steps','3500'],cwd=B,stdout=f,stderr=subprocess.STDOUT)
 for _ in range(300):
  if r.exists():break
  if sim.poll() is not None:raise RuntimeError('Launch failed')
  time.sleep(.1)
 shutil.copytree(B/'scripts',r/'scripts_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
 for n in ['optimized_config.json','protocol.json','asset_manifest.json']:shutil.copy2(B/n,r/n)
 if name.startswith('stationary_arc'):
  c=json.loads((r/'optimized_config.json').read_text());c.update(version='v3_continuous_clearance_arc',continuous_arc=True,clearance_arc_m=[0,-.12,.15],approach_steps=330);(r/'optimized_config.json').write_text(json.dumps(c,indent=2))
 if name.startswith('stationary_retry'):
  stress=json.loads((r/'optimized_config.json').read_text());stress['first_attempt_target_offset_m']=[.08,0,0];stress['stress_test']='Deliberately offset only the first grasp target by +8cm X to test supported-bottle physical retry; excluded from nominal multistart success rate.';(r/'optimized_config.json').write_text(json.dumps(stress,indent=2))
 with (B/'runs'/f'{name}_supervisor.log').open('w') as log:
  sup=subprocess.Popen([sys.executable,str(B/'scripts/optimized_protocol.py'),str(r)],cwd=B,stdout=log,stderr=subprocess.STDOUT)
  while not (r/'result.json').exists():
   if sim.poll() is not None:
    (r/'launcher_failure.json').write_text(json.dumps({'status':'simulator_exited_before_result','returncode':sim.returncode,'time':time.time(),'partial_capture':True,'training_eligible':False},indent=2))
    break
   if sup.poll() is not None and sup.returncode!=0:
    # Keep the collector alive until it saves a final step and closes video/HDF5.
    observations=sorted(r.glob('observation_[0-9][0-9][0-9].json'))
    if observations:
     obs=json.loads(observations[-1].read_text());i=obs['observation_id'];decision=r/f'decision_{i:03d}.json'
     if not decision.exists():
      record={'observation_id':i,'action':'stop','steps':1,'goal':{},'reason':'Supervisor failed; finalize partial episode without more grasp commands','author':'launcher error guard','wall_time':time.time()}
      tmp=decision.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(decision)
      (r/'supervisor_error.json').write_text(json.dumps({'returncode':sup.returncode,'step':obs['step'],'partial_capture':True,'training_ready':False},indent=2))
      print('Supervisor error; requesting finalized partial capture',flush=True)
   time.sleep(1)
  if (r/'result.json').exists():
   # Some Kit shutdowns hang after all data is saved; bound only post-result teardown.
   try:sim.wait(timeout=35)
   except subprocess.TimeoutExpired:
    sim.terminate()
    try:sim.wait(timeout=5)
    except subprocess.TimeoutExpired:sim.kill();sim.wait()
  if sup.poll() is None:sup.terminate();sup.wait(timeout=5)
completed=(r/'result.json').exists()
print('FINISHED' if completed else 'INTERRUPTED',name,flush=True)
if not completed:sys.exit(1)
