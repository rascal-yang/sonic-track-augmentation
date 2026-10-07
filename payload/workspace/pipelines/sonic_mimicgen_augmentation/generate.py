"""Single entry point: sample, compile, physically execute, validate and index."""
import argparse
from pathlib import Path
import subprocess
import sys
from common import HERE, read, write

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--count',type=int,default=10,help='Number of offline candidates')
p.add_argument('--seed',type=int,default=100)
p.add_argument('--profile',choices=['identity','smoke','robot','object_se2'],default='smoke')
p.add_argument('--max-attempts',type=int,default=3,help='Bounded serial physical attempts')
p.add_argument('--target-successes',type=int,default=1)
p.add_argument('--timeout-s',type=int,default=1200)
p.add_argument('--randomize-planner-seed',action='store_true')
p.add_argument('--prepare-only',action='store_true')
p.add_argument('--keep-failed-raw',action='store_true')
a=p.parse_args()
if a.output.exists():p.error('Use a new output directory. For resuming existing candidates use batch.py.')
if not 1<=a.count<=1000 or not 1<=a.max_attempts<=1000 or not 1<=a.target_successes<=a.max_attempts or a.timeout_s<30:p.error('Invalid limits')
a.output.mkdir(parents=True)
write(a.output/'request.json',dict(count=a.count,seed=a.seed,profile=a.profile,max_attempts=a.max_attempts,
      target_successes=a.target_successes,timeout_s=a.timeout_s,prepare_only=a.prepare_only,randomize_planner_seed=a.randomize_planner_seed))
prepared=a.output/'candidates'
rc=subprocess.run([sys.executable,str(HERE/'prepare.py'),'--output',str(prepared),'--count',str(a.count),
                   '--seed',str(a.seed),'--profile',a.profile]+(['--randomize-planner-seed'] if a.randomize_planner_seed else [])).returncode
summary=read(prepared/'preparation_summary.json') if (prepared/'preparation_summary.json').exists() else []
if not any(x['status']=='prepared' for x in summary):raise SystemExit(rc or 1)
if a.prepare_only:raise SystemExit(rc)
command=[sys.executable,str(HERE/'batch.py'),'--candidates',str(prepared),'--output',str(a.output/'runs'),
         '--max-attempts',str(a.max_attempts),'--target-successes',str(a.target_successes),'--timeout-s',str(a.timeout_s)]
if a.keep_failed_raw:command.append('--keep-failed-raw')
raise SystemExit(subprocess.run(command).returncode)
