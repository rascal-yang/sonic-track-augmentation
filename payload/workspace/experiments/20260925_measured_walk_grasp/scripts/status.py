from pathlib import Path
import json,time,sys
b=Path(__file__).resolve().parents[1]
for _ in range(45 if '--wait' in sys.argv else 1):
 pending=[]
 for r in sorted((b/'runs').glob('start_*')):
  if not r.is_dir():continue
  for p in r.glob('review_request_*.json'):
   i=int(p.stem.rsplit('_',1)[-1])
   if not (r/f'reviewed_{i:03d}.json').exists():pending.append((r.name,i,json.loads(p.read_text())['label']))
 if pending:break
 time.sleep(1)
print('PENDING',pending)
for r in sorted((b/'runs').glob('start_*')):
 if not r.is_dir():continue
 fs=sorted(r.glob('observation_[0-9][0-9][0-9].json'))
 if fs:
  s=json.loads(fs[-1].read_text());print(r.name,'obs',s['observation_id'],'step',s['step'],'phase',s['phase'],'err',round(s['navigation_error_m'],4),'finalized',(r/'result.json').exists())
print((b/'batch.log').read_text()[-1500:])
