"""Independent physical replay for each generated mobile episode; resumable audits."""
import argparse,fcntl,shutil,sys
from common import *
from batch import invoke
from compact import compact


def replay_one(first,root,name):
    root=Path(root);first=Path(first).resolve()
    for folder in ['candidates','runs','logs']:(root/folder).mkdir(parents=True,exist_ok=True)
    c,r=root/'candidates'/name,root/'runs'/name
    status=root/'logs'/f'{name}.json'
    if status.exists():return read(status)
    if c.exists() or r.exists():raise RuntimeError('Unfinished replay requires inspection: '+name)
    if shutil.disk_usage(root).free/2**30<6:raise RuntimeError('Replay disk reserve reached')
    record=dict(first_execution=str(first),replay=str(r),replay_accepted=False,training_ready=False)
    try:
        assert invoke([HERE/'replay_prepare.py','--first',first,'--output',c],root/'logs'/f'{name}_prepare.log',300)==0
        code=invoke([HERE/'replay_run.py','--candidate',c,'--output',r],root/'logs'/f'{name}_physics.log',3000)
        record['physics_exit']=code
        if (r/'episode_raw.h5').exists():
            record['validation_exit']=invoke([HERE/'replay_validate.py',r],root/'logs'/f'{name}_validation.log',1500)
        if (r/'collection_result.json').exists():
            result=read(r/'collection_result.json');record['replay_accepted']=bool(code==0 and result.get('replay_accepted'))
            record['quality']=result['quality']
            if (r/'replay_comparison.json').exists():record['comparison']=str(r/'replay_comparison.json')
            if result['validation_passed']:
                # The source training capture stays intact. Replay retains all non-pixel
                # state/action evidence and videos after exact numerical comparison.
                record['freed_duplicate_bytes']=compact(r,keep_raw=False)
    except Exception as error:record['error']=repr(error)
    write(status,record)
    return record


def replay_all(root,records):
    accepted=[]
    for source in records:
        if not source['task_success']:continue
        result=replay_one(Path(source['run']),root/'replays',source['name'])
        entry=dict(source,replay=result)
        if result['replay_accepted']:accepted.append(entry)
        write(root/'replay_accepted_index.json',accepted)
        if not result['replay_accepted']:
            write(root/'replay_status.json',dict(status='review_required',accepted=len(accepted),failed=result,training_ready=False))
            raise RuntimeError('Independent replay did not pass: '+source['name'])
    write(root/'replay_status.json',dict(status='complete',accepted=len(accepted),training_ready=False,visual_review='pending'))
    return accepted

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--first',required=True,type=Path);p.add_argument('--root',required=True,type=Path);p.add_argument('--name',required=True)
    a=p.parse_args()
    with (W/'experiments/.sonic_augmentation_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        print(replay_one(a.first,a.root,a.name),flush=True)
