"""Three-stage serial collection with resumable auditable phase boundaries."""
import argparse
import fcntl
import shutil
import sys
from pathlib import Path
from common import *
from batch import invoke
from compact import compact
from replay_batch import replay_one
from lossless_pack import repack


ROBOT_START_OFFSETS = [
    (-.02, 0, -2), (.02, 0, 2), (0, -.025, 0),
    (-.015, .015, 1), (.015, -.015, -1),
    (-.02, -.015, 1), (.02, .015, -1),
    (-.01, -.025, 2), (.01, .02, -2), (0, .025, 1),
]


def mobile_schedule(stationary, extras=False):
    if not extras:
        return [(parent, parent['geometry_id']+'_walk', None) for parent in stationary]
    parents = [x for x in stationary if x['split']=='test'][:5]
    if len(parents) != 5:
        raise ValueError('Robot-start evaluation requires all five held-out geometry groups')
    jobs = []
    for i, offset in enumerate(ROBOT_START_OFFSETS):
        parent = parents[i % 5]
        suffix = '_robot' if i < 5 else '_robot_02'
        jobs.append((parent, parent['geometry_id']+suffix, offset))
    return jobs


def collect_mobile(root, stationary, extras=False):
    folder=root/('robot_starts' if extras else 'mobile')
    for name in ['candidates','runs','logs']:(folder/name).mkdir(parents=True,exist_ok=True)
    records=[];failures=0
    for i,(parent,name,offset) in enumerate(mobile_schedule(stationary, extras)):
        record_path=folder/'logs'/f'{name}.json'
        if record_path.exists():
            record=read(record_path)
        else:
            if shutil.disk_usage(folder).free/2**30<10:raise RuntimeError('Disk reserve reached; preserve records and inspect storage')
            c,r=folder/'candidates'/name,folder/'runs'/name
            if c.exists() or r.exists():raise RuntimeError(f'Unfinished prior mobile attempt {name}; inspect before resume')
            record=dict(name=name,geometry_id=parent['geometry_id'],split=parent['split'],run=str(r),task_success=False)
            try:
                candidate=Path(read(Path(parent['run'])/'collection_result.json')['candidate'])
                args=[HERE/'mobile_prepare.py','--skill',candidate,'--output',c]
                if extras:
                    dx,dy,yaw=offset;args+=['--dx',dx,'--dy',dy,'--yaw',yaw]
                assert invoke(args,folder/'logs'/f'{name}_prepare.log',300)==0
                code=invoke([HERE/'mobile_run.py','--candidate',c,'--output',r],folder/'logs'/f'{name}_physics.log',3000)
                record['physics_exit']=code
                if (r/'episode_raw.h5').exists():
                    record['validation_exit']=invoke([HERE/'mobile_validate.py',r],folder/'logs'/f'{name}_validation.log',1500)
                if (r/'collection_result.json').exists():
                    result=read(r/'collection_result.json')
                    record.update(task_success=bool(code==0 and result['task_success']),quality=result['quality'])
                    if result['validation_passed']:
                        record['freed_duplicate_bytes']=compact(r,keep_raw=True)
                        record['lossless_pack']=repack(r)
            except Exception as error:record['error']=repr(error)
            record['training_ready']=False
            write(record_path,record)
        if record['task_success']:
            replay = replay_one(Path(record['run']),root/'replays',name)
            record['replay'] = replay
            write(record_path,record)
            if not replay['replay_accepted']:
                write(root/'replay_status.json',dict(status='review_required',episode=name,result=replay,training_ready=False))
                raise RuntimeError('Independent replay requires review: '+name)
            accepted_path=root/'replay_accepted_index.json'
            accepted={x['name']:x for x in read(accepted_path)} if accepted_path.exists() else {}
            accepted[name]=record
            write(accepted_path,list(accepted.values()))
            write(root/'replay_status.json',dict(status='collecting',accepted=len(accepted),target=60,training_ready=False))
        records.append(record)
        failures=0 if record['task_success'] else failures+1
        write(folder/'success_index.json',[x for x in records if x['task_success']])
        write(folder/'status.json',dict(attempt_count=len(records),success_count=sum(x['task_success'] for x in records),training_ready=False))
        print(name,record['task_success'],flush=True)
        # First mobile failure is a handoff diagnostic, not grounds to launch 49 copies.
        if (i==0 and not record['task_success']) or failures>=3:
            raise RuntimeError('Mobile handoff requires review before continuing; physical failure preserved')
    if not all(x['task_success'] for x in records):raise RuntimeError('Some mobile poses failed; no complete-dataset claim')
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--stationary-only',action='store_true');a=p.parse_args()
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    # Validate the new replay path on an already successful full mobile episode.
    pilot=root/'mobile_pilot_runs/x_plus_010_r1'
    with (W/'experiments/.sonic_augmentation_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        write(root/'phase_status.json',dict(phase='independent_replay_pilot',training_ready=False))
        write(root/'replay_status.json',dict(status='pilot_running',target=60,training_ready=False))
        pilot_result=replay_one(pilot,root/'replay_pilot','mobile_pilot')
    if not pilot_result['replay_accepted']:
        raise RuntimeError('Replay pilot requires review before batch expansion')
    write(root/'phase_status.json',dict(phase='stationary',stationary=len(read(root/'stationary/success_index.json')) if (root/'stationary/success_index.json').exists() else 0,mobile=0,robot_starts=0,training_ready=False))
    # batch.py owns the GPU lock during stationary collection.
    rc=invoke([HERE/'batch.py','--root',root/'stationary','--limit',50,'--efficient-capture'],root/'stationary_collection.log',24*3600)
    stationary=read(root/'stationary/success_index.json') if (root/'stationary/success_index.json').exists() else []
    if rc or len(stationary)!=50:raise RuntimeError(f'Stationary collection incomplete: {len(stationary)}/50; inspect status and failures')
    write(root/'phase_status.json',dict(phase='mobile_pending',stationary=50,mobile=0,robot_starts=0,training_ready=False))
    if a.stationary_only:return
    with (W/'experiments/.sonic_augmentation_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        mobile=collect_mobile(root,stationary)
        write(root/'phase_status.json',dict(stationary=50,mobile=len(mobile),robot_starts=0,training_ready=False))
        extras=collect_mobile(root,stationary,extras=True)
    write(root/'phase_status.json',dict(stationary=50,mobile=len(mobile),robot_starts=len(extras),training_ready=False,visual_review='pending'))
    write(root/'replay_status.json',dict(status='complete',accepted=len(mobile)+len(extras),target=60,training_ready=False,visual_review='pending'))
    for phase,index in [('stationary',stationary),('mobile',mobile),('robot_starts',extras)]:
        for split in ['train','validation','test']:
            selected=[x for x in index if x['split']==split]
            if not selected:continue
            manifest=root/f'{phase}_{split}_index.json';write(manifest,selected)
            output=root/'lerobot'/f'{phase}_{split}'
            if not output.exists():
                assert invoke([HERE/'export.py','--index',manifest,'--output',output],root/f'{phase}_{split}_export.log',3600)==0
            # A directory left by an interrupted export is not evidence of completion.
            assert invoke([HERE/'check_export.py',output],root/f'{phase}_{split}_loader.log',1800)==0


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        import traceback
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1])
            write(root/'collector_error.json',dict(error=repr(error),traceback=traceback.format_exc(),training_ready=False))
        raise
