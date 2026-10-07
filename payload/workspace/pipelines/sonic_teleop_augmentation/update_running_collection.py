"""Replace only the waiting coordinator; let its independently running batch finish."""
import argparse
import fcntl
import shutil
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from common import HERE, W, read, write


def command(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')


def birth(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--wait-pid',type=int)
    p.add_argument('--birth')
    p.add_argument('--apply',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if a.wait_pid:
        while True:
            try:
                if birth(a.wait_pid)!=a.birth:break
                state=Path(f'/proc/{a.wait_pid}/stat').read_text().rsplit(')',1)[1].split()[0]
                if state=='Z':break
            except FileNotFoundError:break
            time.sleep(5)
        os.execv(sys.executable,[sys.executable,str(HERE/'collect.py'),'--root',str(root)])
    previous=read(root/'collector_process.json');pid=int(previous['pid'])
    if not Path(f'/proc/{pid}/cmdline').exists():
        # A stale PID alone does not establish inactivity across PID namespaces.
        # Probe the shared GPU lock before touching any incomplete output.
        with (W/'experiments/.sonic_augmentation_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not a.apply:
                print('Previous coordinator absent; shared GPU lock is free.');return
            archive=root/'coordinator_updates';archive.mkdir(exist_ok=True)
            write(archive/f'previous_{pid}.json',previous)
            interrupted=[]
            stationary=root/'stationary'
            for candidate in sorted((stationary/'candidates').iterdir()):
                if (stationary/'logs'/f'{candidate.name}.json').exists():continue
                destination=root/'interrupted'/f'{candidate.name}_{time.time_ns()}'
                destination.mkdir(parents=True)
                shutil.move(str(candidate),str(destination/'candidate'))
                run=stationary/'runs'/candidate.name
                if run.exists():shutil.move(str(run),str(destination/'run'))
                write(destination/'interruption.json',dict(reason='Previous collector unavailable and shared GPU lock free; preserve incomplete attempt before retry',task_success=False,grasp_outcome='unknown'))
                interrupted.append(str(destination))
            with (root/'full_collection_10starts.log').open('w') as log:
                proc=subprocess.Popen([sys.executable,str(HERE/'collect.py'),'--root',str(root)],stdout=log,stderr=subprocess.STDOUT,cwd=W,start_new_session=True)
            write(root/'collector_process.json',dict(pid=proc.pid,command=proc.args,status='resumed',previous_coordinator_pid=pid,
                log=str(root/'full_collection_10starts.log'),target_stationary=50,target_mobile=50,target_robot_starts=10,training_ready=False))
            write(archive/f'update_{proc.pid}.json',dict(change='Require independent replay for all60 mobile episodes;10 robot starts retained',previous_coordinator_missing=True,gpu_lock_was_free=True,preserved_interrupted_attempts=interrupted))
            print('Resumed collection with60-episode replay checks; incomplete evidence preserved. PID',proc.pid,flush=True)
        return
    assert str(HERE/'collect.py').encode() in command(pid), 'Expected live original coordinator'
    assert str(root).encode() in command(pid), 'Unexpected experiment root'
    children=[int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    assert len(children)==1, 'Update requires coordinator waiting on its single stationary batch'
    child=children[0]
    assert str(HERE/'batch.py').encode() in command(child)
    assert str(root/'stationary').encode() in command(child)
    stamp=birth(child)
    print(dict(coordinator=pid,stationary_batch=child,target_robot_starts=10),flush=True)
    if not a.apply:return
    archive=root/'coordinator_updates';archive.mkdir(exist_ok=True)
    write(archive/f'previous_{pid}.json',previous)
    # Deliberately signal only the coordinator PID, never its process group or simulator.
    os.kill(pid,signal.SIGTERM)
    with (root/'full_collection_10starts.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--root',str(root),
            '--wait-pid',str(child),'--birth',stamp],stdout=log,stderr=subprocess.STDOUT,
            cwd=W,start_new_session=True)
    write(root/'collector_process.json',dict(pid=proc.pid,command=proc.args,
        status='waiting_for_existing_stationary_batch',existing_batch_pid=child,
        previous_coordinator_pid=pid,log=str(root/'full_collection_10starts.log'),
        target_stationary=50,target_mobile=50,target_robot_starts=10,training_ready=False))
    write(archive/f'update_{proc.pid}.json',dict(change='Require independent replay for all60 mobile episodes; other motion settings unchanged',
          old_coordinator_pid=pid,new_coordinator_pid=proc.pid,preserved_stationary_batch_pid=child))
    print('Coordinator replaced; existing stationary batch preserved. New PID',proc.pid,flush=True)


if __name__=='__main__':main()
