"""Serial, bounded automatic simulation collection with audited success-only index."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from common import HERE, WORKSPACE, BASE, WALK, read, write, sha
from supervisor import Supervisor


def compact_failure(run):
    # Only these generated large intermediates in this run; keep logs, tokens, videos, reports.
    removed=[]
    for name in ['episode_raw.h5','isaac_initial_scene.usda','capture_chunks','synthetic','hdf5','trajectory.jsonl']:
        p=run/name
        if not p.exists() or p.is_symlink():continue
        size=sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.is_dir() else p.stat().st_size
        removed.append(dict(path=name,bytes=size))
        if p.is_dir():shutil.rmtree(p)
        else:p.unlink()
    write(run/'cleanup_manifest.json',dict(removed=removed,total_bytes=sum(x['bytes'] for x in removed),scope='This failed episode only; compact evidence retained'))


def terminate_owned(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGINT)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)


def validate(run):
    checks=[(HERE/'audit.py',[str(run)]),
            (WALK/'audit_capture.py',['--run',str(run)]),
            (BASE/'scripts/validate_brainco.py',['--run',str(run)]),
            (BASE/'scripts/validate_contacts.py',['--run',str(run)])]
    report=[]
    for script,args in checks:
        with (run/(script.stem+'.log')).open('w') as log:
            try:rc=subprocess.run([sys.executable,str(script),*args],stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
            except subprocess.TimeoutExpired:rc=124
        report.append(dict(script=str(script),exit_code=rc))
    write(run/'validation_commands.json',report)
    return all(x['exit_code']==0 for x in report)


def finalize(run, candidate, completed, validation_ok, returncode, error):
    audit=read(run/'audit.json') if (run/'audit.json').exists() else {}
    result=read(run/'result.json') if (run/'result.json').exists() else {}
    success=bool(completed and validation_ok and returncode==0 and not error and not result.get('error') and audit.get('task_success') and audit.get('minimum_pelvis_z_m',0)>.45)
    # Source collector fields are inherited; give the new automatic controller authoritative metadata.
    controller=read(run/'token_controller.json') if (run/'token_controller.json').exists() else {}
    inherited=run/'inherited_metadata';inherited.mkdir(exist_ok=True)
    for name in ['task.json','variant.json','direct_provenance.json','planner_provenance.json','control_definition.json']:
        if (run/name).exists() and not (inherited/name).exists():shutil.copy2(run/name,inherited/name)
    planner=read(run/'planner_provenance.json') if (run/'planner_provenance.json').exists() else {}
    planner.update(IK='No online IK; offline skill-reference adaptation only',low_level=controller.get('controller'),
                   navigation='Independent SONIC reference planning; measured numeric phase gates')
    write(run/'planner_provenance.json',planner)
    write(run/'direct_provenance.json',dict(runtime_IK=False,online_encoder=True,online_planner=True,
          initial_state_restore_once=True,physical_state_reset_during_episode=False,GRAIL_residual=0,
          external_API=False,checkpoint_review='Automatic numeric gates; visual review is separate and pending'))
    write(run/'variant.json',dict(sample=read(candidate/'sample.json'),controller=controller.get('controller'),training_ready=False))
    if result:write(run/'inherited_result.json',result)
    result.update(task_success=success,training_ready=False,control=controller.get('controller','launch failed'),
                  automatic_supervisor=True,external_model_api=False,validation_passed=validation_ok,
                  batch_error=error,exit_code=returncode,authoritative_report='collection_result.json')
    write(run/'result.json',result)
    record=dict(candidate=str(candidate),run=str(run),status='physics_success' if success else ('physics_failed' if audit else ('validation_failed' if (run/'episode_raw.h5').exists() else 'launch_failed')),
                task_success=success,training_ready=False,visual_review='pending',metrics=audit,validation_passed=validation_ok,
                error=error,exit_code=returncode,controller='SONIC planner -> encoder -> decoder; adapted reference skill tokens + separate Revo2 targets')
    guards=sorted(run.glob('guard_*.json'))
    last_guard=read(guards[-1]) if guards else {}
    failure_reasons=[]
    if not success:
        if last_guard.get('action')=='stop':failure_reasons.append(last_guard['reason'])
        if returncode!=0:failure_reasons.append('Process exit '+str(returncode))
        if not validation_ok:failure_reasons.append('See validation_commands.json and audit/validation logs')
        if audit and not audit.get('complete_fast_grasp_sequence'):failure_reasons.append('Incomplete skill token sequence')
        if audit and audit.get('strict_final_hold_s',0)<3:failure_reasons.append('Final strict hold below 3 s')
        if audit and audit.get('monitored_table_force_peak_N',0)>.05:failure_reasons.append('Monitored table contact exceeds threshold')
        if (run/'run.log').exists():
            lines=(run/'run.log').read_text(errors='replace').splitlines()
            failure_reasons += [x for x in lines if 'Error:' in x or 'Exception:' in x][-3:]
    record['failure_reasons']=failure_reasons
    write(run/'collection_result.json',record)
    write(run/'visual_realism_review.json',dict(accepted=False,training_ready=False,temporal_review_status='pending',
          renderer='Inherited RTX realtime DLAA preview, not RT2',note='No automatic visual certification; numeric checks are separate.'))
    write(run/'control_definition.json',dict(controller=record['controller'],automatic_supervisor=True,online_IK=False,
          initial_restore_once=True,physical_state_writes_during_episode=False,training_ready=False))
    write(run/'task.json',dict(instruction='Walk to the bottle, grasp and lift it.',task_success=success,training_ready=False,
          source_candidate=str(candidate),supervisor='Automatic numeric phase and contact gates; no human review claims'))
    write(run/'reflection.json',dict(observed_status=record['status'],evidence='collection_result.json',
          failure_reasons=failure_reasons,hypothesis=read(candidate/'plan.json')['hypothesis'],changed_parameters=read(candidate/'sample.json'),
          next_step='Review actual docking/contact/visual evidence before widening the sampling distribution.',
          uncertainty='One rollout is not a perturbation success rate; reference geometric feasibility alone is not success.'))
    if (run/'episode_raw.h5').exists():
        import h5py
        try:
            f=h5py.File(run/'episode_raw.h5','r+')
        except (OSError,ValueError) as e:
            record.update(task_success=False,status='validation_failed',hdf5_error=repr(e))
            result.update(task_success=False,hdf5_error=repr(e));write(run/'result.json',result)
            write(run/'collection_result.json',record)
            return record
        with f:
            f.attrs['episode_success']=success;f.attrs['training_ready']=False;f.attrs['controller']=record['controller']
            f.attrs['automatic_supervisor']=True;f.attrs['collection_report']='collection_result.json'
    return record


def run_one(candidate, run, timeout_s):
    candidate=candidate.resolve();run=run.resolve()
    if run.exists():raise RuntimeError('Refuse to overwrite '+str(run))
    if read(candidate/'status.json')['status']!='prepared':raise ValueError('Candidate not prepared')
    supervisor=Supervisor(read(candidate/'plan.json'))
    run.parent.mkdir(parents=True,exist_ok=True)
    console=run.parent/(run.name+'.launcher.log')
    start=time.monotonic();error=None;last_id=-1;stop_sent=False
    with console.open('w') as log:
        proc=subprocess.Popen([sys.executable,str(HERE/'run_episode.py'),'--candidate',str(candidate),'--output',str(run)],
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                if time.monotonic()-start>timeout_s:raise TimeoutError('Episode wall-time budget exceeded')
                files=sorted(run.glob('observation_[0-9][0-9][0-9].json'))
                if files:
                    try:s=read(files[-1])
                    except (ValueError,FileNotFoundError):time.sleep(.1);continue
                    idx=s['observation_id']
                    if idx>last_id:
                        decision,diagnostics=supervisor.decide(s)
                        decision['wall_time']=time.time()
                        write(run/f'guard_{idx:03d}.json',diagnostics)
                        write(run/f'decision_{idx:03d}.json',decision)
                        last_id=idx;stop_sent=decision['action']=='stop'
                        if idx%10==0 or decision['action']!='continue':print(json.dumps(diagnostics),flush=True)
                time.sleep(.2)
        except BaseException as e:
            error=repr(e);terminate_owned(proc)
            if isinstance(e,KeyboardInterrupt):
                run.mkdir(exist_ok=True);write(run/'interrupted.json',dict(error=error));raise
        finally:terminate_owned(proc)
    run.mkdir(exist_ok=True)
    completed=(run/'result.json').exists()
    validation_ok=validate(run) if (run/'episode_raw.h5').exists() else False
    record=finalize(run,candidate,completed,validation_ok,proc.returncode,error)
    if record['task_success']:
        with (run/'export.log').open('w') as log:
            try:subprocess.run([sys.executable,str(HERE/'export.py'),str(run)],stdout=log,stderr=subprocess.STDOUT,check=True)
            except subprocess.CalledProcessError as e:
                record.update(status='export_failed',export_error=repr(e));write(run/'collection_result.json',record)
                return record
        record['export']='executed_sonic_tokens.npz';write(run/'collection_result.json',record)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--max-attempts',type=int,default=3);p.add_argument('--target-successes',type=int,default=1)
    p.add_argument('--timeout-s',type=int,default=1200);p.add_argument('--keep-failed-raw',action='store_true')
    p.add_argument('--minimum-free-gb',type=float,default=12.)
    args=p.parse_args()
    if not 1<=args.max_attempts<=1000 or not 1<=args.target_successes<=args.max_attempts or args.timeout_s<30:p.error('Invalid bounded batch parameters')
    args.output.mkdir(parents=True,exist_ok=True)
    candidates=sorted(x.parent for x in args.candidates.glob('candidate_*/status.json') if read(x)['status']=='prepared')
    # One lock across every batch/output on this workstation, never launch overlapping Isaac jobs.
    with (WORKSPACE/'experiments/.sonic_augmentation_gpu.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another augmentation batch owns the GPU lock')
        records=[];successes=0
        for c in candidates:
            r=args.output/c.name
            if r.exists():continue
            if len(records)>=args.max_attempts or successes>=args.target_successes:break
            if shutil.disk_usage(args.output).free<args.minimum_free_gb*1024**3:raise SystemExit('Disk reserve reached')
            record=run_one(c,r,args.timeout_s)
            records.append(record)
            with (args.output/'attempts.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            if record['status']=='physics_success' and record.get('export'):
                successes+=1
                with (args.output/'successes.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            elif record['status'] in ['physics_failed','launch_failed'] and not args.keep_failed_raw:compact_failure(r)
        write(args.output/'batch_summary.json',dict(attempts=len(records),successes=successes,training_ready=False,records=records))
        print(json.dumps(dict(attempts=len(records),successes=successes,output=str(args.output.resolve()))),flush=True)
    if successes<args.target_successes:raise SystemExit(2)

if __name__=='__main__':main()
