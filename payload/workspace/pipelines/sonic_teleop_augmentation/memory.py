"""Append scoped evidence to shared grasp experience without promoting untested claims."""
import fcntl,json
from datetime import datetime,timezone
from pathlib import Path
from common import W,read


def record(run):
    r=Path(run).resolve();result=read(r/'collection_result.json')
    path=W/'policy_memory/grasp_learning/attempts.jsonl'
    entry=dict(id='teleop_aug50_'+r.parent.parent.name+'_'+r.name,time=datetime.now(timezone.utc).isoformat(),
        prompt_version='human_prior_v6',stage='physics',object='23cm rigid bottle; same112x60x75cm desk',
        robot='130cm G1; bilateral Revo2;50Hz encoder/decoder,200Hz physics',source_reference=result.get('candidate'),
        code_snapshot=str(r/'token_source_snapshot'),human_strategy='Preserve deep palm wrap; adapt object-centric approach',
        hypothesis='Bounded Cartesian reference changes preserve successful grip through SONIC',
        changed_parameters=result.get('sample'),predicted_observations='Complete lift, final hold>=3s; opposed grasp; no monitored obstacle contact',
        observed_metrics=result.get('quality'),evidence_paths=[str(r/'collection_result.json'),str(r/'reflection.json')],
        result='success' if result['task_success'] else 'failure',reflection=read(r/'reflection.json'),
        experience_status='observed',applicability='This geometry and controller only; visual temporal review pending')
    with path.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
        if any(json.loads(line).get('id')==entry['id'] for line in f if line.strip()):return
        f.write(json.dumps(entry,ensure_ascii=False)+'\n')
