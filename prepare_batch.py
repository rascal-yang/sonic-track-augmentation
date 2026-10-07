"""Prepare the original 50 geometries plus 10 starts using preserved helpers."""
import argparse
from pathlib import Path
from batch import schedule
from collect import mobile_schedule
from common import read, write
from prepare import compile_candidate
from mobile_prepare import compile_mobile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--count', type=int, choices=range(1,61), default=60)
p.add_argument('--random-seed', type=int, default=20260930)
a = p.parse_args()
root = a.output.resolve()
root.mkdir(parents=True, exist_ok=False)
entries = schedule(root, a.random_seed)['samples']
records = []
for entry in entries[:min(a.count,50)]:
    name = entry['geometry_id']
    skill = compile_candidate(root/'skills'/name, entry)
    candidate = compile_mobile(skill, root/'candidates'/(name+'_walk'))
    records.append(dict(entry, candidate=str(candidate), skill=str(skill)))
    write(root/'candidate_index.json', records)
if a.count > 50:
    for parent, name, offset in mobile_schedule(records, extras=True)[:a.count-50]:
        candidate = compile_mobile(parent['skill'], root/'candidates'/name, *offset)
        records.append(dict(name=name, candidate=str(candidate), geometry_id=parent['geometry_id'],
                            robot_start_offset=list(offset), split=parent['split']))
        write(root/'candidate_index.json', records)
write(root/'status.json', dict(status='prepared', candidates=len(records), task_success=False,
                              training_ready=False))
print(root/'candidates')
