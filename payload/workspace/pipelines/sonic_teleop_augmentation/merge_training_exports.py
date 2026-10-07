"""Merge compatible LeRobot exports into a new all-train dataset, without video encoding."""
import argparse
import json
import shutil

import numpy as np
import pandas as pd

from common import Path, read, write
from export import stats


def merge(sources, output):
    sources = [Path(x).resolve() for x in sources]
    output = Path(output).resolve()
    assert len(sources) >= 2
    infos = [read(x/'meta/info.json') for x in sources]
    modality = read(sources[0]/'meta/modality.json')
    tasks = (sources[0]/'meta/tasks.jsonl').read_text()
    for source, info in zip(sources, infos):
        assert info['fps'] == 50 and info['features'] == infos[0]['features']
        assert read(source/'meta/modality.json') == modality
        assert (source/'meta/tasks.jsonl').read_text() == tasks
        assert read(source/'meta/loader_smoke_test.json')['passed']
    output.mkdir(parents=True, exist_ok=False)
    meta = output/'meta'
    meta.mkdir()
    records, arrays, source_records = [], {}, []
    rows = 0
    for source, info in zip(sources, infos):
        provenance = read(source/'meta/collection_provenance.json')
        proofs = {}
        if provenance.get('source_index'):
            proofs = {str(Path(x['run']).resolve()): x for x in read(Path(provenance['source_index']))}
        episodes = [json.loads(x) for x in (source/'meta/episodes.jsonl').read_text().splitlines()]
        assert len(episodes) == info['total_episodes']
        for original in episodes:
            old_id, ei = original['episode_index'], len(records)
            table = pd.read_parquet(source/info['data_path'].format(episode_chunk=old_id//1000, episode_index=old_id))
            n = len(table)
            assert n == original['length']
            table['episode_index'] = np.full(n, ei, dtype='int64')
            table['index'] = np.arange(rows, rows+n, dtype='int64')
            destination = output/f'data/chunk-{ei//1000:03d}/episode_{ei:06d}.parquet'
            destination.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(destination, index=False)
            for key, feature in info['features'].items():
                if key.startswith(('observation.', 'action.', 'teleop.')) and feature['dtype'] != 'video':
                    arrays.setdefault(key, []).append(np.stack(table[key]))
                elif feature['dtype'] == 'video':
                    video = source/info['video_path'].format(episode_chunk=old_id//1000, episode_index=old_id, video_key=key)
                    new_video = output/f'videos/chunk-{ei//1000:03d}/{key}/episode_{ei:06d}.mp4'
                    new_video.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(video, new_video)
            record = dict(original, episode_index=ei, split='train', training_ready=False)
            # Older export has per-episode replay proof in the collection index.
            if 'replay_status' not in record:
                proof = proofs[str(Path(record['source_run']).resolve())]['replay']
                assert proof['replay_accepted']
                assert read(Path(proof['comparison']))['repeatability_passed']
                record['replay_status'] = 'previously_passed'
            records.append(record)
            rows += n
        source_records.append(dict(dataset=str(source), episodes=len(episodes),
                                   provenance=provenance))
    info = dict(infos[0], total_episodes=len(records), total_frames=rows,
                total_videos=len(records), total_chunks=(len(records)+999)//1000,
                splits={'train':f'0:{len(records)}'})
    write(meta/'info.json', info)
    write(meta/'modality.json', modality)
    write(meta/'stats.json', {key:stats(np.concatenate(values)) for key, values in arrays.items()})
    write(meta/'geometry_splits.json', {'train':list(range(len(records)))})
    (meta/'tasks.jsonl').write_text(tasks)
    (meta/'episodes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    write(meta/'collection_provenance.json', dict(sources=source_records, split='all_train',
          video_reencoded=False, statistics='Recomputed from all merged episodes',
          training_ready=False, visual_temporal_review='pending',
          replay_checked=sum(x['replay_status'] in ['passed', 'previously_passed'] for x in records),
          replay_unchecked=sum(x['replay_status']=='not_sampled' for x in records),
          note='Unselected start variants have first-execution success only; not claimed replayed.'))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', nargs='+', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    print(merge(args.sources, args.output))
