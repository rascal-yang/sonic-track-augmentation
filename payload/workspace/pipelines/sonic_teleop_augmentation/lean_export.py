"""Export minimal training captures; reuse already aligned video without re-encoding."""
import argparse
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
from common import Path, read, write, arrays, sha
from export import GROUPS, TASK, stats


def export(index, output, allow_unreplayed=False):
    episodes = read(index)
    assert episodes
    if allow_unreplayed:
        sampling = read(Path(index).parent/'spotcheck_report.json')
        assert sampling['passed'] and sampling['first_executions'] == len(episodes)
        checked = {r['name'] for r in episodes if r.get('replay_accepted')}
        assert checked == set(sampling['selected']) and checked
        assert all(x['passed'] for x in sampling['checks'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    meta = output/'meta'
    meta.mkdir()
    records, all_values, offsets = [], {}, {}
    rows = 0
    for ei, item in enumerate(episodes):
        run = Path(item['run']).resolve()
        first_result = read(run/'lean_result.json')
        assert first_result['task_success'] and first_result['profile'] == 'generate'
        replay_result = None
        if item.get('replay_accepted'):
            replay = Path(item['replay'])
            replay_result = read(replay/'lean_result.json')
            assert replay_result['task_success'] and replay_result['profile'] == 'replay'
            controller = read(replay/'token_controller.json')
            assert controller['tokens_sha256'] == sha(run/'executed_teleop_tokens.npz')
            assert controller['decoder_sha256'] == read(run/'token_controller.json')['decoder_sha256']
            for name in ['initial_state.npz', 'deployment_parameters.json']:
                assert sha(run/name) == sha(replay/name), name
        else:
            assert allow_unreplayed, 'Unreplayed export requires an explicit sampling policy'
            assert item.get('replay_status') == 'not_sampled', 'Do not export failed or pending replays'
        z = arrays(run/'training_capture.npz')
        executed = arrays(run/'executed_teleop_tokens.npz')
        for key, recorded in [('token', 'sonic_encoded_tokens'),
                              ('left_hand_target', 'revo2_left_active_position_target_rad'),
                              ('right_hand_target', 'revo2_right_active_position_target_rad')]:
            assert np.array_equal(z[key], executed[recorded]), key
        n = len(z['control_index'])
        assert n == first_result['control_steps']
        if replay_result is not None:
            assert n == replay_result['control_steps']
        assert np.array_equal(z['control_index'], np.arange(n))
        names = z['body_joint_names'].tolist()
        parts, offset = [], 0
        for group, group_names in GROUPS.items():
            parts.append(z['state'][:, [names.index(name) for name in group_names]])
            offsets[group] = dict(start=offset, end=offset+len(group_names), original_key='observation.state')
            offset += len(group_names)
        for side, start in [('left', 29), ('right', 35)]:
            parts.append(z['state'][:, start:start+6])
            offsets[side+'_hand'] = dict(start=offset, end=offset+6, original_key='observation.state')
            offset += 6
        values = {'observation.state': np.concatenate(parts, axis=1).astype('float32'),
                  'observation.projected_gravity': z['projected_gravity'].astype('float32'),
                  'action.motion_token': z['token'].astype('float32'),
                  'teleop.left_hand_joints': z['left_hand_target'].astype('float32'),
                  'teleop.right_hand_joints': z['right_hand_target'].astype('float32')}
        assert offset == 41 and all(np.isfinite(v).all() for v in values.values())
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                           '-show_entries', 'stream=nb_frames,width,height,r_frame_rate', '-of', 'json', str(run/'head.mp4')]))['streams'][0]
        assert int(probe['nb_frames']) == n and probe['width'] == 640 and probe['height'] == 480
        assert probe['r_frame_rate'] == '50/1'
        video = output/f'videos/chunk-{ei//1000:03d}/observation.images.ego_view/episode_{ei:06d}.mp4'
        video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run/'head.mp4', video)
        columns = {k: list(v) for k, v in values.items()}
        columns.update(timestamp=np.arange(n, dtype='float32')/50, frame_index=np.arange(n),
                       episode_index=np.full(n, ei), index=np.arange(rows, rows+n),
                       task_index=np.zeros(n, dtype='int64'), camera_source_control_index=np.arange(n)//2*2,
                       camera_age_s=(np.arange(n) % 2)/50)
        path = output/f'data/chunk-{ei//1000:03d}/episode_{ei:06d}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns).to_parquet(path, index=False)
        for k, v in values.items():
            all_values.setdefault(k, []).append(v)
        records.append(dict(episode_index=ei, length=n, tasks=[TASK], source_run=str(run),
                            geometry_id=item.get('geometry_id', run.name), split='train', training_ready=False,
                            replay_status='passed' if replay_result is not None else 'not_sampled',
                            variant_id=item.get('name', run.name), robot_start_offset=item.get('offset')))
        rows += n
    features = {k: dict(dtype='float32', shape=[v[0].shape[1]], names=None) for k, v in all_values.items()}
    features['observation.images.ego_view'] = dict(dtype='video', shape=[480, 640, 3], names=['height','width','channels'],
        info={'video.fps':50,'video.height':480,'video.width':640,'video.codec':'h264','video.pix_fmt':'yuv420p','video.is_depth_map':False,'has_audio':False})
    for k in ['timestamp','frame_index','episode_index','index','task_index','camera_source_control_index','camera_age_s']:
        features[k] = dict(dtype='float32' if k in ['timestamp','camera_age_s'] else 'int64', shape=[1], names=None)
    write(meta/'info.json', dict(codebase_version='v2.1', robot_type='unitree_g1_revo2_sonic', fps=50,
          total_episodes=len(records), total_frames=rows, total_tasks=1, total_videos=len(records),
          total_chunks=(len(records)+999)//1000, chunks_size=1000, splits={'train':f'0:{len(records)}'},
          data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
          video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4', features=features))
    offsets['projected_gravity'] = dict(start=0, end=3, original_key='observation.projected_gravity')
    write(meta/'modality.json', dict(state=offsets, action={
          'motion_token':dict(start=0,end=64,original_key='action.motion_token'),
          'left_hand_joints':dict(start=0,end=6,original_key='teleop.left_hand_joints'),
          'right_hand_joints':dict(start=0,end=6,original_key='teleop.right_hand_joints')},
          video={'ego_view':dict(original_key='observation.images.ego_view')},
          annotation={'human.task_description':dict(original_key='task_index')}))
    write(meta/'stats.json', {k: stats(np.concatenate(v)) for k, v in all_values.items()})
    write(meta/'geometry_splits.json', {'train':list(range(len(records)))})
    (meta/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (meta/'tasks.jsonl').write_text(json.dumps(dict(task_index=0,task=TASK))+'\n')
    write(meta/'collection_provenance.json', dict(training_ready=False, visual_temporal_review='pending',
          capture_profile='training-only', replay_acceptance='terminal success and aggregate guards',
          trajectory_equivalence_required=False, action_dim=76, state_joint_dim=41,
          camera_native_hz=25, control_hz=50, no_depth_or_segmentation=True, source_index=str(Path(index).resolve()),
          per_episode_replay_required=not allow_unreplayed,
          replay_checked=sum(r['replay_status']=='passed' for r in records),
          replay_unchecked=sum(r['replay_status']=='not_sampled' for r in records)))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--index', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    print(export(a.index, a.output))
