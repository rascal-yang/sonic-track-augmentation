"""Export audited G1/Revo2 executions to LeRobot v2.1 for GR00T SONIC actions.
Exports remain research candidates until temporal visual acceptance. RGB is rendered
at 25 Hz and causally held at 50 Hz; no future image is paired with an action.
"""
import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from common import read, write

TASK = 'Pick up the bottle from the desk and hold it steadily.'
GROUPS = {
    'left_leg': ['left_'+s+'_joint' for s in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']],
    'right_leg': ['right_'+s+'_joint' for s in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']],
    'waist': ['waist_'+s+'_joint' for s in ['yaw','roll','pitch']],
    'left_arm': ['left_'+s+'_joint' for s in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']],
    'right_arm': ['right_'+s+'_joint' for s in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']],
}


def stats(values):
    return {key: value.tolist() for key, value in {
        'mean': values.mean(0), 'std': values.std(0), 'min': values.min(0), 'max': values.max(0),
        'q01': np.quantile(values,.01,axis=0), 'q99': np.quantile(values,.99,axis=0)}.items()}


def export(index, output):
    episodes = read(index)
    assert episodes and all(e['task_success'] for e in episodes)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    meta = output/'meta'; meta.mkdir()
    rows, all_values, offsets, records = 0, {}, {}, []
    split_indices = {}
    for ei, episode in enumerate(episodes):
        run = Path(episode['run'])
        assert read(run/'collection_result.json')['validation_passed']
        token_file = run/'executed_teleop_tokens.npz'
        z = np.load(token_file)
        with h5py.File(run/'episode_raw.h5') as f:
            n = len(f['observation/timestamp'])
            joint_names = json.loads(f.attrs['joint_names'])
            parts, offset = [], 0
            for group, names in GROUPS.items():
                ids = [joint_names.index(name) for name in names]
                parts.append(f['observation/joint_position'][:,ids] if ids == sorted(ids)
                             else f['observation/joint_position'][:][:,ids])
                offsets[group] = dict(start=offset,end=offset+len(ids),original_key='observation.state')
                offset += len(ids)
            hand_actions = {}
            for side in ['left','right']:
                names = z[f'revo2_{side}_active_joint_names'].tolist()
                assert len(names) == 6
                ids = [json.loads(f.attrs[f'brainco_{side}_joint_names']).index(name) for name in names]
                parts.append(f[f'brainco/{side}/joint_position'][:][:,ids])
                offsets[f'{side}_hand'] = dict(start=offset,end=offset+6,original_key='observation.state')
                offset += 6
                hand_actions[f'teleop.{side}_hand_joints'] = f[f'brainco/{side}/position_target'][:][:,ids].astype(np.float32)
                assert np.array_equal(hand_actions[f'teleop.{side}_hand_joints'],z[f'revo2_{side}_active_position_target_rad'])
            assert offset == 41
            rotation = Rotation.from_quat(f['observation/root_quaternion_wxyz'][:][:,[1,2,3,0]])
            values = {'observation.state':np.concatenate(parts,axis=1).astype(np.float32),
                      'observation.projected_gravity':rotation.inv().apply(np.tile([0,0,-1],(n,1))).astype(np.float32),
                      'action.motion_token':z['sonic_encoded_tokens'].astype(np.float32), **hand_actions}
            assert values['action.motion_token'].shape == (n,64)
            camera_indices = f['camera/head/control_index'][:]
            selected = np.searchsorted(camera_indices,np.arange(n),side='right')-1
            assert selected.min() >= 0
            age = np.arange(n)-camera_indices[selected]
            assert age.max() <= 1
            video = output/f'videos/chunk-{ei//1000:03d}/observation.images.ego_view/episode_{ei:06d}.mp4'
            video.parent.mkdir(parents=True, exist_ok=True)
            cmd = ['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                   '-s','640x480','-r','50','-i','pipe:0','-an','-c:v','libx264','-crf','18',
                   '-preset','fast','-pix_fmt','yuv420p','-threads','2',str(video)]
            child = subprocess.Popen(cmd,stdin=subprocess.PIPE)
            try:
                prior = -1
                for ci in selected:
                    if ci != prior:
                        pixels = f['camera/head/rgb'][ci].tobytes(); prior = ci
                    child.stdin.write(pixels)
                child.stdin.close()
                assert child.wait() == 0
            except BaseException:
                child.kill(); child.wait(); raise
        columns = {key:list(value) for key,value in values.items()}
        columns.update(timestamp=np.arange(n,dtype=np.float32)/50,
                       frame_index=np.arange(n),episode_index=np.full(n,ei),
                       index=np.arange(rows,rows+n),task_index=np.zeros(n,dtype=np.int64),
                       camera_source_control_index=camera_indices[selected],camera_age_s=age/50)
        path = output/f'data/chunk-{ei//1000:03d}/episode_{ei:06d}.parquet'
        path.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(columns).to_parquet(path,index=False)
        for key,value in values.items(): all_values.setdefault(key,[]).append(value)
        records.append(dict(episode_index=ei,length=n,tasks=[TASK],geometry_id=episode.get('geometry_id',run.name),
                            split=episode.get('split','pilot'),source_run=str(run),training_ready=False))
        split_indices.setdefault(episode.get('split','pilot'),[]).append(ei)
        rows += n
    features = {key:dict(dtype='float32',shape=[value[0].shape[1]],names=None) for key,value in all_values.items()}
    features['observation.images.ego_view'] = dict(dtype='video',shape=[480,640,3],names=['height','width','channels'],
        info={'video.fps':50,'video.height':480,'video.width':640,'video.codec':'h264','video.pix_fmt':'yuv420p','video.is_depth_map':False,'has_audio':False})
    for key in ['timestamp','frame_index','episode_index','index','task_index','camera_source_control_index','camera_age_s']:
        features[key] = dict(dtype='float32' if key in ['timestamp','camera_age_s'] else 'int64',shape=[1],names=None)
    write(meta/'info.json',dict(codebase_version='v2.1',robot_type='unitree_g1_revo2_sonic',fps=50,
        total_episodes=len(records),total_frames=rows,total_tasks=1,total_videos=len(records),
        total_chunks=(len(records)+999)//1000,chunks_size=1000,splits={'train':f'0:{len(records)}'},
        data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',features=features))
    offsets['projected_gravity'] = dict(start=0,end=3,original_key='observation.projected_gravity')
    write(meta/'modality.json',dict(state=offsets,action={
        'motion_token':dict(start=0,end=64,original_key='action.motion_token'),
        'left_hand_joints':dict(start=0,end=6,original_key='teleop.left_hand_joints'),
        'right_hand_joints':dict(start=0,end=6,original_key='teleop.right_hand_joints')},
        video={'ego_view':dict(original_key='observation.images.ego_view')},
        annotation={'human.task_description':dict(original_key='task_index')}))
    write(meta/'stats.json',{key:stats(np.concatenate(values)) for key,values in all_values.items()})
    (meta/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    (meta/'tasks.jsonl').write_text(json.dumps(dict(task_index=0,task=TASK))+'\n')
    write(meta/'geometry_splits.json',split_indices)
    write(meta/'collection_provenance.json',dict(training_ready=False,visual_temporal_review='pending',
        action_dim=76,body_token_dim=64,hand_dim_per_side=6,state_joint_dim=41,
        camera_native_hz=25,control_hz=50,camera_alignment='causal zero-order hold; age 0 or 20ms',
        video_encoding='h264 CRF18 yuv420p from raw RGB; raw lossless RGB retained in source',
        source_index=str(Path(index).resolve()),split_note='Use geometry_splits.json to make separate dataset roots before training; info train range is storage coverage, not approved training split.'))
    return output


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(export(a.index,a.output))
