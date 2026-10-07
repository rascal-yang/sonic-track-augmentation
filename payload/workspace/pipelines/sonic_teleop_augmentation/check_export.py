"""Integration check using the installed repository's real GR00T episode loader."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from common import read,write,HERE
p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('--groot',type=Path,required=True);p.add_argument('--video-backend',choices=['ffmpeg','opencv','opencv_rgb_sequential'],default='opencv_rgb_sequential');a=p.parse_args()
sys.path.insert(0,str(HERE/'.export_deps'));sys.path.insert(0,str(a.groot))
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
if a.video_backend == 'opencv_rgb_sequential':
 import cv2
 import gr00t.data.dataset.lerobot_episode_loader as loader_module
 def sequential_rgb(video_path,indices,video_backend=None,video_backend_kwargs=None):
  indices=np.asarray(indices,dtype=np.int64)
  assert indices.ndim==1 and len(indices)>0 and indices.min()>=0
  cap=cv2.VideoCapture(str(video_path));assert cap.isOpened(),video_path
  slots={}
  for slot,index in enumerate(indices):slots.setdefault(int(index),[]).append(slot)
  result=None
  try:
   for index in range(int(indices.max())+1):
    ok,frame=cap.read();assert ok,(video_path,index)
    if index in slots:
     rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
     if result is None:result=np.empty((len(indices),*rgb.shape),dtype=rgb.dtype)
     for slot in slots[index]:result[slot]=rgb
  finally:cap.release()
  return result
 loader_module.get_frames_by_indices=sequential_rgb
loader=LeRobotEpisodeLoader(a.dataset,MODALITY_CONFIGS['unitree_g1_sonic'],video_backend=a.video_backend)
reports=[]
for ei,meta in enumerate(loader.episodes_metadata):
 frame=loader[ei]
 raw=pd.read_parquet(a.dataset/f'data/chunk-{ei//1000:03d}/episode_{ei:06d}.parquet')
 z=np.load(Path(meta['source_run'])/'executed_teleop_tokens.npz')
 assert len(frame)==len(z['time_s'])
 for name,expected in [('motion_token',z['sonic_encoded_tokens']),('left_hand_joints',z['revo2_left_active_position_target_rad']),('right_hand_joints',z['revo2_right_active_position_target_rad'])]:
  assert np.array_equal(np.stack(frame['action.'+name]),expected.astype(np.float32))
 assert np.stack(frame['video.ego_view']).shape==(len(frame),480,640,3)
 assert np.all(raw.camera_source_control_index<=raw.frame_index)
 assert np.max(raw.camera_age_s)<=.020001
 reports.append(dict(episode=ei,frames=len(frame),token_dim=64,left_hand_dim=6,right_hand_dim=6,video_shape=[480,640,3],labels_match_execution=True))
 print(f'Checked episode {ei+1}/{len(loader.episodes_metadata)}',flush=True)
assert loader.get_dataset_statistics()['action']['motion_token']['mean']
write(a.dataset/'meta/loader_smoke_test.json',dict(passed=True,loader=str(a.groot/'gr00t/data/dataset/lerobot_episode_loader.py'),backend=a.video_backend,episodes=reports,training_run_performed=False))
print('GR00T loader check passed:',len(reports),'episodes')
