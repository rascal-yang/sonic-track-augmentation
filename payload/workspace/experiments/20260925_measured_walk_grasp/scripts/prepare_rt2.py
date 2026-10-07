"""Prepare a manifest for the repository RT2 renderer; no physics advancement."""
from pathlib import Path
import json,sys,h5py
B=Path(__file__).resolve().parents[1];r=B/'runs'/sys.argv[1]
with h5py.File(r/'episode_raw.h5') as f:
 scene_names=list(f.get('scene',{})); assert set(scene_names)<= {'table','plush','basket'},scene_names
 manifest={'source_raw':str(r/'episode_raw.h5'),'stage':str(r/'isaac_initial_scene.usda'),'control_stride':2,'env_prim':'/World/envs/env_0','object_prim':'/World/envs/env_0/Object','camera_prim':'/World/Capture/head','scene_prims':{k:{'table':'/World/envs/env_0/Table','plush':'/World/envs/env_0/Room/Plush','basket':'/World/envs/env_0/Room/Scene/Basket'}[k] for k in scene_names}}
assert Path(manifest['stage']).exists()
(r/'rt2_episode.json').write_text(json.dumps(manifest,indent=2));print(r/'rt2_episode.json')
