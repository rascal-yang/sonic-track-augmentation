"""Independent Isaac physical rollout from the portable reference NPZ."""
from pathlib import Path
import argparse,os,sys
import numpy as np
B=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
a.reference=a.reference.resolve()
with np.load(a.reference,allow_pickle=False) as z:
 n=len(z['time_s']);root=z['root_position_world_m'][0].tolist()
import json
os.environ.update(G1_START_ROOT=json.dumps(root),G1_HEADLESS='1',G1_OUTPUT_DIR=str(B/'runs'/a.name),G1_GROUP='window_scene',G1_ROOM='window_room',G1_DESK_PLUSH='1',GRASP_METHOD='feedback',GRASP_CONTROLLER='sonic_full',G1_AUTO_CAPTURE='0',G1_LIVE_SESSION='1',G1_MAX_STEPS=str(n),G1_GRAIL_ACTIVE='0',G1_REVO2='1',G1_STATIONARY='0',G1_OPTIMIZED='1')
sys.path.insert(0,str(B/'code'))
from trajectory_rollout import install
install(a.reference)
from experiment import run
run(a.name,sonic_planner=True)
