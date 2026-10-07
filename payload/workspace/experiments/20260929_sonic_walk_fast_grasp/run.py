from pathlib import Path
import os,sys,json,argparse
H=Path(__file__).resolve().parent;W=H.parents[1];B=W/'experiments/20260925_measured_walk_grasp'
p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args();plan=json.loads((H/'plan.json').read_text())
os.environ.update(G1_START_ROOT=json.dumps(plan['start_root_m']),G1_HEADLESS='1',G1_OUTPUT_DIR=str(H/'runs'/a.name),G1_GROUP='window_scene',G1_ROOM='window_room',G1_DESK_PLUSH='1',GRASP_METHOD='feedback',GRASP_CONTROLLER='sonic_full',G1_AUTO_CAPTURE='0',G1_LIVE_SESSION='1',G1_MAX_STEPS='1800',G1_GRAIL_ACTIVE='0',G1_REVO2='1',G1_STATIONARY='0',G1_OPTIMIZED='1')
sys.path.insert(0,str(B/'code'))
sys.path.insert(0,str(H))
from controller import install
install()
from experiment import run
run(a.name,sonic_planner=True)
