from pathlib import Path
import os,sys,argparse,json
import numpy as np

HERE=Path(__file__).resolve().parent; W=HERE.parents[1]; B=W/'experiments/20260925_measured_walk_grasp'
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--candidate',required=True);a=p.parse_args()
candidate=HERE/'candidates'/a.candidate
assert (candidate/'geometry_check.json').exists(), 'Run geometric clearance check before physics'
assert json.loads((candidate/'geometry_check.json').read_text())['table_screen_passed']
with np.load(candidate/'grasp_reference.npz') as z: root=z['root_position_world_m'][0].tolist();n=len(z['time_s'])
out=HERE/'runs'/a.name
os.environ.update(G1_START_ROOT=json.dumps(root),G1_HEADLESS='1',G1_OUTPUT_DIR=str(out),G1_GROUP='window_scene',G1_ROOM='window_room',G1_DESK_PLUSH='1',GRASP_METHOD='feedback',GRASP_CONTROLLER='sonic_full',G1_AUTO_CAPTURE='0',G1_LIVE_SESSION='1',G1_MAX_STEPS=str(n),G1_GRAIL_ACTIVE='0',G1_REVO2='1',G1_STATIONARY='0',G1_OPTIMIZED='1')
sys.path.insert(0,str(B/'code'))
from token_controller import install
install(candidate/'grasp_tokens.npz',Path('__RUNTIME__/models/release/model_decoder.onnx'),candidate/'deployment_parameters.json',W/'g1_closed_loop/.deps')
from experiment import run
run(a.name,sonic_planner=True)
