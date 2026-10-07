from pathlib import Path
import os,sys,argparse,json
import numpy as np

HERE=Path(__file__).resolve().parent; W=HERE.parents[1]; B=W/'experiments/20260925_measured_walk_grasp'
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);p.add_argument('--efficient-capture',action='store_true');a=p.parse_args()
candidate=Path(a.candidate).resolve()
from common import read,sha
for name,digest in read(candidate/'prepared_files.json').items():assert sha(candidate/name)==digest, name
assert (candidate/'geometry_check.json').exists(), 'Run geometric clearance check before physics'
assert json.loads((candidate/'geometry_check.json').read_text())['table_screen_passed']
with np.load(candidate/'grasp_reference.npz') as z: root=z['root_position_world_m'][0].tolist();n=len(z['time_s'])
out=Path(a.output).resolve()
os.environ.update(G1_START_ROOT=json.dumps(root),G1_HEADLESS='1',G1_OUTPUT_DIR=str(out),G1_GROUP='window_scene',G1_ROOM='window_room',G1_DESK_PLUSH='1',GRASP_METHOD='feedback',GRASP_CONTROLLER='sonic_full',G1_AUTO_CAPTURE='1',G1_LIVE_SESSION='1',G1_MAX_STEPS=str(n),G1_GRAIL_ACTIVE='0',G1_REVO2='1',G1_STATIONARY='0',G1_OPTIMIZED='1')
sys.path.insert(0,str(B/'code'));sys.path.insert(0,str(HERE))
os.environ['G1_AUG_CANDIDATE']=str(candidate)
from automatic_policy import install as install_policy
install_policy()
from token_controller import install
install(candidate/'grasp_tokens.npz',Path('__RUNTIME__/models/release/model_decoder.onnx'),candidate/'deployment_parameters.json',W/'g1_closed_loop/.deps')
if a.efficient_capture:
 from capture_efficiency import install as install_capture_efficiency
 install_capture_efficiency()
from experiment import run
run(a.name,sonic_planner=True)
