"""Fresh simulator, fixed saved tokens and hands, live decoder; no encoder/planner."""
import argparse,os,sys,json
from common import *
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
c=a.candidate.resolve();out=a.output.resolve();assert not out.exists()
for name,digest in read(c/'prepared_files.json').items():assert sha(c/name)==digest,name
assert sha(MODEL/'model_decoder.onnx')==read(c/'provenance.json')['decoder_sha256']
z=arrays(c/'grasp_tokens.npz');init=arrays(c/'initial_state.npz')
os.environ.update(G1_START_ROOT=json.dumps(init['root_position'].tolist()),G1_HEADLESS='1',G1_OUTPUT_DIR=str(out),G1_GROUP='window_scene',G1_ROOM='window_room',G1_DESK_PLUSH='1',GRASP_METHOD='feedback',GRASP_CONTROLLER='sonic_full',G1_AUTO_CAPTURE='1',G1_LIVE_SESSION='1',G1_MAX_STEPS=str(len(z['time_s'])),G1_GRAIL_ACTIVE='0',G1_REVO2='1',G1_STATIONARY='0',G1_OPTIMIZED='1')
sys.path.insert(0,str(BASE/'code'));sys.path.insert(0,str(HERE))
from automatic_policy import install as install_guard
install_guard()
from replay_controller import install
install(c/'grasp_tokens.npz',MODEL/'model_decoder.onnx',c/'deployment_parameters.json',W/'g1_closed_loop/.deps')
from capture_efficiency import install as install_capture
install_capture()
from experiment import run
run(out.name,sonic_planner=True,seed=17)
