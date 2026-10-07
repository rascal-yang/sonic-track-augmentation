from pathlib import Path
import os,sys,argparse,json
from common import *
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
c=a.candidate.resolve();out=a.output.resolve();plan=read(c/'plan.json')
for name,digest in read(c/'prepared_files.json').items():assert sha(c/name)==digest,name
os.environ.update(G1_AUG_CANDIDATE=str(c),G1_START_ROOT=json.dumps(plan['start_root_m']),G1_HEADLESS='1',G1_OUTPUT_DIR=str(out),G1_GROUP='window_scene',G1_ROOM='window_room',G1_DESK_PLUSH='1',GRASP_METHOD='feedback',GRASP_CONTROLLER='sonic_full',G1_AUTO_CAPTURE='1',G1_LIVE_SESSION='1',G1_MAX_STEPS='1800',G1_GRAIL_ACTIVE='0',G1_REVO2='1',G1_STATIONARY='0',G1_OPTIMIZED='1')
sys.path.insert(0,str(BASE/'code'));sys.path.insert(0,str(HERE))
from mobile_supervisor import install as install_policy
install_policy()
from mobile_controller import install
install()
from capture_efficiency import install as install_efficiency
install_efficiency()
# The mobile runtime registers handoff/start_grasp checkpoint actions.
sys.modules['runtime_adapter']=module('mobile_runtime_adapter',W/'pipelines/sonic_mimicgen_augmentation/runtime_adapter.py')
from experiment import run
run(out.name,sonic_planner=True,seed=17)
