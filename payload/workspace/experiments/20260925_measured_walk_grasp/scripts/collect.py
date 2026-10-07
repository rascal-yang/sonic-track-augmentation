"""Collect a new physical episode; never overwrite an existing run."""
import argparse,os,sys,faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(120,repeat=True)
from pathlib import Path
B=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--steps',type=int,default=1500);p.add_argument('--grail',action='store_true');p.add_argument('--gui',action='store_true');p.add_argument('--brainco',action='store_true');a=p.parse_args()
import json
protocol=json.loads((B/'protocol.json').read_text())
os.environ['G1_START_ROOT']=json.dumps(protocol['starts'][a.name])
sys.path.insert(0,str(B/'code'))
os.environ.update(G1_HEADLESS='0' if a.gui else '1',G1_OUTPUT_DIR=str(B/'runs'/a.name),G1_GROUP='window_scene',G1_ROOM='window_room',G1_DESK_PLUSH='1',GRASP_METHOD='feedback',GRASP_CONTROLLER='ik_upper_gc',G1_AUTO_CAPTURE='0',G1_LIVE_SESSION='1',G1_MAX_STEPS=str(a.steps),G1_GRAIL_ACTIVE='0')
if a.brainco: os.environ['G1_REVO2']='1'
os.environ['G1_STATIONARY']='1' if a.name.startswith('stationary') else '0'
os.environ['G1_OPTIMIZED']='0' if a.name in ['stationary_001','stationary_002'] else '1'
from experiment import run
run(a.name,sonic_planner=True)
