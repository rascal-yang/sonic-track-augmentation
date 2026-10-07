"""One physical episode; caller serializes GPU jobs and supplies automatic decisions."""
from pathlib import Path
import os
import sys
import argparse
from common import HERE, WORKSPACE, BASE, WALK, read, module, sha, MODEL_ROOT

p=argparse.ArgumentParser();p.add_argument('--candidate',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
a=p.parse_args();candidate=a.candidate.resolve();out=a.output.resolve()
if read(candidate/'status.json')['status']!='prepared':raise SystemExit('Candidate must pass preparation')
if out.exists():raise SystemExit('Output exists; refusing overwrite')
for rel,digest in read(candidate/'prepared_files.json').items():
    if sha(candidate/rel)!=digest:raise SystemExit('Prepared input changed: '+rel)
provenance=read(candidate/'provenance.json')
for name,rel in [('encoder','policy/release/model_encoder.onnx'),('decoder','policy/release/model_decoder.onnx'),('planner','planner/target_vel/V2/planner_sonic.onnx')]:
    if sha(MODEL_ROOT/rel)!=provenance[name+'_sha256']:raise SystemExit('Model changed: '+name)
plan=read(candidate/'plan.json')
os.environ.update(G1_AUG_CANDIDATE=str(candidate),G1_START_ROOT=__import__('json').dumps(plan['start_root_m']),
    G1_HEADLESS='1',G1_OUTPUT_DIR=str(out),G1_GROUP='window_scene',G1_ROOM='window_room',G1_DESK_PLUSH='1',
    GRASP_METHOD='feedback',GRASP_CONTROLLER='sonic_full',G1_AUTO_CAPTURE='0',G1_LIVE_SESSION='1',
    G1_MAX_STEPS='2000',G1_GRAIL_ACTIVE='0',G1_REVO2='1',G1_STATIONARY='0',G1_OPTIMIZED='1')
sys.path.insert(0,str(BASE/'code'));sys.path.insert(0,str(HERE))
from controller import install
install()
# Use frozen successful adapter, including physics contact latch and capture path.
sys.modules['runtime_adapter']=module('frozen_runtime_adapter',HERE/'runtime_adapter.py')
from experiment import run
run(out.name,sonic_planner=True,seed=17)
