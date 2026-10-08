"""Minimal generation or terminal-only independent replay, unchanged motion control."""
import argparse
import json
import os
import sys
import fcntl
from common import BASE, HERE, MODEL, W, arrays, module, read, sha

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['generate', 'replay'], required=True)
parser.add_argument('--candidate', type=__import__('pathlib').Path, required=True)
parser.add_argument('--output', type=__import__('pathlib').Path, required=True)
parser.add_argument('--benchmark-reference', type=__import__('pathlib').Path)
args = parser.parse_args()
gpu_lock = (W/'experiments/.sonic_augmentation_gpu.lock').open('a')
fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
candidate, out = args.candidate.resolve(), args.output.resolve()
assert not out.exists(), out
for name, digest in read(candidate/'prepared_files.json').items():
    assert sha(candidate/name) == digest, name
initial = arrays(candidate/'initial_state.npz')
maximum = 1800 if args.mode == 'generate' else len(arrays(candidate/'grasp_tokens.npz')['time_s'])
os.environ.update(G1_LEAN_MODE=args.mode, G1_AUG_CANDIDATE=str(candidate),
    G1_START_ROOT=json.dumps(initial['root_position'].tolist()), G1_HEADLESS='1',
    G1_OUTPUT_DIR=str(out), G1_GROUP='window_scene', G1_ROOM='window_room',
    G1_DESK_PLUSH='1', GRASP_METHOD='feedback', GRASP_CONTROLLER='sonic_full',
    G1_AUTO_CAPTURE='1', G1_LIVE_SESSION='1', G1_MAX_STEPS=str(maximum),
    G1_GRAIL_ACTIVE='0', G1_REVO2='1', G1_STATIONARY='0', G1_OPTIMIZED='1')
os.environ.pop('G1_LEAN_STOP_REASON', None)
if args.benchmark_reference:
    os.environ['G1_LEAN_BENCHMARK_REFERENCE'] = str(args.benchmark_reference.resolve())
else:
    os.environ.pop('G1_LEAN_BENCHMARK_REFERENCE', None)
sys.path.insert(0, str(BASE/'code'))
sys.path.insert(0, str(HERE))
sys.modules['training_capture'] = module('lean_training_capture', HERE/'lean_capture.py')
sys.modules['contact_watch_full'] = module('contact_watch_full', BASE/'code/contact_watch.py')
sys.modules['contact_watch'] = module('lean_contact_watch', HERE/'lean_contact_watch.py')
sys.modules['runtime_adapter'] = module('lean_runtime_adapter', HERE/'lean_runtime_adapter.py')
if args.mode == 'generate':
    from mobile_supervisor import install as policy
    from mobile_controller import install as controller
    policy()
    controller()
else:
    if (candidate/'provenance.json').exists():
        assert sha(MODEL/'model_decoder.onnx') == read(candidate/'provenance.json')['decoder_sha256']
    from automatic_policy import install as policy
    from replay_controller import install as controller
    policy()
    controller(candidate/'grasp_tokens.npz', MODEL/'model_decoder.onnx',
               candidate/'deployment_parameters.json', W/'g1_closed_loop/.deps')
from render_settings import install_launcher
install_launcher()
from experiment import run
run(out.name, sonic_planner=True, seed=17)
