import argparse,os
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('trial');parser.add_argument('--method',required=True);parser.add_argument('--controller',default='sonic_full')
parser.add_argument('--prefix',type=Path);parser.add_argument('--group',required=True);parser.add_argument('--grail',action='store_true');a=parser.parse_args();experiment=Path(__file__).resolve().parent.parent
os.environ['G1_GRAIL_ACTIVE']='1' if a.grail else '0'
os.environ['G1_GROUP']=a.group
os.environ.update(G1_OUTPUT_DIR=str(experiment/a.group/'runs'/a.trial),G1_DESK_PLUSH='1',GRASP_METHOD=a.method,GRASP_CONTROLLER=a.controller)
if a.prefix:
    import threading,json,time
    from experiment import write_json
    commands=json.loads(a.prefix.read_text())
    def replay():
        out=Path(os.environ['G1_OUTPUT_DIR'])
        for i,command in enumerate(commands):
            observation=out/f'observation_{i:03d}.json'
            while not observation.exists():time.sleep(.2)
            decision=dict(command,observation_id=i,author='fixed replay of previously calibrated reference; no live GPT decision',wall_time=time.time())
            write_json(out/f'decision_{i:03d}.json',decision)
    threading.Thread(target=replay,daemon=True).start()
from experiment import run
run(a.trial,sonic_planner=True)
