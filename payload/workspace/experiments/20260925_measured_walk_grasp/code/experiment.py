"""One G1 physical pickup experiment with a file-based agent control boundary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

CODE_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('G1_SOURCE_ROOT', '__RUNTIME__/workspace/g1_closed_loop'))
HOME_DIR = Path.home()
GRAIL = Path(os.environ.get('GRAIL_ROOT', HOME_DIR / 'Projects/GRAIL'))
SONIC = Path(os.environ['SONIC_ROOT'])
INTERN = Path(os.environ.get('INTERN_ROOT', HOME_DIR / 'code/InternDataEngine'))
KEY = 'pickup_table__apple_0__000'
DATA = ROOT / 'data/pickup_table'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def prepare():
    repo = 'nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL'
    base = f'https://huggingface.co/datasets/{repo}/resolve/main/'
    records = []

    def download(remote):
        local = DATA / remote.removeprefix('data/pickup_table/')
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            print('Downloading', remote, flush=True)
            subprocess.run(['curl', '-L', '--fail', '--retry', '3', '--max-time', '120',
                            '-sS', base + remote, '-o', str(local.with_suffix(local.suffix + '.part'))], check=True)
            local.with_suffix(local.suffix + '.part').replace(local)
        records.append({'path': str(local.relative_to(ROOT)), 'url': base + remote,
                        'bytes': local.stat().st_size,
                        'sha256': hashlib.sha256(local.read_bytes()).hexdigest()})

    for folder, suffix in [('robot', 'pkl'), ('objects', 'pkl'), ('meta', 'pkl'),
                           ('object_usd', 'usd'), ('bps', 'npy')]:
        download(f'data/pickup_table/{folder}/{KEY}.{suffix}')
    download('data/pickup_table/bps/_basis.npy')
    texture_path = f'data/pickup_table/object_usd/textures/{KEY}'
    api = f'https://huggingface.co/api/datasets/{repo}/tree/main/{texture_path}?recursive=true'
    textures = json.loads(subprocess.check_output(['curl', '-L', '--fail', '--retry', '3',
                          '--max-time', '60', '-sS', api], text=True))
    for item in textures:
        if item['type'] == 'file':
            download(item['path'])
    write_json(ROOT / 'data/download_manifest.json', records)
    target = ROOT / 'checkpoints/pnp_table_release'
    target.mkdir(parents=True, exist_ok=True)
    for name in ['config.yaml', 'last.pt']:
        shutil.copy2(SONIC / 'models/pnp_table_release' / name, target / name)
    print('Prepared', len(records), 'files', flush=True)


def run(name, online=False, seed=17, sonic_planner=False):
    online = online or sonic_planner
    driver = Path(os.environ.get('G1_DRIVER_LIB', '/tmp/g1-driver-580.173.02/usr/lib/x86_64-linux-gnu'))
    if driver.exists() and not os.environ.get('G1_DRIVER_REEXEC'):
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(driver) + ':' + env.get('LD_LIBRARY_PATH', '')
        env['G1_DRIVER_REEXEC'] = '1'
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    out = Path(os.environ['G1_OUTPUT_DIR']) if os.environ.get('G1_OUTPUT_DIR') else ROOT / 'runs' / name
    if out.exists():
        raise RuntimeError('Use a new run name; existing experiments are never overwritten')
    out.mkdir(parents=True)
    # Keep complete logs even when the invoking tool disconnects.
    log = (out / 'run.log').open('w', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    write_json(out / 'task.json', {
        'instruction': 'Walk from configured start to fixed standing point, then pick up water bottle using authorized side-grasp hints.',
        'motion_key': KEY, 'agent': 'geometric supervisor; no live GPT API' if os.environ.get('G1_AUTO_CAPTURE') == '1' else 'fixed GPT-designed feedback supervisor, with session visual review; no external GPT API',
        'external_model_api': False, 'max_control_steps': 600,
        'success_definition': 'whole bottle lowest point > table + .02m, hand contact > .2N, object table force < .1N, pelvis > .45m, continuous final hold >=3s; opposed thumb/finger contact at end',
        'note': 'Lift criterion is a proxy; does not certify a stable grasp.'})
    if sonic_planner:
        os.environ['G1_SONIC_PLANNER'] = '1'
    motion_data = DATA
    retry = os.environ.get('G1_COMPARE_PROTOCOL') == 'v3_retry'
    if retry:
        from retry_protocol import prepare_retry
        motion_data = prepare_retry(out, DATA, os.environ['G1_COMPARE_MODE'])
    if online:
        from online_goals import prepare_online
        motion_data = prepare_online(out, SONIC, DATA, seed)
        os.environ['G1_ONLINE_GOALS'] = '1'
        if os.environ.get('G1_DESK_PLUSH'):
            from desk_plush_experiment import prepare_reference
            prepare_reference(motion_data, out)
        task = json.loads((out / 'task.json').read_text())
        task.update(reference_source='SONIC V2 navigation and authorized successful bottle-grasp waypoint hints', seed=seed, max_control_steps=int(os.environ.get("G1_MAX_STEPS",2600)))
        write_json(out / 'task.json', task)
    provenance = {}
    sources = out / 'source_snapshot'
    sources.mkdir()
    for file in ('experiment.py', 'runtime_adapter.py', 'online_goals.py', 'sonic_planner_goals.py', 'reference_comparison.py', 'retry_protocol.py', 'desk_plush_experiment.py', 'calibration_config.py'):
        if (CODE_ROOT / file).exists(): shutil.copy2(CODE_ROOT / file, sources / file)
    for label, path in [('InternDataEngine', INTERN), ('GRAIL', GRAIL), ('SONIC', SONIC)]:
        def git(*args):
            return 'packaged snapshot; see source_manifest.json and per-episode evidence/provenance.json'
        provenance[label] = {'path': str(path), 'commit': git('rev-parse', 'HEAD'),
                             'status': git('status', '--porcelain')}
    write_json(out / 'provenance.json', provenance)
    for file in CODE_ROOT.glob('*.py'):
        shutil.copy2(file, sources / file.name)
    os.environ.update(G1_EXPERIMENT_ROOT=str(ROOT), G1_RUN_DIR=str(out),
                      INTERN_ROOT=str(INTERN), PYTHONUNBUFFERED='1', WANDB_MODE='disabled',
                      VK_ICD_FILENAMES=os.environ.get('VK_ICD_FILENAMES', '/usr/share/vulkan/icd.d/nvidia_icd.json'))
    os.chdir(SONIC)
    sys.path.insert(0, str(SONIC))
    from gear_sonic import eval_agent_trl as evaluation
    from runtime_adapter import install
    install(evaluation, out, INTERN)
    args = {
        '+checkpoint': ROOT / 'checkpoints/pnp_table_release/last.pt',
        '+headless': 'False' if os.environ.get('G1_HEADLESS')=='0' else 'True', '++run_eval_loop': 'True', '++run_once': 'True',
        '++max_render_steps': 601, '++num_envs': 1, '++eval_callbacks': '[]',
        'hydra.run.dir': out / 'hydra',
        '++manager_env.commands.motion.motion_lib_cfg.motion_file': motion_data / 'robot',
        '++manager_env.commands.motion.motion_lib_cfg.object_motion_file': motion_data / 'objects',
        '++manager_env.commands.motion.motion_lib_cfg.bps_dir': motion_data / 'bps',
        '++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys': f'[{KEY}]',
        '++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size': 1,
        '++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank': 0,
        '++manager_env.commands.motion.motion_lib_cfg.multi_thread': 'False',
        '++manager_env.config.object_usd_path': DATA / 'object_usd',
        '++manager_env.config.action_transform_module_cfg': SONIC / 'models/sonic_manipulation_base/model_config.yaml',
        '++manager_env.config.action_transform_module_checkpoint': SONIC / 'models/sonic_manipulation_base/last.pt',
        '++manager_env.config.gpu_collision_stack_size_exp': 28,
        '++manager_env.config.render_results': 'True',
        '++manager_env.config.render_width': 960, '++manager_env.config.render_height': 544,
        '++manager_env.config.save_rendering_dir': out,
        '++manager_env.config.max_render_envs': 1,
        '++manager_env.config.eval_camera_offset': '[2,-2,1]',
        '++manager_env.config.fix_camera_after_first_frame': 'True',
        '++manager_env.config.render_frame_skip': 2,
        '++manager_env.recorders.render_envs._target_': 'gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg',
        '++manager_env.recorders.render_envs.video_save_path': out,
        '++manager_env.recorders.render_envs.video_quality': 6,
    }
    if online:
        args['++manager_env.config.table_size'] = '[1.0,0.6,0.04]'
    if os.environ.get('G1_COMPARE_MODE'):
        if online:
            raise ValueError('Comparison requires the released demonstration')
        args['++seed'] = 17
        args['++max_render_steps'] = 400
        task = json.loads((out / 'task.json').read_text())
        task.update(comparison_mode=os.environ['G1_COMPARE_MODE'], seed=17,
                    max_control_steps=400, agent=('current Codex/GPT session' if
                    os.environ['G1_COMPARE_MODE'] in ('gpt', 'direct') else 'fixed automatic schedule'))
        if os.environ.get('G1_COMPARE_PROTOCOL') in ('v2', 'v3_retry'):
            args['++manager_env.terminations.object_pos_deviation.params.threshold'] = .4
            task.update(protocol='v2', offset_world_m=json.loads(os.environ['G1_COMPARE_OFFSET']),
                        object_deviation_threshold_m=.4)
        if retry:
            from retry_protocol import MAX_STEPS
            args['++max_render_steps'] = MAX_STEPS + 1
            task.update(protocol='v3_retry', max_control_steps=MAX_STEPS,
                        record_only_terminations=['object_pos_deviation', 'hand_table_contact_termination'])
        write_json(out / 'task.json', task)
    if sonic_planner:
        args['++max_render_steps'] = 1001
        task = json.loads((out / 'task.json').read_text())
        task.update(control='SONIC planner + IK + SONIC base', grail_interaction_enabled=os.environ.get('G1_GRAIL_ACTIVE')=='1', max_control_steps=1000)
        write_json(out / 'task.json', task)
    if os.environ.get('G1_DESK_PLUSH'):
        args.update({'++max_render_steps':1501,
            '++manager_env.config.table_position':'[0,0,0]',
            '++manager_env.config.table_quat':'[1,0,0,0]',
            '++manager_env.config.table_size':'[1.1070822,0.5780839,0.024]',
            '++manager_env.config.eval_camera_offset':'[0.80,-2.1,1.0]'})
        task=json.loads((out/'task.json').read_text())
        task.update(instruction='Walk from configured start to fixed station, then use authorized bottle side-grasp hints with actual-state arm IK and SONIC; GRAIL residual zero.',
                    max_control_steps=int(os.environ.get('G1_MAX_STEPS','2200')),protocol='bottle_multistart_fixed_v1',
                    success_definition='Entire collision geometry >table+2cm for .5s, hand contact>.2N, table contact<.1N, pelvis>.45m; validation.json authoritative.',
                    privileged_information=True, external_model_api=False,
                    record_only_terminations=['object_pos_deviation','hand_table_contact_termination'])
        write_json(out/'task.json',task)
    if os.environ.get('G1_AUTO_CAPTURE') == '1':
        task=json.loads((out/'task.json').read_text())
        task.update(agent='geometric feedback supervisor; no live GPT API', max_control_steps=int(os.environ.get('G1_MAX_STEPS','1500')), success_definition='All toy collision vertices above tabletop + 2cm, hand contact > .2N, table contact < .1N, pelvis > .45m, for .5s; success_final requires hold at end.', note='Authoritative labels in validation.json; result.task_success_proxy is inherited origin-lift diagnostic.')
        write_json(out/'task.json', task)
    args['++max_render_steps']=int(os.environ.get('G1_MAX_STEPS','1500'))+1
    args['++manager_env.recorders.dataset_export_dir_path']=str(out/'hdf5')
    args['++seed']=17
    sys.argv = [str(SONIC / 'gear_sonic/eval_agent_trl.py')] + [f'{k}={v}' for k,v in args.items()]
    import hydra
    hydra.main(config_path=str(SONIC / 'gear_sonic/config'), config_name='base_eval',
               version_base='1.1')(evaluation.main.__wrapped__)()


def decide(name, observation, action, steps, reason, goal=None):
    out = Path(os.environ['G1_OUTPUT_DIR']) if os.environ.get('G1_OUTPUT_DIR') else ROOT / 'runs' / name
    observed = out / f'observation_{observation:03d}.json'
    if not observed.exists():
        raise ValueError('Observe before deciding: ' + str(observed))
    target = out / f'decision_{observation:03d}.json'
    if target.exists():
        raise ValueError('Decision already exists')
    if not 1 <= steps <= 200:
        raise ValueError('steps must be 1..200')
    write_json(target, {'observation_id': observation, 'action': action, 'steps': steps,
                        'goal': goal, 'reason': reason, 'author': 'current Codex/GPT session',
                        'wall_time': time.time()})


def verify(name):
    out = Path(os.environ['G1_OUTPUT_DIR']) if os.environ.get('G1_OUTPUT_DIR') else ROOT / 'runs' / name
    video = out / '000000.mp4'
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=codec_name,width,height,nb_frames,duration,r_frame_rate',
        '-of', 'json', str(video)], text=True))
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-'], check=True)
    states = [json.loads(line) for line in (out / 'trajectory.jsonl').read_text().splitlines()]
    observations = sorted(out.glob('observation_*.json'))
    decisions = sorted(out.glob('decision_*.json'))
    assert states and decisions and len(observations) >= len(decisions)
    assert int(probe['streams'][0]['nb_frames']) > 1
    report = {'video': str(video), 'bytes': video.stat().st_size,
              'stream': probe['streams'][0], 'full_decode_passed': True,
              'state_records': len(states), 'observations': len(observations),
              'decisions': len(decisions),
              'result': json.loads((out / 'result.json').read_text())}
    write_json(out / 'validation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('prepare')
    r = sub.add_parser('run'); r.add_argument('name'); r.add_argument('--online', action='store_true'); r.add_argument('--sonic-planner', action='store_true'); r.add_argument('--seed', type=int, default=17)
    v = sub.add_parser('verify'); v.add_argument('name')
    d = sub.add_parser('decide'); d.add_argument('name'); d.add_argument('observation', type=int)
    d.add_argument('action', choices=['execute_pickup', 'continue', 'stop', 'navigate', 'shift_stance', 'reach', 'close_hand', 'open_hand', 'lift'])
    d.add_argument('--steps', type=int, default=100); d.add_argument('--reason', required=True)
    d.add_argument('--goal', type=json.loads, help='World xyz target / wrist_euler_xyz / hand / delta JSON')
    a = p.parse_args()
    if a.cmd == 'prepare': prepare()
    elif a.cmd == 'run': run(a.name, a.online, a.seed, a.sonic_planner)
    elif a.cmd == 'verify': verify(a.name)
    else: decide(a.name, a.observation, a.action, a.steps, a.reason, a.goal)
