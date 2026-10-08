"""Thin launcher for the preserved SONIC seed and lean augmentation scripts."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TEXT = {'.py', '.json', '.jsonl', '.yaml', '.yml', '.usda', '.urdf', '.xml',
        '.md', '.txt', '.toml', '.hpp', '.cpp'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_path(path):
    path = Path(path).resolve()
    if 'robot' in str(path):
        raise ValueError('Path must not contain "robot": upstream loader replaces robot with meta.')
    return path


def prepare(runtime):
    runtime = checked_path(runtime)
    manifest = ROOT/'resources.json'
    version = digest(manifest)
    marker = runtime/'repository_runtime.json'
    if runtime.exists():
        recorded = json.loads(marker.read_text()) if marker.is_file() else {}
        if recorded.get('repository') != str(ROOT) or recorded.get('resources_sha256') != version:
            raise RuntimeError('Existing runtime is unrelated or outdated; choose a new directory.')
        integrity = runtime/'runtime_files.json'
        if not integrity.is_file() or digest(integrity) != recorded.get('runtime_files_sha256'):
            raise RuntimeError('Runtime checksum manifest is missing or changed; choose a new directory.')
        for name, expected in json.loads(integrity.read_text())['files'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe runtime path: '+name)
            target = runtime/relative
            if not target.is_file() or digest(target) != expected:
                raise RuntimeError('Runtime checksum mismatch or missing file: '+name)
        return runtime
    files = json.loads(manifest.read_text())['files']
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe resource path: '+name)
        source = ROOT/'payload'/relative
        if not source.is_file() or digest(source) != expected:
            raise RuntimeError('Resource checksum mismatch or missing file: '+name+
                               '; unpack the matching offline resource archive.')
    runtime.mkdir(parents=True)
    for name in files:
        source, target = ROOT/'payload'/name, runtime/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix in TEXT:
            try:
                target.write_text(source.read_text().replace('__RUNTIME__', str(runtime)))
                continue
            except UnicodeDecodeError:
                pass
        # Binaries are copied, so a caller cannot accidentally modify the source archive.
        shutil.copy2(source, target)
    for prepared in runtime.rglob('prepared_files.json'):
        keys = json.loads(prepared.read_text())
        prepared.write_text(json.dumps({name:digest(prepared.parent/name) for name in keys}, indent=2)+'\n')
    (runtime/'workspace/experiments').mkdir(parents=True, exist_ok=True)
    integrity = runtime/'runtime_files.json'
    integrity.write_text(json.dumps({'files':{name:digest(runtime/name) for name in files}}, indent=2)+'\n')
    marker.write_text(json.dumps({'repository':str(ROOT), 'resources_sha256':version,
                                 'runtime_files_sha256':digest(integrity)}, indent=2)+'\n')
    return runtime


def environment(runtime, ort_path=None, seed=None):
    env = os.environ.copy()
    env.update(G1_SOURCE_ROOT=str(runtime/'workspace/g1_closed_loop'),
               SONIC_ROOT=str(runtime/'vendor/SONIC'),
               INTERN_ROOT=str(runtime/'vendor/InternDataEngine'),
               SONIC_PLANNER_MODEL=str(runtime/'models/deploy/planner/target_vel/V2/planner_sonic.onnx'))
    for key in ['G1_BUNDLE_POLICY', 'G1_BUNDLE_MAX_STEPS', 'G1_BUNDLE_RECORD_VIDEO',
                'G1_BUNDLE_ROOT', 'G1_LEAN_BENCHMARK_REFERENCE']:
        env.pop(key, None)
    env['G1_GRASP_SOURCE'] = (str(Path(seed).resolve()) if seed else str(runtime/
        'workspace/experiments/20260929_sonic_grasp_smooth/candidates/safe025'))
    paths = [str(runtime/'vendor/SONIC')]
    if ort_path:
        paths.insert(0, str(Path(ort_path).resolve()))
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    return env


def bind_collection(candidates, output, render_updates=1):
    """Prevent upstream name-only resume from mixing different candidate batches."""
    candidates, output = Path(candidates).resolve(), Path(output).resolve()
    records = {str(path.parent.relative_to(candidates)):digest(path)
               for path in sorted(candidates.glob('*/prepared_files.json'))}
    if not records:
        raise RuntimeError('No prepared candidate manifests found: '+str(candidates))
    request = {'candidates':str(candidates), 'prepared_files_sha256':records,
               'resources_sha256':digest(ROOT/'resources.json'), 'render_updates':render_updates}
    marker = output/'collection_request.json'
    if output.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != request:
            raise RuntimeError('Existing collection has different or unrecorded candidate provenance; choose a new output.')
    else:
        output.mkdir(parents=True)
        marker.write_text(json.dumps(request, indent=2)+'\n')


def run(runtime, script, arguments, env, report=None):
    started = time.perf_counter()
    command = [sys.executable, str(script), *map(str, arguments)]
    print('Running:', ' '.join(command), flush=True)
    result = subprocess.run(command, cwd=runtime/'workspace', env=env)
    elapsed = time.perf_counter()-started
    if report and Path(report).is_dir():
        Path(report, 'launcher_timing.json').write_text(json.dumps({
            'wall_s':elapsed, 'exit_code':result.returncode,
            'scope':'Whole child process, including startup/exit and export when requested; excludes resource preparation.',
            'command':command}, indent=2)+'\n')
    if result.returncode:
        raise SystemExit(result.returncode)


def doctor(runtime, env):
    packages = ['torch', 'isaacsim', 'isaaclab', 'numpy', 'scipy', 'mujoco',
                'onnxruntime', 'pandas', 'pyarrow', 'imageio', 'trimesh', 'joblib',
                'av', 'opencv-python']
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    # Probe imports used by the actual subprocess, including optional dependency paths.
    for package, module in [('onnxruntime', 'onnxruntime'), ('av', 'av'), ('opencv-python', 'cv2')]:
        probe = subprocess.run([sys.executable, '-c',
            f'import {module}; print({module}.__version__)'], env=env, capture_output=True, text=True)
        versions[package] = probe.stdout.strip() if probe.returncode == 0 else None
    missing = [k for k,v in versions.items() if v is None]
    missing_commands = [name for name in ['ffmpeg', 'ffprobe'] if not shutil.which(name)]
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                          '--format=csv,noheader'], capture_output=True, text=True) if shutil.which('nvidia-smi') else None
    result = dict(python=sys.version.split()[0], runtime=str(runtime), packages=versions,
                  missing=missing, missing_commands=missing_commands,
                  gpu=gpu.stdout.strip() if gpu else None,
                  physics_tested=False)
    print(json.dumps(result, indent=2))
    if sys.version_info[:2] != (3,11) or missing or missing_commands or gpu is None or gpu.returncode:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    for name in ['prepare', 'doctor', 'candidate', 'prepare-batch', 'collect', 'trial', 'replay', 'seed-reference', 'check-export']:
        p = subs.add_parser(name)
        p.add_argument('--runtime', type=Path, default=ROOT/'runtime')
        p.add_argument('--ort-path', type=Path)
        p.add_argument('--seed', type=Path, help='Validated grasp-reference directory; default packaged safe025')
        if name in ['candidate', 'prepare-batch', 'collect', 'trial', 'replay', 'seed-reference']:
            p.add_argument('--output', type=Path, required=True)
        if name in ['collect', 'trial', 'replay']:
            p.add_argument('--render-updates', type=int, choices=[1,2,4], default=1)
        if name == 'candidate':
            p.add_argument('--dx', type=float, default=0.)
            p.add_argument('--dy', type=float, default=0.)
            p.add_argument('--yaw', type=float, default=0.)
            p.add_argument('--scale', type=float, default=1.)
            p.add_argument('--start-dx', type=float, default=0.)
            p.add_argument('--start-dy', type=float, default=0.)
            p.add_argument('--start-yaw', type=float, default=0.)
        if name == 'prepare-batch':
            p.add_argument('--count', type=int, choices=range(1,61), default=60)
            p.add_argument('--random-seed', type=int, default=20260930)
        if name == 'collect':
            group = p.add_mutually_exclusive_group(required=True)
            group.add_argument('--candidate', type=Path)
            group.add_argument('--candidates', type=Path)
            p.add_argument('--limit', type=int)
            p.add_argument('--no-export', action='store_true')
        if name == 'trial':
            p.add_argument('--candidate', type=Path, required=True)
        if name == 'replay':
            p.add_argument('--first', type=Path, required=True)
        if name == 'check-export':
            p.add_argument('--dataset', type=Path, required=True)
            p.add_argument('--groot', type=Path, default=ROOT/'dependencies/Isaac-GR00T')
    args = parser.parse_args()
    if hasattr(args, 'output'):
        args.output = checked_path(args.output)
        if args.output.exists() and not (args.action == 'collect' and args.candidates):
            parser.error('--output must be a new directory')
    runtime = prepare(args.runtime)
    env = environment(runtime, args.ort_path, args.seed)
    if hasattr(args, 'render_updates'):
        env['SONIC_RENDER_UPDATES'] = str(args.render_updates)
    pipe = runtime/'workspace/pipelines/sonic_teleop_augmentation'
    if args.action == 'prepare':
        print(runtime)
    elif args.action == 'doctor':
        doctor(runtime, env)
    elif args.action == 'candidate':
        # Original helpers run in a fresh subprocess, isolating their plain common imports.
        code = ('from prepare import compile_candidate; from mobile_prepare import compile_mobile; '
                'import sys; from pathlib import Path; '
                'o=Path(sys.argv[1]); '
                's=compile_candidate(o.parent/(o.name+"_skill"), '
                'dict(object_delta_m=[float(sys.argv[2]),float(sys.argv[3]),0], '
                'grasp_yaw_deg=float(sys.argv[4]),approach_scale=float(sys.argv[5]))); '
                'print(compile_mobile(s,o,float(sys.argv[6]),float(sys.argv[7]),float(sys.argv[8])))')
        env['PYTHONPATH'] = str(pipe)+os.pathsep+env['PYTHONPATH']
        result = subprocess.run([sys.executable, '-c', code, str(args.output),
            *map(str,[args.dx,args.dy,args.yaw,args.scale,args.start_dx,args.start_dy,args.start_yaw])],
            cwd=runtime/'workspace', env=env)
        raise SystemExit(result.returncode)
    elif args.action == 'prepare-batch':
        env['PYTHONPATH'] = str(pipe)+os.pathsep+env['PYTHONPATH']
        run(runtime, ROOT/'prepare_batch.py', ['--output', args.output,
            '--count', args.count, '--random-seed', args.random_seed], env, args.output)
    elif args.action == 'collect':
        if args.candidate:
            arguments = ['--candidate', args.candidate.resolve(), '--output', args.output]
            if args.no_export:
                arguments.append('--no-export')
            script = pipe/'lean_pair.py'
        else:
            if args.no_export:
                parser.error('--no-export is only supported for single-pair collection')
            bind_collection(args.candidates, args.output, args.render_updates)
            arguments = ['--candidates', args.candidates.resolve(), '--output', args.output]
            if args.limit is not None:
                arguments += ['--limit', args.limit]
            script = pipe/'lean_batch.py'
        run(runtime, script, arguments, env, args.output)
    elif args.action in ['trial', 'replay']:
        candidate = args.candidate.resolve() if args.action == 'trial' else args.output.parent/(args.output.name+'_frozen')
        if args.action == 'replay':
            run(runtime, pipe/'lean_prepare_replay.py', ['--first', args.first.resolve(),
                '--output', candidate], env)
        run(runtime, pipe/'lean_run.py', ['--mode', 'generate' if args.action == 'trial' else 'replay',
            '--candidate', candidate, '--output', args.output], env, args.output)
        if not json.loads((args.output/'lean_result.json').read_text())['task_success']:
            raise SystemExit(2)
    elif args.action == 'seed-reference':
        run(runtime, runtime/'workspace/experiments/20260929_sonic_teleop_grasp/prepare.py',
            ['--output', args.output], env, args.output)
    elif args.action == 'check-export':
        run(runtime, pipe/'check_export.py', ['--groot', args.groot.resolve(),
            args.dataset.resolve()], env)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
