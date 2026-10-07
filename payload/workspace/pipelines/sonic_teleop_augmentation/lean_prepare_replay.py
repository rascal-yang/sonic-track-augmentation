"""Freeze actual lean-generation tokens for a fresh decoder-only simulation."""
import argparse
import shutil
from common import Path, MODEL, arrays, read, write, sha


def prepare(first, output):
    first, output = Path(first).resolve(), Path(output).resolve()
    result = read(first/'lean_result.json')
    assert result['task_success'] and result['profile'] == 'generate'
    output.mkdir(parents=True, exist_ok=False)
    for name in ['initial_state.npz', 'deployment_parameters.json', 'plan.json']:
        shutil.copy2(first/name, output/name)
    z = arrays(first/'executed_teleop_tokens.npz')
    assert len(z['time_s']) == result['control_steps']
    shutil.copy2(first/'executed_teleop_tokens.npz', output/'grasp_tokens.npz')
    decoder_hash = sha(MODEL/'model_decoder.onnx')
    assert read(first/'token_controller.json')['decoder_sha256'] == decoder_hash
    write(output/'provenance.json', dict(first_execution=str(first), decoder_sha256=decoder_hash,
          token_sha256=sha(output/'grasp_tokens.npz'), encoder_rerun=False, planner_rerun=False))
    write(output/'prepared_files.json', {n: sha(output/n) for n in
          ['initial_state.npz', 'deployment_parameters.json', 'plan.json', 'grasp_tokens.npz']})
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--first', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    print(prepare(a.first, a.output))
