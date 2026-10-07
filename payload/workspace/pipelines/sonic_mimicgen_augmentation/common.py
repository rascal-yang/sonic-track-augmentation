"""Object-centric transformations; no simulator state writes during execution."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
BASE = WORKSPACE / 'experiments/20260925_measured_walk_grasp'
SEED_PIPELINE = WORKSPACE / 'pipelines/sonic_walk_grasp_success_v1/snapshot/experiments'
SOURCE = SEED_PIPELINE / '20260929_sonic_grasp_smooth/candidates/safe025'
WALK = SEED_PIPELINE / '20260929_sonic_walk_fast_grasp'
MODEL_ROOT = Path('__RUNTIME__/models/deploy')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quat_rotate(q, rotation):
    q = np.asarray(q)
    return (rotation * R.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]].reshape(q.shape)


def points_transform(points, pivot, translation, rotation):
    p = np.asarray(points)
    return rotation.apply((p - pivot).reshape(-1, 3)).reshape(p.shape) + pivot + translation


def transform_initial(z, pivot, translation, rotation, include_object=True):
    out = {k: v.copy() for k, v in z.items()}
    positions = ['root_position', 'body_position_world']
    quats = ['root_quaternion_wxyz', 'body_quaternion_wxyz']
    velocities = ['root_linear_velocity_world', 'root_angular_velocity_world']
    for side in ['left', 'right']:
        prefix = f'brainco_{side}_root_'
        positions.append(prefix + 'position')
        quats.append(prefix + 'quaternion')
        velocities += [prefix + 'linear_velocity', prefix + 'angular_velocity']
    if include_object:
        positions += ['object_position']
        quats += ['object_quaternion_wxyz']
        velocities += ['object_linear_velocity', 'object_angular_velocity']
    for k in positions:
        out[k] = points_transform(z[k], pivot, translation, rotation).astype(z[k].dtype)
    for k in quats:
        out[k] = quat_rotate(z[k], rotation).astype(z[k].dtype)
    for k in velocities:
        out[k] = rotation.apply(z[k]).astype(z[k].dtype)
    return out


def contact_preserving_scale(points, contact_point, times, scale, pregrasp_s=4.9):
    """C2 window; starts/ends unchanged; zero modification throughout contact/hold."""
    t = np.asarray(times)
    u = np.clip(t / pregrasp_s, 0, 1)
    weight = 64 * u**3 * (1 - u)**3
    weight[t >= pregrasp_s] = 0
    return points + (scale - 1) * weight[:, None] * (points - contact_point)


def encode_reference(ref, names, session):
    n = len(ref['time_s'])
    future = np.minimum(np.arange(n)[:, None] + np.arange(0, 50, 5), n - 1)
    order = [ref['g1_joint_names'].tolist().index(k) for k in names]
    rot = R.from_quat(ref['root_quaternion_world_wxyz'][:, [1, 2, 3, 0]]).as_matrix()
    relative = np.einsum('nji,nkjl->nkil', rot, rot[future])
    obs = np.zeros((n, 1762), np.float32)
    obs[:, 4:294] = ref['g1_joint_position_rad'][:, order][future].reshape(n, 290)
    obs[:, 294:584] = ref['g1_joint_velocity_rad_s'][:, order][future].reshape(n, 290)
    obs[:, 601:661] = relative[:, :, :, :2].reshape(n, 60)
    token = np.concatenate([session.run(None, {session.get_inputs()[0].name: x[None]})[0] for x in obs])
    if token.shape != (n, 64) or not np.isfinite(token).all():
        raise ValueError('Invalid encoder output')
    result = dict(time_s=ref['time_s'], control_index=np.arange(n), sonic_encoded_tokens=token,
                  encoder_inputs=obs, future_indices=future, full_future_window_valid=np.arange(n)+45<n,
                  encoder_joint_names=np.array(names))
    for side in ['left', 'right']:
        idx = ref[f'revo2_{side}_active_indices']
        result[f'revo2_{side}_active_joint_names'] = ref[f'revo2_{side}_joint_names'][idx]
        result[f'revo2_{side}_active_position_target_rad'] = ref[f'revo2_{side}_position_target_rad'][:, idx]
    return result
