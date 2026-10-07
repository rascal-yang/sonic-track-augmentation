"""Continuous 30 s recovery experiment; generated references never reset physics."""
import copy
import shutil

import joblib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from experiment import KEY, write_json

MAX_STEPS = 1500
CYCLE_STEPS = 500


def prepare_retry(out, assets, mode):
    """Three 10 s cycles: 6 s approach/lift, 1 s hold, 3 s open-hand return.

    Extend at source 25 Hz so the normal motion loader recomputes velocities,
    contact references and all length metadata. Keep a 4 s planning tail.
    Direct uses only frame zero, with no future demonstration poses.
    """
    robot_source = joblib.load(assets / 'robot' / f'{KEY}.pkl')
    object_source = joblib.load(assets / 'objects' / f'{KEY}.pkl')
    key = next(iter(robot_source))
    robot, objects = robot_source[key], object_source[key]
    assert robot['fps'] == objects['fps'] == 25
    n = len(robot['dof'])
    alpha = np.arange(1, 76) / 75
    alpha = alpha**2 * (3 - 2 * alpha)

    def expand(value, quaternion=False):
        if mode == 'direct':
            return np.repeat(value[:1], 850, axis=0)
        a = alpha.reshape((-1,) + (1,) * (value.ndim - 1))
        recovery = value[149] + a * (value[0] - value[149])
        if quaternion:
            shape = value.shape[1:]
            # Robot is xyzw; object input is wxyz.
            end_points = value[[149, 0]].reshape(2, 4)
            if value.ndim == 3:
                end_points = end_points[:, [1, 2, 3, 0]]
            recovery = Slerp([0, 1], Rotation.from_quat(end_points))(alpha).as_quat()
            if value.ndim == 3:
                recovery = recovery[:, [3, 0, 1, 2]]
            recovery = recovery.reshape((75,) + shape)
        cycle = np.concatenate([value[:150], np.repeat(value[149:150], 25, axis=0), recovery])
        return np.concatenate([cycle] * 3 + [np.repeat(value[:1], 100, axis=0)]).astype(value.dtype)

    generated = []
    for source in (robot, objects):
        result = copy.deepcopy(source)
        for name, value in source.items():
            if isinstance(value, np.ndarray) and value.shape[0] == n:
                result[name] = expand(value, name in ('root_rot', 'root_quat'))
            elif name.startswith('contact_points_'):
                result[name] = {}
                if mode != 'direct':
                    for cycle in range(3):
                        for frame in range(175):
                            points = value.get(min(frame, 149))
                            if points is not None:
                                result[name][cycle * 250 + frame] = points.copy()
        generated.append(result)
    robot_out, object_out = generated
    for side in ('left', 'right'):
        hand = robot_out['hand_action_' + side]
        if mode == 'direct':
            hand[:] = -1
        else:
            for cycle in range(3):
                hand[cycle * 250 + 175:(cycle + 1) * 250] = -1
            hand[750:] = -1
    # Keep root rotation and its axis-angle representation consistent.
    robot_out['pose_aa'][:, 0] = Rotation.from_quat(robot_out['root_rot']).as_rotvec()
    data = out / 'retry_reference'
    for folder in ('robot', 'objects', 'meta', 'bps'):
        (data / folder).mkdir(parents=True)
    joblib.dump({key: robot_out}, data / 'robot' / f'{KEY}.pkl')
    joblib.dump({key: object_out}, data / 'objects' / f'{KEY}.pkl')
    shutil.copy2(assets / 'meta' / f'{KEY}.pkl', data / 'meta' / f'{KEY}.pkl')
    for name in (f'{KEY}.npy', '_basis.npy'):
        shutil.copy2(assets / 'bps' / name, data / 'bps' / name)
    write_json(out / 'retry_protocol.json', {
        'protocol': 'v3_retry', 'max_control_steps': MAX_STEPS, 'seconds': 30,
        'attempt_start_seconds': [0, 10, 20],
        'reference_phases': {'approach_and_lift': [0, 6], 'hold': [6, 7], 'open_and_return': [7, 10]},
        'direct_phases': 'GPT issues hand targets and stance IK online; no cyclic demo',
        'record_only_termination_terms': ['hand_table_contact_termination', 'object_pos_deviation'],
        'retained_termination_terms': ['anchor_pos', 'anchor_ori_full', 'ee_body_pos', 'time_out'],
        'physical_resets_between_attempts': 0, 'source_frames': 850,
        'reference_source': 'only initial demo frame' if mode == 'direct' else 'first 6 s of released robot trajectory + smooth return',
        'caveat': 'Recovery is a constructed reference, outside the original training distribution; no guarantee of recovery.'})
    return data
