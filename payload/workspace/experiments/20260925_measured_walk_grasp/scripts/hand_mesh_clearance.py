"""Conservative separating-plane checks from actual joints and URDF mesh vertices."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base/'code'))
from brainco_adapter import MOUNT_ROTATIONS
from calibration_config import ROBOT_SCALE

run = Path(sys.argv[1])
obs_path = run/f'observation_{int(sys.argv[2]):03d}.json' if len(sys.argv)>2 else sorted(run.glob('observation_[0-9][0-9][0-9].json'))[-1]
obs = json.loads(obs_path.read_text())
report = {'observation_id': obs['observation_id'], 'step': obs['step'],
          'method': 'Actual joint positions, hand root translation, fixed mount rotation and original collision-mesh vertices. Bounds, not signed cooked-PhysX collision distances.'}
front = obs['privileged']['tabletop_aabb_m'][0][1]
top = obs['privileged']['tabletop_aabb_m'][1][2]
for side in ['left', 'right']:
    tree = ET.parse(base/'assets/brainco_candidate'/f'revo2_{side}_clean.urdf').getroot()
    hand = obs['brainco'][side]
    q = dict(zip(hand['joint_names'], hand['joint_position']))
    w = np.array(obs['wrist_quat_wxyz'][side])
    r = Rotation.from_quat(w[[1, 2, 3, 0]]) * MOUNT_ROTATIONS[side]
    poses = {side+'_base_link': (r.as_matrix(), np.array(hand['root_position']))}
    for joint in tree.findall('joint'):
        mat, pos = poses[joint.find('parent').get('link')]
        o = joint.find('origin')
        delta = np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
        rot = Rotation.from_euler('xyz', np.fromstring(o.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        if joint.get('type') != 'fixed':
            rot = rot @ Rotation.from_rotvec(np.fromstring(joint.find('axis').get('xyz'), sep=' ') * q[joint.get('name')]).as_matrix()
        poses[joint.find('child').get('link')] = (mat @ rot, pos + mat @ (delta*ROBOT_SCALE))
    cloud = []
    for link in tree.findall('link'):
        mat, pos = poses[link.get('name')]
        for col in link.findall('collision'):
            mesh = col.find('geometry/mesh')
            if mesh is None:
                raise ValueError('Add primitive support before claiming complete bounds')
            v = np.asarray(trimesh.load(mesh.get('filename'), process=False).vertices)
            v = v*np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            o = col.find('origin')
            if o is not None:
                v = Rotation.from_euler('xyz', np.fromstring(o.get('rpy', '0 0 0'), sep=' ')).apply(v) + np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
            cloud.append((v*ROBOT_SCALE) @ mat.T + pos)
    points = np.concatenate(cloud)
    report[side] = {'bounds_world_m': [points.min(0).tolist(), points.max(0).tolist()],
                    'whole_hand_behind_front_edge_margin_m': float(front-points[:, 1].max()),
                    'whole_hand_above_table_margin_m': float(points[:, 2].min()-top)}
(run/f'mesh_clearance_{obs["observation_id"]:03d}.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
