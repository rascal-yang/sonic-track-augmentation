"""Privileged semantic part and actual Revo2 fingertip geometry, in meters."""
import json
from calibration_config import ROBOT_SCALE
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

BASE = Path(__file__).resolve().parents[1]
VERTICES = np.load(BASE / 'assets/water_bottle_v1/collision_source.npz')['vertices']
REGIONS = json.loads((BASE / 'assets/protrusion_regions.json').read_text())['regions']
URDF = ET.parse(BASE / 'assets/brainco_candidate/revo2_right_clean.urdf').getroot()
MOUNT = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])


def local_tips(values):
    poses = {'right_base_link': (np.eye(3), np.zeros(3))}
    for joint in URDF.findall('joint'):
        r, p = poses[joint.find('parent').get('link')]
        origin = joint.find('origin')
        offset = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        r1 = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        motion = np.eye(3)
        if joint.get('type') != 'fixed':
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            motion = Rotation.from_rotvec(axis * values[joint.get('name')]).as_matrix()
        poses[joint.find('child').get('link')] = (r @ r1 @ motion, p + r @ offset)
    return {f: poses['right_' + f + '_tip'][1]*ROBOT_SCALE for f in ['thumb', 'index', 'middle', 'ring', 'pinky']}


def nominal_tips(closure, opposition=.75):
    values = {}
    for j in URDF.findall('joint'):
        if j.get('type') == 'fixed':
            continue
        limit = j.find('limit')
        fraction = opposition if 'thumb_metacarpal' in j.get('name') else closure
        values[j.get('name')] = fraction * max([float(limit.get('lower')), float(limit.get('upper'))], key=abs)
    for j in URDF.findall('joint'):
        mimic = j.find('mimic')
        if mimic is not None:
            values[j.get('name')] = values[mimic.get('joint')] * float(mimic.get('multiplier', 1))
    return {f: MOUNT @ p - np.array([.045, 0, 0])*ROBOT_SCALE for f, p in local_tips(values).items()}


def add_protrusions(data):
    hand = data.get('brainco', {}).get('right')
    if hand is None:
        return
    q = np.array(data['wrist_quat_wxyz']['right'])
    wrist_r = Rotation.from_quat(q[[1, 2, 3, 0]])
    tips = {f: np.array(hand['root_position']) + wrist_r.apply(MOUNT @ p)
            for f, p in local_tips(dict(zip(hand['joint_names'], hand['joint_position']))).items()}
    pinch = (tips['thumb'] + tips['index']) / 2
    q = np.array(data['object_quat_wxyz'])
    obj_r = Rotation.from_quat(q[[1, 2, 3, 0]])
    palm = np.array(data['palm_positions']['right'])
    result = {'source': 'simulated object pose + annotated mesh regions + measured physical hand joint FK',
              'distance_kind': 'Euclidean point to center and nearest sampled visual mesh vertex; not signed collision clearance',
              'tips_world_m': {f: p.tolist() for f, p in tips.items()},
              'pinch_definition': 'thumb + index tip midpoint; middle/ring/pinky may be folded independently',
              'pinch_center_world_m': pinch.tolist(),
              'pinch_offset_wrist_m': wrist_r.inv().apply(pinch - palm).tolist(), 'parts': {}}
    for region in REGIONS:
        selected = VERTICES[np.all((VERTICES >= region['lo']) & (VERTICES <= region['hi']), axis=1)]
        center = (selected.min(0) + selected.max(0)) / 2
        world = obj_r.apply(selected) + data['object_pos']
        target = obj_r.apply(center) + data['object_pos']
        result['parts'][region['name']] = {
            'center_local_m': center.tolist(), 'center_world_m': target.tolist(),
            'extent_local_m': np.ptp(selected, axis=0).tolist(),
            'lowest_surface_above_table_m': float(world[:, 2].min() - .75),
            'palm_to_center_m': float(np.linalg.norm(palm - target)),
            'palm_to_surface_vertex_m': float(np.linalg.norm(world - palm, axis=1).min()),
            'pinch_to_center_m': float(np.linalg.norm(pinch - target)),
            'target_minus_pinch_m': (target - pinch).tolist(),
            'fingertip_to_center_m': {f: float(np.linalg.norm(p - target)) for f, p in tips.items()},
            'fingertip_to_surface_vertex_m': {f: float(np.linalg.norm(world - p, axis=1).min()) for f, p in tips.items()}}
    data['protrusion_geometry'] = result
