"""Package two successful simulation references; never connect to hardware."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile
import xml.etree.ElementTree as ET
import h5py
import numpy as np

B = Path(__file__).resolve().parents[1]
W = B.parents[1]
OUT = W / 'exports/g1_revo2_bottle_trajectories_20260925'
OUT.mkdir(exist_ok=False)

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def copy(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

runs = [('full_waypoints', 'walk_009', 'successful_walk009'),
        ('simplified_arc', 'walk_010_arc', 'successful_walk010_arc')]
validation = []
for label, run, prefix in runs:
    src = B / 'runs' / run
    dst = OUT / label
    dst.mkdir()
    for suffix in ['_actual_hint.npz', '_actual_hint.json', '_config.json']:
        copy(B / 'reference' / (prefix + suffix), dst / 'original_reference' / (prefix + suffix))
    evidence = dst / 'evidence'
    evidence.mkdir()
    # Preserve small, exact runtime evidence and the executed source snapshots.
    for pattern in ['*.json', '*.jsonl', '*.md']:
        for p in src.glob(pattern): copy(p, evidence / p.name)
    for folder in ['source_snapshot', 'scripts_snapshot']:
        shutil.copytree(src / folder, dst / folder, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    copy(src / 'handoff_reference.npz', evidence / 'handoff_reference.npz')
    copy(src / '000000.mp4', dst / 'external_view.mp4')
    copy(src / 'head.mp4', dst / 'head_original_rtx.mp4')
    with h5py.File(src / 'episode_raw.h5', 'r') as f:
        names = json.loads(f.attrs['joint_names'])
        body_names = json.loads(f.attrs['body_names'])
        ids = [i for i, n in enumerate(names) if '_hand_' not in n]
        assert len(ids) == 29
        n = len(f['observation/timestamp'])
        a = {
            'time_s': f['observation/timestamp'][:],
            'control_index': f['observation/control_index'][:],
            'g1_joint_names': np.array([names[i] for i in ids]),
            'g1_joint_position_rad': f['observation/joint_position'][:, ids],
            'g1_joint_velocity_rad_s': f['observation/joint_velocity'][:, ids],
            'root_position_world_m': f['observation/root_position'][:],
            'root_quaternion_world_wxyz': f['observation/root_quaternion_wxyz'][:],
            'body_names': np.array(body_names),
            'body_position_world_m': f['observation/body_position_world'][:],
            'body_quaternion_world_wxyz': f['observation/body_quaternion_wxyz'][:],
            'object_position_world_m': f['observation/object_position'][:],
            'object_quaternion_world_wxyz': f['observation/object_quaternion_wxyz'][:],
        }
        mapping = {'g1_29_in_export_order': [{'export_index': j, 'source_index_43': i, 'name': names[i], 'hardware_motor_index': None} for j,i in enumerate(ids)],
                   'legacy_placeholder_indices_not_revo2': [i for i in range(len(names)) if i not in ids],
                   'hardware_mapping_status': 'UNVERIFIED: map by name against destination SONIC and Revo2 driver; no hardware IDs invented'}
        for side in ['left', 'right']:
            hn = json.loads(f.attrs[f'brainco_{side}_joint_names'])
            tree = ET.parse(B / 'assets/brainco_candidate' / f'revo2_{side}_clean.urdf')
            spec = {}
            for joint in tree.findall('joint'):
                if joint.get('type') == 'fixed': continue
                mimic = joint.find('mimic')
                spec[joint.get('name')] = {'limit': dict(joint.find('limit').attrib), 'mimic': None if mimic is None else dict(mimic.attrib)}
            active = [i for i,name in enumerate(hn) if spec[name]['mimic'] is None]
            assert len(active) == 6 and len(hn) == 11
            a[f'revo2_{side}_joint_names'] = np.array(hn)
            a[f'revo2_{side}_active_indices'] = np.array(active, dtype=np.int64)
            a[f'revo2_{side}_joint_position_rad'] = f[f'brainco/{side}/joint_position'][:]
            a[f'revo2_{side}_joint_velocity_rad_s'] = f[f'brainco/{side}/joint_velocity'][:]
            a[f'revo2_{side}_position_target_rad'] = f[f'brainco/{side}/position_target'][:]
            mapping[f'revo2_{side}'] = [{'export_index':i,'name':name,'active':i in active, **spec[name], 'hardware_channel':None} for i,name in enumerate(hn)]
        np.savez_compressed(dst / 'trajectory_named.npz', **a)
        # Keep every non-camera dataset, including exact 200 Hz actuator substeps.
        with h5py.File(dst / 'episode_motion.h5', 'w') as g:
            for k,v in f.attrs.items(): g.attrs[k] = v
            for key in f:
                if key != 'camera': f.copy(key, g)
            g.attrs['transfer_note'] = 'Lossless non-camera subset of episode_raw.h5; source attrs retained as historical provenance; not a hardware command file'
        checked = []
        with h5py.File(dst / 'episode_motion.h5', 'r') as g:
            def check(name,obj):
                if isinstance(obj,h5py.Dataset):
                    assert np.array_equal(obj[:], f[name][:], equal_nan=True), name
                    checked.append(name)
            g.visititems(check)
        # Convenience segment starts after measured support handoff. No new initial posture transition generated.
        start = 830
        sample_keys = [k for k,v in a.items() if v.ndim > 0 and v.shape[0] == n]
        segment = {k: v[start:] if k in sample_keys else v for k,v in a.items()}
        segment['source_time_s'] = segment['time_s'].copy()
        segment['time_s'] = segment['time_s'] - segment['time_s'][0]
        np.savez_compressed(dst / 'grasp_segment_named.npz', **segment)
        with np.load(dst / 'trajectory_named.npz',allow_pickle=False) as z:
            for k,v in a.items(): assert np.array_equal(z[k],v), k
        for k in sample_keys:
            assert np.isfinite(a[k]).all(), k
        assert np.allclose(np.diff(a['time_s']), .02)
        events = [json.loads(line) for line in (src/'hand_command_log.jsonl').read_text().splitlines()]
        write_json(dst/'events.json', [{'control_step':e['frame'],'time_s':e['frame']*.02,'observation_id':e['observation_id'],'action':e['action'],'requested_goal':e['requested_goal'],'applied_goal':e['applied_goal'],'effective_finger_overrides':e['effective_finger_overrides']} for e in events])
        write_json(dst/'joint_mapping.json',mapping)
        write_json(dst/'schema.json', {
            'source_run':run,'control_dt_s':.02,'physics_dt_s':.005,'samples':n,
            'world_axes':'right-handed Z up; original simulation world, not destination robot frame',
            'quaternion_order':'wxyz','angles':'rad','positions':'m','velocities':'rad/s or m/s',
            'timing':'trajectory_named.npz uses observation at beginning of each control interval; Revo2 state and current target captured in begin() at that same boundary. Position targets are snapshot values, not certified future hardware commands. episode_motion.h5 also preserves next_observation at t+0.02 and all G1 physics substeps.',
            'revo2':'11 joints per hand = 6 active + 5 passive mimic. The separate brainco HDF5 groups are the real simulated Revo2. The last14 legacy G1 joints are compatibility placeholders, not Revo2.',
            'grasp_segment':{'source_start_index':start,'source_start_time_s':16.6,'time_s_rebased':True,'source_time_and_control_index_preserved':True,'entry_transition':'NOT generated; destination must reach compatible stable support/arm posture first'},
            'sample_axis_keys':sample_keys,
            'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in a.items()},
            'hardware_ready':False})
        validation.append({'run':run,'samples':n,'noncamera_datasets_compared_exactly':len(checked),'named_npz_roundtrip_exact':True,'dt_verified':True,'finite_sample_arrays':True,'g1_body_joints':29,'revo2_joints_per_hand':11,'active_per_hand':6})

for p in ['RESULTS.md','FAILURE_ANALYSIS.md','README.md']:
    copy(B/p,OUT/'context'/p)
copy(B/'reports/simplification_comparison.json',OUT/'context/simplification_comparison.json')
for side in ['left','right']:
    copy(B/f'assets/brainco_candidate/revo2_{side}_clean.urdf', OUT/f'model_metadata/revo2_{side}_clean.urdf')
copy(B/'renders/walk_010_arc_rt2/video.mp4',OUT/'simplified_arc/head_rt2.mp4')
copy(B/'renders/walk_010_arc_rt2/validation.json',OUT/'simplified_arc/rt2_validation.json')
copy(Path(__file__),OUT/'provenance/package_transfer.py')
write_json(OUT/'validation.json',validation)
print(OUT)
