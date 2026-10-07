"""Evaluate continuous table-contact quality separately from lift success."""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
a = p.parse_args()
rows = [json.loads(s) for s in (a.run/'table_contacts_physics.jsonl').read_text().splitlines()]
result = json.loads((a.run/'result.json').read_text())
capture = json.loads((a.run/'validation.json').read_text())
hand = json.loads((a.run/'hand_validation.json').read_text())
config = json.loads((a.run/'contact_watch_config.json').read_text())
dt = 1/config['frequency_hz']
assert len(rows) == result['physics_substeps']
assert [r['physics_index'] for r in rows] == list(range(len(rows)))
assert np.allclose([r['time_s'] for r in rows], np.arange(1, len(rows)+1)*dt)
assert [r['control_index'] for r in rows] == np.repeat(np.arange(result['control_steps']), 4).tolist()
force = np.array([r['total_table_force_N'] for r in rows])
assert np.isfinite(force).all() and np.min(force) >= 0
hits = force > .05
alerts_path = a.run/'contact_alerts.jsonl'
alerts = [json.loads(x) for x in alerts_path.read_text().splitlines()] if alerts_path.exists() else []
link_peaks = {}
for r in rows:
    for k, value in r['links_N'].items():
        link_peaks[k] = max(value, link_peaks.get(k, 0.))
quality = {'physics_samples': len(rows), 'frequency_Hz': config['frequency_hz'],
           'covered_links': len(config['links']), 'sampling_alignment_passed': True,
           'table_contact_reporting_threshold_N': .05,
           'table_force_max_N': float(force.max()),
           'table_contact_duration_s_above_threshold': float(hits.sum()*dt),
           'first_table_contact_s': next((r['time_s'] for r in rows if r['total_table_force_N'] > .05), None),
           'per_link_peak_table_force_N': link_peaks, 'early_checkpoint_alerts': len(alerts),
           'lift_success_final': capture['success_final'],
           'mount_rotation_checks_passed': all(hand[s]['mount_orientation_max_error_rad'] < .03 for s in ['left','right']),
           'clean_grasp_success': bool(capture['success_final'] and not hits.any()),
           'coverage': 'Both Revo 2 hands, G1 elbow and wrist bodies against table. This does not certify all robot-room collisions.'}
(a.run/'contact_validation.json').write_text(json.dumps(quality, indent=2))
print(json.dumps(quality, indent=2))
