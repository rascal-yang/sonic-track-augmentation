"""Reapply retained calibrated initial translation; deleted trial data not required."""
from pathlib import Path
import json,numpy as np
h=Path(__file__).resolve().parent
p=json.loads((h/'plan.json').read_text())
z=dict(np.load(Path(p['source_fast'])/'initial_state.npz'))
delta=np.asarray(p['start_root_m'])-z['root_position']
for k in ['root_position','body_position_world','brainco_left_root_position','brainco_right_root_position']:z[k]=z[k]+delta
np.savez_compressed(h/'initial_state.npz',**z)
print('Restored calibrated initial translation from retained plan; no failed-run dependency.')
