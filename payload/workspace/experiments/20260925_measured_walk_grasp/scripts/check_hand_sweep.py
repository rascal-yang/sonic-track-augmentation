from pathlib import Path
import json,sys,runpy
import numpy as np
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1];r=Path(sys.argv[1]);state=runpy.run_path(str(B/'scripts/hand_mesh_clearance.py'))
s=state['obs'];points=state['points'];palm=np.array(s['palm_positions']['right']);q=np.array(s['wrist_quat_wxyz']['right']);rot=Rotation.from_quat(q[[1,2,3,0]]);goal=Rotation.from_euler('xyz',[0,0,1.57]);rv=(rot.inv()*goal).as_rotvec();cfg=json.loads((r/'optimized_config.json').read_text());target=np.array(s['object_pos'])+cfg['palm_offset_from_bottle_m'];local=rot.inv().apply(points-palm);mins=[]
for a in np.linspace(0,1,41):
 rr=rot*Rotation.from_rotvec(a*rv);cloud=rr.apply(local)+palm+a*(target-palm);mins.append(float(cloud[:,2].min()-.75))
report={'observation_id':s['observation_id'],'method':'41 samples of rigid open-hand collision-mesh swept bounds along planned palm translation/orientation; not actual tracking or forearm collision proof','minimum_whole_hand_tabletop_height_margin_m':min(mins),'wrist_turn_deg':float(np.rad2deg(np.linalg.norm(rv))),'target':target.tolist(),'passed_geometric_screen':min(mins)>.025}
(r/f'hand_sweep_{s["observation_id"]:03d}.json').write_text(json.dumps(report,indent=2));print(report)
