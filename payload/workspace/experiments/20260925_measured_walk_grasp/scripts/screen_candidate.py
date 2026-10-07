from pathlib import Path
import sys,json
import numpy as np,h5py
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'code'))
from online_goals import Kinematics
from experiment import SONIC
r=B/'runs'/sys.argv[1];obs=json.loads(sorted(r.glob('observation_[0-9][0-9][0-9].json'))[-1].read_text()) if len(sys.argv)<3 else json.loads((r/f'observation_{int(sys.argv[2]):03d}.json').read_text())
with h5py.File(B/'runs/walk_004/episode_raw.h5') as f:names=json.loads(f.attrs['joint_names'])
k=Kinematics(SONIC);q=np.array([obs['joint_pos'][names.index(n)] for n in k.names]);cfg=json.loads((B/'optimized_config.json').read_text());target=np.array(obs['object_pos'])+cfg['palm_offset_from_bottle_m'];out=[]
for dz in [0,.01,-.01]:
 t=target+[0,0,dz];sol,report=k.solve(q,obs['robot_root_pos'],obs['robot_root_quat_wxyz'],'right',t,[0,0,1.57],allow_waist=False);mat=k.data.xmat[k.model.body('right_wrist_yaw_link').id].reshape(3,3);report.update(target=t.tolist(),height_shift_m=dz,orientation_error_rad=float((Rotation.from_euler('xyz',[0,0,1.57]).inv()*Rotation.from_matrix(mat)).magnitude()),q=sol.tolist());out.append(report)
report={'observation':obs['observation_id'],'run':r.name,'scope':'actual saved body pose, right arm only, numerical IK screening; not physics or contact acceptance','candidates':out};(r/f'candidate_ik_{obs["observation_id"]:03d}.json').write_text(json.dumps(report,indent=2));print(json.dumps({**report,'candidates':[{k:v for k,v in s.items() if k!='q'} for s in out]},indent=2))
