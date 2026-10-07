"""Screen a continuous approach from a saved actual post-walk pose, not a physical success."""
from pathlib import Path
import sys,json,runpy,contextlib,io
import numpy as np
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1];r=B/'runs'/sys.argv[1];obsid=int(sys.argv[2]);args=sys.argv[:]
sys.argv=[str(B/'scripts/hand_mesh_clearance.py'),str(r),str(obsid)]
with contextlib.redirect_stdout(io.StringIO()): mesh=runpy.run_path(str(B/'scripts/hand_mesh_clearance.py'))
sys.argv=args;s=mesh['obs'];points=mesh['points'];sys.path.insert(0,str(B/'code'))
from online_goals import Kinematics
from experiment import SONIC
k=Kinematics(SONIC);q0=np.array([s['joint_pos'][s['privileged']['joint_names'].index(n)] for n in k.names]);origin=np.array(s['palm_positions']['right']);q=np.array(s['wrist_quat_wxyz']['right']);r0=Rotation.from_quat(q[[1,2,3,0]]);local=r0.inv().apply(points-origin);cfg=json.loads((r/'optimized_config.json').read_text());target=np.array(s['object_pos'])+cfg['palm_offset_from_bottle_m'];rv=(r0.inv()*Rotation.from_euler('xyz',[0,0,1.57])).as_rotvec();lo,hi=np.asarray(s['privileged']['tabletop_aabb_m']);reports=[]
for arc in [[.04,-.10,.15],[.04,-.12,.18],[.03,-.10,.20],[.0,-.12,.18]]:
 q=q0.copy();rows=[]
 for a in np.linspace(0,1,81):
  p=origin+a*(target-origin)+np.array(arc)*np.sin(np.pi*a)**2;v=np.clip(a/.4,0,1);alpha=v**3*(10-15*v+6*v*v);rot=r0*Rotation.from_rotvec(rv*alpha);q,rep=k.solve(q,s['robot_root_pos'],s['robot_root_quat_wxyz'],'right',p,rot.as_euler('xyz'),allow_waist=False);w=Rotation.from_matrix(k.data.xmat[k.model.body('right_wrist_yaw_link').id].reshape(3,3));cloud=w.apply(local)+k.palm('right');gap=np.maximum(lo-cloud.max(0),cloud.min(0)-hi);rows.append({'progress':float(a),'palm_target':p.tolist(),'palm_achieved':k.palm('right').tolist(),'separating_margin_m':float(gap.max()),'ik_error_m':rep['ik_palm_error_m']})
 rep={'arc_m':arc,'orientation_finish_fraction':.4,'minimum_table_box_margin_m':min(x['separating_margin_m'] for x in rows),'maximum_ik_error_m':max(x['ik_error_m'] for x in rows),'samples':rows};reports.append(rep)
report={'run':r.name,'start_observation':obsid,'method':'Sequential IK at actual saved root; actual open-hand joint mesh bounds. No forearm or future tracking/balance guarantee','candidates':reports};(B/'reports'/f'{r.name}_arc_screen.json').write_text(json.dumps(report,indent=2));print([{k:v for k,v in x.items() if k!='samples'} for x in reports])
