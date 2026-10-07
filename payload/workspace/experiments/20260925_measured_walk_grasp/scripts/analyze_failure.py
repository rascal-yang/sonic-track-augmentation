from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1];r=B/'runs/start_05_right'
rows=[json.loads(l) for l in (r/'trajectory.jsonl').read_text().splitlines()]
arm={x['frame']:x for x in [json.loads(l) for l in (r/'direct_arm_execution.jsonl').read_text().splitlines()]}
q=np.array([s['object_quat_wxyz'] for s in rows]);axis=Rotation.from_quat(q[:,[1,2,3,0]]).apply(np.tile([0,0,1],(len(q),1)));tilt=np.degrees(np.arccos(np.clip(axis[:,2],-1,1)))
force=np.array([s['grasp_metrics']['object_hand_normal_force_N'] for s in rows])
def detail(i):
 s=rows[int(i)];m=s['grasp_metrics'];frame=s['step'];a=arm.get(frame,arm.get(frame-1,{}));R=Rotation.from_quat(np.array(s['wrist_quat_wxyz']['right'])[[1,2,3,0]])
 return {'step':frame,'time_s':s['sim_time'],'bottle_tilt_deg':float(tilt[i]),'hand_force_N':float(force[i]),'root':s['robot_root_pos'],'navigation_error_m':s['navigation_error_m'],'palm':s['palm_positions']['right'],'wrist_euler_xyz':R.as_euler('xyz').tolist(),'arm_execution':a,'finger_contacts_nonzero':{k:v for k,v in m.get('finger_contacts',{}).items() if v.get('object_normal_force_N',0)>.01}}
out={}
for key,mask in [('first_hand_contact_above_0p2N',force>.2),('first_tilt_5deg',tilt>5),('first_tilt_30deg',tilt>30),('first_tilt_60deg',tilt>60)]:
 ids=np.flatnonzero(mask);out[key]=detail(ids[0]) if len(ids) else None
out['peak_hand_contact']=detail(int(np.argmax(force)))
out['max_ik_position_error_m']=max(x['ik_error_m'] for x in arm.values())
out['largest_ik_error']=max(arm.values(),key=lambda x:x['ik_error_m'])
out['last']=detail(len(rows)-1)
(r/'failure_analysis.json').write_text(json.dumps(out,indent=2))
for k,v in out.items():
 if isinstance(v,dict):print(k,{x:y for x,y in v.items() if x!='finger_contacts_nonzero'})
 else:print(k,v)
