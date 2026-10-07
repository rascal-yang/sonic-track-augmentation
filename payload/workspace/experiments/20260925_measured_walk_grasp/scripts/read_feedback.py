"""Print measured feedback and geometric aids; does not choose any action."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
files=sorted(a.run.glob('observation_[0-9][0-9][0-9].json'))
if not files: print('initializing');raise SystemExit()
f=files[-1];d=json.loads(f.read_text());g=d['grasp_information'];m=g['metrics'];q=np.array(d['object_quat_wxyz']);w=Rotation.from_quat(q[[1,2,3,0]]).apply([-.108,.005,.16])+d['object_pos'];pad=np.array(g['pad_based_grasp_center_m'])
x={k:d.get(k) for k in ['observation_id','step','robot_root_pos','object_pos','palm_positions','active_goal','goal_error_m','navigation_error_m']};x.update(canonical_grasp_center=w.tolist(),pad_center=pad.tolist(),grasp_center_error=(w-pad).tolist(),metrics={k:m[k] for k in ['whole_object_clearance_m','object_hand_normal_force_N','object_table_normal_force_N','thumb_contact_N','index_middle_contact_N','opposed_digits_contact']},decision_exists=(a.run/f'decision_{d["observation_id"]:03d}.json').exists())
if 'brainco' in d:
 import xml.etree.ElementTree as ET
 robot=ET.parse(Path(__file__).resolve().parents[1]/'assets/brainco_candidate/revo2_right_clean.urdf').getroot()
 hand=d['brainco']['right'];values=dict(zip(hand['joint_names'],hand['joint_position']));poses={'right_base_link':(np.eye(3),np.zeros(3))}
 for joint in robot.findall('joint'):
  parent=joint.find('parent').get('link');child=joint.find('child').get('link');r0,p0=poses[parent];origin=joint.find('origin');offset=np.fromstring(origin.get('xyz','0 0 0'),sep=' ');r1=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix();rotation=np.eye(3)
  if joint.get('type')!='fixed':rotation=Rotation.from_rotvec(np.fromstring(joint.find('axis').get('xyz'),sep=' ')*values[joint.get('name')]).as_matrix()
  poses[child]=(r0@r1@rotation,p0+r0@offset)
 quat=np.array(d['wrist_quat_wxyz']['right']);rw=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix();mount=np.array([[0,0,1],[1,0,0],[0,1,0]]);rh=rw@mount;ph=np.array(hand['root_position'])
 tips={finger:(ph+rh@poses['right_'+finger+'_tip'][1]).tolist() for finger in ['thumb','index','middle','ring','pinky']}
 center=(np.array(tips['thumb'])+(np.array(tips['index'])+tips['middle'])/2)/2
 x.update(revo2_URDF_tip_centers=tips,revo2_URDF_pinch_center=center.tolist(),revo2_URDF_pinch_error=(w-center).tolist(),revo2_pinch_offset_from_IK_palm_in_wrist_axes=(rw.T@(center-d['palm_positions']['right'])).tolist())
 (a.run/f'geometry_aid_{d["observation_id"]:03d}.json').write_text(json.dumps(x,indent=2))
print(json.dumps(x,indent=2))
