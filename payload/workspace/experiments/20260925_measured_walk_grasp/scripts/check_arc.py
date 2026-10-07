"""Nominal swept-hand separating-box check before physical arc trial (not a guarantee)."""
from pathlib import Path
import sys,json,os,xml.etree.ElementTree as ET
import numpy as np,trimesh
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'code'))
from online_goals import Kinematics
from brainco_adapter import MOUNT_ROTATIONS,MOUNT_P
from experiment import SONIC
s=json.loads((B/'runs/stationary_002/observation_001.json').read_text());kin=Kinematics(SONIC);q=np.array([s['joint_pos'][s['privileged']['joint_names'].index(n)] for n in kin.names]);root=np.array(s['robot_root_pos']);quat=np.array(s['robot_root_quat_wxyz']);origin=np.array(s['palm_positions']['right']);target=np.array([.415,-.19,.86]);arc=np.array(json.loads(os.environ.get('ARC','[.05,-.11,.18]')));orient_fraction=float(os.environ.get('ARC_ORIENT_FRACTION','1')); r0=Rotation.from_quat(np.array(s['wrist_quat_wxyz']['right'])[[1,2,3,0]]);dr=(r0.inv()*Rotation.from_euler('xyz',[0,0,1.57])).as_rotvec()
tree=ET.parse(B/'assets/brainco_candidate/revo2_right_clean.urdf').getroot();poses={'right_base_link':(np.eye(3),np.zeros(3))}
for j in tree.findall('joint'):
 R,p=poses[j.find('parent').get('link')];o=j.find('origin');rot=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix();value=0
 if j.get('type')!='fixed' and 'thumb_metacarpal' in j.get('name'):
  lim=j.find('limit');value=.75*max([float(lim.get('lower')),float(lim.get('upper'))],key=abs)
 if j.get('type')!='fixed':rot=rot@Rotation.from_rotvec(np.fromstring(j.find('axis').get('xyz'),sep=' ')*value).as_matrix()
 poses[j.find('child').get('link')]=(R@rot,p+R@np.fromstring(o.get('xyz','0 0 0'),sep=' '))
cloud=[]
for l in tree.findall('link'):
 R,p=poses[l.get('name')]
 for c in l.findall('collision'):
  m=c.find('geometry/mesh');v=np.asarray(trimesh.load(m.get('filename'),process=False).vertices)*np.fromstring(m.get('scale','1 1 1'),sep=' ');o=c.find('origin')
  if o is not None:v=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).apply(v)+np.fromstring(o.get('xyz','0 0 0'),sep=' ')
  cloud.append(v@R.T+p)
local=np.concatenate(cloud);lo=np.array([-.5616,-.29423,.70]);hi=np.array([.5616,.29423,.75]);rows=[]
for u in np.linspace(0,1,67):
 a=35*u**3-105*u**4+126*u**5-70*u**6+15*u**7;p=origin+a*(target-origin)+arc*np.sin(np.pi*a)**2;v=np.clip(a/orient_fraction,0,1);ra=v*v*v*(10-15*v+6*v*v);r=r0*Rotation.from_rotvec(ra*dr);q,rep=kin.solve(q,root,quat,'right',p,r.as_euler('xyz'),allow_waist=False);bid=kin.model.body('right_wrist_yaw_link').id;wr=Rotation.from_matrix(kin.data.xmat[bid].reshape(3,3));bp=kin.data.xpos[bid]+wr.apply(MOUNT_P);v=(wr*MOUNT_ROTATIONS['right']).apply(local)+bp;gap=np.maximum(lo-v.max(0),v.min(0)-hi);rows.append({'u':float(u),'palm_target':p.tolist(),'separating_box_margin_m':float(gap.max()),'ik_error_m':rep['ik_palm_error_m']})
out={'method':'Open-hand collision mesh vertices with nominal root and sequential IK; enclosing AABB separating-plane check. Does not certify dynamics, cooked colliders or all arm meshes.','minimum_separating_margin_m':min(x['separating_box_margin_m'] for x in rows),'max_ik_error_m':max(x['ik_error_m'] for x in rows),'samples':rows};out['arc_m']=arc.tolist();out['orient_fraction']=orient_fraction;(B/'reports'/('nominal_arc_'+os.environ.get('TAG','default')+'.json')).write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k!='samples'})
