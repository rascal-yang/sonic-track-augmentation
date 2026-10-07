"""Collision-mesh separating-plane screen, not a cooked PhysX collision proof."""
from pathlib import Path
import argparse,json,sys,xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

HERE=Path(__file__).resolve().parent;W=HERE.parents[1];B=W/'experiments/20260925_measured_walk_grasp'
sys.path.insert(0,str(B/'code'))
from brainco_adapter import MOUNT_ROTATIONS,MOUNT_P
from calibration_config import ROBOT_SCALE as scale
p=argparse.ArgumentParser();p.add_argument('candidate');a=p.parse_args();c=HERE/'candidates'/a.candidate
z=np.load(c/'grasp_reference.npz')
obs=json.loads((W/'experiments/20260929_sonic_grasp_token/runs/grasp_token_001/observation_000.json').read_text())
low,high=np.array(obs['privileged']['tabletop_aabb_m'])
report={'method':'Every2controlsteps, per-link collision-mesh AABB separating-plane margin to tabletop AABB; source finger joints and adjusted wrist pose; excludes forearm/room and decoder tracking error.','table_aabb_m':[low.tolist(),high.tolist()],'sides':{}}
for side in ['left','right']:
 root=ET.parse(B/'assets/brainco_candidate'/f'revo2_{side}_clean.urdf').getroot();meshes={}
 for link in root.findall('link'):
  clouds=[]
  for col in link.findall('collision'):
   mesh=col.find('geometry/mesh');assert mesh is not None
   mesh_path=Path(mesh.get('filename'))
   if not mesh_path.exists():mesh_path=(B/'assets/brainco_candidate').resolve()/'meshes'/mesh_path.name
   assert mesh_path.exists(),str(mesh_path)
   v=np.array(trimesh.load(mesh_path,process=False).vertices)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
   origin=col.find('origin')
   if origin is not None:v=R.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).apply(v)+np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
   clouds.append(v*scale)
  if clouds:meshes[link.get('name')]=np.concatenate(clouds)
 joints=root.findall('joint');bn=z['body_names'].tolist().index(side+'_wrist_yaw_link');hn=z[f'revo2_{side}_joint_names'].tolist();worst=(1.,None,None)
 for i in range(0,len(z['time_s']),2):
  wr=R.from_quat(z['body_quaternion_world_wxyz'][i,bn][[1,2,3,0]]);wp=z['body_position_world_m'][i,bn]
  tr={side+'_base_link':(wp+wr.apply(MOUNT_P*scale),wr*MOUNT_ROTATIONS[side])};q=dict(zip(hn,z[f'revo2_{side}_joint_position_rad'][i]))
  pending=joints.copy()
  while pending:
   progressed=False
   for j in pending[:]:
    parent=j.find('parent').get('link')
    if parent not in tr:continue
    pp,pr=tr[parent];o=j.find('origin');v=np.fromstring(o.get('xyz','0 0 0'),sep=' ')*scale;rot=R.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' '));ax=j.find('axis')
    if j.get('type')!='fixed':rot=rot*R.from_rotvec(np.fromstring(ax.get('xyz'),sep=' ')*q[j.get('name')])
    tr[j.find('child').get('link')]=(pp+pr.apply(v),pr*rot);pending.remove(j);progressed=True
   assert progressed
  for link,points in meshes.items():
   lp,lr=tr[link];v=lr.apply(points)+lp
   margin=float(max(np.max(low-v.max(0)),np.max(v.min(0)-high)))
   if margin<worst[0]:worst=(margin,i,link)
 report['sides'][side]={'minimum_separating_margin_m':worst[0],'step':worst[1],'link':worst[2]}
report['table_screen_passed']=all(x['minimum_separating_margin_m']>0 for x in report['sides'].values())
(c/'geometry_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
assert report['table_screen_passed']
