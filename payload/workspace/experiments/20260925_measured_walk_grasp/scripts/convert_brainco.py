"""Convert the locally available Revo2 URDF, retaining physical mimic joints."""
import os,sys
from pathlib import Path
import xml.etree.ElementTree as ET
B=Path(__file__).resolve().parents[1]
if not os.environ.get('BRAINCO_DRIVER'):
 e=os.environ.copy();e.update(BRAINCO_DRIVER='1',LD_LIBRARY_PATH='/tmp/g1-driver-580.173.02/usr/lib/x86_64-linux-gnu:'+e.get('LD_LIBRARY_PATH',''));os.execve(sys.executable,[sys.executable,*sys.argv],e)
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
try:
 for side in ['right','left']:
  source=B/'assets/brainco_candidate'/f'brainco_{side}.urdf';tree=ET.parse(source);r=tree.getroot()
  for item in list(r):
   if (item.tag=='link' and item.get('name')=='base_link') or (item.tag=='joint' and item.get('name')=='base') or item.tag=='mujoco':r.remove(item)
  for mesh in r.findall('.//mesh'):
   mesh.set('filename',str((source.parent/mesh.get('filename')).resolve()))
  clean=source.with_name(f'revo2_{side}_clean.urdf');tree.write(clean)
  cfg=UrdfConverterCfg(asset_path=str(clean),usd_dir=str(B/'assets/brainco_usd'/side),usd_file_name='hand.usd',fix_base=False,merge_fixed_joints=True,force_usd_conversion=True,make_instanceable=False,convert_mimic_joints_to_normal_joints=False,collider_type='convex_hull',self_collision=False,joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=4.,damping=.12)))
  c=UrdfConverter(cfg);print('CONVERTED',side,c.usd_path,flush=True)
finally:app.close()
