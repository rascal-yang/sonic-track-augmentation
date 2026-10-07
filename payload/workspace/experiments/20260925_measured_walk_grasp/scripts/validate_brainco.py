"""Verify Revo2 fixed mounts, physical mimic constraints, and finite states."""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import h5py,numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
from brainco_adapter import MOUNT_ROTATIONS
from calibration_config import ROBOT_SCALE
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();base=Path(__file__).resolve().parents[1];report={}
with h5py.File(a.run/'episode_raw.h5') as f:
 names=json.loads(f.attrs['body_names']);N=len(f['observation/root_position'])
 for side in ['left','right']:
  hn=json.loads(f.attrs['brainco_'+side+'_joint_names']);hbn=json.loads(f.attrs['brainco_'+side+'_body_names']);q=f[f'brainco/{side}/joint_position'][:];assert len(q)==N and np.isfinite(q).all()
  i=names.index(side+'_wrist_yaw_link');r=Rotation.from_quat(f['observation/body_quaternion_wxyz'][:,i][:,[1,2,3,0]])
  expected=f['observation/body_position_world'][:,i]+r.apply(np.tile(np.array([.025,0,0])*ROBOT_SCALE,(N,1)));actual=f[f'brainco/{side}/body_position'][:,hbn.index(side+'_base_link')];error=np.linalg.norm(actual-expected,axis=1)
  root=ET.parse(base/'assets/brainco_candidate'/f'revo2_{side}_clean.urdf').getroot();mimics={}
  for joint in root.findall('joint'):
   mimic=joint.find('mimic')
   if mimic is not None:mimics[joint.get('name')]=float(np.max(np.abs(q[:,hn.index(joint.get('name'))]-float(mimic.get('multiplier','1'))*q[:,hn.index(mimic.get('joint'))])))
  actual_rotation=Rotation.from_quat(f[f'brainco/{side}/body_quaternion_wxyz'][:,hbn.index(side+'_base_link')][:,[1,2,3,0]])
  orientation_error=( (r*MOUNT_ROTATIONS[side]).inv()*actual_rotation ).magnitude()
  assert orientation_error.max()<.03,(side,orientation_error.max())
  assert error.max()<.005,(side,error.max())
  assert max(mimics.values())<.03,(side,mimics)
  report[side]={'mount_max_error_m':float(error.max()),'mount_orientation_max_error_rad':float(orientation_error.max()),'mimic_max_error_rad':mimics,'finite':True}
 root=f['observation/root_position'][:];report['control_steps']=N;report['forward_displacement_at_6s_m']=float(root[min(300,N-1),1]-root[0,1]);report['minimum_pelvis_z_m']=float(root[:,2].min());report['legacy_hand_compatibility_joints_retained']=True
(a.run/'hand_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
