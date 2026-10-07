from pathlib import Path
import json,sys,hashlib
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,Gf
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'code'))
from calibration_config import ROBOT_SCALE,ASSETS
from online_goals import Kinematics
from experiment import SONIC
from brainco_adapter import MOUNT_ROTATIONS,SPECS
# Use actual visual mesh samples for descriptor and whole-object bounds; exact collider samples separately.
s=Usd.Stage.Open(str(B/'assets/water_bottle_v1/bottle.usda')); x=UsdGeom.XformCache(); points=[]
for p in s.Traverse():
 if p.IsA(UsdGeom.Mesh):
  tf=x.GetLocalToWorldTransform(p); points.extend([tuple(tf.Transform(Gf.Vec3d(*v))) for v in UsdGeom.Mesh(p).GetPointsAttr().Get()])
points=np.asarray(points,dtype=np.float32);np.savez_compressed(B/'assets/water_bottle_v1/collision_source.npz',vertices=points)
col=s.GetPrimAtPath('/Bottle/Collision'); cyl=UsdGeom.Cylinder(col); tf=x.GetLocalToWorldTransform(col); radius=cyl.GetRadiusAttr().Get(); height=cyl.GetHeightAttr().Get(); cp=[]
for z in [-height/2,height/2]:
 for a in np.linspace(0,2*np.pi,128,endpoint=False):cp.append(tuple(tf.Transform(Gf.Vec3d(radius*np.cos(a),radius*np.sin(a),z))))
(B/'assets/water_bottle_v1/collision_hulls.json').write_text(json.dumps([{'vertices':cp}]))
kin=Kinematics(SONIC);assert np.isfinite(kin.model.body_pos).all()
# Independent bilateral anatomical directions: preserve forward and opposing thumb axes.
right=MOUNT_ROTATIONS['right'].as_matrix();left=MOUNT_ROTATIONS['left'].as_matrix()
assert np.allclose(right[:,2],left[:,2]) and np.allclose(right[:,0],-left[:,0])
checks={'robot_scale':ROBOT_SCALE,'bottle_visual_bounds_m':[points.min(0).tolist(),points.max(0).tolist()],'bottle_collision_bounds_m':[np.min(cp,axis=0).tolist(),np.max(cp,axis=0).tolist()],'bilateral_mount_forward_same':True,'bilateral_thumb_axes_opposed':True,'hands':{}}
for side in ['left','right']:
 spec=SPECS[side]; checks['hands'][side]={'active':len([x for x in spec.values() if not x['mimic']]),'mimic':len([x for x in spec.values() if x['mimic']]),'rotation':MOUNT_ROTATIONS[side].as_matrix().tolist()}
assert abs(np.ptp(points,axis=0)[2]-.23)<1e-6
(B/'reports/preparation.json').write_text(json.dumps(checks,indent=2));print(checks)
records=json.loads((B/'asset_manifest.json').read_text())
for r in records:r['sha256']=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()
(B/'asset_manifest.json').write_text(json.dumps(records,indent=2))
