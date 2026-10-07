from pathlib import Path
from pxr import Usd,UsdGeom,Gf,Vt
import json
B=Path(__file__).resolve().parents[1];A=B.parents[1]/'asset_tasks/20260925_measured_scene/assets'
z=json.loads((A/'calibration.json').read_text())['bottle']['scale_xyz'][2]
s=Usd.Stage.Open(str(A/'source_scene/bottle/bottle.usda'));flat=s.Flatten();flat.Export(str(B/'assets/water_bottle_v1/bottle.usda'))
s=Usd.Stage.Open(str(B/'assets/water_bottle_v1/bottle.usda'))
for p in s.Traverse():
 if p.IsA(UsdGeom.Mesh):
  mesh=UsdGeom.Mesh(p);mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(v[0],v[1],v[2]*z) for v in mesh.GetPointsAttr().Get()]))
 if p.IsA(UsdGeom.Cylinder):
  c=UsdGeom.Cylinder(p);c.GetHeightAttr().Set(c.GetHeightAttr().Get()*z)
 for op in UsdGeom.Xformable(p).GetOrderedXformOps():
  if op.GetOpType()==UsdGeom.XformOp.TypeTranslate:
   v=op.Get();op.Set(Gf.Vec3d(v[0],v[1],v[2]*z))
s.GetRootLayer().Save()
b=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeWorldBound(s.GetDefaultPrim()).ComputeAlignedRange();assert abs(b.GetSize()[2]-.23)<1e-6
(B/'reports/baked_bottle.json').write_text(json.dumps({'reason':'GeometryObject resets root transform order; bake calibrated Z scale into mesh/cylinder geometry to preserve 23 cm in runtime','height_m':b.GetSize()[2],'root_scale':[1,1,1]},indent=2))
print('BAKED',b.GetSize())
