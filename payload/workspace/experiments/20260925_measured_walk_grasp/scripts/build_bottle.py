"""Deterministic metric bottle proxy. Original geometry; no fluid simulation."""
from pathlib import Path
import json,numpy as np,trimesh
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Sdf,Gf
A=Path(__file__).resolve().parents[1]/'assets/water_bottle_v1';A.mkdir(exist_ok=True)
s=Usd.Stage.CreateNew(str(A/'bottle.usda'));root=UsdGeom.Xform.Define(s,'/Bottle');s.SetDefaultPrim(root.GetPrim());UsdGeom.SetStageUpAxis(s,'Z');UsdGeom.SetStageMetersPerUnit(s,1)
UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(True);UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(.25)
def material(name,color,rough=.3):
 m=UsdShade.Material.Define(s,'/Bottle/Looks/'+name);sh=UsdShade.Shader.Define(s,m.GetPath().AppendChild('Shader'));sh.CreateIdAttr('UsdPreviewSurface');sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(color);sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(rough);m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface');return m
blue=material('BlueCap',(.015,.07,.4));body=material('BlueTintPET',(.38,.69,.83),.18);label=material('WhiteBlueLabel',(.85,.93,.98),.55)
phys=UsdShade.Material.Define(s,'/Bottle/PhysicsMaterial');api=UsdPhysics.MaterialAPI.Apply(phys.GetPrim());api.CreateStaticFrictionAttr(.6);api.CreateDynamicFrictionAttr(.5);api.CreateRestitutionAttr(.02)
verts=[];hulls=[]
def mesh(name,profile,mat,collide=True):
 t=trimesh.creation.revolve(np.array(profile),sections=64);m=UsdGeom.Mesh.Define(s,'/Bottle/'+name);m.CreatePointsAttr(t.vertices.tolist());m.CreateFaceVertexCountsAttr([3]*len(t.faces));m.CreateFaceVertexIndicesAttr(t.faces.flatten().tolist());m.CreateSubdivisionSchemeAttr('none');UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
 if collide:
  UsdPhysics.CollisionAPI.Apply(m.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull');UsdShade.MaterialBindingAPI(m.GetPrim()).Bind(phys,materialPurpose='physics');verts.append(t.vertices);hulls.append({'vertices':t.vertices.tolist()})
 return t
mesh('Body',[(0,0),(.026,0),(.030,.008),(.030,.16),(.029,.169),(0,.169)],body)
mesh('ShoulderNeck',[(0,.165),(.029,.165),(.015,.19),(.015,.21),(0,.21)],body)
mesh('Cap',[(0,.207),(.017,.207),(.017,.22),(0,.22)],blue)
mesh('Label',[(.0301,.052),(.0301,.116)],label,False)
v=np.vstack(verts);np.savez(A/'collision_source.npz',vertices=v);(A/'collision_hulls.json').write_text(json.dumps(hulls));s.GetRootLayer().Save()
(A/'validation_isaac.json').write_text(json.dumps({'settled_position_m':[.05,-.15,.752],'settled_quaternion_wxyz':[1,0,0,0],'pose_status':'initial spawn, settling checked in rollout; not prevalidated'}))
(A.parent/'protrusion_regions.json').write_text(json.dumps({'regions':[{'name':n,'lo':[-.04,-.04,lo],'hi':[.04,.04,hi]} for n,lo,hi in [('body',.008,.169),('neck',.19,.21),('cap',.207,.22)]]}))
(A/'manifest.json').write_text(json.dumps({'source':'original procedural geometry created for this experiment','height_m':.22,'diameter_m':.06,'mass_kg_assumed':.25,'static_friction_assumed':.6,'dynamic_friction_assumed':.5,'limitations':['rigid proxy','water represented only by total mass','no measured real bottle','opaque tinted PET visual approximation'],'geometry_bounds_m':[v.min(0).tolist(),v.max(0).tolist()]},indent=2))
print(A)
