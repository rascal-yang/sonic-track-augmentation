"""Photo-constrained room; editable USD, PBR textures and separate collision geometry."""
from pathlib import Path
import json, math, hashlib
import cv2
import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdShade, UsdLux, UsdPhysics, Sdf, Gf

B = Path(__file__).resolve().parents[1]
A = B / 'assets'
T = A / 'textures'
rng = np.random.default_rng(17)
# Surface patches retain photographic appearance. They are not calibrated albedo.
photo = cv2.imread(str(B/'inputs/room_03.jpg'))
floor_photo=cv2.imread(str(B/'inputs/room_04.jpg'))
patch = floor_photo[750:920,1160:1370].copy()
low=cv2.GaussianBlur(patch.astype(np.float32),(0,0),25)
patch=np.uint8(np.clip(patch.astype(np.float32)/np.maximum(low,1)*np.array([70,65,58]),0,255))
patch = cv2.resize(patch, (1024,1024))
# Periodic mirrored tile avoids cut seams while preserving observed granules.
patch = np.concatenate([patch, patch[:,::-1]],1)
patch = np.concatenate([patch,patch[::-1]],0)
cv2.imwrite(str(T/'rubber_photo.jpg'),patch,[cv2.IMWRITE_JPEG_QUALITY,96])
gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY).astype(float)/255
# Fine normal/roughness maps are appearance estimates, never physical identification.
gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)*.5;gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)*.5
n=np.stack([-gx,-gy,np.ones_like(gx)],-1);n/=np.linalg.norm(n,axis=-1,keepdims=True)
cv2.imwrite(str(T/'rubber_normal.png'),((n[...,::-1]+1)*127.5).astype(np.uint8))
cv2.imwrite(str(T/'rubber_roughness.png'),np.uint8(np.clip(.63+.18*(1-gray),0,1)*255))
# Photo-derived exterior is a distant visual card, not reconstructed outdoor geometry.
im=cv2.imread(str(B/'inputs/room_01.jpg'));cv2.imwrite(str(T/'outside_photo.jpg'),im[0:229,112:378])
wall=cv2.imread(str(B/'inputs/room_01.jpg'))[365:605,1280:1515]
cv2.imwrite(str(T/'wall_photo.jpg'),cv2.resize(wall,(768,768)))
# Rectify the photographed tabletop, then extract an unobstructed wood patch.
H=cv2.getPerspectiveTransform(np.float32([[811,255],[1498,453],[1139,710],[522,344]]),np.float32([[0,0],[1107,0],[1107,578],[0,578]]))
rectified=cv2.warpPerspective(photo,H,(1107,578))
cv2.imwrite(str(T/'table_rectified_reference.jpg'),rectified)
cv2.imwrite(str(T/'wood_photo.jpg'),cv2.resize(rectified[330:545,750:1060],(1024,512)))


s=Usd.Stage.CreateNew(str(A/'room.usda'))
root=UsdGeom.Xform.Define(s,'/Room');s.SetDefaultPrim(root.GetPrim())
UsdGeom.SetStageUpAxis(s,'Z');UsdGeom.SetStageMetersPerUnit(s,1)
manifest=[]

def mat(name,color,rough=.65,metal=0,texture=None,normal=None,roughtex=None,opacity=1):
    p='/Room/Materials/'+name;m=UsdShade.Material.Define(s,p)
    shader=UsdShade.Shader.Define(s,p+'/Surface');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(rough)
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(metal)
    shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(opacity)
    m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    if texture or normal or roughtex:
        uv=UsdShade.Shader.Define(s,p+'/UV');uv.CreateIdAttr('UsdPrimvarReader_float2');uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');uv.CreateOutput('result',Sdf.ValueTypeNames.Float2)
        for kind,file,output in [('diffuseColor',texture,'rgb'),('normal',normal,'rgb'),('roughness',roughtex,'r')]:
            if file:
                tex=UsdShade.Shader.Define(s,p+'/'+kind);tex.CreateIdAttr('UsdUVTexture')
                tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('textures/'+file))
                tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB' if kind=='diffuseColor' else 'raw')
                tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
                tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat');tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
                if kind=='normal':
                    tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set((2,2,2,1));tex.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set((-1,-1,-1,0))
                tex.CreateOutput(output,Sdf.ValueTypeNames.Float3 if output=='rgb' else Sdf.ValueTypeNames.Float)
                shader.CreateInput(kind,Sdf.ValueTypeNames.Normal3f if kind=='normal' else (Sdf.ValueTypeNames.Color3f if kind=='diffuseColor' else Sdf.ValueTypeNames.Float)).ConnectToSource(tex.ConnectableAPI(),output)
    return m

paint=mat('WarmPaint',(.75,.72,.66),.9)
wallmat=mat('WallPhoto',(.75,.72,.66),.94,texture='wall_photo.jpg')
rubber=mat('RubberPhoto',(.035,.04,.043),.8,texture='rubber_photo.jpg',normal='rubber_normal.png',roughtex='rubber_roughness.png')
seam=mat('MatSeam',(.009,.012,.014),.94)
white=mat('PVCFrame',(.78,.8,.77),.32)
metal=mat('Aluminium',(.37,.4,.4),.28,.8)
black=mat('Gasket',(.018,.024,.024),.8)
glass=mat('GlassApprox',(.78,.9,.93),.08,.1,opacity=.10)
wicker=[mat('Willow'+str(i),(.42+.033*i,.17+.015*i,.047+.006*i),.58) for i in range(5)]
inside=mat('ApplianceBlack',(.012,.022,.029),.22,.25)
exterior=mat('ExteriorPhoto',(.7,.7,.7),1,texture='outside_photo.jpg')

def semantic(prim,label,evidence):
    prim.CreateAttribute('semantic:Semantics:params:semanticType',Sdf.ValueTypeNames.String).Set('class')
    prim.CreateAttribute('semantic:Semantics:params:semanticData',Sdf.ValueTypeNames.String).Set(label)
    prim.AddAppliedSchema('SemanticsAPI:Semantics')
    prim.CreateAttribute('asset:evidence',Sdf.ValueTypeNames.String).Set(evidence)

def mesh(name,verts,faces,material,label='room',collide=False,uv=None,evidence='photo proportions; metric scale estimated'):
    p='/Room/'+name.replace('.', '_');m=UsdGeom.Mesh.Define(s,p)
    m.CreatePointsAttr(np.asarray(verts,dtype=np.float32));m.CreateFaceVertexCountsAttr([len(f) for f in faces]);m.CreateFaceVertexIndicesAttr(np.concatenate(faces).astype(int));m.CreateSubdivisionSchemeAttr('none')
    m.CreateDoubleSidedAttr(True)
    if uv is not None:UsdGeom.PrimvarsAPI(m).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'faceVarying').Set(np.asarray(uv,dtype=np.float32))
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    semantic(m.GetPrim(),label,evidence)
    if collide:UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    return m

def box(name,pos,size,material,label='room',collide=True,rot=0,uvscale=1,evidence='photo proportions; metric scale estimated'):
    tm=trimesh.creation.box(extents=size);verts=tm.vertices.copy()
    c=math.cos(rot);sn=math.sin(rot);verts[:,:2]=verts[:,:2]@np.array([[c,sn],[-sn,c]])
    verts+=pos
    uv=[]
    for face,normal in zip(tm.faces,tm.face_normals):
        ids=[i for i in range(3) if i!=np.argmax(abs(normal))]
        uv.extend(((tm.vertices[face][:,ids]/np.array(size)[ids]+.5)*uvscale).tolist())
    m=mesh(name,verts,tm.faces,material,label,collide,uv,evidence)
    manifest.append(dict(name=name,center=pos,size=size,collision=collide,class_name=label,evidence=evidence))
    return m

def tubes(name,paths,radius,material,label='basket',sides=6,collide=False):
    verts=[];faces=[]
    for path in paths:
        path=np.asarray(path)
        if np.max(np.ptp(path,axis=0))<1e-7: continue
        direc=np.gradient(path,axis=0);direc/=np.maximum(np.linalg.norm(direc,axis=1,keepdims=True),1e-9)
        guide=np.tile([0,0,1.],(len(path),1));guide[abs(direc[:,2])>.9]=[0,1.,0]
        u=np.cross(direc,guide);u/=np.linalg.norm(u,axis=1,keepdims=True);v=np.cross(direc,u)
        ring=path[:,None,:]+radius*(np.cos(np.arange(sides)*2*np.pi/sides)[None,:,None]*u[:,None,:]+np.sin(np.arange(sides)*2*np.pi/sides)[None,:,None]*v[:,None,:])
        off=len(verts);verts.extend(ring.reshape(-1,3))
        for i in range(len(path)-1):
            for k in range(sides):
                a=off+i*sides+k;b=off+i*sides+(k+1)%sides;c=b+sides;d=a+sides
                faces.extend([[a,b,c],[a,c,d]])
    return mesh(name,verts,faces,material,label,collide)

# Z=0 remains the original simulation support plane; the visual mats are just below it.
box('FloorFoundation',[-.7,-1.55,-.035],[4.9,3.8,.06],seam,'floor')
for i,x in enumerate(np.arange(-2.95,1.8,.5)):
    for j,y in enumerate(np.arange(-3.25,.5,.5)):
        box(f'Mats/Tile_{i}_{j}',[float(x),float(y),-.005],[.497,.497,.009],rubber,'floor_mat',False,uvscale=1)
# Back wall with a genuine window opening.
box('Walls/BelowWindow',[-.7,.37,.55],[4.9,.12,1.10],paint,'wall')
box('Walls/AboveWindow',[-.7,.37,2.4],[4.9,.12,.80],paint,'wall')
box('Walls/RightOfWindow',[.57,.37,1.55],[1.16,.12,.90],paint,'wall')
box('Walls/LeftOfWindow',[-2.23,.37,1.55],[1.84,.12,.90],paint,'wall')
box('Walls/RightReturn',[1.20,-1.55,1.4],[.12,3.8,2.8],paint,'wall')
box('Walls/Column',[1.065,-1.13,1.4],[.23,.50,2.8],paint,'pillar')
box('Walls/SkirtingBack',[-.7,.299,.027],[4.9,.014,.054],white,'skirting')
box('Walls/SkirtingSide',[1.131,-1.55,.027],[.014,3.8,.054],white,'skirting')
box('Walls/ColumnSkirting',[.945,-1.13,.027],[.012,.52,.054],white,'skirting')
# Unseen boundaries retained as separate explicit inferred enclosure.
box('Inferred/LeftWall',[-3.1,-1.55,1.4],[.10,3.8,2.8],paint,'wall',evidence='unobserved room boundary, assumed')
box('Inferred/FrontWall',[-.95,-3.45,1.4],[4.4,.1,2.8],paint,'wall',evidence='unobserved room boundary, assumed')
box('Inferred/Ceiling',[-.95,-1.55,2.83],[4.4,3.8,.06],paint,'ceiling',evidence='ceiling not photographed, assumed')
# Revised smaller window: opening x[-1.31,-.01], z[1.10,2.00].
box('Window/Sill',[-.66,.27,1.09],[1.43,.26,.036],white,'window_sill')
for name,pos,size in [('Bottom',[-.66,.32,1.13],[1.30,.075,.055]),('Top',[-.66,.32,1.97],[1.30,.075,.055]),('Left',[-1.283,.32,1.55],[.055,.075,.90]),('Right',[-.037,.32,1.55],[.055,.075,.90]),('Mullion',[-.66,.32,1.55],[.055,.075,.90])]:
    box('Window/'+name,pos,size,white,'window_frame')
box('Window/FixedGlass',[-.965,.33,1.55],[.555,.006,.78],glass,'glass')
hinge=np.array([-.065,.29]);ang=np.deg2rad(45);rr=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
def leaf(name,local,size,material):
    p=hinge+rr@np.array(local[:2]);return box('Window/OpenLeaf/'+name,[float(p[0]),float(p[1]),local[2]],size,material,'window',rot=ang)
for name,loc,sz in [('Bottom',[-.285,0,1.16],[.60,.05,.055]),('Top',[-.285,0,1.94],[.60,.05,.055]),('HingeStile',[0,0,1.55],[.055,.05,.835]),('LatchStile',[-.57,0,1.55],[.055,.05,.835])]:leaf(name,loc,sz,white)
leaf('Glass',[-.285,0,1.55],[.515,.007,.725],glass)
leaf('Handle',[-.525,-.045,1.49],[.026,.055,.15],metal)
for z in [1.25,1.84]:leaf('Hinge'+str(z),[.012,0,z],[.025,.065,.11],metal)
box('Exterior/PhotoCard',[-1.1,1.2,1.6],[3.4,.008,2.0],exterior,'outside_visual',False,evidence='source image crop on distant card; outside geometry not reconstructed')
z=np.linspace(1.15,2.0,100);cord=np.c_[np.full(100,.018),np.full(100,.245),z]
tubes('Window/BlindCord',[cord],.0014,white,'window_detail',5)
# Woven basket to the right/rear of the toy. Hollow physical structure.
center=np.array([.25,.075,.755]);rx,ry=.155,.105;height=.145
horizontal=[[] for _ in wicker]
for j,z in enumerate(np.linspace(.009,height,44)):
    th=np.linspace(0,2*np.pi,321);scale=.82+.18*z/height
    ripple=.0010*np.sin(th*96+j*np.pi)
    pts=np.c_[(rx*scale+ripple)*np.cos(th),(ry*scale+ripple)*np.sin(th),np.full_like(th,z)]+center
    horizontal[j%5].append(pts)
for j,paths in enumerate(horizontal):tubes('Basket/Weft'+str(j),paths,.0011,wicker[j])
vertical=[]
for theta in np.linspace(0,2*np.pi,96,endpoint=False):
    zs=np.linspace(0,height,120);scale=.82+.18*zs/height;ripple=.0015*np.sin(zs/height*44*np.pi)
    vertical.append(np.c_[(rx*scale+ripple)*np.cos(theta),(ry*scale+ripple)*np.sin(theta),zs]+center)
tubes('Basket/Warp',vertical,.00145,wicker[2])
for z in [.006,height]:
    th=np.linspace(0,2*np.pi,180);sc=.82 if z<.01 else 1
    tubes('Basket/Rim'+str(z),[np.c_[rx*sc*np.cos(th),ry*sc*np.sin(th),np.full_like(th,z)]+center],.0035,wicker[3],sides=10)
# Dense woven base, not a solid visual fill.
paths=[]
for x in np.arange(-rx*.8,rx*.8,.005):
    yy=ry*.8*np.sqrt(max(0,1-(x/(rx*.8))**2));paths.append(np.array([[x,-yy,.003],[x,yy,.003]])+center)
for y in np.arange(-ry*.8,ry*.8,.005):
    xx=rx*.8*np.sqrt(max(0,1-(y/(ry*.8))**2));paths.append(np.array([[-xx,y,.005],[xx,y,.005]])+center)
tubes('Basket/BaseWeave',paths,.0018,wicker[1])
# Arched wrapped handle, plane along short axis (as seen in side photo).
th=np.linspace(0,np.pi,160)
path=np.c_[np.zeros_like(th),ry*np.cos(th),height+.25*np.sin(th)]+center
tubes('Basket/HandleCore',[path],.0055,wicker[2],sides=10)
coils=[]
for i in range(150):
    t=(i+.5)/150*np.pi;middle=np.array([0,ry*np.cos(t),height+.25*np.sin(t)])+center
    tangent=np.array([0,-ry*np.sin(t),.25*np.cos(t)]);tangent/=np.linalg.norm(tangent);u=np.array([1.,0,0]);v=np.cross(tangent,u)
    phi=np.linspace(0,2*np.pi,12);coils.append(middle+.0058*(np.cos(phi)[:,None]*u+np.sin(phi)[:,None]*v))
tubes('Basket/HandleWrapping',coils,.0008,wicker[3],sides=5)
# Hidden collider ring with a genuinely empty interior; no full-basket convex hull.
coll=UsdGeom.Xform.Define(s,'/Room/Basket/Collision');coll.CreateVisibilityAttr('invisible')
for i,t in enumerate(np.linspace(0,2*np.pi,40,endpoint=False)):
    x,y=rx*.90*np.cos(t),ry*.90*np.sin(t)
    tm=trimesh.creation.box(extents=[.026,.005,height]);angle=math.atan2(ry*np.cos(t),-rx*np.sin(t))
    c,sn=np.cos(angle),np.sin(angle);v=tm.vertices.copy();v[:,:2]=v[:,:2]@np.array([[c,sn],[-sn,c]]);v+=center+[x,y,height/2]
    m=mesh('Basket/Collision/Side'+str(i),v,tm.faces,wicker[0],'basket_collision',True)
box('Basket/Collision/Base',(center+[0,0,.001]).tolist(),[rx*1.6,ry*1.6,.005],wicker[0],'basket_collision')
# Left-side appliance, tray on top, cable and wall socket (partially photographed).
box('Appliance/Body',[-2.25,-.03,.39],[.45,.46,.78],white,'appliance')
box('Appliance/FrontGlass',[-2.25,-.267,.43],[.34,.01,.36],inside,'appliance_window')
for i in range(3):box('Appliance/Vent'+str(i),[-2.25,-.277,.65+i*.025],[.36,.012,.009],black,'appliance_detail',False)
box('Appliance/TopPrinter',[-2.25,-.02,.87],[.33,.31,.16],inside,'printer')
box('Appliance/PaperTray',[-2.25,.06,1.0],[.29,.025,.15],black,'printer')
box('Socket/Base',[-1.9,.292,.72],[.066,.018,.10],white,'socket')
box('Socket/Plug',[-1.9,.266,.70],[.027,.035,.042],black,'plug')
cable=np.array([[-1.9,.24,.68],[-1.91,.22,.49],[-1.85,.17,.30],[-1.88,.12,.08],[-2.02,.04,.015],[-2.20,-.10,.02]])
# Interpolated cable curve.
from scipy.interpolate import make_interp_spline
cable=make_interp_spline(np.arange(len(cable)),cable,k=2)(np.linspace(0,len(cable)-1,90))
tubes('Appliance/PowerCable',[cable],.003,black,'cable',7)
for i in range(2):box('LooseMats/Piece'+str(i),[-2.0+i*.08,-.73+i*.03,.013+i*.013],[.36,.28,.024],rubber,'loose_mat',rot=.16+i*.15)
box('FloorDebris/Paper',[.45,-1.15,.002],[.025,.013,.001],white,'paper',False,rot=.4)
# Light enters through the opening; bounded exposure avoids washed-out tabletop.
key=UsdLux.RectLight.Define(s,'/Room/Lights/WindowDaylight');key.CreateWidthAttr(1.6);key.CreateHeightAttr(1.05);key.CreateIntensityAttr(1600);key.CreateColorAttr((.91,.95,1.0))
key.AddTranslateOp().Set(Gf.Vec3d(-.95,.26,1.62));key.AddRotateXYZOp().Set(Gf.Vec3f(-90,0,0))
ceiling=UsdLux.RectLight.Define(s,'/Room/Lights/InteriorFill');ceiling.CreateWidthAttr(2.5);ceiling.CreateHeightAttr(2.5);ceiling.CreateIntensityAttr(1100);ceiling.CreateColorAttr((1.,.96,.9));ceiling.AddTranslateOp().Set(Gf.Vec3d(-.5,-.8,2.65))
fill=UsdLux.DomeLight.Define(s,'/Room/Lights/Ambient');fill.CreateIntensityAttr(180);fill.CreateColorAttr((.86,.9,1.0))
sun=UsdLux.DistantLight.Define(s,'/Room/Lights/Sun');sun.CreateIntensityAttr(550);sun.CreateAngleAttr(7);sun.AddRotateXYZOp().Set(Gf.Vec3f(35,-20,15))
for light in [key,ceiling]:light.GetPrim().CreateAttribute('visibleInPrimaryRay',Sdf.ValueTypeNames.Bool).Set(False)
s.GetRootLayer().Save()
# Reuse complete desk with all materials/colliders in a relocatable local wrapper.
ts=Usd.Stage.CreateNew(str(A/'table.usda'));t=UsdGeom.Xform.Define(ts,'/Table');ts.SetDefaultPrim(t.GetPrim());UsdGeom.SetStageUpAxis(ts,'Z');UsdGeom.SetStageMetersPerUnit(ts,1)
t.GetPrim().GetReferences().AddReference('reused/office_desk_v1/desk.usda','/Desk')
# Override only the local table material, retaining the original asset.
p=ts.GetPrimAtPath('/Table/Visuals/_materials/Source_video_light_wood/Image_Texture');p.GetAttribute('inputs:file').Set(Sdf.AssetPath('textures/wood_photo.jpg'))
shader=ts.GetPrimAtPath('/Table/Visuals/_materials/Source_video_light_wood/Principled_BSDF');shader.GetAttribute('inputs:roughness').Set(.40)
ts.GetRootLayer().Save()
# Complete, directly openable Isaac Sim scene with a useful camera and saved toy pose.
scene=Usd.Stage.CreateNew(str(A/'scene.usda'));w=UsdGeom.Xform.Define(scene,'/World');scene.SetDefaultPrim(w.GetPrim());UsdGeom.SetStageUpAxis(scene,'Z');UsdGeom.SetStageMetersPerUnit(scene,1)
pose=json.loads((A/'reused/wanou_v1/validation_isaac.json').read_text())
# Move reused toy to the left of basket, preserving settled attitude.
shift=np.array([-.15,0,0]);pose['settled_position_m']=(np.array(pose['settled_position_m'])+shift).tolist()
for name,file,prim in [('Room','room.usda','/Room'),('Table','table.usda','/Table'),('Plush','reused/wanou_v1/plush.usda','/Plush')]:
    x=UsdGeom.Xform.Define(scene,'/World/'+name);x.GetPrim().GetReferences().AddReference(file,prim)
    if name=='Plush':
        x.ClearXformOpOrder();x.AddTranslateOp().Set(Gf.Vec3d(*pose['settled_position_m']));q=pose['settled_quaternion_wxyz'];x.AddOrientOp().Set(Gf.Quatf(q[0],Gf.Vec3f(*q[1:])))
cam=UsdGeom.Camera.Define(scene,'/World/Camera');cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.02,-1.8,1.45),Gf.Vec3d(-.35,.05,.92),Gf.Vec3d(0,0,1)).GetInverse());cam.CreateFocalLengthAttr(22);cam.CreateClippingRangeAttr((.01,100))
physics=UsdPhysics.Scene.Define(scene,'/World/PhysicsScene');physics.CreateGravityDirectionAttr((0,0,-1));physics.CreateGravityMagnitudeAttr(9.81)
scene.GetRootLayer().Save()
config={'window_room':{'table_center':[0,0,0],'table_size':[1.1070822169594245,.578083867209969,.75],'room_asset':str(A/'room.usda'),'table_asset':str(A/'table.usda'),'object_shift':shift.tolist(),'estimated':True}}
(B/'rooms.json').write_text(json.dumps(config,indent=2))
(A/'scene_manifest.json').write_text(json.dumps({'components':manifest,'toy_pose':pose,'scale_anchor':'previous desk height ASSUMED 0.75m, not measured','basket_estimates_m':{'radii':[rx,ry],'body_height':height,'handle_rise':.25},'limits':['4 photos only; no metric survey','unobserved enclosure inferred','outdoor is photo card','textures retain source illumination','rigid basket and plush; friction estimated'],'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (B/'inputs').glob('*.jpg')}},indent=2))
print('Built',A/'scene.usda','components',len(manifest))
