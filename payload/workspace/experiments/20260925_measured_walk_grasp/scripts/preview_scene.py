import os,sys,json
from pathlib import Path
B=Path(__file__).resolve().parents[1]
driver='/tmp/g1-driver-580.173.02/usr/lib/x86_64-linux-gnu'
if not os.environ.get('SCENE_DRIVER_REEXEC'):
 e=os.environ.copy();e.update(SCENE_DRIVER_REEXEC='1',LD_LIBRARY_PATH=driver+':'+e.get('LD_LIBRARY_PATH',''),VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json');os.execve(sys.executable,[sys.executable,*sys.argv],e)
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True,enable_cameras=True,device='cuda:0');app=launcher.app
import carb,omni.usd
import omni.replicator.core as rep
from pxr import UsdGeom,Gf
import numpy as np
import imageio.v2 as imageio
from isaacsim.core.api import World
try:
 omni.usd.get_context().open_stage(str(B/'assets/scene.usda'))
 for _ in range(20):app.update()
 world=World(physics_dt=1/120,rendering_dt=1/25,stage_units_in_meters=1,physics_prim_path='/World/PhysicsScene');world.reset()
 settings=carb.settings.get_settings();settings.set_bool('/rtx/dataWindow/fitOutputToDataWindow',True);settings.set_float('/rtx/post/tonemap/exposureKey',.6)
 stage=omni.usd.get_context().get_stage();cam=UsdGeom.Camera(stage.GetPrimAtPath('/World/Camera'));cam.CreateHorizontalApertureAttr(36);cam.CreateVerticalApertureAttr(27);cam.CreateFocalLengthAttr(26)
 settings.set_string('/rtx/rendermode','RaytracedLighting');settings.set_int('/rtx/post/aa/op',3)
 rp=rep.create.render_product('/World/Camera',(1200,900));ann=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu');ann.attach([rp]);op=cam.GetOrderedXformOps()[0]
 cameras=[('overview',[.90,-1.85,1.40],[-.35,.06,1.02]),('left',[-1.1,-1.28,1.55],[.05,0,.85]),('table',[.66,-.85,1.2],[.0,.04,.83])]
 for name,eye,target in cameras:
  op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())
  for _ in range(12):world.render()
  rgb=np.asarray(ann.get_data())[:,:,:3];assert rgb.shape==(900,1200,3);imageio.imwrite(B/'assets'/f'preview_{name}.png',rgb)
  print('PREVIEW',name,flush=True)
finally:app.close()
