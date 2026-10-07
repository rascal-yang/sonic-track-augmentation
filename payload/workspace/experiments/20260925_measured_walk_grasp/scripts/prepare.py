from pathlib import Path
import shutil,json,hashlib,xml.etree.ElementTree as ET
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,Gf
B=Path(__file__).resolve().parents[1]; W=B.parents[1]; S=W/'experiments/20260921_new_scene_closed_loop'; A=W/'asset_tasks/20260925_measured_scene/assets'
cal=json.loads((A/'calibration.json').read_text()); scale=cal['g1']['uniform_scale']; bs=cal['bottle']['scale_xyz'][2]
for name in ['code','scripts']:
 shutil.copytree(S/name,B/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','setup.py','prepare.py'))
for name in ['assets','runs','reports']:(B/name).mkdir(exist_ok=True)
shutil.copy2(S/'pipeline.sh',B/'pipeline.sh')
for name in ['brainco_candidate','brainco_usd']:
 (B/'assets'/name).symlink_to((S/'assets'/name).resolve())
shutil.copytree(S/'assets/water_bottle_v1',B/'assets/water_bottle_v1')
f=B/'assets/water_bottle_v1/bottle.usda'; f.unlink(); st=Usd.Stage.CreateNew(str(f)); root=UsdGeom.Xform.Define(st,'/Bottle');root.GetPrim().GetReferences().AddReference(str(A/'bottle/bottle.usda'),'/Bottle');st.SetDefaultPrim(root.GetPrim());UsdGeom.SetStageMetersPerUnit(st,1);UsdGeom.SetStageUpAxis(st,'Z');st.GetRootLayer().Save()
f=B/'assets/water_bottle_v1/collision_source.npz';data=dict(np.load(f));print('old bottle geometry',np.ptp(data['vertices'],axis=0));data['vertices'][:,2]*=bs;np.savez_compressed(f,**data)
f=B/'assets/water_bottle_v1/collision_hulls.json';obj=json.loads(f.read_text());print('hull metadata type',type(obj).__name__)
# collision_source supplies BPS, semantic regions and full-object lift checks; USD owns actual collider.
shutil.copy2(S/'assets/protrusion_regions.json',B/'assets/protrusion_regions.json')
f=B/'assets/protrusion_regions.json'; d=json.loads(f.read_text())
for region in d['regions']:
 for key in ['lo','hi']:region[key][2]*=bs
f.write_text(json.dumps(d,indent=2))
for name in ['room','table']:
 t=(S/'assets'/f'{name}.usda').read_text().replace(str(W/'asset_tasks/20260920_photo_scene/assets/scene.usda'),str(A/'scene.usda'));(B/'assets'/f'{name}.usda').write_text(t)
rooms=json.loads((S/'rooms.json').read_text()); rr=rooms['window_room'];rr.update(table_size=[1.12,.60,.75],estimated=False,dimension_source='User measurements 2026-09-25; mass and friction remain estimated')
for name in ['room','table']:rr[name+'_asset']=str(B/'assets'/f'{name}.usda')
(B/'rooms.json').write_text(json.dumps(rooms,indent=2))
stand=[.37,-.55,.76*scale]; protocol=json.loads((S/'protocol.json').read_text()); protocol.update(starts={'walk_001':[.37,-1.,.76*scale]},target_root_m=stand,evaluation_runs=['walk_001'],hint_source=str(S/'runs/walk_02_middle_v3'),navigation_paths={},supplementary_runs=[],interrupted_runs=[],development_navigation_only=[])
(B/'protocol.json').write_text(json.dumps(protocol,indent=2))
cfg=json.loads((S/'runs/walk_02_middle_v3/optimized_config.json').read_text());cfg.update(stand_root_m=stand,side_raise_m=[.595,-.455,.92],max_retries=0,version='measured_assets_old_success_prior_v1');cfg['palm_offset_from_bottle_m']=[.055*scale,-.04*scale,.11*bs];(B/'optimized_config.json').write_text(json.dumps(cfg,indent=2))
(B/'code/calibration_config.py').write_text(f'ROBOT_SCALE = {scale!r}\nASSETS = {str(A)!r}\n')
def patch(name,old,new):
 f=B/'code'/name;t=f.read_text();assert old in t,(name,old);f.write_text(t.replace(old,new))
patch('runtime_adapter.py',"mod.configure_scene_assets(cfg)","mod.configure_scene_assets(cfg)\n                from calibration_config import ASSETS\n                cfg.robot.spawn.usd_path=str(Path(ASSETS)/'robot/g1.usda')\n                for side in ['left','right']:\n                    getattr(cfg,'brainco_'+side).spawn.usd_path=str(Path(ASSETS)/f'robot/revo2_{side}.usda')")
patch('online_goals.py','import json','import json\nfrom calibration_config import ROBOT_SCALE')
patch('online_goals.py','self.data = mujoco.MjData(self.model)', 'self.model.body_pos[:] *= ROBOT_SCALE\n        self.model.body_ipos[:] *= ROBOT_SCALE\n        self.model.jnt_pos[:] *= ROBOT_SCALE\n        self.model.body_inertia[:] *= ROBOT_SCALE**2\n        self.data = mujoco.MjData(self.model)\n        mujoco.mj_setConst(self.model,self.data)')
patch('online_goals.py','np.array([.07,0,0])','np.array([.07,0,0])*ROBOT_SCALE')
patch('online_goals.py','apply([.07,0,0])','apply(np.array([.07,0,0])*ROBOT_SCALE)')
patch('walk_grasp.py','self.stand_root=np.array([.37,-.54,.76])',f'self.stand_root=np.array({stand!r})')
patch('brainco_adapter.py','import json','import json\nfrom calibration_config import ROBOT_SCALE')
# Joint localPos0 is in scaled wrist-local USD units, while reset world offset is scaled explicitly.
patch('brainco_adapter.py','p+r.apply(MOUNT_P)','p+r.apply(MOUNT_P*ROBOT_SCALE)')
patch('brainco_adapter.py','extension.hand_offset=np.array([.075,.02,0])','extension.hand_offset=np.array([.075,.02,0])*ROBOT_SCALE')
patch('protrusion_geometry.py','import json','import json\nfrom calibration_config import ROBOT_SCALE')
patch('protrusion_geometry.py',"poses['right_' + f + '_tip'][1]", "poses['right_' + f + '_tip'][1]*ROBOT_SCALE")
patch('protrusion_geometry.py','np.array([.045, 0, 0])','np.array([.045, 0, 0])*ROBOT_SCALE')
patch('desk_plush_experiment.py',"'bottle_height_assumed_m':.22","'bottle_height_measured_m':.23")
patch('stationary_hint.py','historical_trajectories_loaded=False', 'historical_trajectories_loaded=False, historical_waypoint_prior=True')
patch('experiment.py',"'desk_plush_experiment.py'):","'desk_plush_experiment.py', 'calibration_config.py'):")
f=B/'scripts/validate_brainco.py';t=f.read_text().replace('from brainco_adapter import MOUNT_ROTATIONS','from brainco_adapter import MOUNT_ROTATIONS\nfrom calibration_config import ROBOT_SCALE').replace('np.tile([.025,0,0],(N,1))','np.tile(np.array([.025,0,0])*ROBOT_SCALE,(N,1))');f.write_text(t)
f=B/'scripts/optimized_protocol.py';t=f.read_text().replace("act('navigate',100", "act('navigate',25").replace("return act(steps=75,reason='Continue feedback walking')", "return act(steps=10,reason='Continue feedback walking; fine near-table observation')")
f.write_text(t)
# Copy actual reference evidence, not actuator commands.
(B/'reference').mkdir(exist_ok=True)
for name in ['optimized_config.json','summary.json','handoff_reference.npz']:
 if (S/'runs/walk_02_middle_v3'/name).exists():shutil.copy2(S/'runs/walk_02_middle_v3'/name,B/'reference'/name)
for name in ['bottle_upright_stationary004_reference.npz','bottle_upright_stationary004_reference.json']:shutil.copy2(W/'policy_memory'/name,B/'reference'/name)
records=[]
for p in [A/'scene.usda',A/'bottle/bottle.usda',A/'robot/g1.usda',A/'robot/revo2_left.usda',A/'robot/revo2_right.usda',B/'assets/water_bottle_v1/collision_source.npz']:
 records.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(B/'asset_manifest.json').write_text(json.dumps(records,indent=2))
(B/'plan.json').write_text(json.dumps({'task':'walk to desk and lift calibrated 23 cm bottle','prompt_version':'human_prior_v5','reference_run':str(S/'runs/walk_02_middle_v3'),'candidates':[{'strategy':'mid-body side wrap','selected':True,'reason':'matched prior clean success, upright rigid bottle'},{'strategy':'higher neck pinch','selected':False,'reason':'narrower neck and reduced opposing contact area, only consider after reach diagnosis'},{'strategy':'two-hand cradle','selected':False,'reason':'needs new bilateral contact strategy; no matching successful prior'}],'prediction':'Scaled actual-state IK and slowed feedback preserve clear approach, opposed closure and >=3 s full-bottle hold','changes':{'table_m':[1.12,.6,.75],'bottle_height_m':.23,'robot_height_m':1.3,'robot_scale':scale,'stand_root':stand,'hand_mount_reset_world_offset_scaled':True,'kinematics_scaled':True,'policy_weights':'unchanged SONIC; GRAIL zero','reference':'successful motion phase/waypoint prior; no historical actuator replay'},'matched_lessons':['upright_success','force_alone_insufficient','postwalk_loaded_vertical_reference_20260921','physics_substep_limiter_units_20260921'],'stop_guards':{'table_contact_N':.5,'pelvis_z_m':.45,'object_tilt_rad':.4,'max_batch_attempts':3},'rejected_hypotheses':['force alone proves grip','IK feasibility proves execution','reuse old global coordinates after scale change'],'known_limits':['SONIC trained on original morphology','mass/friction remain estimated','reference motion library internal FK retains original body model; live task IK uses scaled geometry']},indent=2))
print('PREPARED',B)
