"""Lean runtime: preserve controller/physics, omit diagnostic video and dense logs."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

from experiment import write_json


def install(evaluation, out, intern):
    original_create = evaluation.train_agent_trl.create_manager_env

    def create(config, device, args_cli):
        # Isaac Sim has started by this point; extension imports are now valid.
        import numpy as np
        import torch
        import imageio.v2 as imageio
        import isaaclab.sim as sim
        from isaaclab.sim.utils import clone
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension('omni.isaac.core')
        from gear_sonic.envs.manager_env import modular_tracking_env_cfg as scene_module
        from gear_sonic.envs.manager_env.mdp.recorders import RenderEnvsRecorderTerm
        from isaaclab.terrains import TerrainImporter

        # Avoid an online dependency on NVIDIA's default ground-plane USD.
        def local_ground(self, name, size=(200.0, 200.0)):
            path = self.cfg.prim_path + '/' + name
            self.terrain_prim_paths.append(path)
            current = sim.get_current_stage()
            floor = UsdGeom.Plane.Define(current, path)
            floor.CreateAxisAttr('Z')
            floor.CreateVisibilityAttr('invisible')
            floor.CreateWidthAttr(200.0); floor.CreateLengthAttr(200.0)
            floor.CreateDisplayColorAttr([Gf.Vec3f(0.28, 0.32, 0.36)])
            UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
            if self.cfg.physics_material is not None:
                material_path = path + '/PhysicsMaterial'
                self.cfg.physics_material.func(material_path, self.cfg.physics_material)
                sim.bind_physics_material(path, material_path)
        TerrainImporter.import_ground_plane = local_ground

        # Load only the two upstream modules needed. Importing core.objects.__init__
        # eagerly loads unrelated controllers and dependencies.
        source_dir = intern / 'workflows/simbox/core/objects'
        saved = {key: sys.modules.get(key) for key in ('core', 'core.objects', 'core.objects.base_object')}
        for key in ('core', 'core.objects'):
            mod = types.ModuleType(key); mod.__path__ = [str(source_dir)]
            sys.modules[key] = mod
        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
        try:
            load('core.objects.base_object', source_dir / 'base_object.py')
            geometry = load('interndata_geometry_object', source_dir / 'geometry_object.py')
        finally:
            for key, previous in saved.items():
                if previous is None: sys.modules.pop(key, None)
                else: sys.modules[key] = previous

        # Save an actual table USD, loaded using InternDataEngine GeometryObject.
        table_path = out / 'table.usda'
        stage = Usd.Stage.CreateNew(str(table_path))
        table = UsdGeom.Xform.Define(stage, '/Table')
        stage.SetDefaultPrim(table.GetPrim())
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        cube = UsdGeom.Cube.Define(stage, '/Table/Top')
        cube.CreateSizeAttr(1)
        cube.AddScaleOp().Set(Gf.Vec3f(*config.manager_env.config.table_size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.40, 0.23, 0.10)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(table.GetPrim())
        stage.GetRootLayer().Save()
        if os.environ.get('G1_DESK_PLUSH'):
            from desk_plush_experiment import make_table_wrapper
            del stage
            table_path = out / 'reconstructed_table.usda'
            make_table_wrapper(table_path)
        holders = []
        spawn_events = []

        @clone
        def interndata_spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
            asset = Path(cfg.usd_path)
            obj = geometry.GeometryObject(
                asset_root=str(asset.parent), root_prim_path=str(Path(prim_path).parent),
                cfg={'name': Path(prim_path).name, 'path': asset.name},
                translation=np.asarray(translation) if translation is not None else None,
                orientation=np.asarray(orientation) if orientation is not None else None)
            holders.append(obj)
            spawn_events.append({'prim': prim_path, 'asset': str(asset),
                                  'loader': str(source_dir / 'geometry_object.py')})
            write_json(out / 'scene_integration.json', spawn_events)
            # Isaac Lab applies rigid-body/contact properties to the prim already
            # created by InternDataEngine, then uses it in the shared PhysX scene.
            return sim.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)

        old_scene_init = scene_module.MySceneCfg.__init__
        def scene_init(self, *args, **kwargs):
            old_scene_init(self, *args, **kwargs)
            old = self.table.spawn
            self.table.spawn = sim.UsdFileCfg(
                usd_path=str(table_path), func=interndata_spawn,
                activate_contact_sensors=True, rigid_props=old.rigid_props,
                collision_props=old.collision_props, mass_props=old.mass_props)
            old = self.object.spawn
            usd_files = list((Path(os.environ['G1_EXPERIMENT_ROOT']) / 'data/pickup_table/object_usd').glob('*.usd'))
            if os.environ.get('G1_DESK_PLUSH'):
                usd_files = [__import__('common').BASE / 'assets/water_bottle_v1/bottle.usda']
            assert len(usd_files) == 1
            self.object.spawn = sim.UsdFileCfg(
                usd_path=str(usd_files[0]), func=interndata_spawn,
                activate_contact_sensors=True, rigid_props=old.rigid_props,
                collision_props=old.collision_props, mass_props=(sim.MassPropertiesCfg(mass=.25) if os.environ.get('G1_DESK_PLUSH') else old.mass_props))
        from grasp_extensions import configure_sensors
        base_scene_init = scene_init
        def enhanced_scene_init(self, *args, **kwargs):
            base_scene_init(self, *args, **kwargs)
            if os.environ.get('G1_REVO2') == '1':
                from brainco_adapter import configure_scene
                configure_scene(self)
            else:
                configure_sensors(self)
            from isaaclab.assets import AssetBaseCfg
            from desk_plush_experiment import ROOM
            self.room=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Room',spawn=sim.UsdFileCfg(usd_path=ROOM['room_asset']))
            write_json(out/'room_setup.json',ROOM)
            self.eval_camera = None
            from isaaclab.assets import RigidObjectCfg
        scene_module.MySceneCfg.__init__ = enhanced_scene_init

        state = {'step': 0, 'observation': 0, 'until': 0, 'initial_z': None,
                 'peak_lift': 0.0, 'held_steps': 0, 'max_held_steps': 0,
                 'ever_low_pelvis': False, 'stop': False, 'error': None, 'ended': False, 'termination_terms': []}
        planner = []
        comparison = []
        video_recorder = []
        capture = []
        from auto_policy import AutoGraspPolicy
        auto_policy = AutoGraspPolicy()
        telemetry = open(os.devnull, 'w')

        def observe(env):
            robot = env.scene['robot'].data
            obj = env.scene['object'].data
            table_data = env.scene['table'].data
            pos = obj.root_pos_w[0].detach().cpu().tolist()
            root = robot.root_pos_w[0].detach().cpu().tolist()
            data = {'step': state['step'], 'sim_time': state['step'] * env.step_dt,
                    'robot_root_pos': root, 'robot_root_quat_wxyz': robot.root_quat_w[0].detach().cpu().tolist(),
                    'object_pos': pos, 'object_quat_wxyz': obj.root_quat_w[0].detach().cpu().tolist(),
                    'table_pos': table_data.root_pos_w[0].detach().cpu().tolist(),
                    'joint_pos': robot.joint_pos[0].detach().cpu().tolist(),
                    'peak_lift_m': state['peak_lift'], 'ever_low_pelvis': state['ever_low_pelvis']}
            if planner: data.update(planner[0].observation())
            if comparison: data.update(comparison[0].observation())
            return data

        old_record = RenderEnvsRecorderTerm.record_post_step
        def record(self):
            # Invoked after physics, before auto-reset; no render products or video.
            if capture and capture[0].active:
                capture[0].post_step()
            return "record_post_step", torch.ones(self.env.num_envs, 1, device=self.env.device)
        RenderEnvsRecorderTerm.record_post_step = record

        if os.environ.get('G1_REVO2') == '1':
            from isaaclab.scene import InteractiveScene
            from brainco_adapter import patch_robot_spawn
            original_scene_constructor=InteractiveScene.__init__
            def scene_constructor(self,cfg):
                patch_robot_spawn(cfg)
                import importlib.util
                spec=importlib.util.spec_from_file_location('photo_integrate','__RUNTIME__/workspace/asset_tasks/20260921_robot_materials/scripts/integrate.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
                mod.configure_scene_assets(cfg)
                from calibration_config import ASSETS
                cfg.robot.spawn.usd_path=str(Path(ASSETS)/'robot/g1.usda')
                for side in ['left','right']:
                    getattr(cfg,'brainco_'+side).spawn.usd_path=str(Path(ASSETS)/f'robot/revo2_{side}.usda')
                original_scene_constructor(self,cfg)
            InteractiveScene.__init__=scene_constructor
        wrapped = original_create(config, device, args_cli)
        # Keep source-backed GeometryObject wrappers alive throughout the scene.
        wrapped._interndata_objects = holders
        if os.environ.get('G1_ONLINE_GOALS'):
            from online_goals import OnlinePlanner
            if os.environ.get('G1_SONIC_PLANNER'):
                from walk_grasp import WalkGraspPlanner as SonicPlannerIK
                if os.environ.get('G1_STATIONARY')=='1':
                    from direct_bottle import DirectBottlePlanner as SonicPlannerIK
                planner.append(SonicPlannerIK(wrapped.env, out, Path.cwd()))
            else:
                planner.append(OnlinePlanner(wrapped.env, out, Path.cwd()))
        from grasp_extensions import Extension
        extension = Extension(wrapped, planner[0], out)
        if os.environ.get('G1_REVO2') == '1':
            from brainco_adapter import install_extension
            install_extension(extension)
        from training_capture import TrainingCapture
        capture.append(TrainingCapture(wrapped.env, out))
        from contact_watch import ContactWatch
        contact_watch=ContactWatch(wrapped.env, capture[0], out)
        from pxr import UsdLux
        current_stage=sim.get_current_stage()
        for prim in current_stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI) and "/Room/" not in str(prim.GetPath()):
                UsdLux.LightAPI(prim).GetIntensityAttr().Set(0)
        term_manager = wrapped.env.termination_manager
        retry = os.environ.get('G1_COMPARE_PROTOCOL') == 'v3_retry'
        events = {}
        if retry or os.environ.get('G1_DESK_PLUSH'):
            # Wrap only two functions, leaving the manager's buffers and retained
            # termination functions intact. Record every positive step compactly.
            for name in ('object_pos_deviation', 'hand_table_contact_termination', 'ee_body_pos'):
                if name not in term_manager.active_terms: continue
                cfg = term_manager.get_term_cfg(name)
                original_func = cfg.func
                def record_only(env, _func=original_func, _name=name, **kwargs):
                    value = _func(env, **kwargs)
                    if bool(value[0]):
                        info = events.setdefault(_name, {'first_step': state['step'], 'last_step': 0, 'positive_steps': 0})
                        info['last_step'] = state['step']
                        info['positive_steps'] += 1
                    return torch.zeros_like(value)
                cfg.func = record_only
                term_manager.set_term_cfg(name, cfg)
        original_termination_compute = term_manager.compute
        def termination_compute():
            result = original_termination_compute()
            if result.any():
                state['termination_terms'] = [name for name in term_manager.active_terms if bool(term_manager.get_term(name)[0])]
                write_json(out / 'termination.json', {'control_step': state['step'], 'terms': state['termination_terms']})
            return result
        term_manager.compute = termination_compute
        grail_enabled=os.environ.get('G1_GRAIL_ACTIVE')=='1'
        submitted=None
        original_forward=wrapped.action_transform_module.forward
        def audited_forward(*args,**kwargs):
            residual=kwargs.get('latent_residual')
            if residual is not None:
                actual=residual.detach().cpu().numpy()[0]
                expected=submitted*float(wrapped._latent_residual_scale)
                assert np.allclose(actual,expected,atol=1e-7)
            return original_forward(*args,**kwargs)
        wrapped.action_transform_module.forward=audited_forward
        write_json(out/'control_definition.json',{'group':os.environ['G1_GROUP'],'grail_interaction_enabled':grail_enabled,
            'grail_checkpoint':'pnp_table_release/last.pt','latent_dimension':64,'latent_scale':float(wrapped._latent_residual_scale),
            'residual_mode':wrapped._latent_residual_mode,'hand_actions_source':'task-reference closure commands, not GRAIL hand outputs',
            'controller':extension.controller,'external_GPT6_API':False,
            'reference_source':'See token_controller.json: SONIC planner and teleop encoder, release decoder; no online IK or GRAIL.',
            'shape_descriptor':'BPS recomputed from rigid water bottle surface vertices',
            'transfer_limit':'released interaction policy not trained or fine-tuned on reconstructed desk/plush'})
        original_step = wrapped.step
        original_end = wrapped.end_render_results

        def checkpoint():
            idx = state['observation']
            if state['step'] == 0 and hasattr(extension,'align_brainco'):
                extension.align_brainco()
            data = observe(wrapped.env)
            if state['initial_z'] is None: state['initial_z'] = data['object_pos'][2]
            data['grasp_information'] = {'metrics': extension.metrics()}
            data['contact_watch'] = contact_watch.observation()
            data['observation_id'] = idx
            data['available_actions'] = ['continue', 'stop', 'shift_stance', 'reach', 'close_hand', 'open_hand', 'lift'] if planner else ['execute_pickup', 'continue', 'stop']
            if comparison and comparison[0].mode == 'gpt':
                data['available_actions'] = ['continue', 'stop', 'reach', 'close_hand', 'open_hand']
            if comparison and comparison[0].mode == 'direct':
                data['available_actions'] = ['continue', 'stop', 'reach', 'close_hand', 'open_hand', 'lift', 'shift_stance']
            if os.environ.get('G1_SONIC_PLANNER'):
                data['available_actions'] = ['navigate','brake','settle','reach', 'close_hand', 'open_hand', 'lift', 'continue', 'stop', 'handoff', 'start_grasp']
            decision = auto_policy.decide(data)
            decision.update(observation_id=idx, author='In-process numeric supervisor', wall_time=time.time())
            if decision['action'] != 'continue':
                with (out/'phase_events.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(step=state['step'], **decision))+'\n')
            if decision['action'] == 'stop':
                os.environ['G1_LEAN_STOP_REASON'] = decision.get('reason', 'guard stop')
            if decision['observation_id'] != idx or decision['action'] not in data['available_actions']:
                raise ValueError('Invalid agent decision')
            if state['step']>=300 and decision['action'] != 'stop':
                assert 5 <= int(decision['steps']) <= 200, 'Bounded feedback interval (5..200 control steps)'
                # Explicit GPT stop is allowed when the attempt is complete or unsafe.
            contact_watch.acknowledge(state['step'])
            if planner: planner[0].apply(decision)
            extension.apply(decision)
            if comparison: comparison[0].apply(decision)
            state['stop'] = decision['action'] == 'stop'
            state['until'] = state['step'] + min(200, max(1, int(decision['steps'])))
            state['observation'] += 1
            if decision['action'] != 'continue': print('PHASE', state['step'], decision['action'], flush=True)

        def step(actor_state):
            nonlocal submitted
            try:
                if os.environ.get('G1_COMPARE_MODE') and not comparison:
                    from reference_comparison import ReferenceComparison
                    comparison.append(ReferenceComparison(wrapped.env, out, Path.cwd(), os.environ['G1_COMPARE_MODE']))
                    # All evaluator resets have now finished. The first residual was
                    # computed before perturbation; all arms share this one-step delay.
                    raw = wrapped.env.observation_manager.compute(update_history=False)
                    obs = wrapped.process_raw_obs(raw, True)
                    wrapped._last_obs_dict = obs
                    actor_state['obs_dict'] = obs
                if state['step'] >= state['until']: checkpoint()
                if os.environ.get('G1_SONIC_PLANNER'):
                    planner[0].tick()
                    actor_state['actions'] = actor_state['actions'].clone()
                    actor_state['actions'][:, :64] *= planner[0].grail_gain if grail_enabled else 0.0
                    state['applied_residual_max_abs'] = float(actor_state['actions'][:, :64].abs().max())
                if comparison and comparison[0].mode == 'direct':
                    actor_state['actions'] = actor_state['actions'].clone()
                    if not grail_enabled: actor_state['actions'][:, :64] = 0.0
                    state['applied_residual_max_abs'] = float(actor_state['actions'][:, :64].abs().max())
                submitted=actor_state['actions'][0,:64].detach().cpu().numpy().copy()
                if state['step']==0:
                    write_json(out/'policy_observation_schema.json',{k:list(v.shape) for k,v in actor_state['obs_dict'].items() if hasattr(v,'shape')})
                state['step'] += 1
                extension.before_step()
                capture[0].begin()
                result = original_step(actor_state)
                capture[0].end()
                if contact_watch.latched is not None:
                    state['until'] = state['step']
                if comparison and state['step'] >= (1500 if retry else 400): result[2][:] = True
                if os.environ.get('G1_DESK_PLUSH') and state['step'] >= int(os.environ.get('G1_MAX_STEPS','1500')): result[2][:] = True
                if state['stop']: result[2][:] = True
                return result
            except Exception as exc:
                state['error'] = repr(exc)
                end()
                raise

        def end():
            if state['ended']: return
            state['ended'] = True
            original_end()
            if capture: capture[0].close()
            contact_watch.close()
            for rec in video_recorder: rec.close_writers()
            telemetry.flush()
            write_json(out/'runtime_summary.json', {
                'error': state['error'], 'control_steps': state['step'],
                'termination_terms': state['termination_terms'],
                'record_only_termination_events': events,
                'agent_requested_stop': state['stop'],
                'success_report': 'lean_result.json',
                'control_definition': 'token_controller.json'})
        wrapped.step = step
        wrapped.end_render_results = end
        return wrapped

    evaluation.train_agent_trl.create_manager_env = create
