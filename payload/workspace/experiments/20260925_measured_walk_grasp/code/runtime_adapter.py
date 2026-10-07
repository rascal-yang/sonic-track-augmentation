"""Runtime adapters; upstream repositories are imported, never edited."""
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
                usd_files = [Path(__file__).resolve().parent.parent / 'assets/water_bottle_v1/bottle.usda']
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
        telemetry = (out / 'trajectory.jsonl').open('w', buffering=1)

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
            if not video_recorder: video_recorder.append(self)
            result = old_record(self)
            # This callback runs before automatic reset; retain actual failure poses.
            data = observe(self.env)
            if state['initial_z'] is None: state['initial_z'] = data['object_pos'][2]
            lift = data['object_pos'][2] - state['initial_z']
            state['peak_lift'] = max(state['peak_lift'], lift)
            state['ever_low_pelvis'] |= data['robot_root_pos'][2] < 0.45
            state['held_steps'] = state['held_steps'] + 1 if lift >= 0.08 and data['robot_root_pos'][2] >= 0.45 else 0
            state['max_held_steps'] = max(state['max_held_steps'], state['held_steps'])
            if hasattr(self.env, '_grasp_extension'):
                data['grasp_metrics'] = self.env._grasp_extension.metrics()
            telemetry.write(json.dumps(data) + '\n')
            if capture and capture[0].active: capture[0].post_step()
            if self._capture_annotator is not None:
                rgb = self._capture_annotator.get_data()
                if isinstance(rgb, dict): rgb = rgb['data']
                rgb = np.asarray(rgb)
                if rgb.ndim == 3 and rgb.shape[0] > 0:
                    imageio.imwrite(out / 'latest.png', rgb[:,:,:3])
            return result
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
                        with (out / 'nonterminal_events.jsonl').open('a') as stream:
                            stream.write(json.dumps({'step': state['step'], 'term': _name}) + '\n')
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
                with (out/'grail_execution.jsonl').open('a') as f:
                    f.write(json.dumps({'step':state['step'],'enabled':grail_enabled, 'phase':planner[0].phase, 'phase_gain':planner[0].grail_gain,
                        'scaled_residual':actual.tolist(),'maximum_absolute_scaled_residual':float(np.max(np.abs(actual))),
                        'l2_scaled_residual':float(np.linalg.norm(actual)),
                        'matches_submitted_times_scale':True})+'\n')
            return original_forward(*args,**kwargs)
        wrapped.action_transform_module.forward=audited_forward
        write_json(out/'control_definition.json',{'group':os.environ['G1_GROUP'],'grail_interaction_enabled':grail_enabled,
            'grail_checkpoint':'pnp_table_release/last.pt','latent_dimension':64,'latent_scale':float(wrapped._latent_residual_scale),
            'residual_mode':wrapped._latent_residual_mode,'hand_actions_source':'task-reference closure commands, not GRAIL hand outputs',
            'controller':extension.controller,'external_GPT6_API':False,
            'reference_source':'SONIC V2 ONNX walking; 3s reference handoff +1s standing; actual-state right-arm IK and bounded gravity compensation, SONIC other joints; GRAIL residual zero throughout. Authorized bottle grasp hints from trial_003.',
            'shape_descriptor':'BPS recomputed from rigid water bottle surface vertices',
            'transfer_limit':'released interaction policy not trained or fine-tuned on reconstructed desk/plush'})
        original_step = wrapped.step
        original_end = wrapped.end_render_results

        def checkpoint():
            idx = state['observation']
            if state['step'] == 0 and hasattr(extension,'align_brainco'):
                extension.align_brainco()
            # Render initial observation without advancing physics/reference time.
            if state['step'] == 0:
                terms = wrapped.env.recorder_manager._terms
                values = terms.values() if isinstance(terms, dict) else terms
                for term in values:
                    if isinstance(term, RenderEnvsRecorderTerm):
                        for _ in range(6): old_record(term)
                        if not video_recorder: video_recorder.append(term)
                        rgb = np.asarray(term._capture_annotator.get_data())
                        if rgb.ndim == 3 and rgb.shape[0]: imageio.imwrite(out / 'latest.png', rgb[:,:,:3])
            data = observe(wrapped.env)
            if state['initial_z'] is None: state['initial_z'] = data['object_pos'][2]
            if os.environ.get('G1_DESK_PLUSH'):
                from desk_plush_experiment import add_privileged
                add_privileged(wrapped.env, data)
            extension.observe(data)
            from protrusion_geometry import add_protrusions
            add_protrusions(data)
            data['contact_watch'] = contact_watch.observation()
            data['observation_id'] = idx
            data['available_actions'] = ['continue', 'stop', 'shift_stance', 'reach', 'close_hand', 'open_hand', 'lift'] if planner else ['execute_pickup', 'continue', 'stop']
            if comparison and comparison[0].mode == 'gpt':
                data['available_actions'] = ['continue', 'stop', 'reach', 'close_hand', 'open_hand']
            if comparison and comparison[0].mode == 'direct':
                data['available_actions'] = ['continue', 'stop', 'reach', 'close_hand', 'open_hand', 'lift', 'shift_stance']
            if os.environ.get('G1_SONIC_PLANNER'):
                data['available_actions'] = ['navigate','brake','settle','reach', 'close_hand', 'open_hand', 'lift', 'continue', 'stop']
            image_path = out / 'latest.png'
            if image_path.exists():
                import shutil
                shutil.copy2(image_path, out / f'observation_{idx:03d}.png')
            head_path=out/'latest_head.png'
            if head_path.exists():
                import shutil
                shutil.copy2(head_path,out/f'observation_{idx:03d}_head.png')
            write_json(out / f'observation_{idx:03d}.json', data)
            write_json(out/f'decision_input_{idx:03d}.json',{'image':str(out/f'observation_{idx:03d}.png'),
                'head_image':str(out/f'observation_{idx:03d}_head.png'),'state':str(out/f'observation_{idx:03d}.json'),'proprioception':'root pose, joints, velocities, torque estimates',
                'privileged':'actual rigid toy pose, scene distances, finger contact forces, geometry',
                'image_timing_note':'latest rendered frame, may precede state by one render interval'})
            print(f'AGENT_WAIT observation={idx} step={state["step"]}', flush=True)
            decision_path = out / f'decision_{idx:03d}.json'
            if os.environ.get('G1_AUTO_CAPTURE')=='1':
                decision=auto_policy.decide(data)
                decision.update(observation_id=idx,author='geometric closed-loop collection supervisor; no GPT API',wall_time=time.time())
                write_json(decision_path,decision)
            if comparison and comparison[0].mode not in ('gpt', 'direct'):
                write_json(decision_path, {'observation_id': idx, 'action': 'continue',
                    'steps': 100, 'reason': 'Fixed schedule; no model intervention',
                    'author': 'automatic comparison baseline'})
            deadline = time.monotonic() + 1200
            print('DECISION_FILE',str(decision_path),flush=True)
            while True:
                try:
                    decision_text=decision_path.read_text()
                    break
                except FileNotFoundError:
                    if time.monotonic() > deadline: raise TimeoutError('No agent decision within 20 minutes')
                    time.sleep(0.2)
            decision = json.loads(decision_text)
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
            print('AGENT_DECISION', decision['action'], decision['reason'], flush=True)

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
            success = state['max_held_steps'] * wrapped.env.step_dt >= 0.5 and not state['ever_low_pelvis']
            write_json(out / 'result.json', {
                'task_success_proxy': bool(success), 'error': state['error'], 'control_steps': state['step'],
                'physics_substeps': state['step'] * int(round(wrapped.env.step_dt / wrapped.env.physics_dt)),
                'peak_lift_m': state['peak_lift'], 'ever_low_pelvis': state['ever_low_pelvis'],
                'max_stable_lift_seconds': state['max_held_steps'] * wrapped.env.step_dt,
                'termination_terms': state['termination_terms'],
                'record_only_termination_events': events,
                'protocol': os.environ.get('G1_COMPARE_PROTOCOL'),
                'agent_decisions': state['observation'], 'agent_requested_stop': state['stop'],
                'comparison_mode': os.environ.get('G1_COMPARE_MODE'),
                'direct_residual_max_abs': state.get('direct_residual_max_abs'),
                'grail_interaction_enabled':grail_enabled,
                'actual_control_definition':'control_definition.json',
                'control': 'SONIC V2 ONNX walking; 3s reference handoff +1s standing; actual-state right-arm IK and bounded gravity compensation, SONIC other joints; GRAIL residual zero throughout.',
                'reference_teleport_during_execution': False})
        wrapped.step = step
        wrapped.end_render_results = end
        return wrapped

    evaluation.train_agent_trl.create_manager_env = create
