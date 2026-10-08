"""Training-only capture or camera-free replay, with bounded terminal grasp checks."""
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from common import write
from render_settings import configure_rt2, resolve_updates, updates_for_frame


def cpu(value):
    return value.detach().cpu().numpy().copy()


class TrainingCapture:
    def __init__(self, env, out):
        self.env, self.out = env, Path(out)
        self.robot = env.scene['robot']
        self.ext = env._grasp_extension
        self.control_index = 0
        self.active = False
        self.closed = False
        self.profile = os.environ['G1_LEAN_MODE']
        self.started = time.perf_counter()
        self.tail = deque(maxlen=150)
        self.min_pelvis = float('inf')
        self.held = 0
        self.records = []
        self.probe = []
        self.benchmark = os.environ.get('G1_LEAN_BENCHMARK_REFERENCE')
        self.body_names = self.ext.lean_body_names
        self.body_ids = [self.robot.joint_names.index(n) for n in self.body_names]
        self.hand_names = self.ext.lean_hand_names
        self.hand_ids = {s: [env.scene['brainco_'+s].joint_names.index(n)
                            for n in self.hand_names[s]] for s in ['left', 'right']}
        self.wrist_id = self.robot.body_names.index('right_wrist_yaw_link')
        self.camera = None
        self.writer = None
        self.rgb = None
        assert abs(env.step_dt-.02) < 1e-9 and abs(env.physics_dt-.005) < 1e-9
        if self.profile == 'generate':
            self.init_camera()

    def init_camera(self):
        from pxr import UsdGeom
        import omni.replicator.core as rep
        import omni.usd
        import carb
        import imageio.v2 as imageio
        settings = carb.settings.get_settings()
        self.render_updates = resolve_updates()
        actual = configure_rt2(settings)
        write(self.out/'renderer_settings.json', dict(
            profile='rt2-native-dlaa', actual_settings=actual,
            updates_per_native_frame=self.render_updates,
            initial_warmup_updates=updates_for_frame(0, self.render_updates),
            physics_control_changed=False, training_ready=False))
        camera = UsdGeom.Camera.Define(omni.usd.get_context().get_stage(), '/World/Capture/head')
        camera.CreateHorizontalApertureAttr(36)
        camera.CreateVerticalApertureAttr(27)
        camera.CreateFocalLengthAttr(18)
        camera.CreateClippingRangeAttr((.015, 40))
        self.camera = camera.AddTransformOp()
        product = rep.create.render_product('/World/Capture/head', (640, 480))
        self.annotator = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
        self.annotator.attach([product])
        self.writer = imageio.get_writer(self.out/'head.mp4', fps=50, codec='libx264',
                                         quality=None, pixelformat='yuv420p', macro_block_size=1,
                                         ffmpeg_params=['-crf', '18', '-preset', 'fast', '-threads', '2'])
        write(self.out/'camera_calibration.json', dict(body_name='torso_link',
              body_local_translation_m=[.095, 0, .45], head_pitch_down_deg=35,
              resolution=[640, 480], native_hz=25, stored_video_hz=50,
              alignment='Current pre-action image; causal hold on odd control steps'))

    def render(self):
        from pxr import Gf
        d = self.robot.data
        i = self.robot.body_names.index('torso_link')
        rot = R.from_quat(cpu(d.body_quat_w[0, i])[[1, 2, 3, 0]])
        eye = cpu(d.body_pos_w[0, i])+rot.apply([.095, 0, .45])
        direction = rot.apply([np.cos(np.deg2rad(35)), 0, -np.sin(np.deg2rad(35))])
        up = rot.apply([0, 0, 1])
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*(eye+direction)), Gf.Vec3d(*up)).GetInverse()
        self.camera.Set(matrix)
        for _ in range(updates_for_frame(self.control_index, self.render_updates)):
            self.env.sim.render()
        raw = self.annotator.get_data()
        if isinstance(raw, dict):
            raw = raw['data']
        self.rgb = np.asarray(raw)[:, :, :3].copy()
        assert self.rgb.shape == (480, 640, 3)

    def begin(self):
        d = self.robot.data
        if self.profile == 'generate':
            cmd = self.ext.lean_command
            assert cmd['control_index'] == self.control_index
            state = [cpu(d.joint_pos[0, self.body_ids])]
            for side in ['left', 'right']:
                state.append(cpu(self.env.scene['brainco_'+side].data.joint_pos[0, self.hand_ids[side]]))
            gravity = R.from_quat(cpu(d.root_quat_w[0])[[1, 2, 3, 0]]).inv().apply([0, 0, -1])
            self.records.append(dict(state=np.concatenate(state), projected_gravity=gravity,
                                     **cmd))
            if self.control_index % 2 == 0:
                self.render()
            self.writer.append_data(self.rgb)
        if self.benchmark:
            self.probe.append(dict(q=cpu(d.joint_pos[0, self.body_ids]),
                                  root=cpu(d.root_pos_w[0]),
                                  wrist=cpu(d.body_pos_w[0, self.wrist_id]),
                                  object=cpu(self.env.scene['object'].data.root_pos_w[0])))
        self.active = True

    def post_step(self):
        # Callback runs before automatic episode reset, including the final state.
        d = self.robot.data
        obj = self.env.scene['object'].data
        m = self.ext.metrics()
        self.min_pelvis = min(self.min_pelvis, float(d.root_pos_w[0, 2]))
        self.held = self.held+1 if m['strict_lift_now'] else 0
        wrist = R.from_quat(cpu(d.body_quat_w[0, self.wrist_id])[[1, 2, 3, 0]])
        relative_p = wrist.inv().apply(cpu(obj.root_pos_w[0])-cpu(d.body_pos_w[0, self.wrist_id]))
        relative_r = wrist.inv()*R.from_quat(cpu(obj.root_quat_w[0])[[1, 2, 3, 0]])
        self.tail.append(dict(clearance=m['whole_object_clearance_m'],
                              opposed=bool(m['opposed_digits_contact']),
                              position=relative_p, rotation=relative_r.as_quat()))

    def end(self):
        self.active = False
        self.control_index += 1
        if self.control_index % 250 == 0:
            print('LEAN_PROGRESS', self.control_index, flush=True)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.writer is not None:
            self.writer.close()
        if self.records:
            arrays = {k: np.stack([r[k] for r in self.records]) for k in self.records[0]}
            arrays.update(body_joint_names=np.array(self.body_names),
                          left_hand_joint_names=np.array(self.hand_names['left']),
                          right_hand_joint_names=np.array(self.hand_names['right']))
            np.savez_compressed(self.out/'training_capture.npz', **arrays)
            np.savez_compressed(self.out/'executed_teleop_tokens.npz',
                time_s=np.arange(self.control_index)*.02,
                sonic_encoded_tokens=arrays['token'], phase=arrays['phase'],
                grasp_index=arrays['grasp_index'], encoder_joint_names=np.array(self.body_names),
                revo2_left_active_joint_names=np.array(self.hand_names['left']),
                revo2_right_active_joint_names=np.array(self.hand_names['right']),
                revo2_left_active_position_target_rad=arrays['left_hand_target'],
                revo2_right_active_position_target_rad=arrays['right_hand_target'])
        report = self.final_result()
        if self.benchmark and self.probe:
            import h5py
            values = {k: np.stack([x[k] for x in self.probe]) for k in self.probe[0]}
            np.savez_compressed(self.out/'benchmark_states.npz', **values)
            with h5py.File(Path(self.benchmark)/'episode_raw.h5') as f:
                names = json.loads(f.attrs['joint_names'])
                bodies = json.loads(f.attrs['body_names'])
                refs = dict(q=f['observation/joint_position'][:][:, [names.index(n) for n in self.body_names]],
                            root=f['observation/root_position'][:], object=f['observation/object_position'][:],
                            wrist=f['observation/body_position_world'][:, bodies.index('right_wrist_yaw_link')])
            same_length = len(values['q']) == len(refs['q'])
            errors = {k: float(np.max(abs(values[k]-refs[k]))) for k in values} if same_length else {}
            write(self.out/'benchmark_comparison.json', dict(equal_length=same_length,
                  max_abs_errors=errors, bitwise_equal=bool(same_length and all(np.array_equal(values[k], refs[k]) for k in values)),
                  reference=self.benchmark, note='Benchmark only, not part of production replay acceptance'))
        write(self.out/'lean_result.json', report)

    def final_result(self):
        complete = self.control_index == int(os.environ['G1_MAX_STEPS'])
        tail = list(self.tail)
        slip, twist, opposed = float('inf'), float('inf'), 0.
        if tail:
            positions = np.stack([x['position'] for x in tail])
            rotations = R.from_quat(np.stack([x['rotation'] for x in tail]))
            slip = float(np.linalg.norm(positions-positions[0], axis=1).max())
            twist = float(np.rad2deg((rotations[0].inv()*rotations).magnitude().max()))
            opposed = float(np.mean([x['opposed'] for x in tail]))
        peak = self.contact_watch.peak
        success = bool(complete and len(tail) == 150 and self.held*.02 >= 3 and
                       tail[-1]['opposed'] and opposed >= .95 and slip <= .015 and twist <= 10 and
                       self.min_pelvis > .45 and peak <= .05 and not os.environ.get('G1_LEAN_STOP_REASON'))
        return dict(task_success=success, complete_sequence=complete, control_steps=self.control_index,
                    duration_s=self.control_index*.02, strict_final_hold_s=self.held*.02,
                    final_clearance_m=tail[-1]['clearance'] if tail else None,
                    final3s_opposed_fraction=opposed, final3s_slip_m=slip,
                    final3s_rotation_deg=twist, minimum_pelvis_z_m=self.min_pelvis,
                    monitored_table_force_peak_N=peak, contact_samples=self.contact_watch.substep_count,
                    stop_reason=os.environ.get('G1_LEAN_STOP_REASON'), profile=self.profile,
                    elapsed_capture_s=time.perf_counter()-self.started,
                    training_ready=False, visual_review='pending' if self.profile == 'generate' else 'no_video_by_design',
                    validation_scope='Terminal 3s grasp and aggregate online guards; no full trajectory equivalence check')
