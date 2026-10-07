"""Fixed standing reference with slow, continuous task-space waypoint adaptation."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline
from scipy.spatial.transform import Rotation
from online_goals import OnlinePlanner
from experiment import write_json


class StationaryHintPlanner(OnlinePlanner):
    def __init__(self, env, out, sonic):
        super().__init__(env, out, sonic)
        self.goal_start = 0
        self.duration = 200
        self.hand_origin = None
        self.last_report = None
        self.serial = 0
        self.reference_origin = self.lib.body_pos_w_full[0, 0].cpu().numpy().copy()
        self.reference_quat = self.lib.body_quat_w_full[0, 0].cpu().numpy().copy()
        write_json(out / 'planner_provenance.json', {
            'planner': 'fixed standing reference; no SONIC ONNX planning calls',
            'low_level': 'SONIC sonic_manipulation_base; GRAIL residual exactly zero',
            'root_reference_m': self.reference_origin.tolist(),
            'joint_reference_increment_bound_rad_per_0p1s': .065,
            'joint_interpolation': 'shape-preserving Hermite; continuous velocity, zero endpoint velocity',
            'orientation': 'quintic progression on quaternion geodesic',
            'leg_waist_references': 'held at initial standing configuration',
            'actuator_output_filter': False,
            'external_GPT6_API': False})

    def tick(self):
        # Physics and the learned controller keep running. Reference changes
        # occur only when the current GPT session supplies another waypoint.
        pass

    def observation(self):
        obs = super().observation()
        obs.pop('stance_limit_m', None)
        obs.update(planner='GPT fresh task-space targets + bounded smooth IK reference',
                   navigation_disabled=True, last_planner_report=self.last_report, historical_trajectories_loaded=False, historical_waypoint_prior=True)
        return obs

    def apply(self, decision):
        import torch
        action = decision['action']
        if action in ('continue', 'stop') or (action in ('close_hand', 'open_hand') and not (decision.get('goal') or {}).get('target')):
            return
        if action not in ('reach', 'close_hand', 'open_hand', 'lift'):
            raise ValueError('Walking and stance shifts disabled in this experiment')
        params = decision.get('goal') or {}
        side = params.get('hand', 'right')
        frame = int((self.cmd.motion_start_time_steps + self.cmd.time_steps)[0])
        start = frame + 2
        count = len(self.lib.dof_pos) - start
        if count < 3:
            raise ValueError('Reference buffer exhausted')
        duration = int(params.get('duration_steps', 200))
        if not 50 <= duration <= 400:
            raise ValueError('Slow transition duration must be 50..400 steps')
        mapping = np.argsort(np.asarray(self.lib.m_cfg.mujoco_to_isaaclab_dof))
        initial = self.lib.dof_pos[start].cpu().numpy()[mapping].astype(np.float64)
        self.kin.fk(initial, self.reference_origin, self.reference_quat)
        origin = self.kin.palm(side).copy()
        bid = self.kin.model.body(side + '_wrist_yaw_link').id
        orientation = Rotation.from_matrix(self.kin.data.xmat[bid].reshape(3, 3))
        target = np.asarray(params.get('target', origin), dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError('Finite hand target required')
        if np.linalg.norm(target - self.reference_origin) > 1.1:
            raise ValueError('Target outside task workspace')
        euler = params.get('wrist_euler_xyz')
        desired = orientation if euler is None else Rotation.from_euler('xyz', euler)
        # Hold actual wrist orientation if a large turn is requested in contact.
        ext = self.env._grasp_extension
        observed = super().observation()
        actual_q = np.asarray(observed['wrist_quat_wxyz'][side])
        actual = Rotation.from_quat(actual_q[[1, 2, 3, 0]])
        force = ext.metrics()['object_hand_normal_force_N']
        turn = float((actual.inv() * desired).magnitude())
        reference_turn=float((orientation.inv()*desired).magnitude())
        if force > .5 and turn > np.deg2rad(15) and reference_turn > np.deg2rad(2):
            angle=float(np.linalg.norm((orientation.inv()*desired).as_rotvec()))
            desired=orientation*Rotation.from_rotvec((orientation.inv()*desired).as_rotvec()*min(1.,np.deg2rad(2)/max(angle,1e-9)))
            write_json(self.out / f'contact_turn_guard_{decision["observation_id"]:03d}.json',
                       {'force_N': force, 'requested_rotation_deg': float(np.rad2deg(turn)),
                        'reference_turn_limited_deg': 2., 'orientation_held_to_actual': False})
        rotvec = (orientation.inv() * desired).as_rotvec()
        arm = [i for i, n in enumerate(self.kin.names) if n.startswith(side + '_')
               and any(k in n for k in ('shoulder', 'elbow', 'wrist'))]
        horizon = min(count, max(duration + 150, 300))
        indices = np.unique(np.r_[np.arange(0, horizon, 5), horizon - 1])
        keys = [initial.copy()]
        errors = []
        previous = initial.copy()
        for index in indices[1:]:
            progress = float(np.clip(index / duration, 0, 1))
            alpha = progress ** 3 * (10 + progress * (-15 + 6 * progress))
            if params.get('progress_profile')=='early_deceleration':
                u=progress;alpha=35*u**3-105*u**4+126*u**5-70*u**6+15*u**7
            desired_palm = origin + alpha * (target - origin)
            desired_palm += np.asarray(params.get('clearance_arc_m',[0,0,0]))*np.sin(np.pi*alpha)**2
            rotation_alpha=alpha
            if params.get('orientation_finish_fraction',1.)<1:
                v=np.clip(alpha/params['orientation_finish_fraction'],0,1);rotation_alpha=v**3*(10-15*v+6*v*v)
            desired_r = orientation * Rotation.from_rotvec(rotation_alpha * rotvec)
            interval = index - indices[len(keys) - 1]
            bound = .065 * interval / 5
            limits = self.kin.model.jnt_range[1:][arm]
            lo = np.maximum(limits[:, 0] + 1e-6, previous[arm] - bound)
            hi = np.minimum(limits[:, 1] - 1e-6, previous[arm] + bound)
            def residual(x):
                q = previous.copy(); q[arm] = x
                self.kin.fk(q, self.reference_origin, self.reference_quat)
                current_r = Rotation.from_matrix(self.kin.data.xmat[bid].reshape(3, 3))
                return np.r_[10 * (self.kin.palm(side) - desired_palm),
                             .6 * (desired_r.inv() * current_r).as_rotvec(),
                             .025 * (x - previous[arm])]
            solution = least_squares(residual, np.clip(previous[arm], lo+1e-8, hi-1e-8),
                                     bounds=(lo, hi), max_nfev=70)
            residual(solution.x)
            errors.append(float(np.linalg.norm(self.kin.palm(side) - desired_palm)))
            previous = previous.copy(); previous[arm] = solution.x
            keys.append(previous)
        keys = np.asarray(keys)
        # Preserve shape without introducing a post-policy actuator lag filter.
        slopes = PchipInterpolator(indices, keys, axis=0).derivative()(indices)
        slopes[0] = 0
        slopes[-1] = 0
        spline = CubicHermiteSpline(indices, keys, slopes, axis=0)
        qs = spline(np.minimum(np.arange(count), indices[-1]))
        ps = np.tile(self.reference_origin, (count, 1))
        pose = np.zeros((count, 30, 3), np.float32)
        pose[:, 0] = Rotation.from_quat(self.reference_quat[[1,2,3,0]]).as_rotvec()
        pose[:, 1:] = qs[:, :, None] * self.kin.axes
        self.write_reference(start, qs, ps, pose)
        if action == 'lift':
            lift = float(params.get('lift_m', .06))
            if not 0 <= lift <= .2:
                raise ValueError('Lift reference out of range')
            alpha = np.clip(np.arange(count) / duration, 0, 1)
            alpha = alpha ** 3 * (10 + alpha * (-15 + 6 * alpha))
            p = self.env.scene['object'].data.root_pos_w[0].clone()
            self.lib._motion_object_root_pos[start:, 0] = p
            self.lib._motion_object_root_pos[start:, 0, 2] += torch.as_tensor(alpha * lift, device=p.device)
            self.lib._motion_object_lin_vel[start:, 0] = 0
            self.lib._motion_object_lin_vel[start:, 0, 2] = torch.as_tensor(np.gradient(alpha * lift, .02), device=p.device)
        self.goal = {'action': action, 'hand': side, 'target': target.tolist(),
                     'wrist_euler_xyz': desired.as_euler('xyz').tolist()}
        self.hand_origin = origin
        self.goal_start = frame
        self.duration = duration
        self.last_report = {'reference_start_frame': start, 'duration_steps': duration,
                            'horizon_steps': horizon, 'goal': self.goal,
                            'actual_start_palm': observed['palm_positions'][side],
                            'reference_start_palm': origin.tolist(),
                            'max_ik_path_error_m': max(errors),
                            'final_ik_error_m': errors[-1],
                            'no_navigation': True, 'no_periodic_replanning': True,
                            'reference_joint_speed_max_rad_s': float(np.abs(np.diff(qs,axis=0)/.02).max())}
        write_json(self.out / f'goal_application_{decision["observation_id"]:03d}.json', self.last_report)
        np.savez_compressed(self.out / f'reference_{decision["observation_id"]:03d}.npz',
                            joint_pos=qs, root_pos=ps, pose_aa=pose)
        self.serial += 1
