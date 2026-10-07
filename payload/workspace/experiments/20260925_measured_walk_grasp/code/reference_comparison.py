"""Small-offset comparison using the same frozen controllers and demonstration."""
import os
import json
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from experiment import write_json
from online_goals import OnlinePlanner


class ReferenceComparison(OnlinePlanner):
    def __init__(self, env, out, sonic, mode):
        super().__init__(env, out, sonic)
        if mode not in ('original', 'geometry', 'gpt', 'direct'):
            raise ValueError(mode)
        self.mode = mode
        self.offset = np.array(json.loads(os.environ.get('G1_COMPARE_OFFSET', '[0.04,0,0]')), dtype=float)
        if self.offset.shape != (3,) or not np.isfinite(self.offset).all() or np.linalg.norm(self.offset) > .300001:
            raise ValueError('Comparison offset must be finite XYZ within 30cm')
        self.current_offset = np.zeros(3)
        lib = self.lib
        self.q = lib.dof_pos.cpu().numpy()[:, np.argsort(lib.m_cfg.mujoco_to_isaaclab_dof)].copy()
        self.pos = lib.body_pos_w_full[:, 0].cpu().numpy().copy()
        self.quat = lib.body_quat_w_full[:, 0].cpu().numpy().copy()
        self.object_reference = lib._motion_object_root_pos.clone()
        self.original_hand = lib._motion_hand_action_right.clone()
        obj = env.scene['object']
        before = obj.data.root_pos_w[0].clone()
        pose = obj.data.root_state_w[:, :7].clone()
        pose[:, :3] += pose.new_tensor(self.offset)
        obj.write_root_pose_to_sim(pose)
        env.scene.update(0.)
        write_json(out / 'comparison_setup.json', {
            'mode': mode, 'offset_world_m': self.offset.tolist(),
            'original_object_pos': before.cpu().tolist(),
            'shifted_object_pos': obj.data.root_pos_w[0].cpu().tolist(),
            'robot_root_pos': env.scene['robot'].data.root_pos_w[0].cpu().tolist(),
            'robot_joint_pos': env.scene['robot'].data.joint_pos[0].cpu().tolist(),
            'max_control_steps': 1500 if os.environ.get('G1_COMPARE_PROTOCOL') == 'v3_retry' else 400, 'seed': 17,
            'physical_object_write': 'once after all resets, before first physical control step',
            'initial_grail_residual_delay_steps': 1,
            'shared_controller': 'frozen sonic_manipulation_base; GRAIL residual disabled for direct' if mode == 'direct' else 'pnp_table_release + frozen sonic_manipulation_base',
            'gpt_api': False, 'source': 'released robot demonstration; no fresh human retarget',
            'scope': 'one perturbation per condition; exploratory, not success-rate evaluation',
            'protocol': os.environ.get('G1_COMPARE_PROTOCOL', 'v1'),
            'object_deviation_threshold_m': .4 if os.environ.get('G1_COMPARE_PROTOCOL') in ('v2', 'v3_retry') else .1,
            'direct_definition': 'initial pose only + GPT task-space targets + IK + SONIC; no future demo or GRAIL residual' if mode == 'direct' else None})
        if mode == 'geometry':
            self.adapt(self.offset, 0, 'fixed rule: match the known initial apple displacement')
        if mode == 'direct':
            self.q[:] = self.q[0]
            self.pos[:] = self.pos[0]
            self.quat[:] = self.quat[0]
            pose_aa = np.zeros((len(self.q), 30, 3), np.float32)
            pose_aa[:, 0] = Rotation.from_quat(self.quat[:, [1,2,3,0]]).as_rotvec()
            pose_aa[:, 1:] = self.q[:, :, None]*self.kin.axes
            self.write_reference(0, self.q, self.pos, pose_aa)
            lib._motion_object_root_pos[:] = obj.data.root_pos_w[0]
            lib._motion_object_lin_vel[:] = 0
            lib._motion_object_ang_vel[:] = 0
            lib._motion_object_root_quat[:] = lib._motion_object_root_quat[0].clone()
            for side in ('left', 'right'):
                getattr(lib, '_motion_hand_action_'+side)[:] = -1.
                getattr(lib, '_motion_object_in_contact_'+side)[:] = 0.
            np.savez_compressed(out/'direct_initial_reference.npz', joint_pos=self.q,
                                root_pos=self.pos, root_quat=self.quat)

    def observation(self):
        obs = super().observation()
        frame = int((self.cmd.motion_start_time_steps + self.cmd.time_steps +
                     self.lib.length_starts[self.cmd.motion_ids])[0])
        frame = min(frame, len(self.q)-1)
        obs.update(comparison_mode=self.mode, applied_reference_offset=self.current_offset.tolist(),
                   reference_object_pos=self.lib._motion_object_root_pos[frame, 0].cpu().tolist(),
                   reference_hand_right=float(self.lib._motion_hand_action_right[frame]))
        if os.environ.get('G1_COMPARE_PROTOCOL') == 'v3_retry':
            phase = frame % 500
            obs.update(retry_attempt=min(frame // 500 + 1, 3),
                       retry_phase=('GPT online goals' if self.mode == 'direct' else
                                    'approach_and_lift' if phase < 300 else
                                    'hold' if phase < 350 else 'open_and_return'))
        return obs

    def adapt(self, offset, observation_id, reason):
        """Preserve root, legs, torso and left arm; offset right palm with smooth onset."""
        import torch
        offset = np.asarray(offset, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > .300001:
            raise ValueError('Absolute reference offset must be finite XYZ, maximum 30cm')
        lib = self.lib
        start = int((self.cmd.motion_start_time_steps + self.cmd.time_steps +
                     lib.length_starts[self.cmd.motion_ids])[0])
        count = len(self.q) - start
        alpha = np.clip(np.arange(count)/50., 0., 1.)
        alpha = alpha*alpha*(3-2*alpha)
        qs = self.q[start:].astype(float).copy()
        ids = [i for i, n in enumerate(self.kin.names)
               if n.startswith('right_') and any(s in n for s in ('shoulder', 'elbow', 'wrist'))]
        untouched = [i for i in range(29) if i not in ids]
        current = lib.dof_pos[start:].cpu().numpy()[:, np.argsort(lib.m_cfg.mujoco_to_isaaclab_dof)].copy()
        errors = []
        limits = self.kin.model.jnt_range[1:][ids]
        for j in range(count):
            f = start+j
            base = self.q[f].astype(float).copy()
            self.kin.fk(base, self.pos[f], self.quat[f])
            target = self.kin.palm('right').copy() + offset
            bid = self.kin.model.body('right_wrist_yaw_link').id
            orientation = self.kin.data.xmat[bid].reshape(3, 3).copy()
            def residual(x):
                trial = base.copy(); trial[ids] = x
                self.kin.fk(trial, self.pos[f], self.quat[f])
                rot = self.kin.data.xmat[bid].reshape(3, 3)
                return np.r_[10*(self.kin.palm('right')-target),
                             .5*Rotation.from_matrix(orientation.T @ rot).as_rotvec(),
                             .03*(x-base[ids])]
            lower, upper = limits[:, 0]+1e-5, limits[:, 1]-1e-5
            solution = least_squares(residual, np.clip(base[ids], lower+1e-7, upper-1e-7),
                                     bounds=(lower, upper), max_nfev=60)
            residual(solution.x)
            errors.append(float(np.linalg.norm(self.kin.palm('right')-target)))
            qs[j, ids] = current[j, ids]*(1-alpha[j]) + solution.x*alpha[j]
        assert np.allclose(qs[:, untouched], self.q[start:, untouched])
        pose = np.zeros((count, 30, 3), np.float32)
        pose[:, 0] = Rotation.from_quat(self.quat[start:, [1,2,3,0]]).as_rotvec()
        pose[:, 1:] = qs[:, :, None]*self.kin.axes
        self.write_reference(start, qs, self.pos[start:], pose)
        # Align the intended object path to its new initial position; no physical writes.
        lib._motion_object_root_pos[:] = self.object_reference + lib._motion_object_root_pos.new_tensor(self.offset)
        self.current_offset = offset
        np.savez_compressed(self.out/f'adapted_reference_{observation_id:03d}.npz',
                            joint_pos=qs, root_pos=self.pos[start:], offset=offset)
        write_json(self.out/f'adaptation_{observation_id:03d}.json', {
            'reason': reason, 'offset_world_m': offset.tolist(), 'reference_start': start,
            'blend_control_steps': 50, 'max_unblended_ik_error_m': max(errors),
            'median_unblended_ik_error_m': float(np.median(errors)),
            'root_legs_torso_left_arm_preserved': True,
            'reference_hand_timing_changed': False,
            'policy_observation_delay_steps': 1 if self.mode == 'gpt' else 0})

    def apply(self, decision):
        if self.mode == 'direct':
            return super().apply(decision)
        if decision['action'] == 'reach':
            self.adapt(decision['goal']['delta'], decision['observation_id'], decision['reason'])
        elif decision['action'] in ('open_hand', 'close_hand'):
            start = int((self.cmd.motion_start_time_steps + self.cmd.time_steps +
                         self.lib.length_starts[self.cmd.motion_ids])[0])
            duration = min(50, int(decision['steps']))
            self.lib._motion_hand_action_right[start:start+duration] = (
                1. if decision['action'] == 'close_hand' else -1.)
            write_json(self.out/f'hand_override_{decision["observation_id"]:03d}.json', {
                'start': start, 'duration': duration, 'action': decision['action'],
                'contact_reference_unchanged': True})
