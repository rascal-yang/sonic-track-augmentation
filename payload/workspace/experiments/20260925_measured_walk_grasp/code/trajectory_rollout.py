"""Opt-in saved-state reference rollout. Never teleports physical state."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from online_goals import OnlinePlanner
from direct_bottle import DirectBottlePlanner
from experiment import write_json


def install(reference):
    reference = Path(reference).resolve()
    with np.load(reference, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    assert np.allclose(np.diff(arrays['time_s']), .02)
    import walk_grasp
    import brainco_adapter

    class ReferencePlanner(DirectBottlePlanner):
        def __init__(self, env, out, sonic):
            super().__init__(env, out, sonic)
            self.phase = 'reference_rollout'
            original_load = self.lib.load_motions
            def reload_reference(*args, **kwargs):
                result = original_load(*args, **kwargs)
                self.load_reference()
                return result
            self.lib.load_motions = reload_reference
            self.load_reference()

        def load_reference(self):
            out = self.out
            names = arrays['g1_joint_names'].tolist()
            order = [names.index(name) for name in self.kin.names]
            n = len(self.lib.dof_pos)
            indices = np.minimum(np.arange(n), len(arrays['time_s']) - 1)
            qs = arrays['g1_joint_position_rad'][indices][:, order]
            ps = arrays['root_position_world_m'][indices]
            quat = arrays['root_quaternion_world_wxyz'][indices]
            pose = np.zeros((n, 30, 3), np.float32)
            pose[:, 0] = Rotation.from_quat(quat[:, [1,2,3,0]]).as_rotvec()
            pose[:, 1:] = qs[:, :, None] * self.kin.axes
            self.write_reference(0, qs, ps, pose)
            # Object motion is reference-only; physical object state is never set here.
            import torch
            obj = arrays['object_position_world_m'][indices]
            self.lib._motion_object_root_pos[:, 0] = torch.as_tensor(obj, device=self.lib.dof_pos.device)
            self.lib._motion_object_lin_vel[:, 0] = torch.as_tensor(np.gradient(obj,.02,axis=0), device=self.lib.dof_pos.device)
            self.reference_origin = ps[0].copy()
            self.reference_quat = quat[0].copy()
            write_json(out / 'reference_rollout.json', {
                'reference': str(reference), 'sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
                'samples': len(arrays['time_s']), 'control_dt_s': .02,
                'body_reference': 'Recorded actual 29 body joint states and root poses, name-mapped into SONIC motion library; FK rebuilt using existing calibrated model',
                'hands': 'Recorded Revo2 position_target_rad snapshots at 50Hz; passive joints remain PhysX mimic',
                'physical_state_writes_during_rollout': False,
                'online_navigation': False, 'actual_state_arm_IK': False,
                'controller': 'live SONIC full-body feedback; no saved G1 actuator commands',
                'initialization': 'same standard experiment reset; no extra state restore',
                'not_claimed': 'not exact original closed-loop controller replay, not RT2 visual replay, not a deployed hardware conversion'})
            # This report supersedes inherited generic planner provenance.
            self.goal = None

        def frame(self):
            return int((self.cmd.motion_start_time_steps + self.cmd.time_steps)[0])

        def apply(self, decision):
            if decision['action'] not in ('continue', 'stop'):
                raise ValueError('Reference rollout accepts no replanning commands')

        def tick(self):
            i = self.frame()
            source_i = min(i, len(arrays['time_s'])-1)
            names = arrays['g1_joint_names'].tolist()
            q = arrays['g1_joint_position_rad'][source_i, [names.index(n) for n in self.kin.names]]
            expected = q[self.lib.m_cfg.mujoco_to_isaaclab_dof]
            error = float(np.max(np.abs(self.lib.dof_pos[i].cpu().numpy()-expected)))
            assert error < 1e-6, ('reference overwritten', i, error)
            with (self.out/'reference_integrity.jsonl').open('a') as f:
                f.write(json.dumps({'frame':i,'source_index':source_i,'joint_reference_max_error_rad':error})+'\n')

    original = brainco_adapter.install_extension
    def hand_extension(extension):
        original(extension)
        extension.controller = 'sonic_full'
        original_before = extension.before_step
        log = (extension.out/'reference_hand_execution.jsonl').open('w',buffering=1)
        def before():
            original_before()
            import torch
            i = min(max(extension.frame(), 0),len(arrays['time_s'])-1)
            for side in ('left','right'):
                hand = extension.env.scene['brainco_'+side]
                names = arrays[f'revo2_{side}_joint_names'].tolist()
                order = [names.index(name) for name in hand.joint_names]
                target = arrays[f'revo2_{side}_position_target_rad'][i,order]
                hand.set_joint_position_target(torch.as_tensor(target, device=hand.data.joint_pos.device)[None])
            log.write(json.dumps({'frame':extension.frame(),'source_index':i,'physical_state_reset':False})+'\n')
        extension.before_step = before
    walk_grasp.WalkGraspPlanner = ReferencePlanner
    brainco_adapter.install_extension = hand_extension
