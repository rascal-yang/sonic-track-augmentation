"""Termination conditions for motion-tracking and HOI reinforcement learning environments."""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_error_magnitude,
    quat_mul,
)

from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand, _get_body_indexes
from gear_sonic.trl.utils.torch_transform import get_heading_q


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = None
    motion_time_out = None
    anchor_pos = None
    anchor_pos_adaptive = None
    anchor_pos_orig = None
    ee_body_pos_adaptive = None
    ee_body_pos_orig = None
    cumm_body_errors = None
    anchor_ori = None
    anchor_ori_full = None
    ee_body_pos = None
    anchor_pos_xy = None
    foot_pos_xyz = None
    cumm_body_pos_error = None
    cumm_body_ori_error = None
    cumm_body_pos_error_local = None
    cumm_body_ori_error_local = None
    # HOI terminations
    object_pos_deviation = None
    object_z_pos_deviation = None
    object_not_lifted = None
    robot_table_contact_before_object = None
    hand_table_contact_termination = None
    grasp_failure_after_contact = None


def exceeded_anchor_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float
) -> torch.Tensor:
    """Terminate if anchor (root) position error exceeds a distance threshold.

    Compute the L2 norm between the reference and robot anchor positions in world
    frame and flag environments where the error exceeds ``threshold``.

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.
        threshold: Maximum allowed position error in meters.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    pos_diff = command.anchor_pos_w - command.robot_anchor_pos_w
    return pos_diff.norm(dim=1).gt(threshold)


def exceeded_anchor_pos_xy(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Terminate if anchor XY (horizontal) position error exceeds a distance threshold.

    Only the X and Y components of the world-frame anchor position are compared,
    ignoring vertical drift.

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.
        threshold: Maximum allowed horizontal position error in meters.
        body_names: Unused; kept for config compatibility.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    xy_diff = command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2]
    return xy_diff.norm(dim=1).gt(threshold)


def exceeded_anchor_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    threshold_adaptive: bool = False,
    down_threshold: float = 0.5,
    root_height_threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate if anchor Z-height error exceeds a threshold.

    When ``threshold_adaptive`` is True, use a looser ``down_threshold`` for
    environments whose reference root height is below ``root_height_threshold``
    (e.g. crouching or sitting motions).

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.
        threshold: Default maximum allowed height error in meters.
        threshold_adaptive: Enable per-env adaptive thresholding.
        down_threshold: Looser threshold applied when the reference root is low.
        root_height_threshold: Height below which ``down_threshold`` is used.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    height_diff = (command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]).abs()
    if threshold_adaptive:
        thresh = torch.full_like(height_diff, threshold)
        thresh[command.running_ref_root_height < root_height_threshold] = down_threshold
        return height_diff.gt(thresh)
    return height_diff.gt(threshold)


def exceeded_anchor_tilt(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """Terminate if the anchor tilt deviates from the reference gravity projection.

    Compare the Z-component of the gravity vector rotated into the reference and
    robot anchor frames. A large difference indicates excessive torso tilt.

    Args:
        env: The manager-based RL environment.
        asset_cfg: Scene entity config used to obtain the gravity vector.
        command_name: Name of the tracking command term.
        threshold: Maximum allowed absolute difference in projected gravity Z.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    command: TrackingCommand = env.command_manager.get_term(command_name)
    quat_apply_fn = (
        math_utils.quat_apply_inverse
        if hasattr(math_utils, "quat_apply_inverse")
        else math_utils.quat_rotate_inverse
    )
    ref_grav = quat_apply_fn(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)
    robot_grav = quat_apply_fn(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    return (ref_grav[:, 2] - robot_grav[:, 2]).abs().gt(threshold)


def exceeded_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """Terminate if the squared anchor orientation error exceeds a threshold.

    Compute the full quaternion error magnitude between reference and robot
    anchor orientations, then compare the squared value against ``threshold``.

    Args:
        env: The manager-based RL environment.
        asset_cfg: Scene entity config (unused but required by termination API).
        command_name: Name of the tracking command term.
        threshold: Maximum allowed squared orientation error (radians^2).

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    angular_err = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
    return angular_err.square().gt(threshold)


def exceeded_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Terminate if any tracked body position error exceeds a distance threshold.

    Compare per-body world-frame positions between the reference motion and the
    robot, and terminate if *any* body exceeds ``threshold``.

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.
        threshold: Maximum allowed position error per body in meters.
        body_names: Optional list of body names to check. If ``None``, all
            tracked bodies from the command are used.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = command.body_pos_relative_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    return pos_diff.norm(dim=-1).gt(threshold).any(dim=-1)


def exceeded_body_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    threshold_adaptive: bool = False,
    down_threshold: float = 0.5,
    body_names: list[str] | None = None,
    root_height_threshold: float = 0.5,
) -> torch.Tensor:
    """Terminate if any tracked body Z-height error exceeds a threshold.

    When ``threshold_adaptive`` is True, use a looser ``down_threshold`` for
    environments whose reference root height is below ``root_height_threshold``.

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.
        threshold: Default maximum allowed height error per body in meters.
        threshold_adaptive: Enable per-env adaptive thresholding.
        down_threshold: Looser threshold applied when the reference root is low.
        body_names: Optional list of body names to check. If ``None``, all
            tracked bodies from the command are used.
        root_height_threshold: Height below which ``down_threshold`` is used.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    height_err = (
        command.body_pos_relative_w[:, tracked, 2] - command.robot_body_pos_w[:, tracked, 2]
    ).abs()
    if threshold_adaptive:
        thresh = torch.full_like(height_err, threshold)
        thresh[command.running_ref_root_height < root_height_threshold] = down_threshold
        return height_err.gt(thresh).any(dim=-1)
    return height_err.gt(threshold).any(dim=-1)


def tracking_time_out(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate when the motion clip has been fully played.

    Compare the elapsed simulation steps (including the motion start offset)
    against the total length of each environment's assigned motion clip.

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    elapsed = command.time_steps + command.motion_start_time_steps + 1
    total = command.motion_lib.get_time_step_total(command.motion_ids)
    return elapsed >= total


def _resolve_matching_names(patterns: str | Sequence[str], candidates: Sequence[str]) -> list[str]:
    """Resolve body name patterns (exact or regex) against available names."""
    if isinstance(patterns, str | bytes):
        patterns = [patterns]
    resolved: list[str] = []
    for pattern in patterns:
        if pattern == ".*":
            resolved.extend(candidates)
            continue
        regex = re.compile(pattern)
        resolved.extend([name for name in candidates if regex.fullmatch(name)])
    # preserve order and drop duplicates
    seen = set()
    ordered_unique = []
    for name in resolved:
        if name not in seen:
            ordered_unique.append(name)
            seen.add(name)
    return ordered_unique


class _CummErrorMixin(ManagerTermBase):
    """Shared logic for cumulative error-based terminations."""

    def __init__(self, cfg: TerminationTermCfg, env):
        """Initialize cumulative error tracking buffers.

        Args:
            cfg: Termination term config. Expected ``params`` keys:
                ``min_steps``, ``threshold``, ``command_name``.
            env: The manager-based RL environment.
        """
        super().__init__(cfg=cfg, env=env)
        self.min_steps: int = cfg.params.get("min_steps")
        self.threshold: float = cfg.params.get("threshold")
        self.command_name = cfg.params.get("command_name")
        self.command: TrackingCommand = env.command_manager.get_term(self.command_name)

        device = self.command.device
        self.error = torch.zeros(env.num_envs, device=device)
        self._cum_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=device)

    def _update_counters(self) -> torch.Tensor:
        """Accumulate consecutive steps above the threshold and return done mask."""
        exceeded = self.error >= self.threshold
        self._cum_steps[exceeded] += 1
        self._cum_steps[~exceeded] = 0
        return self._cum_steps >= self.min_steps

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset cumulative step counters for the given environments.

        Args:
            env_ids: Environment indices to reset, or ``None`` for all.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._cum_steps[env_ids] = 0


class CummBodyPosError(_CummErrorMixin):
    """Terminate if body position error in world frame exceeds threshold for min_steps."""

    def __init__(self, cfg: TerminationTermCfg, env):
        """Initialize body index mapping for world-frame position error tracking.

        Args:
            cfg: Termination term config. Additional ``params`` key:
                ``body_names`` -- regex or list of body names to track
                (default ``".*"`` for all).
            env: The manager-based RL environment.
        """
        super().__init__(cfg=cfg, env=env)
        motion_names = self.command.cfg.body_names
        body_names = cfg.params.get("body_names", ".*")
        selected = _resolve_matching_names(body_names, motion_names)
        self.motion_body_indices = [motion_names.index(name) for name in selected]
        # command.body_indexes maps motion body order to robot body indices
        self.robot_body_indices = self.motion_body_indices

    def __call__(self, env, body_names=None, min_steps=None, threshold=None, command_name=None):
        """Compute max world-frame body position error and check cumulative threshold.

        Returns:
            Boolean tensor of shape ``(num_envs,)``.
        """
        ref_body_pos = self.command.body_pos_w[:, self.motion_body_indices]
        robot_body_pos = self.command.robot_body_pos_w[:, self.robot_body_indices]
        body_pos_error = (ref_body_pos - robot_body_pos).norm(dim=-1)
        self.error[:] = body_pos_error.max(dim=1).values
        return self._update_counters()


class CummBodyOriError(_CummErrorMixin):
    """Terminate if body orientation error in world frame exceeds threshold for min_steps."""

    def __init__(self, cfg: TerminationTermCfg, env):
        """Initialize body index mapping for world-frame orientation error tracking.

        Args:
            cfg: Termination term config. Additional ``params`` key:
                ``body_names`` -- regex or list of body names to track
                (default ``".*"`` for all).
            env: The manager-based RL environment.
        """
        super().__init__(cfg=cfg, env=env)
        motion_names = self.command.cfg.body_names
        body_names = cfg.params.get("body_names", ".*")
        selected = _resolve_matching_names(body_names, motion_names)
        self.motion_body_indices = [motion_names.index(name) for name in selected]
        # self.robot_body_indices = self.command.body_indexes[self.motion_body_indices].tolist()
        self.robot_body_indices = self.motion_body_indices

    def __call__(self, env, body_names=None, min_steps=None, threshold=None, command_name=None):
        """Compute max world-frame body orientation error and check cumulative threshold.

        Returns:
            Boolean tensor of shape ``(num_envs,)``.
        """
        ref_body_quat = self.command.body_quat_w[:, self.motion_body_indices]
        robot_body_quat = self.command.robot_body_quat_w[:, self.robot_body_indices]
        quat_diff = quat_mul(quat_conjugate(ref_body_quat), robot_body_quat)
        body_ori_error = axis_angle_from_quat(quat_diff).norm(dim=-1)
        self.error[:] = body_ori_error.max(dim=1).values
        return self._update_counters()


class CummBodyPosErrorLocal(_CummErrorMixin):
    """Terminate if body position error in the root-yaw frame exceeds threshold for min_steps."""

    def __init__(self, cfg, env):
        """Initialize body index mapping for root-local position error tracking.

        Args:
            cfg: Termination term config. Additional ``params`` key:
                ``body_names`` -- regex or list of body names to track
                (default ``".*"`` for all).
            env: The manager-based RL environment.
        """
        super().__init__(cfg=cfg, env=env)
        body_names = cfg.params.get("body_names", ".*")
        motion_names = self.command.cfg.body_names
        selected = _resolve_matching_names(body_names, motion_names)
        self.motion_body_indices = [motion_names.index(name) for name in selected]
        # self.robot_body_indices = self.command.body_indexes[self.motion_body_indices].tolist()
        self.robot_body_indices = self.motion_body_indices

    def __call__(self, env, body_names=None, min_steps=None, threshold=None, command_name=None):
        """Compute max root-yaw-local body position error and check cumulative threshold.

        Transform body positions into the heading-aligned root frame before
        computing errors, making the check invariant to global position and yaw.

        Returns:
            Boolean tensor of shape ``(num_envs,)``.
        """
        ref_body_pos = self.command.body_pos_w[:, self.motion_body_indices]
        robot_body_pos = self.command.robot_body_pos_w[:, self.robot_body_indices]

        ref_root_pos = self.command.anchor_pos_w.view(self.num_envs, 1, 3).clone()
        robot_root_pos = self.command.robot_anchor_pos_w.view(self.num_envs, 1, 3).clone()
        ref_root_pos[..., 2] = 0.0
        robot_root_pos[..., 2] = 0.0

        ref_root_quat = get_heading_q(self.command.anchor_quat_w.view(self.num_envs, 1, 4))
        robot_root_quat = get_heading_q(self.command.robot_anchor_quat_w.view(self.num_envs, 1, 4))
        # expand root quaternions to match per-body vectors for quat_apply_inverse
        ref_root_quat = ref_root_quat.expand(ref_body_pos.shape[0], ref_body_pos.shape[1], -1)
        robot_root_quat = robot_root_quat.expand(
            robot_body_pos.shape[0], robot_body_pos.shape[1], -1
        )

        ref_body_local = quat_apply_inverse(ref_root_quat, ref_body_pos - ref_root_pos)
        robot_body_local = quat_apply_inverse(robot_root_quat, robot_body_pos - robot_root_pos)

        body_pos_error = (ref_body_local - robot_body_local).norm(dim=-1)
        self.error[:] = body_pos_error.max(dim=1).values
        return self._update_counters()


class CummBodyOriErrorLocal(_CummErrorMixin):
    """Terminate if body orientation error in the root-yaw frame exceeds threshold for min_steps."""

    def __init__(self, cfg: TerminationTermCfg, env):
        """Initialize body index mapping for root-local orientation error tracking.

        Args:
            cfg: Termination term config. Additional ``params`` key:
                ``body_names`` -- regex or list of body names to track
                (default ``".*"`` for all).
            env: The manager-based RL environment.
        """
        super().__init__(cfg=cfg, env=env)
        motion_names = self.command.cfg.body_names
        body_names = cfg.params.get("body_names", ".*")
        selected = _resolve_matching_names(body_names, motion_names)
        self.motion_body_indices = [motion_names.index(name) for name in selected]
        # self.robot_body_indices = self.command.body_indexes[self.motion_body_indices].tolist()
        self.robot_body_indices = self.motion_body_indices

    def __call__(self, env, body_names=None, min_steps=None, threshold=None, command_name=None):
        """Compute max root-yaw-local body orientation error and check cumulative threshold.

        Transform body quaternions into the heading-aligned root frame before
        computing axis-angle errors, making the check invariant to global yaw.

        Returns:
            Boolean tensor of shape ``(num_envs,)``.
        """
        ref_body_quat = self.command.body_quat_w[:, self.motion_body_indices]
        robot_body_quat = self.command.robot_body_quat_w[:, self.robot_body_indices]

        ref_root_quat = get_heading_q(self.command.anchor_quat_w.view(self.num_envs, 1, 4))
        robot_root_quat = get_heading_q(self.command.robot_anchor_quat_w.view(self.num_envs, 1, 4))
        ref_root_quat = ref_root_quat.expand_as(ref_body_quat)
        robot_root_quat = robot_root_quat.expand_as(robot_body_quat)

        ref_body_quat_local = quat_mul(quat_conjugate(ref_root_quat), ref_body_quat)
        robot_body_quat_local = quat_mul(quat_conjugate(robot_root_quat), robot_body_quat)

        quat_diff = quat_mul(quat_conjugate(ref_body_quat_local), robot_body_quat_local)
        body_ori_error = axis_angle_from_quat(quat_diff).norm(dim=-1)
        self.error[:] = body_ori_error.max(dim=1).values
        return self._update_counters()




# ==================== HOI manipulation terminations ====================


def object_pos_deviation(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.2,
) -> torch.Tensor:
    """Terminate if object position deviates from reference by more than threshold.

    This implements condition (i) from Interaction Early Termination (IET):
    Object points deviate from their references by more than threshold on average.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        threshold: Maximum allowed deviation in meters.

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    ref_object_pos = command.object_root_pos[:, 0, :]

    if "object" not in env.scene.rigid_objects:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    obj = env.scene["object"]
    current_object_pos = obj.data.root_pos_w

    pos_error = torch.norm(ref_object_pos - current_object_pos, dim=-1)
    return pos_error > threshold


def object_z_pos_deviation(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.2,
) -> torch.Tensor:
    """Terminate if object Z position deviates from reference by more than threshold.

    Z-only variant of :func:`object_pos_deviation` — implements condition (i)
    from Interaction Early Termination (IET) projected onto the vertical axis.
    Used by termination bundles that tolerate XY drift but not dropping /
    lifting the object above threshold.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        threshold: Maximum allowed Z deviation in meters (default 0.2).

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    ref_object_z_pos = command.object_root_pos[:, 0, 2]

    if "object" not in env.scene.rigid_objects:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    obj = env.scene["object"]
    current_object_z_pos = obj.data.root_pos_w[:, 2]

    z_pos_error = torch.abs(ref_object_z_pos - current_object_z_pos)
    return z_pos_error > threshold


def hand_table_contact_termination(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate if right hand contacts the table.

    Uses force_matrix_w (filtered contact forces between table and hand links only),
    NOT net_forces_w (which includes all forces on the table such as gravity and
    the object resting on it).

    Args:
        env: The environment.
        sensor_cfg: Contact sensor config (e.g. table_to_hand_contact_sensor).
        force_threshold: Minimum total contact force (N) to trigger termination.

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    contact_force = sensor.data.force_matrix_w
    force_magnitude = torch.norm(contact_force, dim=-1).sum(dim=(-1, -2))
    return force_magnitude > force_threshold


class GraspFailureAfterContact(ManagerTermBase):
    """Terminate if robot fails to achieve grasp after reference motion's contact phase begins.

    Uses the reference motion's ``in_contact`` label (from motion lib) to detect when
    the contact phase starts, then checks if a proper grasp (``min_contacts`` fingers
    with force > ``contact_threshold``) is ever achieved within ``grace_frames``.
    Falls back to the sim contact sensor for first-contact detection when in_contact
    labels are unavailable.

    Does NOT terminate if a grasp was achieved at any point (``has_ever_grasped``),
    even if the object is later released. State resets on env reset.
    """

    def __init__(self, cfg: TerminationTermCfg, env):
        """Initialize grasp-failure tracking state.

        Args:
            cfg: Termination term config. Optional ``params`` keys:
                ``sensor_name``, ``contact_threshold``, ``min_contacts``,
                ``grace_frames``, ``command_name``, ``hand``.
            env: The manager-based RL environment.
        """
        super().__init__(cfg=cfg, env=env)
        self._first_contact_step = torch.full(
            (env.num_envs,), fill_value=-1, dtype=torch.long, device=env.device
        )
        self._has_ever_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.sensor_name = cfg.params.get("sensor_name", "object_to_hand_contact_sensor")
        self.contact_threshold = cfg.params.get("contact_threshold", 0.1)
        self.min_contacts = cfg.params.get("min_contacts", 3)
        self.grace_frames = cfg.params.get("grace_frames", 50)
        self.command_name = cfg.params.get("command_name", "motion")
        self.hand = cfg.params.get("hand", "right_hand")

    def __call__(
        self,
        env,
        sensor_name: str = None,
        contact_threshold: float = None,
        min_contacts: int = None,
        grace_frames: int = None,
        command_name: str = None,
        hand: str = None,
    ) -> torch.Tensor:
        """Check whether a grasp was achieved within the grace period after first contact.

        Uses reference motion ``in_contact`` labels when available; falls back to
        the sim contact sensor otherwise.

        Returns:
            Boolean tensor of shape ``(num_envs,)``.
        """
        current_step = env.episode_length_buf

        command: TrackingCommand = env.command_manager.get_term(self.command_name)
        in_contact = command.get_in_contact(self.hand)

        if in_contact is not None:
            ref_in_contact = in_contact > 0.5
            newly_in_contact = ref_in_contact & (self._first_contact_step < 0)
            self._first_contact_step[newly_in_contact] = current_step[newly_in_contact]
        else:
            sensor: ContactSensor = env.scene[self.sensor_name]
            contact_force = sensor.data.force_matrix_w
            force_magnitude = torch.norm(contact_force, dim=-1).sum(dim=(-1, -2))
            has_any_contact = force_magnitude > self.contact_threshold
            newly_in_contact = has_any_contact & (self._first_contact_step < 0)
            self._first_contact_step[newly_in_contact] = current_step[newly_in_contact]

        sensor: ContactSensor = env.scene[self.sensor_name]
        contact_force = sensor.data.force_matrix_w
        force_magnitude = torch.norm(contact_force, dim=-1)
        force_magnitude = force_magnitude.squeeze(1)
        fingers_in_contact = (force_magnitude > self.contact_threshold).float()
        num_fingers = fingers_in_contact.sum(dim=-1)
        is_grasping = num_fingers >= self.min_contacts

        self._has_ever_grasped = self._has_ever_grasped | is_grasping

        has_contact = self._first_contact_step >= 0
        steps_since_contact = current_step - self._first_contact_step
        past_grace = steps_since_contact > self.grace_frames

        terminate = has_contact & past_grace & ~self._has_ever_grasped

        return terminate

    def reset(self, env_ids):
        """Reset first-contact and grasp state for the given environments."""
        self._first_contact_step[env_ids] = -1
        self._has_ever_grasped[env_ids] = False
