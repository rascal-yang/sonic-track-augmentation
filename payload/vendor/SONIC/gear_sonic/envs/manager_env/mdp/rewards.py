"""Reward functions for the manager-based RL environment MDP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    quat_rotate_inverse,
)
import torch

from gear_sonic.envs.manager_env.mdp.commands import (
    ForceTrackingCommand,
    TrackingCommand,
    _get_body_indexes,
)
from gear_sonic.envs.manager_env.mdp.observations import get_hand_object_transform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    tracking_anchor_pos = None
    tracking_anchor_ori = None
    tracking_relative_body_pos = None
    tracking_relative_body_ori = None
    tracking_relative_body_ori_weighted = None
    tracking_body_linvel = None
    tracking_body_angvel = None
    action_rate_l2 = None
    joint_limit = None
    undesired_contacts = None
    undesired_contacts_no_hands = None
    undesired_contacts_no_ankle_hand = None
    undesired_contacts_no_ankle_yam = None
    tracking_body_pos = None
    tracking_body_ori = None
    tracking_vr_3point_global = None
    tracking_vr_3point_local = None
    tracking_vr_3point_force = None
    tracking_vr_2wrists_ori_tight = None
    tracking_vr_2wrists_local_ori = None
    tracking_head_local_ori = None
    feet_contact_duration = None
    jitter_penalty = None
    energy_consumption = None
    anti_shake_ang_vel = None
    feet_contact_requirement = None
    feet_contact_requirement_motion_lib = None
    tracking_vr_5point_local = None
    motion_5point_local_pos = None
    feet_acc = None
    # HOI manipulation rewards
    hand_fingers_object_distance = None
    grasp_finger_direction = None
    grasp_finger_direction_left = None
    grasp_reward = None
    grasp_reward_left = None
    grasp_contact_center_right = None
    grasp_contact_center_left = None
    lift_reward = None
    lift_z_elevation = None
    hand_table_contact_penalty = None
    # Object motion tracking reward (OmniGrasp-style r_t^obj)
    object_tracking_reward = None
    object_lift_contact_reward = None
    # Finger primitive action limit penalty
    finger_primitive_limit = None
    # Meta action (token + residual) smoothness penalty
    meta_action_rate_l2 = None
    # Full latent (decoder input) smoothness penalty
    full_latent_rate_l2 = None
    # Early termination penalty (penalizes non-timeout terminations)
    is_terminated = None
    # Wrist alignment reward (encourage wrist to face object during approach)
    wrist_look_at_object = None
    # Wrist smoothness penalty around contact transition
    wrist_smoothness_at_contact = None
    # Approach behavior rewards (sim2real: prevent swatting, keep hand open)
    approach_velocity_penalty = None
    open_hand_until_close = None
    upright_penalty = None
    # Foot slippage penalty (penalize foot velocity while in ground contact)
    foot_slippage_penalty = None


def tracking_anchor_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Compute anchor position tracking reward using a Gaussian kernel.

    Encourages the robot's anchor (root) position to match the reference motion anchor.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel. Smaller values produce
            sharper falloff and stricter tracking.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    diff = command.anchor_pos_w - command.robot_anchor_pos_w
    sq_dist = (diff * diff).sum(dim=-1)
    return torch.exp(-sq_dist / (std * std))


def tracking_anchor_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Compute anchor orientation tracking reward using a Gaussian kernel.

    Encourages the robot's anchor (root) orientation to match the reference motion anchor.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    angular_err = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
    return torch.exp(-angular_err.square() / (std * std))


def upright_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_name: str | None = None,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize tilt of bodies away from upright.

    Compute the squared magnitude of the x/y components of the gravity vector
    in each body's local frame, summed across all specified bodies. When a body
    is perfectly upright the local gravity is [0, 0, -1] and the penalty is 0.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        body_name: Single body name (for backwards compatibility).
        body_names: List of body names. If both are None, defaults to ["pelvis"].

    Returns:
        Penalty tensor of shape (num_envs,). Zero when upright, positive when tilted.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]

    if body_names is None:
        body_names = [body_name] if body_name else ["pelvis"]

    total_penalty = torch.zeros(env.num_envs, device=env.device)
    for name in body_names:
        body_idx = robot.body_names.index(name)
        body_quat = robot.data.body_quat_w[:, body_idx]
        g_local = quat_apply(quat_inv(body_quat), command.down_dir)
        total_penalty += g_local[:, 0] ** 2 + g_local[:, 1] ** 2

    return total_penalty


def tracking_body_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body position tracking reward in world frame using a Gaussian kernel.

    Encourages tracked body positions to match the reference motion. The reward is
    the mean squared distance across all tracked bodies, passed through an exponential.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = command.body_pos_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    per_body_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_vr_3point_error(env: ManagerBasedRLEnv, command_name: str, std: float):
    """Compute VR 3-point tracking reward in world frame using a Gaussian kernel.

    Encourages the robot's 3 VR tracking points (typically left wrist, right wrist,
    head) to match their reference positions in world frame.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    pos_diff = command.robot_vr_3point_pos_w - command.vr_3point_body_pos_w
    per_point_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_point_err.mean(dim=-1) / (std * std))


def tracking_vr_2wrists_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute wrist orientation tracking reward in world frame.

    Measure the orientation error of 2 wrist bodies against the reference motion,
    similar to tracking_relative_body_ori_error but restricted to wrist links.

    NOTE: The rigid extension defined in vr_3point_body_offset can be skipped for
    orientation error since it does not affect rotations.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: List of wrist body names (must be provided).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    assert body_names is not None, "body_names must be provided"
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_w[:, tracked], command.robot_body_quat_w[:, tracked]
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_local_vr_2wrists_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute wrist orientation tracking reward in the anchor's local frame.

    Transform both reference and robot wrist orientations into the anchor (root)
    frame before computing the angular error. This makes the reward invariant to
    global root orientation.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: List of wrist body names (must be provided).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    assert body_names is not None, "body_names must be provided"
    body_indexes = _get_body_indexes(command, body_names)
    num_bodies = len(body_indexes)

    # reference motion
    ref_wrist_quat_w = command.body_quat_w[:, body_indexes]
    ref_anchor_quat_w = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, num_bodies, 1)
    ref_wrist_quat_local = quat_mul(quat_inv(ref_anchor_quat_w), ref_wrist_quat_w)

    # robot
    robot_wrist_quat_w = command.robot_body_quat_w[:, body_indexes]
    robot_anchor_quat_w = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, num_bodies, 1
    )
    robot_wrist_quat_local = quat_mul(quat_inv(robot_anchor_quat_w), robot_wrist_quat_w)

    error = quat_error_magnitude(ref_wrist_quat_local, robot_wrist_quat_local) ** 2
    return torch.exp(-error.mean(-1) / std**2)


def tracking_local_head_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Compute head orientation tracking reward in the anchor's local frame.

    Transform the head (torso_link) orientation into the anchor's local frame for
    both the reference motion and the robot, then compute the angular error. This
    encourages the robot to match the head-to-root relative orientation.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get the head body index (torso_link)
    head_body_names = ["torso_link"]
    body_indexes = _get_body_indexes(command, head_body_names)

    # reference motion: head orientation in world frame, transformed to anchor's local frame
    ref_head_quat_w = command.body_quat_w[:, body_indexes]  # [num_envs, 1, 4]
    ref_anchor_quat_w = command.anchor_quat_w.view(env.num_envs, 1, 4)
    ref_head_quat_local = quat_mul(quat_inv(ref_anchor_quat_w), ref_head_quat_w)

    # robot: head orientation in world frame, transformed to anchor's local frame
    robot_head_quat_w = command.robot_body_quat_w[:, body_indexes]  # [num_envs, 1, 4]
    robot_anchor_quat_w = command.robot_anchor_quat_w.view(env.num_envs, 1, 4)
    robot_head_quat_local = quat_mul(quat_inv(robot_anchor_quat_w), robot_head_quat_w)

    error = quat_error_magnitude(ref_head_quat_local, robot_head_quat_local) ** 2
    return torch.exp(-error.squeeze(-1) / std**2)


def tracking_local_vr_3point_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    point_weights: list[float] | None = None,
):
    """Compute VR 3-point tracking reward in the anchor's local frame.

    Transform tracking points into the anchor (root) local frame before computing
    position error, making the reward invariant to global root position/orientation.
    Supports optional per-point weighting.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        point_weights: Optional weights for each tracking point. Order matches
            vr_3point_body config (typically [left_wrist, right_wrist, head]).
            If None, all points are weighted equally.
            Example: [2, 2, 1] gives wrists 2x importance vs head.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    ref_3point_pos = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    robot_root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    robot_3point_diff = command.robot_vr_3point_pos_w - command.robot_anchor_pos_w[:, None, :]
    robot_3point_pos = quat_apply(quat_inv(robot_root_quat), robot_3point_diff)
    diff = robot_3point_pos - ref_3point_pos
    error = torch.sum(torch.square(diff), dim=-1)  # [num_envs, num_points]

    if point_weights is not None:
        # Weighted mean: sum(w_i * e_i) / sum(w_i)
        weights = torch.tensor(point_weights, dtype=error.dtype, device=error.device)
        weighted_error = (error * weights).sum(dim=-1) / weights.sum()
    else:
        # Simple mean (equal weights)
        weighted_error = error.mean(dim=-1)

    return torch.exp(-weighted_error / std**2)


def tracking_local_vr_5point_error(env: ManagerBasedRLEnv, command_name: str, std: float):
    """Compute VR 5-point tracking reward in the anchor's local frame.

    Same approach as tracking_local_vr_3point_error but with 5 tracking points
    (e.g., 2 wrists + head + 2 feet).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ref_5point_diff = command.reward_point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.reward_point_body), 1
    )
    ref_5point_pos = quat_apply(quat_inv(ref_root_quat), ref_5point_diff)
    robot_root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.reward_point_body), 1
    )
    robot_5point_diff = (
        command.robot_reward_point_body_pos_w - command.robot_anchor_pos_w[:, None, :]
    )
    robot_5point_pos = quat_apply(quat_inv(robot_root_quat), robot_5point_diff)
    diff = robot_5point_pos - ref_5point_pos
    error = torch.sum(torch.square(diff), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def tracking_vr_3point_error_pos_force(
    env: ManagerBasedRLEnv, motion_command_name: str, force_command_name: str, std: float
):
    """Compute VR 3-point tracking reward with force-based compliance correction.

    Add a force-proportional offset to the wrist tracking error so that applied
    external forces shift the tracking target, enabling compliant behavior under
    force perturbations.

    Args:
        env: The environment.
        motion_command_name: Name of the motion tracking command term.
        force_command_name: Name of the force tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    motion_command: TrackingCommand = env.command_manager.get_term(motion_command_name)
    force_command: ForceTrackingCommand = env.command_manager.get_term(force_command_name)
    diff = motion_command.robot_vr_3point_pos_w - motion_command.vr_3point_body_pos_w
    force_error_wrists = (
        force_command.last_force_applied * force_command.eef_stiffness_buf[:, :, None]
    )
    diff[:, :2] += force_error_wrists
    error = torch.sum(torch.square(diff), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def tracking_body_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body orientation tracking reward in world frame using a Gaussian kernel.

    Encourages tracked body orientations to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_w[:, tracked], command.robot_body_quat_w[:, tracked]
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_relative_body_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body position tracking reward using anchor-relative reference positions.

    Use reference body positions that have been shifted to share the robot's anchor
    (root) position, so only the relative pose matters rather than absolute position.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = command.body_pos_relative_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    per_body_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_relative_body_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body orientation tracking reward using anchor-relative reference orientations.

    Use reference body orientations that have been transformed to share the robot's
    anchor (root) orientation, making the reward invariant to global heading.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_relative_w[:, tracked],
        command.robot_body_quat_w[:, tracked],
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_relative_body_ori_weighted_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    body_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute anchor-relative body orientation tracking reward with per-body weights.

    Same as tracking_relative_body_ori_error but allows different bodies to contribute
    differently to the mean error. Useful for relaxing tracking on certain joints
    (e.g., wrists during manipulation).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.
        body_weights: Dict mapping body name to weight multiplier. Bodies not listed
            default to 1.0. E.g. {"left_wrist_yaw_link": 0.1} to relax wrist tracking.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            command.body_quat_relative_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    if body_weights is not None:
        tracked_names = [command.cfg.body_names[i] for i in body_indexes]
        weights = torch.tensor(
            [body_weights.get(name, 1.0) for name in tracked_names],
            device=error.device,
            dtype=error.dtype,
        )
        weighted_error = (error * weights).sum(-1) / weights.sum()
    else:
        weighted_error = error.mean(-1)
    return torch.exp(-weighted_error / std**2)


def tracking_body_linvel_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body linear velocity tracking reward using a Gaussian kernel.

    Encourages tracked body linear velocities to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    vel_diff = command.body_lin_vel_w[:, tracked] - command.robot_body_lin_vel_w[:, tracked]
    per_body_err = (vel_diff * vel_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_body_angvel_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body angular velocity tracking reward using a Gaussian kernel.

    Encourages tracked body angular velocities to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    vel_diff = command.body_ang_vel_w[:, tracked] - command.robot_body_ang_vel_w[:, tracked]
    per_body_err = (vel_diff * vel_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def feet_contact_duration(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Compute reward for maintaining foot contact within a time threshold.

    Penalize feet that are airborne but whose last contact was recent (within
    threshold), encouraging stable ground contact timing.

    Args:
        env: The environment.
        sensor_cfg: Contact sensor configuration for the feet.
        threshold: Maximum time (seconds) since last contact to still count as
            recently airborne.

    Returns:
        Reward tensor of shape (num_envs,). Count of feet meeting the criterion.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    airborne_mask = sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    time_since_contact = sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return ((time_since_contact < threshold) * airborne_mask).sum(dim=-1)


def feet_contact_requirement(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    min_ref_feet_height_no_contact: float,
    max_ref_feet_height_when_contact: float,
):
    """Compute penalty for incorrect foot contact relative to reference motion height.

    Penalize two error cases: (1) foot is on the ground when the reference foot is
    above min_ref_feet_height_no_contact (should be airborne), and (2) foot is in
    the air when the reference foot is below max_ref_feet_height_when_contact (should
    be grounded). Skips the first 10 steps and ignores crouching motions
    (ref root height < 0.3m).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        sensor_cfg: Contact sensor configuration for the feet.
        min_ref_feet_height_no_contact: Reference foot height above which the real
            foot should not be in contact.
        max_ref_feet_height_when_contact: Reference foot height below which the real
            foot must be in contact.

    Returns:
        Penalty tensor of shape (num_envs,). Negative values indicate violations.
    """
    # Add validation for threshold ordering
    assert (
        min_ref_feet_height_no_contact > max_ref_feet_height_when_contact
    ), "min_ref_feet_height_no_contact should be greater than max_ref_feet_height_when_contact"

    command: TrackingCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Remove redundant slicing
    current_feet_contact_forces = contact_sensor.data.net_forces_w
    feet_index = contact_sensor.find_bodies(".*ankle_roll.*")[0]
    current_feet_in_the_air = current_feet_contact_forces[:, feet_index, 2] <= 1.0

    body_indexes = _get_body_indexes(command, sensor_cfg.body_names)
    # Fix: Extract only Z-coordinate for height comparison
    ref_feet_height = command.body_pos_w[:, body_indexes, 2]

    feet_not_need_contact = ref_feet_height > min_ref_feet_height_no_contact

    feet_must_contact = ref_feet_height < max_ref_feet_height_when_contact
    feet_wrong_contact = torch.logical_and(feet_not_need_contact, ~current_feet_in_the_air)
    feet_wrong_air = torch.logical_and(feet_must_contact, current_feet_in_the_air)

    feet_wrong_contact_reward = -torch.logical_or(feet_wrong_air, feet_wrong_contact).to(
        torch.float64
    )

    after_10_steps = env.episode_length_buf > 10

    current_reward = torch.sum(feet_wrong_contact_reward, dim=1) * after_10_steps.to(torch.float64)

    current_reward[command.running_ref_root_height < 0.3] = 0.0

    return current_reward


def feet_contact_requirement_motion_lib(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
):
    """Compute penalty for incorrect foot contact using motion library contact labels.

    Use precomputed left/right foot contact labels from the motion library instead
    of height-based heuristics. Penalize cases where the reference says a foot
    should be in contact but the robot's foot is airborne. Skips the first 10 steps
    and ignores crouching motions (ref root height < 0.3m).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        sensor_cfg: Contact sensor configuration for the feet.

    Returns:
        Penalty tensor of shape (num_envs,). Negative values indicate violations.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_feet_contact_forces = contact_sensor.data.net_forces_w
    feet_index = contact_sensor.find_bodies(".*ankle_roll.*")[0]
    current_feet_in_the_air = current_feet_contact_forces[:, feet_index, 2] <= 1.0
    ref_feet_l = command.feet_l  # Left foot contact from reference
    ref_feet_r = command.feet_r  # Right foot contact from reference
    ref_foot_contact = torch.stack([ref_feet_l, ref_feet_r], dim=1)  # Shape: [num_envs, 2]
    feet_wrong_contact_reward = -torch.logical_and(
        ref_foot_contact.squeeze(-1), current_feet_in_the_air
    ).to(torch.float64)

    after_10_steps = env.episode_length_buf > 10
    current_reward = torch.sum(feet_wrong_contact_reward, dim=1) * after_10_steps.to(torch.float64)
    current_reward[command.running_ref_root_height < 0.3] = 0.0
    return current_reward


def jitter_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Compute total squared body position error across all bodies and coordinates.

    Penalize large deviations from the reference body positions. Unlike the per-body
    mean used in tracking rewards, this sums across all coordinates for a raw
    aggregate error signal.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.

    Returns:
        Penalty tensor of shape (num_envs, num_bodies). Squared error per body.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    return torch.sum(torch.square(command.body_pos_w - command.robot_body_pos_w), dim=-1)


def anti_shake_ang_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 1.5,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize excessive angular velocity on selected bodies with a deadzone.

    Discourage high-frequency jitter on small links (wrists, head) while allowing
    normal intentional motion within the threshold. Speeds below the threshold
    incur zero penalty.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        threshold: Angular velocity deadzone (rad/s). No penalty below this.
        body_names: Bodies to penalize. If None, uses all tracked bodies.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (use negative weight).
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    # [E, B, 3]
    ang_vel = command.robot_body_ang_vel_w[:, body_indexes]
    # magnitude per body: [E, B]
    speed = torch.linalg.norm(ang_vel, dim=-1)
    # deadzone then square: [E, B]
    excess = torch.relu(speed - threshold)
    penalty = (excess * excess).mean(dim=-1)
    return penalty


# ==================== HOI Manipulation Rewards ====================


def reward_hand_fingers_object_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_to_hand_frame_transformer"),
    exp_coeff: float = -10.0,
    command_name: str = "motion",
) -> torch.Tensor:
    """Compute reward for hand proximity to the object.

    Encourage the hand to approach the object using an exponential distance kernel.
    Only applied when the contact label indicates contact should happen
    (current_frame >= first_contact_frame).

    Args:
        env: The environment.
        asset_cfg: Frame transformer config for the object-to-hand transform.
        exp_coeff: Exponential coefficient for distance-to-reward mapping (negative).
        command_name: Name of the tracking command term.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Check if contact data is available - use precomputed per-env first contact frame
    per_env_first_contact = getattr(command, "_per_env_first_contact", None)

    if per_env_first_contact is not None:
        current_time = command.motion_start_time_steps + command.time_steps
        should_contact = (current_time >= per_env_first_contact).float()
    else:
        # No contact data, always apply reward
        should_contact = torch.ones(env.num_envs, device=env.device)

    hand_object_transform = get_hand_object_transform(env, asset_cfg)
    object_pos_in_hand = hand_object_transform[:, :3]
    object_pos_in_hand_distance = torch.norm(object_pos_in_hand, dim=-1)
    reward = torch.exp(exp_coeff * object_pos_in_hand_distance)
    # Only apply reward when contact should happen
    return reward * should_contact


def reward_grasp(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 0.1,  # Min force to count as contact
    min_contacts: int = 3,  # Minimum number of finger contacts required
    gate_with_contact_label: bool = False,
    anti_contact_factor: float | None = None,
    hand: str = "right_hand",
    command_name: str = "motion",
) -> torch.Tensor:
    """Compute reward for successful grasp based on finger contact count.

    Reward is proportional to the number of fingers in contact, reaching 1.0 when
    at least min_contacts fingers touch the object. Optionally gates the reward by
    motion contact labels from object motion data, so that reward activation follows
    dataset timing rather than pure simulation contact.

    Args:
        env: The environment.
        sensor_cfg: Contact sensor config for finger-object contact.
        contact_threshold: Minimum force (N) to count a finger as in contact.
        min_contacts: Number of finger contacts required for full reward.
        gate_with_contact_label: If True, only apply reward when the motion data's
            contact label is active for the given hand.
        hand: Which hand's contact label to use ("left_hand" or "right_hand").
        command_name: Name of the tracking command term.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    if anti_contact_factor is not None and not gate_with_contact_label:
        raise ValueError(
            "reward_grasp: anti_contact_factor requires gate_with_contact_label=True"
        )

    sensor: ContactSensor = env.scene[sensor_cfg.name]
    contact_force = sensor.data.force_matrix_w
    force_magnitude = torch.norm(contact_force, dim=-1)  # [num_envs, 1, 4]
    force_magnitude = force_magnitude.squeeze(1)  # [num_envs, 4]

    fingers_in_contact = (force_magnitude > contact_threshold).float()  # [num_envs, 4]
    num_fingers_in_contact = fingers_in_contact.sum(dim=-1)  # [num_envs]

    reward = torch.clamp(num_fingers_in_contact / min_contacts, max=1.0)

    reward_out = reward

    if gate_with_contact_label:
        command: TrackingCommand = env.command_manager.get_term(command_name)
        in_contact = command.get_in_contact(hand)
        if in_contact is None:
            raise RuntimeError(
                "reward_grasp: gate_with_contact_label=True but no contact label found. "
                "Expected object motion data to include object_in_contact_left/right "
                "(derived from contact_points_left_hand/right_hand)."
            )

        in_contact_f = in_contact.float()
        if anti_contact_factor is not None:
            # in_contact=1: normal reward; in_contact=0: penalty for unwanted contact
            reward_out = reward * in_contact_f + anti_contact_factor * reward * (1.0 - in_contact_f)
        else:
            reward_out = reward * in_contact_f

    return reward_out


def reward_grasp_finger_direction(
    env: ManagerBasedRLEnv,
    thumb_link: str = "right_hand_thumb_2_link",
    index_link: str = "right_hand_index_1_link",
    middle_link: str | None = "right_hand_middle_1_link",
    hand: str = "right_hand",
    command_name: str = "motion",
    thumb_offset: tuple[float, float, float] | None = None,
    index_offset: tuple[float, float, float] | None = None,
    middle_offset: tuple[float, float, float] | None = None,
    use_contact_center: bool = False,
) -> torch.Tensor:
    """Compute reward for fingers opposing around the object for a pinch grasp.

    Encourage the thumb and index/middle fingers to be on opposite sides of the
    object by computing the negative dot product of the object-to-thumb and
    object-to-index/middle directions. When fingers oppose, the dot product is
    negative and the reward is positive.

    Only applied when the in_contact label is active for the given hand.

    Args:
        env: The environment.
        thumb_link: Name of the thumb tip link.
        index_link: Name of the index finger tip link.
        middle_link: Name of the middle finger tip link.
        hand: Which hand's in_contact label to use ("left_hand" or "right_hand").
        command_name: Name of the tracking command term.
        thumb_offset: Optional (x, y, z) offset in the thumb link's local frame,
            e.g. to shift from the link origin to the fingertip contact point.
        index_offset: Optional (x, y, z) offset in the index link's local frame.
        middle_offset: Optional (x, y, z) offset in the middle link's local frame.
        use_contact_center: If True, use the motion data's contact center point
            (mean of contact points on the object surface) instead of the object
            center of mass. Falls back to object CoM per-env when contact center
            is unavailable or zero.

    Returns:
        Reward tensor of shape (num_envs,). Positive when fingers oppose.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Use per-frame in_contact label from motion data
    in_contact = command.get_in_contact(hand)
    if in_contact is not None:
        should_contact = in_contact
    else:
        # Fallback to legacy per-env first contact frame
        per_env_first_contact = getattr(command, "_per_env_first_contact", None)
        if per_env_first_contact is not None:
            current_time = command.motion_start_time_steps + command.time_steps
            should_contact = (current_time >= per_env_first_contact).float()
        else:
            should_contact = torch.ones(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    obj = env.scene["object"]

    # Reference point: contact center from motion data, or object CoM fallback
    ref_pos = obj.data.root_pos_w[:, :3]  # (num_envs, 3)
    if use_contact_center:
        side = "left" if hand == "left_hand" else "right"
        contact_center = getattr(command, f"object_contact_center_{side}", None)
        if contact_center is not None:
            valid = torch.norm(contact_center, dim=-1, keepdim=True) > 1e-6
            ref_pos = torch.where(valid, contact_center, ref_pos)

    # Get finger link indices
    body_names = robot.body_names
    if thumb_link not in body_names or index_link not in body_names:
        return torch.zeros(env.num_envs, device=env.device)
    thumb_idx = body_names.index(thumb_link)
    index_idx = body_names.index(index_link)
    middle_idx = body_names.index(middle_link) if middle_link in body_names else None

    # Get finger positions, optionally offset from link origin to fingertip
    thumb_pos = robot.data.body_pos_w[:, thumb_idx, :]  # (num_envs, 3)
    if thumb_offset is not None:
        offset = torch.tensor(thumb_offset, device=env.device, dtype=thumb_pos.dtype).unsqueeze(0).expand_as(thumb_pos)
        thumb_pos = thumb_pos + quat_apply(robot.data.body_quat_w[:, thumb_idx, :], offset)

    index_pos = robot.data.body_pos_w[:, index_idx, :]  # (num_envs, 3)
    if index_offset is not None:
        offset = torch.tensor(index_offset, device=env.device, dtype=index_pos.dtype).unsqueeze(0).expand_as(index_pos)
        index_pos = index_pos + quat_apply(robot.data.body_quat_w[:, index_idx, :], offset)

    # Compute directions from reference point to fingers
    ref_to_thumb = thumb_pos - ref_pos
    ref_to_index = index_pos - ref_pos
    support_dirs = [ref_to_index]
    if middle_idx is not None:
        middle_pos = robot.data.body_pos_w[:, middle_idx, :]  # (num_envs, 3)
        if middle_offset is not None:
            offset = torch.tensor(middle_offset, device=env.device, dtype=middle_pos.dtype).unsqueeze(0).expand_as(middle_pos)
            middle_pos = middle_pos + quat_apply(robot.data.body_quat_w[:, middle_idx, :], offset)
        support_dirs.append(middle_pos - ref_pos)
    ref_to_index_middle = torch.stack(support_dirs, dim=0).mean(dim=0)

    # Normalize directions
    ref_to_thumb = ref_to_thumb / (torch.norm(ref_to_thumb, dim=-1, keepdim=True) + 1e-6)
    ref_to_index_middle = ref_to_index_middle / (
        torch.norm(ref_to_index_middle, dim=-1, keepdim=True) + 1e-6
    )

    # Inner product
    inner_product = torch.sum(ref_to_thumb * ref_to_index_middle, dim=-1)

    # Return negative inner product so that opposite sides give positive reward
    # Only apply when contact should happen
    return -inner_product * should_contact


def reward_wrist_look_at_object(
    env: ManagerBasedRLEnv,
    wrist_link: str = "right_wrist_yaw_link",
    forward_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    hand: str = "right_hand",
    command_name: str = "motion",
) -> torch.Tensor:
    """Reward for wrist orientation pointing toward the object center during approach.

    Computes the dot product between the wrist's local forward axis (rotated to world
    frame) and the unit direction from wrist to object center. Returns a value in [-1, 1]
    where +1 means the wrist is perfectly aimed at the object.

    Gated by the motion data in_contact label: active only BEFORE contact (approach phase).
    Once in_contact is True, the reward is zeroed so it doesn't interfere with manipulation.

    Args:
        env: The environment.
        wrist_link: Name of the wrist rigid body link.
        forward_axis: Local-frame axis that should point toward the object (default: +X,
            verified visually for right_wrist_yaw_link on YAM robot).
        hand: Which hand's in_contact label to use ("right_hand" or "left_hand").
        command_name: Name of the motion command (for in_contact data).

    Returns:
        Reward tensor of shape (num_envs,). Range [-1, 1], use positive weight.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get in_contact label — fail loudly if missing
    in_contact = command.get_in_contact(hand)
    if in_contact is None:
        raise RuntimeError(
            f"reward_wrist_look_at_object: in_contact label not available for hand='{hand}' "
            f"on command='{command_name}'. The motion data must provide per-frame contact labels."
        )
    # Active only BEFORE contact (approach phase)
    approach_mask = 1.0 - in_contact

    robot = env.scene["robot"]
    obj = env.scene["object"]

    # Validate wrist link exists
    body_names = robot.body_names
    if wrist_link not in body_names:
        raise RuntimeError(
            f"reward_wrist_look_at_object: wrist_link='{wrist_link}' not found in robot body names. "
            f"Available: {body_names}"
        )
    wrist_idx = body_names.index(wrist_link)

    # Get wrist position and orientation
    wrist_pos = robot.data.body_pos_w[:, wrist_idx, :]    # [num_envs, 3]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx, :]  # [num_envs, 4]

    # Get object position
    obj_pos = obj.data.root_pos_w[:, :3]  # [num_envs, 3]

    # Rotate local forward axis to world frame
    forward_local = torch.tensor(
        forward_axis, device=env.device, dtype=wrist_pos.dtype
    ).unsqueeze(0).expand(env.num_envs, -1)  # [num_envs, 3]
    forward_world = quat_apply(wrist_quat, forward_local)  # [num_envs, 3]
    forward_world = forward_world / (torch.norm(forward_world, dim=-1, keepdim=True) + 1e-6)

    # Direction from wrist to object
    wrist_to_obj = obj_pos - wrist_pos  # [num_envs, 3]
    wrist_to_obj = wrist_to_obj / (torch.norm(wrist_to_obj, dim=-1, keepdim=True) + 1e-6)

    # Dot product: +1 when wrist forward axis points at object
    alignment = torch.sum(forward_world * wrist_to_obj, dim=-1)  # [num_envs]

    return alignment * approach_mask


def reward_wrist_smoothness_at_contact(
    env: ManagerBasedRLEnv,
    wrist_link: str = "right_wrist_yaw_link",
    command_name: str = "motion",
    window_seconds: float = 1.0,
    ee_radius: float = 0.15,
) -> torch.Tensor:
    """Penalize wrist velocity in a time window around the contact transition.

    Discourages sudden jerky wrist movements at the approach→grasp boundary.
    Active for ±window_seconds/2 around the first contact frame (from reference motion).
    Outside this window, returns zero.

    Combines linear and angular velocity into a unified penalty using an
    equivalent-radius approximation: angular velocity is scaled by ee_radius²
    to convert to equivalent linear velocity² at the end effector tip.
    penalty = lin_vel² + ee_radius² * ang_vel²

    Requires _first_contact_lookup to be populated on the TrackingCommand
    (automatically derived when object motions pkl has contact_points_right_hand).

    Args:
        env: The environment.
        wrist_link: Name of the wrist rigid body link.
        command_name: Name of the motion command.
        window_seconds: Total window duration in seconds, centered on contact transition.
        ee_radius: Distance (meters) from wrist center to end effector tip.
            Used to scale angular velocity to equivalent linear velocity.
            Default 0.15m (~15cm for YAM gripper).

    Returns:
        Penalty tensor of shape (num_envs,). Positive values, use negative weight.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get first contact lookup
    first_contact_lookup = getattr(command, "_first_contact_lookup", None)
    if first_contact_lookup is None:
        raise RuntimeError(
            "reward_wrist_smoothness_at_contact requires _first_contact_lookup but it is None. "
            "The object motions pkl must contain contact_points_right_hand."
        )

    robot = env.scene["robot"]

    # Validate wrist link
    body_names = robot.body_names
    if wrist_link not in body_names:
        raise RuntimeError(
            f"reward_wrist_smoothness_at_contact: wrist_link='{wrist_link}' not found. "
            f"Available: {body_names}"
        )
    wrist_idx = body_names.index(wrist_link)

    # Current frame in motion space
    current_frame = command.motion_start_time_steps + command.time_steps  # [num_envs]
    first_contact = first_contact_lookup[command.motion_ids]  # [num_envs]

    # Compute window in frames: ±half_window around contact
    sim_fps = 1.0 / env.step_dt  # typically 50
    half_window_frames = int(window_seconds * sim_fps / 2.0)

    # Signed distance to contact: negative=before, positive=after
    distance = current_frame - first_contact  # [num_envs]

    # Active only within window
    in_window = ((distance >= -half_window_frames) & (distance <= half_window_frames)).float()

    # Wrist velocity penalty: linear + angular (scaled by ee_radius² for equivalence)
    wrist_lin_vel = robot.data.body_lin_vel_w[:, wrist_idx, :]   # [num_envs, 3]
    wrist_ang_vel = robot.data.body_ang_vel_w[:, wrist_idx, :]   # [num_envs, 3]
    lin_vel_sq = torch.sum(torch.square(wrist_lin_vel), dim=-1)   # [num_envs]
    ang_vel_sq = torch.sum(torch.square(wrist_ang_vel), dim=-1)   # [num_envs]

    penalty = (lin_vel_sq + ee_radius * ee_radius * ang_vel_sq) * in_window

    return penalty


def reward_grasp_contact_center(
    env: ManagerBasedRLEnv,
    finger_links: list[str] | None = None,
    hand: str = "right_hand",
    command_name: str = "motion",
    exp_coeff: float = -10.0,
) -> torch.Tensor:
    """Compute reward for fingers approaching the reference contact center.

    Use the per-hand contact center (mean of all contact points) from the motion
    library to guide grasping. The reward uses an exponential kernel on the average
    distance from fingertips to the contact center.

    Only applied when the contact label indicates contact should happen.

    Args:
        env: The environment.
        finger_links: List of finger link names to track. Defaults to the
            right hand fingertips if hand="right_hand", left otherwise.
        hand: Which hand's contact center to use ("left_hand" or "right_hand").
        command_name: Name of the tracking command term.
        exp_coeff: Exponential coefficient for distance-to-reward mapping (negative).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    if finger_links is None:
        if hand == "left_hand":
            finger_links = [
                "left_hand_thumb_2_link",
                "left_hand_index_1_link",
                "left_hand_middle_1_link",
            ]
        else:
            finger_links = [
                "right_hand_thumb_2_link",
                "right_hand_index_1_link",
                "right_hand_middle_1_link",
            ]

    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get per-hand contact center in world frame
    if hand == "left_hand":
        contact_center = command.object_contact_center_left
    else:
        contact_center = command.object_contact_center_right

    if contact_center is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Use per-frame in_contact label from motion data
    in_contact = command.get_in_contact(hand)
    if in_contact is not None:
        should_contact = in_contact
    else:
        # Fallback to legacy per-env first contact frame
        per_env_first_contact = getattr(command, "_per_env_first_contact", None)
        if per_env_first_contact is not None:
            current_time = command.motion_start_time_steps + command.time_steps
            should_contact = (current_time >= per_env_first_contact).float()
        else:
            should_contact = torch.ones(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    body_names = robot.body_names

    # Get finger positions
    finger_indices = [body_names.index(link) for link in finger_links]
    finger_positions = robot.data.body_pos_w[:, finger_indices, :]  # (num_envs, num_fingers, 3)

    # Compute distance from each finger to the contact center
    contact_center_expanded = contact_center.unsqueeze(1)  # (num_envs, 1, 3)
    distances = torch.norm(
        finger_positions - contact_center_expanded, dim=-1
    )  # (num_envs, num_fingers)

    # Average over fingers
    avg_distance = distances.mean(dim=-1)  # (num_envs,)

    # Convert distance to reward (closer = higher reward)
    reward = torch.exp(exp_coeff * avg_distance)

    return reward * should_contact


def reward_lift_distance(env: ManagerBasedRLEnv, exp_coeff: float = -10.0) -> torch.Tensor:
    """Compute reward for lifting the object toward a goal position in the robot's local frame.

    Transform the object position into the robot's root frame and compute exponential
    reward based on distance to a fixed local goal (0.3m forward, 0.5m up).

    Args:
        env: The environment.
        exp_coeff: Exponential coefficient for distance-to-reward mapping (negative).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    obj = env.scene["object"]
    obj_pos = obj.data.root_pos_w[:, :3]
    goal_pos = [0.3, 0.0, 0.5]
    # Transform obj_pos to local frame relative to robot root
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    obj_pos_local = quat_rotate_inverse(robot_quat, obj_pos - robot_pos, w_last=True)
    # Calculate distance between obj_pos_local and goal_pos
    distance = torch.norm(obj_pos_local - goal_pos, dim=-1)
    return torch.exp(exp_coeff * -distance)


def reward_lift_z_elevation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Compute reward proportional to object z elevation above the table.

    Reward increases linearly with object height above 0.8m (assumed table +
    half-object height), clamped to a maximum of 0.3m.

    Args:
        env: The environment.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 0.3].
    """
    obj = env.scene["object"]
    obj_z = obj.data.root_pos_w[:, 2]
    # Reward lifting above table height + half of the object height (assume it is 0.8m)
    return torch.clamp(obj_z - 0.8, min=0, max=0.3)


def reward_hand_table_contact_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Compute penalty for hand contacting the table surface.

    Discourage the hand from pressing against the table. Uses force_matrix_w
    (filtered forces between table and hand links only), not net_forces_w which
    includes all forces on the table.

    Args:
        env: The environment.
        sensor_cfg: Contact sensor config for table-hand contact.

    Returns:
        Penalty tensor of shape (num_envs,) in [0, 1]. Clamped total force magnitude.
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    # force_matrix_w: [num_envs, num_sensor_bodies, num_filter_bodies, 3]
    contact_force = sensor.data.force_matrix_w
    force_magnitude = torch.norm(contact_force, dim=-1).sum(dim=(-1, -2))
    return torch.clamp(force_magnitude, max=1.0)


def object_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    w_op: float = 0.5,  # object position weight
    w_or: float = 0.3,  # object rotation weight
    w_ov: float = 0.05,  # object linear velocity weight
    w_oav: float = 0.05,  # object angular velocity weight
    pos_exp_coeff: float = -100.0,
    ori_exp_coeff: float = -100.0,
    vel_exp_coeff: float = -5.0,
    ang_vel_exp_coeff: float = -5.0,
) -> torch.Tensor:
    """Compute object tracking reward following the OmniGrasp formulation.

    Weighted sum of exponential rewards for object position, orientation, linear
    velocity, and angular velocity tracking, gated by a contact indicator. Only
    active when the robot is in contact with the object. Contact encouragement
    is handled separately by reward_grasp.

    r_t^obj = (w_op * exp(c_p * ||err_p||) + w_or * exp(c_r * ||err_r||)
             + w_ov * exp(c_v * ||err_v||) + w_oav * exp(c_av * ||err_av||)) * 1{C}

    Args:
        env: The environment.
        command_name: Name of the tracking command containing object trajectory.
        sensor_cfg: Contact sensor config for finger-object contact detection.
        w_op: Weight for object position tracking.
        w_or: Weight for object orientation tracking.
        w_ov: Weight for object linear velocity tracking.
        w_oav: Weight for object angular velocity tracking.
        pos_exp_coeff: Exponential coefficient for position error.
        ori_exp_coeff: Exponential coefficient for orientation error.
        vel_exp_coeff: Exponential coefficient for linear velocity error.
        ang_vel_exp_coeff: Exponential coefficient for angular velocity error.

    Returns:
        Reward tensor of shape (num_envs,). Zero when not in contact.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get current object state from scene
    obj = env.scene["object"]
    current_obj_pos = obj.data.root_pos_w[:, :3]
    current_obj_quat = obj.data.root_quat_w
    current_obj_lin_vel = obj.data.root_lin_vel_w[:, :3]
    current_obj_ang_vel = obj.data.root_ang_vel_w[:, :3]

    # Get target object state from motion library
    target_obj_pos = command.object_root_pos[:, 0, :3]
    target_obj_quat = command.object_root_quat[:, 0]

    # Get target velocities from precomputed motion library data
    current_time = command.motion_start_time_steps + command.time_steps
    target_obj_lin_vel = command.motion_lib.get_object_lin_vel(command.motion_ids, current_time)[
        :, 0, :
    ]
    target_obj_ang_vel = command.motion_lib.get_object_ang_vel(command.motion_ids, current_time)[
        :, 0, :
    ]

    # Position tracking
    pos_error = torch.norm(target_obj_pos - current_obj_pos, dim=-1)
    pos_reward = torch.exp(pos_exp_coeff * pos_error)

    # Orientation tracking
    ori_error = quat_error_magnitude(target_obj_quat, current_obj_quat)
    ori_reward = torch.exp(ori_exp_coeff * ori_error)

    # Linear velocity tracking
    lin_vel_error = torch.norm(target_obj_lin_vel - current_obj_lin_vel, dim=-1)
    lin_vel_reward = torch.exp(vel_exp_coeff * lin_vel_error)

    # Angular velocity tracking
    ang_vel_error = torch.norm(target_obj_ang_vel - current_obj_ang_vel, dim=-1)
    ang_vel_reward = torch.exp(ang_vel_exp_coeff * ang_vel_error)

    # print("obj_po", target_obj_pos, current_obj_pos)
    # print("d0:", pos_error, pos_reward, ori_error, ori_reward)
    # print("d1:", lin_vel_error, lin_vel_reward, ang_vel_error, ang_vel_reward)

    # Contact indicator 1{C}: finger-object contact from object_to_hand_contact_sensor
    # This sensor tracks contact between object and finger links (thumb, index, middle, palm)
    # force_matrix_w: [num_envs, 1, 4, 3] - force on Object from each of 4 finger links
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    contact_force = sensor.data.force_matrix_w
    force_magnitude = torch.norm(contact_force, dim=-1).sum(dim=(-1, -2))
    in_contact = (force_magnitude > 1.0).float()

    # Tracking reward gated by contact: (tracking) * 1{C}
    tracking_reward = (
        w_op * pos_reward + w_or * ori_reward + w_ov * lin_vel_reward + w_oav * ang_vel_reward
    )

    return tracking_reward * in_contact


def object_lift_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_lift_height: float = 0.15,  # 15 cm max reward height
) -> torch.Tensor:
    """Compute reward for lifting the object above the table after first contact.

    Linear reward based on object height above the table surface, normalized by
    max_lift_height. Zero before the first contact frame from the motion data.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        max_lift_height: Maximum height (meters) for full reward. Heights above
            this are clamped.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    current_time = command.motion_start_time_steps + command.time_steps

    # Use precomputed per-env first contact frame
    per_env_first_contact = getattr(command, "_per_env_first_contact", None)
    if per_env_first_contact is not None:
        should_lift = current_time > per_env_first_contact
    else:
        # No contact data, always apply reward
        should_lift = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    # Get object and table positions
    obj = env.scene["object"]
    table = env.scene["table"]
    object_z = obj.data.root_pos_w[:, 2]  # Object height
    table_z = table.data.root_pos_w[:, 2]  # Table height

    # Linear lift reward: how high object is above table, clipped to max_lift_height
    height_above_table = torch.clamp(object_z - table_z, min=0.0, max=max_lift_height)
    lift_height_reward = height_above_table / max_lift_height  # Normalize to [0, 1]

    # Only apply reward after first contact frame
    lift_reward = torch.where(
        should_lift, lift_height_reward.reshape(-1), torch.zeros_like(lift_height_reward)
    )
    return lift_reward


def reward_finger_primitive_limit(
    env: ManagerBasedRLEnv,
    mode: str = "discrete",
    action_scale: float = 1.0,
    discrete_threshold: float = 0.5,
) -> torch.Tensor:
    """Penalize finger primitive actions that exceed valid limits.

    Encourage the policy to output actions within the valid range naturally,
    rather than relying on clamping. This helps with gradient flow during training.

    Args:
        env: The environment.
        mode: "linear" or "discrete". Determines the valid action range.
        action_scale: For linear mode, valid range is [-1/action_scale, 1/action_scale].
        discrete_threshold: For discrete mode, valid range is [-threshold, threshold].

    Returns:
        Penalty tensor of shape (num_envs,). Positive values indicate out-of-limit.
    """
    # Get raw primitive actions stored by wrapper before processing
    primitive_actions = getattr(env, "_finger_primitive_actions_raw", None)

    if primitive_actions is None:
        # Finger primitives not enabled, return zero penalty
        return torch.zeros(env.num_envs, device=env.device)

    # Determine valid range based on mode
    if mode == "linear":
        lower_limit = -1.0 / action_scale
        upper_limit = 1.0 / action_scale
    elif mode == "discrete":
        lower_limit = -discrete_threshold
        upper_limit = discrete_threshold
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'linear' or 'discrete'.")

    below_limit = (-primitive_actions + lower_limit).clamp(min=0.0)  # positive when below lower
    above_limit = (primitive_actions - upper_limit).clamp(min=0.0)  # positive when above upper

    # Sum penalties across all primitive actions
    penalty = (below_limit + above_limit).sum(dim=-1)

    return penalty


def energy_consumption(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute energy consumption penalty based on mechanical power.

    Sum absolute joint torque-velocity products (P = |tau * omega|) across all
    joints. Encourages the policy to minimize energy usage while performing the task.

    Args:
        env: The environment.
        asset_cfg: The asset config for the robot (default: "robot").

    Returns:
        Penalty tensor of shape (num_envs,). Higher values = more energy used.
    """
    # Handle both dict (from YAML) and SceneEntityCfg object
    # Get the robot asset
    # Handle both dict (from YAML) and SceneEntityCfg object
    if isinstance(asset_cfg, dict):
        robot_name = asset_cfg.get("name", "robot")
    else:
        robot_name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[robot_name]

    # Get joint torques (applied torques) and joint velocities
    # joint_acc_pos_w: [num_envs, num_joints] - applied torques
    # joint_vel: [num_envs, num_joints] - joint velocities
    joint_torques = robot.data.applied_torque
    joint_velocities = robot.data.joint_vel

    # Compute instantaneous mechanical power: P = |tau * omega|
    # Taking absolute value since we care about energy magnitude regardless of direction
    power = torch.abs(joint_torques * joint_velocities)

    # Sum over all joints to get total energy consumption
    total_energy = power.sum(dim=-1)

    return total_energy


def meta_action_rate_l2(
    env: ManagerBasedRLEnv,
    token_only: bool = False,
) -> torch.Tensor:
    """Penalize the rate of change of meta actions (token + residual) via L2-squared.

    The meta action is the policy's raw output: latent residual (e.g., 64 dims)
    + finger primitives (e.g., 2 dims). Encourages smooth token trajectories in
    latent space, reducing jitter in the decoded joint-level actions.

    Since full_token = encoded_token + scaled_residual and encoded_token is constant
    for the same proprioception, penalizing the residual rate effectively penalizes
    the full_token rate.

    Args:
        env: The environment.
        token_only: If True, only penalize the latent/token portion (excluding
            finger primitives). If False, penalize the full meta action.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values = higher rate.
    """
    last_meta = getattr(env, "_last_meta_action", None)
    prev_meta = getattr(env, "_prev_meta_action", None)

    if last_meta is None or prev_meta is None:
        return torch.zeros(env.num_envs, device=env.device)

    if token_only:
        # Exclude finger primitives (last 2 dims by default)
        # meta_action_dim = tokenizer_action_dim + hand_action_dim
        # hand_action_dim is typically 2
        hand_action_dim = getattr(env, "_hand_action_dim", 2)
        tokenizer_dim = last_meta.shape[-1] - hand_action_dim
        diff = last_meta[:, :tokenizer_dim] - prev_meta[:, :tokenizer_dim]
    else:
        diff = last_meta - prev_meta

    return torch.sum(torch.square(diff), dim=-1)


def full_latent_rate_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize the rate of change of the full decoder-input latent via L2-squared.

    The full latent is the post-quantization token sent to the decoder:
        full_latent = FSQ_quantize(encoder_output + scaled_residual)

    Unlike meta_action_rate_l2 which penalizes the policy's residual output, this
    penalizes the actual decoder input including the encoder's contribution.

    NOTE: Buffers (_full_latent, _prev_full_latent) are maintained by
    ManagerEnvWrapper.

    Args:
        env: The environment.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values = higher rate.
    """
    full_latent = getattr(env, "_full_latent", None)
    prev_full_latent = getattr(env, "_prev_full_latent", None)

    if full_latent is None or prev_full_latent is None:
        return torch.zeros(env.num_envs, device=env.device)

    diff = full_latent - prev_full_latent
    return torch.sum(torch.square(diff), dim=-1)


# ==================== Approach Behavior Rewards ====================


def reward_approach_velocity_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    wrist_link: str = "right_hand_palm_link",
    sigma: float = 0.25,
    contact_threshold: float = 0.1,
    min_contacts: int = 3,
    disable_after_contact: bool = True,
) -> torch.Tensor:
    """Penalize high wrist velocity near the object to prevent swatting.

    Apply a Gaussian distance-gated velocity penalty: strongest at the object and
    decaying with distance. By default, disabled after actual grasp (physics contact)
    to allow free movement during lifting/manipulation.

    Formula: penalty = velocity^2 * exp(-distance^2 / sigma^2) * contact_gate

    NOTE: Disabling penalty after grasp may incentivize rushing to touch the object
    to escape the penalty. If observed, increase min_contacts or set
    disable_after_contact=False.

    Args:
        env: The environment.
        sensor_cfg: Contact sensor config for finger-object contact detection.
        wrist_link: Name of the wrist/palm link to track velocity of.
        sigma: Characteristic distance (meters). Penalty is ~37% at this distance.
        contact_threshold: Minimum contact force (N) to count as touching.
        min_contacts: Minimum finger contacts to count as "grasping".
        disable_after_contact: If True, penalty turns off after grasp. If False,
            penalty stays active during grasp.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (use negative weight).
    """
    # Get robot and object from scene
    robot = env.scene["robot"]
    obj = env.scene["object"]

    # Get wrist body index from robot.body_names (ALL robot bodies)
    body_names = robot.body_names
    if wrist_link not in body_names:
        return torch.zeros(env.num_envs, device=env.device)
    wrist_index = body_names.index(wrist_link)

    # Get wrist position and velocity in world frame
    wrist_pos = robot.data.body_pos_w[:, wrist_index, :]  # [num_envs, 3]
    wrist_vel = robot.data.body_lin_vel_w[:, wrist_index, :]  # [num_envs, 3]

    # Get object position
    object_pos = obj.data.root_pos_w[:, :3]  # [num_envs, 3]

    # Compute squared distance and squared velocity (L2)
    diff = wrist_pos - object_pos
    distance_sq = torch.sum(torch.square(diff), dim=-1)  # [num_envs]
    velocity_sq = torch.sum(torch.square(wrist_vel), dim=-1)  # [num_envs]

    # Gaussian distance factor: strongest at object, decays with distance
    sigma_sq = sigma * sigma
    distance_factor = torch.exp(-distance_sq / sigma_sq)

    # Contact gating using ACTUAL physics contact (same as grasp_reward)
    if disable_after_contact:
        sensor: ContactSensor = env.scene[sensor_cfg.name]
        contact_force = sensor.data.force_matrix_w
        force_magnitude = torch.norm(contact_force, dim=-1).squeeze(1)  # [num_envs, num_fingers]

        # Count fingers in contact (force above threshold)
        fingers_in_contact = (force_magnitude > contact_threshold).float()
        num_fingers_in_contact = fingers_in_contact.sum(dim=-1)

        # Grasping = at least min_contacts fingers touching
        is_grasping = (num_fingers_in_contact >= min_contacts).float()
        contact_gate = 1.0 - is_grasping
    else:
        # No contact gating - penalty stays active even during/after grasp
        contact_gate = 1.0

    # Final penalty: velocity^2 * distance_factor * contact_gate
    penalty = velocity_sq * distance_factor * contact_gate

    return penalty


def reward_open_hand_until_close(
    env: ManagerBasedRLEnv,
    wrist_link: str = "right_hand_palm_link",
    sigma: float = 0.25,
    hand: str = "right",
) -> torch.Tensor:
    """Penalize closing the hand when far from the object.

    Encourage the robot to keep its hand open during approach and only close when
    close to the object (within grasp range). Uses an inverse Gaussian: penalty is
    zero at the object and increases with distance.

    Formula: penalty = hand_closed * (1 - exp(-distance^2 / sigma^2))

    Args:
        env: The environment.
        wrist_link: Name of the wrist/palm link to measure distance from.
        sigma: Characteristic distance (meters). Penalty reaches ~63% at this distance.
        hand: Which hand to check ("left" or "right").

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (use negative weight).
    """
    # Get hand action state (0 = left, 1 = right due to sorted keys)
    hand_actions = getattr(env, "_finger_primitive_actions_raw", None)
    if hand_actions is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Hand index: sorted(["left", "right"]) = ["left", "right"]
    # Index 0 = left, Index 1 = right
    hand_idx = 1 if hand == "right" else 0

    # Binary closed detection: action >= 0 means closed (discrete mode threshold)
    hand_closed = (hand_actions[:, hand_idx] >= 0).float()  # [num_envs]

    # Get robot and object from scene
    robot = env.scene["robot"]
    obj = env.scene["object"]

    # Get wrist body index from robot.body_names (ALL robot bodies)
    body_names = robot.body_names
    if wrist_link not in body_names:
        return torch.zeros(env.num_envs, device=env.device)
    wrist_index = body_names.index(wrist_link)

    wrist_pos = robot.data.body_pos_w[:, wrist_index, :]  # [num_envs, 3]

    # Get object position
    object_pos = obj.data.root_pos_w[:, :3]  # [num_envs, 3]

    # Compute squared distance
    diff = wrist_pos - object_pos
    distance_sq = torch.sum(torch.square(diff), dim=-1)  # [num_envs]

    # Inverse Gaussian: penalty when FAR, not when close
    # At distance=0: factor=0 (no penalty for closing)
    # At distance=sigma: factor=0.63
    # At distance=2*sigma: factor=0.98
    sigma_sq = sigma * sigma
    far_factor = 1.0 - torch.exp(-distance_sq / sigma_sq)

    # Final penalty: closed * far_factor
    penalty = hand_closed * far_factor

    return penalty


def reward_open_hand_until_close_contact_gated(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    wrist_link: str = "right_hand_palm_link",
    sigma: float = 0.25,
    hand: str = "right",
    contact_threshold: float = 0.1,
    min_contacts: int = 3,
) -> torch.Tensor:
    """Penalize closing the hand when far from the object, gated by physics contact.

    Same as reward_open_hand_until_close but also disables the penalty once the
    robot has established a grasp (detected via contact sensor), allowing free
    hand closure during manipulation.

    Formula: penalty = hand_closed * far_factor * not_grasping

    Args:
        env: The environment.
        sensor_cfg: Contact sensor configuration for detecting grasp.
        wrist_link: Name of the wrist/palm link to measure distance from.
        sigma: Characteristic distance (meters). Penalty reaches ~63% at this distance.
        hand: Which hand to check ("left" or "right").
        contact_threshold: Force threshold (N) to consider a finger in contact.
        min_contacts: Minimum number of fingers in contact to consider grasping.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (use negative weight).
    """
    # Get hand action state (0 = left, 1 = right due to sorted keys)
    hand_actions = getattr(env, "_finger_primitive_actions_raw", None)
    if hand_actions is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Hand index: sorted(["left", "right"]) = ["left", "right"]
    # Index 0 = left, Index 1 = right
    hand_idx = 1 if hand == "right" else 0

    # Binary closed detection: action >= 0 means closed (discrete mode threshold)
    hand_closed = (hand_actions[:, hand_idx] >= 0).float()  # [num_envs]

    # Get robot and object from scene
    robot = env.scene["robot"]
    obj = env.scene["object"]

    # Get wrist body index from robot.body_names (ALL robot bodies)
    body_names = robot.body_names
    if wrist_link not in body_names:
        return torch.zeros(env.num_envs, device=env.device)
    wrist_index = body_names.index(wrist_link)

    wrist_pos = robot.data.body_pos_w[:, wrist_index, :]  # [num_envs, 3]

    # Get object position
    object_pos = obj.data.root_pos_w[:, :3]  # [num_envs, 3]

    # Compute squared distance
    diff = wrist_pos - object_pos
    distance_sq = torch.sum(torch.square(diff), dim=-1)  # [num_envs]

    # Inverse Gaussian: penalty when FAR, not when close
    sigma_sq = sigma * sigma
    far_factor = 1.0 - torch.exp(-distance_sq / sigma_sq)

    # Physics-based grasp detection using contact sensor
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    contact_force = sensor.data.force_matrix_w
    force_magnitude = torch.norm(contact_force, dim=-1).squeeze(1)

    # Count fingers in contact (force above threshold)
    fingers_in_contact = (force_magnitude > contact_threshold).float()
    num_fingers_in_contact = fingers_in_contact.sum(dim=-1)

    # Grasping = at least min_contacts fingers touching
    is_grasping = (num_fingers_in_contact >= min_contacts).float()
    not_grasping = 1.0 - is_grasping

    # Final penalty: closed * far_factor * not_grasping
    penalty = hand_closed * far_factor * not_grasping

    return penalty


def reward_foot_slippage_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize foot velocity while the foot is in ground contact.

    Discourages the robot from shuffling or skating its feet along the ground.
    When a foot has ground contact force above the threshold, any velocity of
    that foot contributes to the penalty.

    Formula: penalty = sum_feet( ||foot_vel|| * (||contact_force|| > threshold) )

    Args:
        env: The environment.
        sensor_cfg: Contact sensor configuration with body_names for feet.
        contact_threshold: Force magnitude (N) above which the foot is considered
            in ground contact. Default 1.0N matches existing foot contact checks.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (use negative weight).
    """
    robot = env.scene["robot"]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Find foot body indices in both the articulation and the contact sensor
    feet_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    feet_idx_robot = [robot.body_names.index(n) for n in feet_names]
    feet_idx_sensor = contact_sensor.find_bodies(".*ankle_roll.*")[0]

    # Foot velocities — full 3D
    foot_vel = robot.data.body_lin_vel_w[:, feet_idx_robot, :]  # [num_envs, 2, 3]
    foot_speed = torch.norm(foot_vel, dim=-1)  # [num_envs, 2]

    # Contact mask: is foot touching ground?
    contact_force = contact_sensor.data.net_forces_w[:, feet_idx_sensor, :]  # [num_envs, 2, 3]
    in_contact = (torch.norm(contact_force, dim=-1) > contact_threshold).float()  # [num_envs, 2]

    # Penalty = speed * in_contact, summed over both feet
    return (foot_speed * in_contact).sum(dim=-1)
