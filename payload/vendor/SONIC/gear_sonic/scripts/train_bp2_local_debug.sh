#!/usr/bin/env bash
# ============================================================
#  Local smoke test for the bp2 pick-and-place chain.
#
#  Runs gear_sonic.train_agent_trl with the
#    pnp_260303_randrigid_heading_bps (bp2) config, capped to
#    num_envs=4, headless=True, num_learning_iterations=3.
#  No evaluation is launched.
#
#  Prerequisites:
#    - gearenv conda env (isaaclab + isaac sim)
#    - Data at data/motion_lib_genhoi/p019_0324_ha/{robot,objects,object_usd}
#    - BPS at data/motion_lib_genhoi/p019_0324/bps/
#    - SONIC checkpoint at models/sonic_sm_3pt_heading_hh-20260203_131215/
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

HYDRA_CONFIG="manager/universal_token/hoi/pnp_260303_randrigid_heading_bps"
SONIC_DIR="models/sonic_sm_3pt_heading_hh-20260203_131215"
DATA_DIR="data/motion_lib_genhoi/p019_0324_ha"
BPS_DIR="data/motion_lib_genhoi/p019_0324/bps"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gearenv/bin/python}"

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

"${PYTHON_BIN}" -u train_agent_trl.py \
    "+exp=${HYDRA_CONFIG}" \
    num_envs=4 \
    headless=True \
    "++algo.config.num_learning_iterations=3" \
    "++manager_env.config.gpu_collision_stack_size_exp=28" \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=${DATA_DIR}/robot" \
    "++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${DATA_DIR}/objects" \
    "++manager_env.config.object_usd_path=${DATA_DIR}/object_usd" \
    "++manager_env.config.action_transform_module_cfg=${SONIC_DIR}/model_config.yaml" \
    "++manager_env.config.action_transform_module_checkpoint=${SONIC_DIR}/model_step_092000.pt" \
    "++manager_env.commands.motion.motion_lib_cfg.bps_dir=${BPS_DIR}" \
    "++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=${REPO_DIR}/data/assets/robot_description/mjcf/" \
    "++manager_env.rewards.upright_penalty.weight=0" \
    "++manager_env.terminations.object_pos_deviation.params.threshold=0.1" \
    "++manager_env.rewards.grasp_finger_direction.weight=10.0" \
    "++manager_env.rewards.grasp_finger_direction.params.use_contact_center=true" \
    "++callbacks.im_resample._target_=gear_sonic.trl.callbacks.im_resample_callback.ImResampleCallback" \
    "++callbacks.im_resample.motion_resample_frequency=250" \
    "++manager_env.config.safe_nan=true" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy" \
    "++manager_env.config.motion_meta_info_path=null" \
    "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable=false" \
    "++manager_env.commands.motion.pose_range.x=[0,0]" \
    "++manager_env.commands.motion.pose_range.y=[0,0]" \
    "++manager_env.commands.motion.pose_range.z=[0,0]" \
    "++manager_env.commands.motion.pose_range.roll=[0,0]" \
    "++manager_env.commands.motion.pose_range.pitch=[0,0]" \
    "++manager_env.commands.motion.pose_range.yaw=[0,0]" \
    "++manager_env.commands.motion.velocity_range.x=[0,0]" \
    "++manager_env.commands.motion.velocity_range.y=[0,0]" \
    "++manager_env.commands.motion.velocity_range.z=[0,0]" \
    "++manager_env.commands.motion.velocity_range.roll=[0,0]" \
    "++manager_env.commands.motion.velocity_range.pitch=[0,0]" \
    "++manager_env.commands.motion.velocity_range.yaw=[0,0]" \
    "++manager_env.commands.motion.joint_position_range=[0,0]" \
    "++manager_env.commands.motion.joint_velocity_range=[0,0]" \
    "++manager_env.commands.motion.object_position_randomize=false" \
    "++manager_env.events.add_joint_default_pos=null" \
    "++manager_env.events.base_com=null" \
    "++manager_env.events.randomize_rigid_body_mass=null" \
    "++manager_env.events.randomize_table_size=null" \
    "++manager_env.events.randomize_object_size=null" \
    "++manager_env.events.physics_material=null" \
    "++manager_env.observations.policy.enable_corruption=false" \
    "++manager_env.observations.tokenizer.enable_corruption=false" \
    "++manager_env.observations.policy_atm.enable_corruption=false" \
    "++manager_env.commands.motion.sample_before_contact=true" \
    "++manager_env.commands.motion.start_from_first_frame=false"
