#!/usr/bin/env bash
# Real-robot evaluation with the dual-policy asynchronous slow-fast scheduler.
# Slow: RISE-2 (global vision policy). Fast: Force Policy @ 50Hz (local force policy).
# Requires easyrobot + flexivrdk and correct camera/robot serials in configs/agent/* and
# configs/agent_class/*. Point --ckpt_path / --fast_ckpt_path to your trained checkpoints.

FAST_POLICY=force_policy.vision_feat_wrist_vision_cond_comb_lowdim_cond_gated_with_global_sep_action_mip_other_1_step

# Push and Flip
python run_eval.py \
    --processor flip.RISE2 --policy vision_policy.RISE2_robot_only --wrapper vision_policy.RISE2_robot_only \
    --agent dual_right_two_camera --agent_obs_keys vision_RISE2 --agent_class flip \
    --scheduler flip.slow_fast --ckpt_path logs/flip_v3_rise2/policy_last.ckpt \
    --fast_processor flip.force_policy.wrist --fast_policy ${FAST_POLICY} \
    --fast_wrapper force_policy.force_policy --fast_agent_obs_keys force_policy \
    --fast_ckpt_path logs/flip_v3_force_policy/policy_last.ckpt \
    --visualize --vis_robot right

# Plug in EV Charger
# python run_eval.py \
#     --processor charger.RISE2_top --policy vision_policy.RISE2_robot_only --wrapper vision_policy.RISE2_robot_only \
#     --agent single_charger_top --agent_obs_keys vision_RISE2 --agent_class charger \
#     --scheduler charger.slow_fast --ckpt_path logs/charger_v2_rise2/policy_last.ckpt \
#     --fast_processor charger.force_policy.wrist --fast_policy ${FAST_POLICY} \
#     --fast_wrapper force_policy.force_policy --fast_agent_obs_keys force_policy \
#     --fast_ckpt_path logs/charger_v2_force_policy/policy_last.ckpt \
#     --visualize --vis_robot right

# Scrape off Sticker
# python run_eval.py \
#     --processor shovel.RISE2 --policy vision_policy.RISE2_robot_only --wrapper vision_policy.RISE2_robot_only \
#     --agent dual_right_robot_only_two_camera --agent_obs_keys vision_RISE2 --agent_class shovel \
#     --scheduler shovel.slow_fast --ckpt_path logs/shovel_rise2/policy_last.ckpt \
#     --fast_processor shovel.force_policy.wrist --fast_policy ${FAST_POLICY} \
#     --fast_wrapper force_policy.force_policy --fast_agent_obs_keys force_policy \
#     --fast_ckpt_path logs/shovel_force_policy/policy_last.ckpt \
#     --visualize --vis_robot right

# --- Optional: vision-only baselines (single-policy schedulers) ---
# Synchronous baseline:   --scheduler <task>.vanilla_no_ensemble  (no --fast_* args)
# Asynchronous baseline:  --scheduler <task>.adaptive             (no --fast_* args)
# Add --record --record_freq 15 to save evaluation logs (convert via scripts/convert_eval_log_to_video.py).
