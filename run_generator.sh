#!/usr/bin/env bash
# Stage 1.5: export the global RISE-2 feature phi(I) into each scene's lowdim/ directory
# (vision_feat_<cam>.h5), consumed by the local Force Policy during Stage 2.
# Run after train.sh; point --ckpt_path to the trained RISE-2 checkpoint.

# Push and Flip
CUDA_VISIBLE_DEVICES=0 python run_generator.py \
    --dataset flip_v3_vision_only \
    --processor flip.RISE2 \
    --policy vision_policy.RISE2_robot_only \
    --wrapper vision_policy.RISE2_robot_only \
    --ckpt_path logs/flip_v3_rise2/policy_last.ckpt

# Plug in EV Charger
# CUDA_VISIBLE_DEVICES=0 python run_generator.py \
#     --dataset charger_v2_vision_only \
#     --processor charger.RISE2_top \
#     --policy vision_policy.RISE2_robot_only \
#     --wrapper vision_policy.RISE2_robot_only \
#     --ckpt_path logs/charger_v2_rise2/policy_last.ckpt

# Scrape off Sticker
# CUDA_VISIBLE_DEVICES=0 python run_generator.py \
#     --dataset shovel_vision_only \
#     --processor shovel.RISE2 \
#     --policy vision_policy.RISE2_robot_only \
#     --wrapper vision_policy.RISE2_robot_only \
#     --ckpt_path logs/shovel_rise2/policy_last.ckpt
