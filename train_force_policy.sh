#!/usr/bin/env bash
# Stage 2: train the local Force Policy, conditioned on the frozen RISE-2 feature phi(I).
# Requires Stage 1 (train.sh) and feature export (run_generator.sh) to be done first.
# Main configuration: separate MIP (action) + MLP (frame/wrench/mask) heads, one-step reference GT.

POLICY=force_policy.vision_feat_wrist_vision_cond_comb_lowdim_cond_gated_with_global_sep_action_mip_other_1_step

# Push and Flip
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_addr 127.0.0.1 --master_port 24008 --nproc_per_node 2 run_trainer.py \
    --trainer force_policy --dataset flip_v3_wrist --processor flip.force_policy.wrist \
    --policy ${POLICY} --wrapper force_policy.force_policy \
    --ckpt_dir logs/flip_v3_force_policy/

# Plug in EV Charger
# CUDA_VISIBLE_DEVICES=0,1 torchrun --master_addr 127.0.0.1 --master_port 24008 --nproc_per_node 2 run_trainer.py \
#     --trainer force_policy --dataset charger_v2_wrist --processor charger.force_policy.wrist \
#     --policy ${POLICY} --wrapper force_policy.force_policy \
#     --ckpt_dir logs/charger_v2_force_policy/

# Scrape off Sticker
# CUDA_VISIBLE_DEVICES=0,1 torchrun --master_addr 127.0.0.1 --master_port 24008 --nproc_per_node 2 run_trainer.py \
#     --trainer force_policy --dataset shovel_wrist --processor shovel.force_policy.wrist \
#     --policy ${POLICY} --wrapper force_policy.force_policy \
#     --ckpt_dir logs/shovel_force_policy/
