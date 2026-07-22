#!/usr/bin/env bash
# Stage 1: train the global vision policy (RISE-2) for each task.
# RISE-2 is frozen afterwards; its 512-D action feature is exported by run_generator.sh.
# Adjust CUDA_VISIBLE_DEVICES / --nproc_per_node / --ckpt_dir for your setup.

# Push and Flip
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_addr 127.0.0.1 --master_port 24003 --nproc_per_node 4 run_trainer.py \
    --trainer RISE2 --dataset flip_v3_vision_only --processor flip.RISE2 \
    --policy vision_policy.RISE2_robot_only --wrapper vision_policy.RISE2_robot_only \
    --ckpt_dir logs/flip_v3_rise2/

# Plug in EV Charger
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_addr 127.0.0.1 --master_port 24003 --nproc_per_node 4 run_trainer.py \
#     --trainer RISE2 --dataset charger_v2_vision_only --processor charger.RISE2_top \
#     --policy vision_policy.RISE2_robot_only --wrapper vision_policy.RISE2_robot_only \
#     --ckpt_dir logs/charger_v2_rise2/

# Scrape off Sticker
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_addr 127.0.0.1 --master_port 24003 --nproc_per_node 4 run_trainer.py \
#     --trainer RISE2 --dataset shovel_vision_only --processor shovel.RISE2 \
#     --policy vision_policy.RISE2_robot_only --wrapper vision_policy.RISE2_robot_only \
#     --ckpt_dir logs/shovel_rise2/
