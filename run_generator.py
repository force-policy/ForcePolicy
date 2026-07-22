"""
Run vision feature generator.

Usage:
    python run_generator.py \
           --dataset [dataset name] \
           --processor [processor name] \
           --policy [policy name] \
           --wrapper [policy wrapper name] \
           --ckpt_path [checkpoint path for the policy]
"""
import argparse

from vision_feat_generator import VisionFeatGenerator
from vision_feat_generator.configs import VisionFeatGeneratorConfig

from configs import get_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Run vision feature generator.")
    parser.add_argument("--dataset", type = str, required = True)
    parser.add_argument("--processor", type = str, required = True)
    parser.add_argument("--policy", type = str, required = True)
    parser.add_argument("--wrapper", type = str, required = True)
    parser.add_argument("--ckpt_path", type = str, required = True)
    args = parser.parse_args()

    vision_feat_generator_config = VisionFeatGeneratorConfig(
        dataset_config = get_config(args.dataset, "dataset"),
        processor_config = get_config(args.processor, "processor"),
        policy_config = get_config(args.policy, "policy"),
        policy_wrapper_config = get_config(args.wrapper, "wrapper"),
        seed = 233,
        batch_size = 1,
        num_workers = 16,
        ckpt_path = args.ckpt_path,
        env_vars = {
            'NCCL_P2P_DISABLE': '1',
            'NCCL_IB_DISABLE': '1',
            'TOKENIZERS_PARALLELISM': '0'
        }
    )

    vision_feat_generator = VisionFeatGenerator(vision_feat_generator_config)
    vision_feat_generator.generate()
