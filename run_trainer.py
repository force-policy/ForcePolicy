"""
Run trainer.

Usage:
    python run_trainer.py \
           --trainer [trainer config] \
           --dataset [dataset config] \
           --processor [processor config] \
           --policy [policy config] \
           --wrapper [wrapper config] \
           --ckpt_dir [checkpoint directory] \
           (--wandb_project [wandb project]) \
           (--wandb_name [wandb name]) \
           (--resume_ckpt [resume checkpoint])
"""
import argparse
from configs import get_config
from trainer.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Run training.")
    parser.add_argument("--trainer", type = str, required = True, help = "trainer config")
    parser.add_argument("--dataset", type = str, required = True, help = "dataset config")
    parser.add_argument("--processor", type = str, required = True, help = "processor config")
    parser.add_argument("--policy", type = str, required = True, help = "policy config")
    parser.add_argument("--wrapper", type = str, required = True, help = "wrapper config")
    parser.add_argument("--ckpt_dir", type = str, required = True, help = "checkpoint directory")
    parser.add_argument("--wandb_project", type = str, default = None, help = "wandb project")
    parser.add_argument("--wandb_name", type = str, default = None, help = "wandb name")
    parser.add_argument("--resume_ckpt", type = str, default = None, help = "resume checkpoint")
    args = parser.parse_args()

    trainer_config = get_config(args.trainer, "trainer")
    trainer_config.dataset_config = get_config(args.dataset, "dataset")
    trainer_config.processor_config = get_config(args.processor, "processor")
    trainer_config.policy_config = get_config(args.policy, "policy")
    trainer_config.policy_wrapper_config = get_config(args.wrapper, "wrapper")
    trainer_config.ckpt_dir = args.ckpt_dir

    if args.resume_ckpt:
        trainer_config.resume_ckpt = args.resume_ckpt

    if args.wandb_project is not None and args.wandb_name is not None:
        wandb_configs = {
            "project": args.wandb_project,
            "name": args.wandb_name,
            "config": trainer_config.__dict__
        }
    else:
        wandb_configs = None

    trainer = Trainer(trainer_config, wandb_configs)
    trainer.train()