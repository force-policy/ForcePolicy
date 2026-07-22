"""
Trainer.
"""
from typing import Dict, Any, Optional

import os
import time
import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from trainer.utils import sync_loss
from trainer.configs import TrainerConfig

from logger import logger
from policy import get_policy, PolicyWrapper
from data_infra.processor import DataProcessor
from data_infra.dataset import get_dataset
from utils.common import set_seed, to_device, num_params, num_trainable_params


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        wandb_config: Dict[str, Any] = None
    ) -> None:
        """ Initialization. """
        self.config = config
        self.enable_wandb = wandb_config is not None

        self._set_env_vars()
        self._setup_dist()
        set_seed(config.seed)

        torch.cuda.set_device(self.LOCAL_RANK)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._build_dataset()
        self._build_model()
        self._build_optimizer()

        if self.RANK == 0 and self.enable_wandb:
            wandb.init(**wandb_config)
        
        if self.RANK == 0:
            os.makedirs(config.ckpt_dir, exist_ok = True)


    def _set_env_vars(self) -> None:
        """ Set environment variables. """
        for key, value in self.config.env_vars.items():
            os.environ[key] = value


    def _setup_dist(self) -> None:
        """ Setup distributed training. """
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
        self.RANK = int(os.environ.get('RANK', 0))
        self.LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
        
        if not dist.is_initialized():
            dist.init_process_group(
                backend = 'nccl', 
                init_method = 'env://', 
                world_size = self.WORLD_SIZE, 
                rank = self.RANK
            )
        
        self.batch_size_per_device = self.config.batch_size // self.WORLD_SIZE


    def _build_dataset(self) -> None:
        """ Build dataset. """
        if self.RANK == 0:
            logger.info("Loading dataset ...")
        
        self.dataset = get_dataset(self.config.dataset_config)
        self.sampler = DistributedSampler(
            self.dataset, 
            num_replicas = self.WORLD_SIZE, 
            rank = self.RANK, 
            shuffle = True
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size = self.batch_size_per_device,
            num_workers = self.config.num_workers,
            sampler = self.sampler,
            drop_last = True
        )
        self.processor = DataProcessor(self.config.processor_config)


    def _build_model(self) -> None:
        """ Build policy model. """
        if self.RANK == 0:
            logger.info("Loading policy ...")
        
        self.policy = get_policy(self.config.policy_config).to(self.device)
        
        if self.RANK == 0:
            n_trainable_parameters = num_trainable_params(self.policy)
            n_parameters = num_params(self.policy)
            logger.info(f"Number of parameters: {n_parameters / 1e6:.2f}M")
            logger.info(f"Number of trainable parameters: {n_trainable_parameters / 1e6:.2f}M")

        self.policy = nn.parallel.DistributedDataParallel(
            self.policy, 
            device_ids = [self.LOCAL_RANK], 
            output_device = self.LOCAL_RANK, 
            find_unused_parameters = True
        )

        self.policy_wrapper = PolicyWrapper(
            self.policy.module, 
            self.config.policy_wrapper_config
        )


    def _build_optimizer(self) -> None:
        """ Build optimizer and scheduler. """
        if self.RANK == 0:
            logger.info("Loading optimizer and scheduler ...")
        
        # optimizer
        opt_cfg = self.config.optimizer_config
        if hasattr(torch.optim, opt_cfg.type):
            optimizer_cls = getattr(torch.optim, opt_cfg.type)
            self.optimizer = optimizer_cls(
                self.policy.parameters(), 
                lr = self.config.lr, 
                ** opt_cfg.kwargs
            )
        else:
            logger.error(f"Optimizer {opt_cfg.type} not found in torch.optim")
            raise NotImplementedError

        # scheduler
        sched_cfg = self.config.scheduler_config
        self.lr_scheduler = get_scheduler(
            name = sched_cfg.type,
            optimizer = self.optimizer,
            num_training_steps = self.config.num_steps,
            **sched_cfg.kwargs
        )

    def _load_checkpoint(self) -> int:
        start_step = 0
        resume_ckpt = self.config.resume_ckpt
        
        if resume_ckpt is None:
            if os.path.exists(os.path.join(self.config.ckpt_dir, "policy_last.ckpt")):
                resume_ckpt = os.path.join(self.config.ckpt_dir, "policy_last.ckpt")
        
        if resume_ckpt and os.path.exists(resume_ckpt):
            if self.RANK == 0:
                logger.info(f"Resuming from checkpoint: {resume_ckpt}")
            
            checkpoint = torch.load(resume_ckpt, map_location = self.device)
            self.policy.module.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
            if "step" in checkpoint:
                start_step = checkpoint["step"]
            
            if self.RANK == 0:
                logger.info(f"Resumed from step {start_step}")
                
        return start_step


    def _save_checkpoint(self, step: int, is_last: bool = False) -> str:
        """ Save checkpoint. """
        if self.RANK != 0:
            return ""
        
        filename = "policy_last.ckpt" if is_last else f"policy_step_{step}_seed_{self.config.seed}.ckpt"
        path = os.path.join(self.config.ckpt_dir, filename)
        
        save_dict = {
            "state_dict": self.policy.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "step": step,
            "config": self.config
        }
        
        torch.save(save_dict, path)
        return f"Checkpoint saved at step {step}.\n" if not is_last else f"Final checkpoint saved at step {step}.\n"


    def train(self) -> None:
        """ Train. """
        train_history = []
        loss_history = {}
        steps_per_epoch = len(self.dataloader)
        cur_step = self._load_checkpoint()
        start_epoch = cur_step // steps_per_epoch
        num_epochs = int(np.ceil(self.config.num_steps / len(self.dataloader)))

        self.policy.train()
        for epoch in range(start_epoch, num_epochs):
            if self.RANK == 0:
                logger.info(f"Training Epoch {epoch}:")
            time.sleep(0.5)
            
            self.sampler.set_epoch(epoch)
            
            pbar = tqdm(self.dataloader) if self.RANK == 0 else self.dataloader
            avg_loss = 0.0
            epoch_losses: Dict[str, list] = {} 
            logs = ""

            for data in pbar:
                # Training step
                loss_dict = self._train_step(data)
                loss = loss_dict["loss"]
                
                avg_loss += loss.item()
                
                # Track individual losses
                for loss_name, loss_value in loss_dict.items():
                    if loss_name not in epoch_losses:
                        epoch_losses[loss_name] = []
                    val = loss_value.item() if torch.is_tensor(loss_value) else loss_value
                    epoch_losses[loss_name].append(val)

                # Save checkpoint
                if (cur_step + 1) % self.config.save_steps == 0:
                    logs += self._save_checkpoint(cur_step + 1)
                
                # Update progress bar
                if self.RANK == 0:
                    self._update_pbar(pbar, cur_step, loss_dict)
                
                cur_step += 1

            # End of epoch logging
            self._log_epoch(
                epoch, cur_step, avg_loss, steps_per_epoch, 
                epoch_losses, train_history, loss_history, logs
            )

        # Save final checkpoint
        self._save_checkpoint(cur_step, is_last = True)


    def _train_step(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """ Train (a forward step). """
        data = to_device(data, device = self.device)
        
        self.optimizer.zero_grad()
        
        obs_dict, action_dict = self.processor(data["obs"], data["action"], enable_aug = True)
        loss_dict = self.policy_wrapper(obs_dict, action_dict, batch_size = self.batch_size_per_device)
        
        loss = loss_dict["loss"]
        loss.backward()
        
        self.optimizer.step()
        self.lr_scheduler.step()
        
        return loss_dict


    def _update_pbar(self, pbar, step: int, loss_dict: Dict[str, Any]) -> None:
        """ Update progress bar. """
        postfix_dict = {
            'step': step + 1,
            'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
        }
        for key, val in loss_dict.items():
            v = val.item() if torch.is_tensor(val) else val
            postfix_dict[key] = f'{v:.2e}'
        pbar.set_postfix(postfix_dict)


    def _log_epoch(
        self, 
        epoch: int, 
        step: int, 
        total_loss: float, 
        steps_per_epoch: int,
        epoch_losses: Dict[str, list],
        train_history: list,
        loss_history: Dict[str, list],
        logs: str
    ) -> None:
        """ Logging. """
        # Calculate average losses
        avg_loss = total_loss / steps_per_epoch
        avg_loss = sync_loss(avg_loss, self.device).item()
        train_history.append(avg_loss)
        
        epoch_avg_losses = {}
        for loss_name, loss_values in epoch_losses.items():
            avg_val = sum(loss_values) / len(loss_values)
            avg_val = sync_loss(avg_val, self.device).item()
            epoch_avg_losses[loss_name] = avg_val
            
            if loss_name not in loss_history:
                loss_history[loss_name] = []
            loss_history[loss_name].append(avg_val)
        
        if self.RANK == 0:
            logs += f"# Steps: {step}. Epoch {epoch} Loss Summary:\n"
            logs += f"  Total Loss: {avg_loss:.6f}\n"
            for loss_name, val in epoch_avg_losses.items():
                logs += f"  {loss_name}: {val:.6f}\n"
            logger.info(logs)

            if self.enable_wandb:
                wandb_log_dict = {"train/step": step} 
                for loss_name, val in epoch_avg_losses.items():
                    wandb_log_dict[f"train/{loss_name}"] = val
                wandb.log(wandb_log_dict)
