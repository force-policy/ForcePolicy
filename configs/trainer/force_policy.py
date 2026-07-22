from trainer.configs import TrainerConfig, OptimizerConfig, SchedulerConfig


trainer_config = TrainerConfig(
    optimizer_config = OptimizerConfig(
        type = "AdamW",
        kwargs = {
            "weight_decay": 1e-6,
            "betas": (0.95, 0.999)
        }
    ),
    scheduler_config = SchedulerConfig(
        type = "cosine",
        kwargs = {
            "num_warmup_steps": 0
        }
    ),
    lr = 1e-4,
    seed = 233,
    batch_size = 240,
    num_workers = 16,
    num_steps = 60000,
    save_steps = 2500,
    env_vars = {
        'NCCL_P2P_DISABLE': '1',
        'NCCL_IB_DISABLE': '1',
        'TOKENIZERS_PARALLELISM': '0'
    }
)
