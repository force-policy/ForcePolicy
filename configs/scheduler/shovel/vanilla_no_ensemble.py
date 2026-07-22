
from runner.configs.scheduler import VanillaSchedulerConfig, VisualizationConfig, SyncActionConfig
from runner.configs.ensemble_buffer import EnsembleBufferConfig
from data_infra.configs.point_mask import CubePointMaskConfig
from utils.transforms.rotation import RotationType


scheduler_config = VanillaSchedulerConfig(
    seed = 233,
    frequency = 10,
    max_steps = 3000,
    num_inference_step = 20,
    ensemble_buffer_config = EnsembleBufferConfig(
        params_dict = {
            "robot/right/tcp_pose": {
                "mode": "new"
            }
        }
    ),
    action_params = {
        "robot/right/tcp_pose": {
            "max_vel": 0.5,
            "max_acc": 1.0,
            "max_angular_vel": 1.0,
            "max_angular_acc": 2.0
        }
    },
    sync_frequency = 10,
    sync_horizon = 50,    
    sync_action_config = {
        "robot/right/tcp_pose": SyncActionConfig(
            source_frequency = 10,
            source_length = 50,
            interpolation = "pose_linear",
            rotation_rep = RotationType.QUATERNION
        )
    },
    # visualization = VisualizationConfig(
    #     mode = "3d",
    #     camera_names = ["main"],
    #     action_pose_keys = ["robot/right/tcp_pose"],
    #     world_mask_config = CubePointMaskConfig(
    #         min_bounds = [0, -0.7, -0.5],
    #         max_bounds = [1.0, 0.7, 1.2]
    #     )
    # ),    
    # visualization = VisualizationConfig(
    #     mode = "2d",
    #     camera_names = ["main"],
    #     action_pose_keys = ["robot/right/tcp_pose"]
    # )
)