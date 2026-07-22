from data_infra.configs.base import *
from data_infra.configs.data import *
from data_infra.configs.data_source import *
from data_infra.configs.augmentation import *
from data_infra.configs.processor import *
from data_infra.configs.normalization import *
from utils.transforms.rotation import RotationType


processor_config = DataProcessorConfig(
    obs_data_configs = {
        "image_wrist": ImageConfig(
            src = ImageSourceConfig(
                camera_name = "wrist",
                aug_groups = ["color_jitter"]
            ),
            size = (84, 84),
            interp_mode = "bilinear",
            norm_config = GaussianNormalizationConfig(
                mean_value = [0.485, 0.456, 0.406],
                std_value = [0.229, 0.224, 0.225],
                channel_last = False
            )
        ),
        "vision_feat": VectorConfig(
            src = VectorSourceConfig(
                input_key = "vision_feat"
            )
        ),
        "proprio_tcp": PoseConfig(
            src = PoseSourceConfig(
                input_key = "tcp_pose",
                frame = "robot_base/right",
                rotation_rep = RotationType.QUATERNION,
                convention = None
            ),
            frame = "robot_tcp/right/aux_tcp_pose/quaternion",
            rotation_rep = RotationType.ROTATION_6D,
            convention = None,
            aug_groups = [],
            norm_config = LinearNormalizationConfig(
                min_value = [-0.3, -0.3, -0.3, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                max_value = [0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
        ),
        "force_torque": WrenchConfig(
            src = WrenchSourceConfig(
                input_key = "force_torque",
                frame = "robot_tcp/right/tcp_pose/quaternion"
            ),
            frame = "robot_tcp/right/tcp_pose/quaternion",
            rotation_only = True,
            norm_config = LinearNormalizationConfig(
                min_value = [-50.0, -50.0, -50.0, -5.0, -5.0, -5.0], 
                max_value = [50.0, 50.0, 50.0, 5.0, 5.0, 5.0]
            ),
            aug_groups = []
        )
    },
    action_data_configs = {
        "action_tcp": PoseConfig(
            src = PoseSourceConfig(
                input_key = "action_tcp_pose",
                frame = "robot_base/right",
                rotation_rep = RotationType.QUATERNION,
                convention = None
            ),
            frame = "robot_tcp/right/aux_tcp_pose/quaternion",
            rotation_rep = RotationType.ROTATION_6D,
            convention = None,
            aug_groups = [],
            norm_config = LinearNormalizationConfig(
                min_value = [-0.3, -0.3, -0.3, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                max_value = [0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
        ),
        "action_force_torque": WrenchConfig(
            src = WrenchSourceConfig(
                input_key = "action_force_torque",
                frame = "interaction_frame"
            ),
            frame = "interaction_frame",
            rotation_only = True,
            norm_config = LinearNormalizationConfig(
                min_value = [-50.0, -50.0, -50.0, -5.0, -5.0, -5.0], 
                max_value = [50.0, 50.0, 50.0, 5.0, 5.0, 5.0]
            ),
            aug_groups = []
        ),
        "action_frame": PoseConfig(
            src = PoseSourceConfig(
                input_key = "action_frame",
                frame = "robot_tcp/right/aux_tcp_pose/quaternion",
                rotation_rep = RotationType.MATRIX,
                convention = None
            ),
            frame = "robot_tcp/right/aux_tcp_pose/quaternion",
            rotation_rep = RotationType.ROTATION_6D,
            convention = None,
            aug_groups = [],
            norm_config = LinearNormalizationConfig(
                min_value = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                max_value = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
        ),
        "action_frame_mask": VectorConfig(
            src = VectorSourceConfig(
                input_key = "action_frame_mask"
            ),
            # norm_config = LinearNormalizationConfig(
            #     min_value = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #     max_value = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            # )
        ),
    },
    augmentation_configs = {
        "color_jitter": ColorJitterAugmentationConfig(
            prob = 0.2,
            brightness = 0.4,
            contrast = 0.2,
            saturation = 0.2,
            hue = 0.1
        )
    },
    action_data_reverse_configs = {
        "action_tcp": PoseInferenceConfig(
            src_key = "action_tcp",
            src_frame = "robot_tcp/right/aux_tcp_pose/quaternion",
            src_rotation_rep = RotationType.ROTATION_6D,
            src_convention = None,
            frame = "robot_base/right",
            rotation_rep = RotationType.QUATERNION,
            convention = None,
            norm_config = LinearNormalizationConfig(
                min_value = [-0.3, -0.3, -0.3, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                max_value = [0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
        ),
        "action_force_torque": WrenchInferenceConfig(
            src_key = "action_force_torque",
            src_frame = "interaction_frame",
            frame = "interaction_frame",
            rotation_only = True,
            norm_config = LinearNormalizationConfig(
                min_value = [-50.0, -50.0, -50.0, -5.0, -5.0, -5.0], 
                max_value = [50.0, 50.0, 50.0, 5.0, 5.0, 5.0]
            )
        ),
        "action_frame": PoseInferenceConfig(
            src_key = "action_frame",
            src_frame = "robot_tcp/right/aux_tcp_pose/quaternion",
            src_rotation_rep = RotationType.ROTATION_6D,
            src_convention = None,
            frame = "robot_tcp/right/aux_tcp_pose/quaternion",
            rotation_rep = RotationType.QUATERNION,
            convention = None,
            norm_config = LinearNormalizationConfig(
                min_value = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                max_value = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
        ),
        "action_frame_mask": VectorInferenceConfig(
            src_key = "action_frame_mask"
        ),
        "switch_signal": VectorInferenceConfig(
            src_key = "switch_signal",
            norm_config = EmptyNormalizationConfig()
        )
    }
)