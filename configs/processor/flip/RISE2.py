from data_infra.configs.base import *
from data_infra.configs.data import *
from data_infra.configs.data_source import *
from data_infra.configs.augmentation import *
from data_infra.configs.processor import *
from data_infra.configs.normalization import *
from utils.transforms.rotation import RotationType


processor_config = DataProcessorConfig(
    obs_data_configs = {
        "point_cloud": PointCloudConfig(
            src = [
                Depth2PointSourceConfig(
                    camera_name = "main",
                    frame = "world",
                    world_mask_config = CubePointMaskConfig(
                        min_bounds = [0, -0.7, -0.5],
                        max_bounds = [1.0, 0.7, 1.2]
                    )
                )
            ],
            frame = "camera/main",
            voxelization = True,
            voxel_size = 0.005,
            fixed_number = False,
            backend = "MinkowskiEngine",
            aug_groups = ["random_transform"],
            norm_config = EmptyNormalizationConfig()
        ),
        "image": ImageConfig(
            src = ImageSourceConfig(
                camera_name = "main",   
                aug_groups = ["color_jitter"]
            ),
            size = (252, 448),
            interp_mode = "bilinear",
            norm_config = GaussianNormalizationConfig(
                mean_value = [0.485, 0.456, 0.406],
                std_value = [0.229, 0.224, 0.225],
                channel_last = False
            )
        ),
        "image_coord": PointConfig(
            src = Depth2PointSourceConfig(
                camera_name = "main",
                frame = "camera/main",
                size = (252, 448),
                fill_hole = True,
                flatten = False,
                pooling_size = (18, 32)
            ),
            frame = "camera/main",
            aug_groups = ["random_transform"]
        )
    },
    action_data_configs = {
        "action_tcp": PoseConfig(
            src = PoseSourceConfig(
                input_key = "tcp_pose",
                frame = "robot_base/right",
                rotation_rep = RotationType.QUATERNION,
                convention = None
            ),
            frame = "camera/main",
            rotation_rep = RotationType.ROTATION_6D,
            convention = None,
            aug_groups = ["random_transform"],
            norm_config = LinearNormalizationConfig(
                min_value = [-1, -1, 0.2, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                max_value = [1.2, 1, 2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
        )
    },
    augmentation_configs = {
        "color_jitter": ColorJitterAugmentationConfig(
            prob = 0.2,
            brightness = 0.4,
            contrast = 0.2,
            saturation = 0.2,
            hue = 0.1
        ),
        "random_transform": RandomTransformAugmentationConfig(
            trans_x_range = (-0.2, 0.2),
            trans_y_range = (-0.2, 0.2),
            trans_z_range = (-0.2, 0.2),
            rot_x_range = (-30, 30),
            rot_y_range = (-30, 30),
            rot_z_range = (-30, 30)
        )
    },
    action_data_reverse_configs = {
        "action_tcp": PoseInferenceConfig(
            src_key = "action_tcp",
            src_frame = "camera/main",
            src_rotation_rep = RotationType.ROTATION_6D,
            src_convention = None,
            frame = "robot_base/right",
            rotation_rep = RotationType.QUATERNION,
            convention = None,
            norm_config = LinearNormalizationConfig(
                min_value = [-1, -1, 0.2, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                max_value = [1.2, 1, 2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
        ),
        "vision_feat": VectorInferenceConfig(
            src_key = "vision_feat",
            norm_config = EmptyNormalizationConfig()
        )
    },
    spatial_aug_frame = "camera/main"
)