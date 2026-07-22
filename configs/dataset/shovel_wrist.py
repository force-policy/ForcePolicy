from data_infra.configs.dataset import *


dataset_config = DatasetConfig(
    type = "vision",
    data_path = "data/shovel",
    robot_poses = {
        "right": np.eye(4, dtype = np.float32),
    },
    obs = ObservationConfig(
        vision = VisionConfig(
            main_cameras = [],
            aux_cameras = {"wrist": "104122064489"},
            colors = ["wrist"],
            depths = [],
            extrinsics = [],
            intrinsics = []
        ),
        lowdim = {
            "vision_feat": LowdimConfig(
                lowdim_name = "vision_feat_104122060902",
                field = "vision_feat",
                length = 1,
                freq = 10,
                data_freq = 10,
                direction = -1,
                remove_first = False
            ),
            "tcp_pose": LowdimConfig(
                lowdim_name = "lowdim_filled",
                field = "tcp_pose_062046",
                length = 100,
                freq = 50,
                data_freq = 1000,
                direction = -1,
                remove_first = False
            ),
            "force_torque": LowdimConfig(
                lowdim_name = "lowdim_filled",
                field = "force_torque_062046",
                length = 100,
                freq = 50,
                data_freq = 1000,
                direction = -1,
                remove_first = False
            ),
            "aux_tcp_pose": LowdimConfig(
                lowdim_name = "lowdim_filled",
                field = "tcp_pose_062046",
                length = 1,
                freq = 50,
                data_freq = 1000,
                direction = -1,
                remove_first = False
            )
        }
    ),
    action = ActionConfig(
        lowdim = {
            "action_tcp_pose": LowdimConfig(
                lowdim_name = "lowdim_filled",
                field = "tcp_pose_062046",
                length = 50,
                freq = 50,
                data_freq = 1000,
                direction = 1,
                remove_first = True
            ),
            "action_force_torque": LowdimConfig(
                lowdim_name = "lowdim_labeled",
                field = "ref_force_frame",
                length = 50,
                freq = 50,
                data_freq = 1000,
                direction = 1,
                remove_first = True
            ), 
            "action_frame": LowdimConfig(
                lowdim_name = "lowdim_labeled",
                field = "frame_pose",
                length = 50,
                freq = 50,
                data_freq = 1000,
                direction = 1,
                remove_first = True
            ),
            "action_frame_mask": LowdimConfig(
                lowdim_name = "lowdim_labeled",
                field = "mask_frame",
                length = 50,
                freq = 50,
                data_freq = 1000,
                direction = 1,
                remove_first = True
            )
        }
    ),
    repeat_dataset = 10
)
