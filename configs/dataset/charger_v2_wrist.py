from data_infra.configs.dataset import *


dataset_config = DatasetConfig(
    type = "vision",
    data_path = "data/charger_v2",
    robot_poses = {
        "main": np.eye(4, dtype = np.float32),
    },
    obs = ObservationConfig(
        vision = VisionConfig(
            main_cameras = [],
            aux_cameras = {"wrist": "104122062823"},
            colors = ["wrist"],
            depths = [],
            extrinsics = [],
            intrinsics = []
        ),
        lowdim = {
            "vision_feat": LowdimConfig(
                lowdim_name = "vision_feat_035622060973",
                field = "vision_feat",
                length = 1,
                freq = 10,
                data_freq = 10,
                direction = -1,
                remove_first = False
            ),
            "tcp_pose": LowdimConfig(
                lowdim_name = "lowdim",
                field = "tcp_pose_063047",
                length = 100,
                freq = 50,
                data_freq = 1000,
                direction = -1,
                remove_first = False
            ),
            "force_torque": LowdimConfig(
                lowdim_name = "lowdim",
                field = "force_torque_063047",
                length = 100,
                freq = 50,
                data_freq = 1000,
                direction = -1,
                remove_first = False
            ),
            "aux_tcp_pose": LowdimConfig(
                lowdim_name = "lowdim",
                field = "tcp_pose_063047",
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
                lowdim_name = "lowdim",
                field = "tcp_pose_063047",
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
