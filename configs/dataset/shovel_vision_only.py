from data_infra.configs.dataset import *


dataset_config = DatasetConfig(
    type = "vision",
    data_path = "data/shovel",
    robot_poses = {
        "right": np.eye(4, dtype = np.float32),
    },
    obs = ObservationConfig(
        vision = VisionConfig(
            main_cameras = ["104122060902"],
            aux_cameras = {},
            colors = ["main"],
            depths = ["main"],
            extrinsics = ["main"],
            intrinsics = ["main"]
        ),
        lowdim = {}
    ),
    action = ActionConfig(
        lowdim = {
            "tcp_pose": LowdimConfig(
                lowdim_name = "lowdim_filled",
                field = "tcp_pose_062046",
                length = 50,
                freq = 10,
                data_freq = 1000,
                direction = 1,
                remove_first = False
            )
        }
    ),
    repeat_dataset = 10
)
