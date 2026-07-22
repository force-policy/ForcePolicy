from adaptor.configs import *


adaptor_config = DataAdaptorConfig(
    patch_size = 1,
    frame_identifier_config = ForceOnlyFrameIdentifierConfig(),
    frame_labeler_config = VanillaFrameLabelerConfig(
        thres_force = 3.0
    ),
    calc_twist_from_pose = False
)
