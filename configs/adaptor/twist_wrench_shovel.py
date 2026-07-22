from adaptor.configs import *

# Adaptive ("ours") interaction-frame recovery for the "Scrape off Sticker" (shovel) task.
# `specify` selects the reconstruction formulation:
#   - 'auto'   : kinematic threshold switching (offline default).
#   - 'twist'  : dissipative residual dominates (orthogonalize wrench against twist).
#   - 'wrench' : structural residual dominates (orthogonalize twist against wrench).
# Use scripts/classify_power_source.py (Gemini 3 Pro) to determine `specify` per task.

adaptor_config = DataAdaptorConfig(
    patch_size = 50,
    frame_identifier_config = TwistWrenchFrameIdentifierConfig(
        weight_angular = 0.05,
        weight_torque = 3.0,
        thres_parallel = 0.98,
        thres_force = 3.0,
        thres_torque = 0.5,
        thres_lin_vel = 0.01,
        thres_ang_vel = 0.3,
        specify = 'auto'
    ),
    frame_labeler_config = AdvancedFrameLabelerConfig(
        thres_ang_vel = 0.3,
        thres_lin_vel = 0.01,
        thres_torque = 2.0,
        thres_force = 10.0,
        thres_is_parallel = 0.8
    ),
    calc_twist_from_pose = False
)
