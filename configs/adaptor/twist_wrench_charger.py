from adaptor.configs import *

# Adaptive ("ours") interaction-frame recovery for the "Plug in EV Charger" (charger) task.
# `specify` selects the reconstruction formulation:
#   - 'auto'   : kinematic threshold switching (offline default).
#   - 'twist'  : dissipative residual dominates (orthogonalize wrench against twist).
#   - 'wrench' : structural residual dominates (orthogonalize twist against wrench).
# Use scripts/classify_power_source.py (Gemini 3 Pro) to determine `specify` per task.

adaptor_config = DataAdaptorConfig(
    patch_size = 10,
    frame_identifier_config = TwistWrenchFrameIdentifierConfig(
        weight_angular = 0,
        weight_torque = 0,
        thres_parallel = 0.98,
        thres_lin_vel = 0.02,
        thres_ang_vel = 0.25,
        specify = 'auto'
    ),
    frame_labeler_config = AdvancedFrameLabelerConfig(
        thres_ang_vel = 0.25,
        thres_lin_vel = 0.02,
        thres_torque = 2.0,
        thres_force = 10.0,
        thres_is_parallel = 0.8
    ),
    calc_twist_from_pose = False
)
