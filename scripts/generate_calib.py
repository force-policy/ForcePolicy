"""
Script to convert extrinsic (JSON) and intrinsic (NPY) calibration data into a unified calibration file.

Authors: Hongjie Fang.

Usage:
    python -m scripts.generate_calib
           --camera global:323522060212:intrinsic.npy:extrinsic.json
                    inhand:123456789876:intrinsic.npy
           --main_serial 323522060212
           --output_path calib/left.npy
"""

import json
import argparse
import numpy as np

from pathlib import Path

from utils.transformation import xyz_rot_to_mat


def load_extrinsics_from_json(json_path):
    # Example extrinsic.json:
    # {"is_global": true,
    #  "pose_in_link": [0.07783932332093665, 0.2078814260418823, 0.34723683952957585,
    #                   0.2273133855008057, -0.6785647482083789, 0.6637673415778982, -0.21746591345696367],
    #  "error": 0.0024148397685646483,
    #  "parent_link_name": "world"}
    # Only "pose_in_link" ([x, y, z, qw, qx, qy, qz]) is read here.
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Extrinsic JSON file not found: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    if "pose_in_link" not in data:
        raise KeyError("'pose_in_link' field not found in JSON file")
    return np.array(data["pose_in_link"])


def load_intrinsics_from_npy(npy_path):
    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(f"Intrinsic NPY file not found: {npy_path}")
    intrinsics = np.load(npy_path)
    return intrinsics


def load_camera_configs(camera_specs):
    cameras = {}
    for spec in camera_specs:
        parts = spec.split(":")
        if len(parts) < 3:
            raise ValueError(
                f"Invalid camera spec: {spec}. "
                "Expected format: 'global:serial:intrinsic.npy:extrinsic.json' or "
                "'inhand:serial:intrinsic.npy'"
            )
        assert parts[0] in ["global", "inhand"], "Camera type should be either global or inhand."
        
        cameras[parts[1]] = {
            "type": parts[0],
            "intrinsic": load_intrinsics_from_npy(parts[2]),
            "extrinsic": None if len(parts) == 3 else np.linalg.inv(xyz_rot_to_mat(load_extrinsics_from_json(parts[3]), rotation_rep = "quaternion"))
        }
    return cameras


def create_calibration(camera_data, main_serial, output_path):
    global_serials = []
    inhand_serials = []

    for serial, cam_data in camera_data.items():
        if cam_data["type"] == "global":
            global_serials.append(serial)
        else:
            inhand_serials.append(serial)
        if serial == main_serial:
            assert cam_data["type"] == "global", "Currently only support global camera as main camera."
            assert cam_data["extrinsic"] is not None, "Main camera (S/N: {}) extrinsic file not found.".format(main_serial)
    
    assert main_serial in (global_serials + inhand_serials), "Main camera (S/N: {}) not found in camera data.".format(main_serial)

    calib = {
        "main_camera_serial": main_serial,
        "camera_serials_global": global_serials,
        "camera_serials_inhand": inhand_serials,
        "camera_serials": global_serials + inhand_serials,
        "camera_data": camera_data
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    
    np.save(output_path, calib, allow_pickle = True)
    print(f"Calibration saved to: {output_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Convert extrinsic (JSON) and intrinsic (NPY) calibration data into a unified calibration file")
    parser.add_argument("--camera", nargs = "+", required = True, help = "Camera specifications in format '[global/inhand]:serial:intrinsic.npy[:extrinsic.json]'.")
    parser.add_argument("--main_serial", required = True, help = "Main camera serial.")
    parser.add_argument("--output_path", required = True, help = "Output calibration file path")
    args = parser.parse_args()
    create_calibration(
        camera_data = load_camera_configs(args.camera),
        main_serial = args.main_serial,
        output_path = args.output_path
    )
