#!/usr/bin/env python3
"""
Save RealSense camera intrinsic matrix to .npy file.

Usage:
    python scripts/save_intrinsic.py --serial 128422271347 --output calib/camera_intrinsic.npy
"""

import argparse
import numpy as np
from pathlib import Path
from easyrobot.camera.realsense import RealSenseRGBDCamera


def main():
    parser = argparse.ArgumentParser(description="Save RealSense camera intrinsic matrix")
    parser.add_argument(
        "--serial",
        type=str,
        required=True,
        help="Camera serial number"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npy file path"
    )
    
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Initializing camera with serial: {args.serial}")
    
    camera = RealSenseRGBDCamera(args.serial, streaming_freq=15, shm_freq=15, shm_name="camera")
    print("Getting intrinsic matrix...")
    intrinsic_mat = camera.get_intrinsic(return_mat=True)
    
    print("\nIntrinsic Matrix (3x3):")
    print(intrinsic_mat)
    print(f"\nShape: {intrinsic_mat.shape}")
    print(f"Dtype: {intrinsic_mat.dtype}")
    
    np.save(args.output, intrinsic_mat)
    print(f"\nSaved intrinsic matrix to: {args.output}")
    
    loaded = np.load(args.output)
    if np.allclose(loaded, intrinsic_mat):
        print("✓ Verification successful: saved file matches original matrix")
    else:
        print("✗ Warning: saved file does not match original matrix!")
    


if __name__ == "__main__":
    main()

