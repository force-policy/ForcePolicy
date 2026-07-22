#!/usr/bin/env python3
"""
Batch-copy intrinsic.npy / extrinsic.json from a calib directory into every scene's cam_<serial> folder.

Usage:
    python scripts/sync_calib_to_scenes.py \\
        --calib_dir calib/104122060902_leftarm \\
        --data_root /data/shirun/data/h5data/peg_in_hole_e3_v2/train \\
        --camera_serial 104122060902

    python scripts/sync_calib_to_scenes.py --dry_run  # only print the actions to be performed
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Copy calibration files into every scene's cam_* folder")
    parser.add_argument(
        "--calib_dir",
        type=Path,
        default=repo_root / "calib" / "104122060902_leftarm",
        help="Directory containing intrinsic.npy and extrinsic.json",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/data/shirun/data/h5data/peg_in_hole_e3_v2/train"),
        help="Root folder containing scene_* directories",
    )
    parser.add_argument(
        "--camera_serial",
        type=str,
        default="104122060902",
        help="Camera serial; files go to cam_<serial>/ under each scene",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print actions only, do not copy")
    args = parser.parse_args()

    calib_dir = args.calib_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    cam_name = f"cam_{args.camera_serial}"

    intrinsic_src = calib_dir / "intrinsic.npy"
    extrinsic_src = calib_dir / "extrinsic.json"
    for name, p in [("intrinsic.npy", intrinsic_src), ("extrinsic.json", extrinsic_src)]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing calibration file: {p}")

    scenes = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("scene_"))
    if not scenes:
        raise FileNotFoundError(f"No scene_* directories under {data_root}")

    for scene in scenes:
        dest_dir = scene / cam_name
        if not dest_dir.is_dir():
            raise FileNotFoundError(f"Expected camera dir missing: {dest_dir}")
        for src in (intrinsic_src, extrinsic_src):
            dest = dest_dir / src.name
            if args.dry_run:
                print(f"[dry-run] {src} -> {dest}")
            else:
                shutil.copy2(src, dest)
                print(f"copied {src.name} -> {dest}")

    print(f"Done: {len(scenes)} scenes under {data_root}")


if __name__ == "__main__":
    main()
