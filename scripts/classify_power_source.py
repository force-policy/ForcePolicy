"""
Per-patch power-source classification with Gemini 3 Pro for Interaction Frame (IF) recovery.

Following the Force Policy paper (Sec. IV-A, "Interaction Frame Identification"), the interaction
frame is recovered on a per-patch basis: for each Delta-t patch of a demonstration starting at
timestep t, the paper prompts Gemini 3 Pro with the patch's *initial visual context* I_t (plus a
task reference image and the task description) to classify the dominant power source, because the
raw twist/wrench signals alone are ambiguous (identical power-exchange patterns can arise from
friction or from stiffness). The classification maps onto the `specify` mode of the twist-wrench
IF identifier:

    dissipative_residual -> specify = "twist"   (orthogonalize wrench against twist, Eq. 4)
    structural_residual  -> specify = "wrench"  (orthogonalize twist against wrench, Eq. 3)

This script reproduces that per-patch procedure. For every scene it:
  1. reads the low-dim timestamps and splits the trajectory into patches of `--patch_size`
     samples (must match the task's adaptor config `patch_size`);
  2. maps each patch's start timestamp to the nearest RGB frame (the patch's "current execution
     image"), and queries Gemini with (task reference image, current frame, task description);
  3. writes a PER-PATCH `specify` sequence to `<scene>/lowdim/<output>` (JSON), aligned 1:1 with
     the adaptor's patches.

Feed the result to the adaptor via:
    python run_adaptor.py --adaptor twist_wrench_<task> ... --specify_seq_name <output>

If you do NOT run this script, the adaptor simply uses the single global `specify` in the task's
adaptor config (default `specify='auto'`, a kinematic-threshold approximation). This Gemini step
is entirely optional.

Usage:
    export GEMINI_API_KEY=<your_key>
    python scripts/classify_power_source.py \
           --task_path data/flip/train \
           --task_description "flip the box: a constrained rotation of the orange box against the vertical wall boundary to transition it from a horizontal to a vertical pose, maintaining contact with the wall to overcome interface friction" \
           --patch_size 10 \
           --reference_image path/to/task_reference.png \
           --camera_serial 104122060902 \
           --lowdim_name lowdim_filled.h5 \
           --output power_source_specify.json
"""
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from logger import logger

# Map the high-level power-source semantics to the IF identifier `specify` mode.
# Both the verbose labels used by the prompt and their short forms are accepted.
POWER_SOURCE_TO_SPECIFY: Dict[str, str] = {
    "dissipative_residual": "twist",
    "structural_residual": "wrench",
    "dissipative": "twist",
    "structural": "wrench",
}

SYSTEM_PROMPT: str = """You are an expert physics engine and robotic manipulation analyst. Your task is to analyze a Task Reference Image (overall phase scope), a Real-time Image Frame (current execution state), and a Task Description to classify the underlying physical nature of the interaction based on the Principle of Virtual Work and Power Exchange Theory.

Background Knowledge: Power Exchange Theory
In contact-rich manipulation, the interaction is modeled by the dominance of specific power residuals. To classify the interaction, evaluate whether the robot is primarily performing Work to overcome non-conservative forces or Regulating Force against potential energy boundaries:

Structural Residual (structural_residual):
Physics: Dominated by environmental stiffness ($K_{env}$) or static geometric constraints.
Mechanism: The Intended Wrench ($W^*$) is the primary causal driver. The robot acts as a force source against a rigid constraint plane.
Detection Cues: Phases focused on normal force regulation, establishing initial contact, surface pressing, or maintaining a static pose against a hard boundary where the intended velocity is effectively zero.

Dissipative Residual (dissipative_residual):
Physics: Dominated by non-conservative energy losses, such as surface-to-surface friction, viscous damping, or path-dependent resistance.
Mechanism: The Intended Twist ($\\xi^*$) is the primary causal driver. The robot performs active work to move the object through a constrained manifold.
Detection Cues: Any phase involving active state-transitioning motion, surface-constrained sliding, complex multi-axis insertions, or dynamic reorientation of an object where work is continuously performed to overcome frictional or mechanical resistance.

Output Format (Strict JSON):
{
  "identified_phase": "The name of the phase from the reference image labels",
  "classification": "dissipative_residual | structural_residual"
}
"""

USER_TEMPLATE: str = (
    "Inputs:\n"
    "Task Description (T): {{{task_description}}}\n"
    "Task Reference Image: [Image 1]\n"
    "Current Execution Image: [Image 2]"
)


def list_color_frames(scene_dir: str, camera_serial: Optional[str]) -> List[Tuple[int, str]]:
    """Return sorted [(timestamp_ms, path), ...] of RGB frames for a scene, or [] if none.

    Frame files are named `<timestamp_ms>.png` under `cam_<serial>/color/`.
    """
    candidate_cam_dirs: List[str] = []
    if camera_serial is not None:
        candidate_cam_dirs.append(os.path.join(scene_dir, f"cam_{camera_serial}"))
    for entry in sorted(os.listdir(scene_dir)):
        if entry.startswith("cam_"):
            cam_dir = os.path.join(scene_dir, entry)
            if cam_dir not in candidate_cam_dirs:
                candidate_cam_dirs.append(cam_dir)

    for cam_dir in candidate_cam_dirs:
        color_dir = os.path.join(cam_dir, "color")
        if not os.path.isdir(color_dir):
            continue
        frames: List[Tuple[int, str]] = []
        for f in os.listdir(color_dir):
            if not f.endswith(".png"):
                continue
            stem = f[:-4]
            try:
                ts = int(stem)
            except ValueError:
                continue
            frames.append((ts, os.path.join(color_dir, f)))
        if frames:
            frames.sort(key=lambda x: x[0])
            return frames
    return []


def nearest_frame_path(frames: List[Tuple[int, str]], target_ts: float) -> Optional[str]:
    """Return the path of the frame whose timestamp is nearest to `target_ts`."""
    if not frames:
        return None
    ts_array = np.asarray([ts for ts, _ in frames], dtype=np.float64)
    idx = int(np.argmin(np.abs(ts_array - float(target_ts))))
    return frames[idx][1]


def read_num_samples_and_timestamps(lowdim_path: str, timestamp_key: str) -> np.ndarray:
    """Read the timestamp array from a low-dim HDF5 file."""
    with h5py.File(lowdim_path, "r") as f:
        if timestamp_key not in f:
            raise KeyError(
                f"Timestamp key '{timestamp_key}' not found in {lowdim_path}. "
                f"Available keys: {list(f.keys())}"
            )
        return f[timestamp_key][:]


def classify_image_pair(
    client: "object",
    model: str,
    reference_image_path: str,
    current_image_path: str,
    task_description: str,
) -> Dict[str, str]:
    """Query Gemini with (reference image, current frame, task description); parse the result."""
    from google.genai import types

    with open(reference_image_path, "rb") as f:
        ref_bytes = f.read()
    with open(current_image_path, "rb") as f:
        cur_bytes = f.read()

    user_text = USER_TEMPLATE.format(task_description=task_description)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=ref_bytes, mime_type="image/png"),   # [Image 1] reference
            types.Part.from_bytes(data=cur_bytes, mime_type="image/png"),   # [Image 2] current
            user_text,
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
        ),
    )

    result = json.loads(response.text)
    classification = str(result.get("classification", "")).strip().lower()
    if classification not in POWER_SOURCE_TO_SPECIFY:
        raise ValueError(f"Unexpected classification from model: {result}")
    return {
        "identified_phase": str(result.get("identified_phase", "")),
        "classification": classification,
        "specify": POWER_SOURCE_TO_SPECIFY[classification],
    }


def classify_scene(
    client: "object",
    model: str,
    scene_dir: str,
    lowdim_name: str,
    timestamp_key: str,
    patch_size: int,
    reference_image_path: Optional[str],
    camera_serial: Optional[str],
    task_description: str,
) -> Optional[Dict[str, object]]:
    """Produce a per-patch specify sequence for one scene (aligned to the adaptor's patches)."""
    lowdim_path = os.path.join(scene_dir, "lowdim", lowdim_name)
    if not os.path.exists(lowdim_path):
        logger.warning(f"{os.path.basename(scene_dir)}: {lowdim_name} not found, skipping.")
        return None

    timestamps = read_num_samples_and_timestamps(lowdim_path, timestamp_key)
    n_samples = len(timestamps)
    if n_samples == 0:
        logger.warning(f"{os.path.basename(scene_dir)}: empty timestamps, skipping.")
        return None

    frames = list_color_frames(scene_dir, camera_serial)
    if not frames:
        logger.warning(f"{os.path.basename(scene_dir)}: no RGB frames found, skipping.")
        return None

    # Reference image: user-provided (overall phase scope), else fall back to the scene's first frame.
    ref_path = reference_image_path if reference_image_path is not None else frames[0][1]

    patch_starts = list(range(0, n_samples, patch_size))
    per_patch: List[Dict[str, object]] = []
    specify_list: List[str] = []
    # Cache Gemini results by the nearest-frame path (RGB is ~15 Hz while patches are much finer,
    # so many patches share the same image; classify each unique image only once).
    cache: Dict[str, Dict[str, str]] = {}

    for patch_idx, start_idx in enumerate(patch_starts):
        target_ts = timestamps[start_idx]
        cur_path = nearest_frame_path(frames, target_ts)

        if cur_path is None:
            specify = "auto"
            entry = {"classification": "unknown", "identified_phase": "", "specify": "auto"}
        else:
            if cur_path in cache:
                entry = cache[cur_path]
            else:
                try:
                    entry = classify_image_pair(client, model, ref_path, cur_path, task_description)
                except Exception as exc:  # noqa: BLE001 - log and fall back per patch
                    logger.error(f"{os.path.basename(scene_dir)} patch {patch_idx}: "
                                 f"classification failed ({exc}); using specify='auto'.")
                    entry = {"classification": "unknown", "identified_phase": "", "specify": "auto"}
                cache[cur_path] = entry
            specify = entry["specify"]

        specify_list.append(specify)
        per_patch.append({
            "patch_idx": patch_idx,
            "start_index": int(start_idx),
            "timestamp": float(target_ts),
            "frame": os.path.basename(cur_path) if cur_path else None,
            "identified_phase": entry.get("identified_phase", ""),
            "classification": entry.get("classification", "unknown"),
            "specify": specify,
        })

    logger.info(
        f"{os.path.basename(scene_dir)}: {len(specify_list)} patches "
        f"({sum(s == 'twist' for s in specify_list)} twist / "
        f"{sum(s == 'wrench' for s in specify_list)} wrench / "
        f"{sum(s == 'auto' for s in specify_list)} auto), "
        f"{len(cache)} unique Gemini queries."
    )

    return {
        "task_description": task_description,
        "model": model,
        "reference_image": ref_path,
        "patch_size": patch_size,
        "num_patches": len(specify_list),
        "specify": specify_list,
        "per_patch": per_patch,
    }


def build_client(api_key: Optional[str]) -> "object":
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise ImportError(
            "google-genai is required. Install it via `pip install google-genai`."
        ) from exc

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Set GEMINI_API_KEY (or GOOGLE_API_KEY), or pass --api_key.")
    return genai.Client(api_key=key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-patch power-source classification with Gemini 3 Pro (optional IF-recovery prior)."
    )
    parser.add_argument("--task_path", type=str, required=True, help="Task data root containing scene_* dirs.")
    parser.add_argument("--task_description", type=str, required=True, help="Natural-language task description.")
    parser.add_argument("--patch_size", type=int, required=True, help="Patch size in low-dim samples; MUST match the task's adaptor config patch_size.")
    parser.add_argument("--reference_image", type=str, default=None, help="Task reference image (overall phase scope). If omitted, each scene's first frame is used.")
    parser.add_argument("--lowdim_name", type=str, default="lowdim_filled.h5", help="Low-dim HDF5 filename (must match what run_adaptor.py consumes).")
    parser.add_argument("--timestamp_key", type=str, default="timestamp", help="Timestamp dataset key inside the low-dim HDF5.")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview", help="Gemini model id.")
    parser.add_argument("--camera_serial", type=str, default=None, help="Preferred camera serial (cam_<serial>).")
    parser.add_argument("--max_scenes", type=int, default=None, help="Limit the number of scenes to process.")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API key (else read from env).")
    parser.add_argument("--output", type=str, default="power_source_specify.json", help="Per-scene output filename (saved under <scene>/lowdim/).")
    args = parser.parse_args()

    client = build_client(args.api_key)

    scene_names = sorted(
        name for name in os.listdir(args.task_path)
        if os.path.isdir(os.path.join(args.task_path, name)) and name.startswith("scene_")
    )
    if args.max_scenes is not None:
        scene_names = scene_names[: args.max_scenes]

    num_written = 0
    for scene_name in scene_names:
        scene_dir = os.path.join(args.task_path, scene_name)
        result = classify_scene(
            client=client,
            model=args.model,
            scene_dir=scene_dir,
            lowdim_name=args.lowdim_name,
            timestamp_key=args.timestamp_key,
            patch_size=args.patch_size,
            reference_image_path=args.reference_image,
            camera_serial=args.camera_serial,
            task_description=args.task_description,
        )
        if result is None:
            continue
        output_path = os.path.join(scene_dir, "lowdim", args.output)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved per-patch specify to {output_path}")
        num_written += 1

    logger.info(
        f"Done. Wrote {num_written} per-scene specify files ('{args.output}'). "
        f"Run the adaptor with --specify_seq_name {args.output} and the SAME "
        f"--patch_size ({args.patch_size}) to apply them."
    )
