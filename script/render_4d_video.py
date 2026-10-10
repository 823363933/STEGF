#!/usr/bin/env python3
"""Render a smooth space-time trajectory from a trained STEGF checkpoint.

The standard test path keeps one held-out camera fixed while time advances.
This utility instead moves a virtual camera inside the calibrated camera rig
while advancing the model timestamp continuously.  The trajectory scale is
derived from the real camera centers, so it remains scene-relative rather
than relying on the legacy fixed-radius ``colmapmv`` helix.
"""

import json
import math
import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torchvision
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GAUSSIAN_SPLATTING_ROOT = REPO_ROOT / "thirdparty" / "gaussian_splatting"
if str(GAUSSIAN_SPLATTING_ROOT) not in sys.path:
    sys.path.append(str(GAUSSIAN_SPLATTING_ROOT))

from helper_train import getmodel, getrenderpip, trbfunction
from thirdparty.gaussian_splatting.arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
)
from thirdparty.gaussian_splatting.helper3dg import (
    _normalize_existence_mode,
    _normalize_motion_model,
    get_combined_args,
)
from thirdparty.gaussian_splatting.scene import Scene
from thirdparty.gaussian_splatting.scene.cameras import Camera
from thirdparty.gaussian_splatting.utils.general_utils import safe_state


def _parse_args():
    parser = ArgumentParser(
        description=(
            "Render a smooth camera-and-time trajectory from a STEGF "
            "checkpoint and encode it as an MP4 video."
        )
    )
    model_group = ModelParams(parser, sentinel=True)
    OptimizationParams(parser)
    pipeline_group = PipelineParams(parser)

    parser.add_argument("--test_iteration", default=-1, type=int)
    parser.add_argument("--duration", default=None, type=int)
    parser.add_argument("--rgbfunction", default=None, type=str)
    parser.add_argument("--rdpip", default=None, type=str)
    parser.add_argument("--valloader", default=None, type=str)
    parser.add_argument("--configpath", default=None, type=str)
    parser.add_argument("--quiet", action="store_true", default=None)
    parser.add_argument(
        "--frames",
        default=150,
        type=int,
        help="Number of continuously sampled space-time frames.",
    )
    parser.add_argument(
        "--fps",
        default=30,
        type=int,
        help="Encoded video frame rate.",
    )
    parser.add_argument(
        "--trajectory_scale",
        default=0.35,
        type=float,
        help=(
            "Ellipse radius relative to the calibrated camera spread. "
            "Values in [0.2, 0.5] are normally conservative."
        ),
    )
    parser.add_argument(
        "--path_cycles",
        default=1.0,
        type=float,
        help="Number of camera loops during one temporal sweep.",
    )
    parser.add_argument(
        "--start_phase",
        default=0.0,
        type=float,
        help="Initial camera-path phase in degrees.",
    )
    parser.add_argument(
        "--output_scale",
        default=1.0,
        type=float,
        help="Output size relative to the configured test resolution.",
    )
    parser.add_argument(
        "--look_at",
        default="",
        type=str,
        help="Optional world-space target as x,y,z.",
    )
    parser.add_argument(
        "--output_path",
        default="",
        type=str,
        help="Output directory (default: <model_path>/trajectory_4d).",
    )
    parser.add_argument(
        "--crf",
        default=18,
        type=int,
        help="H.264 constant-rate factor; lower is higher quality.",
    )
    parser.add_argument(
        "--keep_frames",
        action="store_true",
        help="Keep rendered PNG frames after MP4 encoding.",
    )
    parser.add_argument(
        "--skip_video",
        action="store_true",
        help="Render PNG frames without invoking ffmpeg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing trajectory generated in the same directory.",
    )

    args = get_combined_args(parser)
    config_path = getattr(args, "configpath", None)
    if config_path and config_path != "None" and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    fallbacks = {
        "duration": 50,
        "rgbfunction": "rgbv1",
        "rdpip": "train_ours_full",
        "valloader": "colmapvalid",
        "configpath": "None",
        "quiet": False,
    }
    for key, value in fallbacks.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    if args.duration < 2:
        parser.error("--duration must be at least 2")
    if args.trajectory_scale <= 0.0:
        parser.error("--trajectory_scale must be positive")
    if args.path_cycles <= 0.0:
        parser.error("--path_cycles must be positive")
    if args.output_scale <= 0.0:
        parser.error("--output_scale must be positive")
    if not 0 <= args.crf <= 51:
        parser.error("--crf must be in [0, 51]")

    args = _normalize_existence_mode(args)
    args = _normalize_motion_model(args)
    return args, model_group.extract(args), pipeline_group.extract(args)


def _module_eval(module):
    if module is not None and hasattr(module, "cuda"):
        module.cuda()
    if module is not None and hasattr(module, "eval"):
        module.eval()


def _prepare_model(gaussians):
    for name in (
        "rgbdecoder",
        "content_exposure_head",
        "euler_field",
        "field_router",
        "field_query_gate",
        "field_decoder",
        "field_static_view_mapper",
        "field_static_app_head",
    ):
        _module_eval(getattr(gaussians, name, None))


def _parse_look_at(value):
    value = str(value).strip()
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--look_at must contain exactly three values: x,y,z")
    target = np.asarray([float(part) for part in parts], dtype=np.float64)
    if not np.all(np.isfinite(target)):
        raise ValueError("--look_at values must be finite")
    return target


def _load_camera_records(model_path):
    path = Path(model_path) / "cameras.json"
    if not path.is_file():
        raise FileNotFoundError(f"Camera metadata not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    unique = {}
    for entry in entries:
        position = np.asarray(entry.get("position"), dtype=np.float64)
        rotation = np.asarray(entry.get("rotation"), dtype=np.float64)
        if position.shape != (3,) or rotation.shape != (3, 3):
            continue
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
            continue
        key = tuple(np.round(position, decimals=7).tolist())
        unique.setdefault(
            key,
            {
                "position": position,
                "rotation": rotation,
                "image_name": str(entry.get("img_name", "unknown")),
            },
        )
    records = list(unique.values())
    if len(records) < 3:
        raise RuntimeError(
            f"At least three unique calibrated cameras are required; found {len(records)}"
        )
    return records, path


def _normalized(vector, fallback=None):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if np.isfinite(norm) and norm > 1e-10:
        return vector / norm
    if fallback is None:
        raise RuntimeError("Cannot normalize a degenerate vector")
    return _normalized(fallback)


def _aligned_mean(vectors):
    reference = _normalized(vectors[0])
    aligned = []
    for vector in vectors:
        vector = _normalized(vector)
        if float(np.dot(vector, reference)) < 0.0:
            vector = -vector
        aligned.append(vector)
    return _normalized(np.mean(np.stack(aligned, axis=0), axis=0), reference)


def _estimate_look_at(records, gaussian_fallback):
    positions = np.stack([record["position"] for record in records], axis=0)
    forwards = np.stack(
        [_normalized(record["rotation"][:, 2]) for record in records],
        axis=0,
    )
    matrix = np.zeros((3, 3), dtype=np.float64)
    vector = np.zeros((3,), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for position, forward in zip(positions, forwards):
        projector = identity - np.outer(forward, forward)
        matrix += projector
        vector += projector @ position

    rig_center = positions.mean(axis=0)
    rig_radius = float(
        np.percentile(np.linalg.norm(positions - rig_center, axis=1), 90)
    )
    rig_radius = max(rig_radius, 1e-6)
    condition = float(np.linalg.cond(matrix))
    target = np.linalg.lstsq(matrix, vector, rcond=None)[0]
    depths = np.sum((target[None, :] - positions) * forwards, axis=1)
    distance = float(np.linalg.norm(target - rig_center))
    valid = (
        np.all(np.isfinite(target))
        and np.isfinite(condition)
        and condition < 1e8
        and float(np.median(depths)) > 0.0
        and 0.25 * rig_radius < distance < 100.0 * rig_radius
    )
    if not valid:
        target = np.asarray(gaussian_fallback, dtype=np.float64)
        method = "canonical_gaussian_median"
    else:
        method = "calibrated_view_ray_intersection"
    return target, method, condition


def _trajectory_geometry(records, scale):
    positions = np.stack([record["position"] for record in records], axis=0)
    center = positions.mean(axis=0)
    centered = positions - center
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    basis = axes[:2]
    projected = centered @ basis.T
    radii = np.percentile(np.abs(projected), 90, axis=0)
    major = max(float(np.max(radii)), 1e-6)
    radii = np.maximum(radii, 0.15 * major) * float(scale)
    up_hint = _aligned_mean(
        [record["rotation"][:, 1] for record in records]
    )
    return center, basis, radii, up_hint


def _look_at_rotation(position, target, up_hint):
    forward = _normalized(target - position)
    right = np.cross(up_hint, forward)
    if float(np.linalg.norm(right)) < 1e-8:
        alternative = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(alternative, forward))) > 0.95:
            alternative = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(alternative, forward)
    right = _normalized(right)
    up = _normalized(np.cross(forward, right))
    return np.stack((right, up, forward), axis=1)


def _trajectory_sample(
    index,
    frames,
    duration,
    center,
    basis,
    radii,
    target,
    up_hint,
    path_cycles,
    start_phase,
):
    fraction = float(index) / float(frames - 1)
    angle = math.radians(float(start_phase)) + 2.0 * math.pi * float(path_cycles) * fraction
    position = (
        center
        + basis[0] * radii[0] * math.cos(angle)
        + basis[1] * radii[1] * math.sin(angle)
    )
    rotation = _look_at_rotation(position, target, up_hint)
    timestamp = fraction * float(duration - 1) / float(duration)
    return position, rotation, timestamp


def _make_camera(reference, position, rotation, timestamp, index, output_scale):
    width = max(int(round(reference.image_width * output_scale)), 2)
    height = max(int(round(reference.image_height * output_scale)), 2)
    translation = -rotation.T @ position
    camera = Camera(
        colmap_id=index,
        R=rotation,
        T=translation,
        FoVx=float(reference.FoVx),
        FoVy=float(reference.FoVy),
        image=(width, height),
        gt_alpha_mask=None,
        image_name=f"trajectory_{index:05d}",
        uid=index,
        data_device="cuda",
        timestamp=float(timestamp),
        rayo=1,
        rayd=1,
        cxr=float(getattr(reference, "cxr", 0.0)),
        cyr=float(getattr(reference, "cyr", 0.0)),
    )
    camera.rays = torch.cat((camera.rayo, camera.rayd), dim=1).cuda()
    return camera


def _prepare_output(output_dir, overwrite):
    output_dir = Path(output_dir).resolve()
    frames_dir = output_dir / "frames"
    known_outputs = (
        output_dir / "trajectory_4d.mp4",
        output_dir / "poster.png",
        output_dir / "trajectory.json",
    )
    occupied = frames_dir.exists() or any(path.exists() for path in known_outputs)
    if occupied and not overwrite:
        raise FileExistsError(
            f"Trajectory output already exists: {output_dir}\n"
            "Pass --overwrite to replace the generated trajectory files."
        )
    if overwrite:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        for path in known_outputs:
            if path.exists():
                path.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, frames_dir


def _encode_video(frames_dir, video_path, fps, crf):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found. PNG frames were kept in "
            f"{frames_dir}; install ffmpeg or encode them manually."
        )
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frames_dir / "%05d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(video_path),
    ]
    print("[STEGF][4D] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def render_trajectory(args, dataset, pipeline):
    if not torch.cuda.is_available():
        raise RuntimeError("4D trajectory rendering requires CUDA")

    GaussianModel = getmodel(dataset.model)
    gaussians = GaussianModel(dataset.sh_degree, args.rgbfunction)
    if hasattr(gaussians, "configure_euler_field"):
        gaussians.configure_euler_field(dataset)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.test_iteration,
        shuffle=False,
        multiview=False,
        duration=args.duration,
        loader=args.valloader,
    )
    reference_views = sorted(
        scene.getTestCameras(),
        key=lambda camera: float(camera.timestamp),
    )
    if not reference_views:
        raise RuntimeError("The selected loader produced no reference camera")
    reference = reference_views[0]
    _prepare_model(gaussians)

    records, camera_path = _load_camera_records(dataset.model_path)
    requested_target = _parse_look_at(args.look_at)
    gaussian_target = (
        gaussians.get_xyz.detach().median(dim=0).values.float().cpu().numpy()
    )
    if requested_target is None:
        target, target_method, ray_condition = _estimate_look_at(
            records,
            gaussian_target,
        )
    else:
        target = requested_target
        target_method = "command_line"
        ray_condition = None

    center, basis, radii, up_hint = _trajectory_geometry(
        records,
        args.trajectory_scale,
    )
    output_dir = args.output_path or str(
        Path(dataset.model_path) / "trajectory_4d"
    )
    output_dir, frames_dir = _prepare_output(output_dir, args.overwrite)
    video_path = output_dir / "trajectory_4d.mp4"
    poster_path = output_dir / "poster.png"
    manifest_path = output_dir / "trajectory.json"

    render_option = {
        "train_ours_full": "test_ours_full",
        "train_ours_fullss": "test_ours_fullss",
    }.get(str(args.rdpip), str(args.rdpip))
    render, settings, rasterizer = getrenderpip(render_option)
    background = torch.zeros((9,), dtype=torch.float32, device="cuda")
    output_width = max(int(round(reference.image_width * args.output_scale)), 2)
    output_height = max(int(round(reference.image_height * args.output_scale)), 2)
    if gaussians.ts is None:
        gaussians.ts = torch.ones(
            (1, 1, output_height, output_width),
            dtype=torch.float32,
            device="cuda",
        )

    print(
        "[STEGF][4D] Rendering trajectory: "
        f"iteration={scene.loaded_iter}, frames={args.frames}, "
        f"resolution={output_width}x{output_height}, "
        f"cameras={len(records)}, target={target_method}",
        flush=True,
    )
    samples = []
    with torch.no_grad():
        for index in tqdm(range(args.frames), desc="Rendering 4D trajectory"):
            position, rotation, timestamp = _trajectory_sample(
                index,
                args.frames,
                args.duration,
                center,
                basis,
                radii,
                target,
                up_hint,
                args.path_cycles,
                args.start_phase,
            )
            camera = _make_camera(
                reference,
                position,
                rotation,
                timestamp,
                index,
                args.output_scale,
            )
            package = render(
                camera,
                gaussians,
                pipeline,
                background,
                scaling_modifier=1.0,
                basicfunction=trbfunction,
                GRsetting=settings,
                GRzer=rasterizer,
            )
            image = package["render"]
            if hasattr(gaussians, "apply_content_exposure"):
                image = gaussians.apply_content_exposure(image)
            image = image.clamp(0.0, 1.0)
            frame_path = frames_dir / f"{index:05d}.png"
            torchvision.utils.save_image(image, frame_path)
            samples.append(
                {
                    "index": int(index),
                    "timestamp": float(timestamp),
                    "source_time": float(timestamp * args.duration),
                    "position": position.tolist(),
                    "rotation": rotation.tolist(),
                }
            )
            del image, package, camera

    shutil.copy2(frames_dir / f"{args.frames // 2:05d}.png", poster_path)
    manifest = {
        "schema": "stegf_4d_trajectory_v1",
        "model_path": str(Path(dataset.model_path).resolve()),
        "source_path": str(Path(dataset.source_path).resolve()),
        "camera_metadata": str(camera_path.resolve()),
        "iteration": int(scene.loaded_iter),
        "duration": int(args.duration),
        "frames": int(args.frames),
        "fps": int(args.fps),
        "resolution": [int(output_width), int(output_height)],
        "trajectory_scale": float(args.trajectory_scale),
        "path_cycles": float(args.path_cycles),
        "start_phase_degrees": float(args.start_phase),
        "calibrated_camera_count": len(records),
        "trajectory_center": center.tolist(),
        "trajectory_basis": basis.tolist(),
        "trajectory_radii": radii.tolist(),
        "look_at": target.tolist(),
        "look_at_method": target_method,
        "view_ray_system_condition": ray_condition,
        "samples": samples,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    if not args.skip_video:
        _encode_video(frames_dir, video_path, args.fps, args.crf)
        if not args.keep_frames:
            shutil.rmtree(frames_dir)

    print(
        "[STEGF][4D] Render complete:\n"
        f"  output: {output_dir}\n"
        f"  poster: {poster_path}\n"
        f"  manifest: {manifest_path}",
        flush=True,
    )
    if video_path.is_file():
        print(f"  video: {video_path}", flush=True)
    elif frames_dir.is_dir():
        print(f"  frames: {frames_dir}", flush=True)


def main():
    args, dataset, pipeline = _parse_args()
    safe_state(bool(args.quiet))
    render_trajectory(args, dataset, pipeline)


if __name__ == "__main__":
    main()
