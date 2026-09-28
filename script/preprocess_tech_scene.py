#!/usr/bin/env python3
"""Build one STEGF scene from an extracted Technicolor sequence.

The sparse reconstruction follows the calibrated Technicolor conversion used
by SpacetimeGaussians: one COLMAP model is triangulated per selected time using
the official ``cameras_parameters.txt`` poses.  The frame-0 dense
reconstruction, voxel downsampling, optional MiDaS inference, resumability, and
headless-host handling reuse the validated STEGF N3D preprocessing path.

Expected raw input for one scene::

    <inputpath>/<scene>/
      cameras_parameters.txt
      <scene>_undist_<frame>_<camera>.png

Published output::

    <datapath>/<scene>/
      cameras_parameters.txt
      colmap_0/ ... colmap_<duration-1>/
      dense_points.ply                 # optional
      midas_beit_large_512/            # optional

Raw PNG files may share the output scene directory, but they are not required
after the published COLMAP slices have been built.
"""

# The Technicolor frame selection and calibrated COLMAP conversion follow
# SpacetimeGaussians, Copyright (c) 2023 OPPO, distributed under the MIT
# License. Dense reconstruction and optional MiDaS stages reuse STEGF's local
# preprocessing implementation.

import argparse
import json
import math
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

from preprocess_n3d_scene import (
    MIDAS_DEFAULT_COMMIT,
    build_dense_points,
    build_midas_depth,
    build_sparse_time_slice,
    feature_enabled,
    resolve_path,
    time_slice_complete,
    write_manifest,
)


TECHNICOLOR_START_FRAMES = {
    "Birthday": 151,
    "Fabien": 51,
    "Painter": 100,
    "Theater": 51,
    "Train": 151,
}

# The official Birthday sequence contains one known corrupt PNG.  Match the
# repair used by SpacetimeGaussians, but write the repaired image only into the
# generated STEGF scene so that the extracted source dataset remains intact.
TECHNICOLOR_IMAGE_REPAIRS = {
    ("Birthday", 173, 9): (172, 9),
}


def resolve_raw_scene(input_path, scene):
    candidate = input_path / scene
    if candidate.is_dir():
        return candidate
    if input_path.is_dir() and input_path.name == scene:
        return input_path
    raise FileNotFoundError(
        f"Extracted Technicolor scene not found: expected {candidate} "
        "or a direct scene path"
    )


def resolve_config(repo_root, scene, explicit_path):
    if explicit_path:
        path = resolve_path(explicit_path, repo_root)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration not found: {path}")
        return path, False
    config_dir = repo_root / "configs" / "tech_ours"
    scene_path = config_dir / f"{scene}.json"
    if scene_path.is_file():
        return scene_path, False
    default_path = config_dir / "default.json"
    if default_path.is_file():
        return default_path, True
    return None, False


def parse_camera_parameters(path, width, height, downscale):
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 12:
            raise ValueError(
                f"Expected 12 camera fields at {path}:{line_number}, "
                f"got {len(fields)}"
            )
        try:
            rows.append([float(value) for value in fields])
        except ValueError as error:
            raise ValueError(
                f"Invalid numeric camera row at {path}:{line_number}"
            ) from error
    if not rows:
        raise ValueError(f"No camera records found in {path}")

    output_width = int(width / float(downscale))
    output_height = int(height / float(downscale))
    if output_width < 1 or output_height < 1:
        raise ValueError(
            f"Invalid downscaled image size: {output_width}x{output_height}"
        )
    cameras = []
    for index, row in enumerate(rows):
        focal, center_x, center_y = row[:3]
        qvec = row[5:9]
        tvec = row[9:12]
        quaternion_norm = math.sqrt(sum(value * value for value in qvec))
        if not math.isfinite(quaternion_norm) or abs(quaternion_norm - 1.0) > 1e-3:
            raise ValueError(
                f"Camera {index:02d} has invalid quaternion norm: "
                f"{quaternion_norm}"
            )
        cameras.append(
            {
                "id": index + 1,
                "camera_index": index,
                "name": f"cam{index:02d}.png",
                "width": output_width,
                "height": output_height,
                # Match the STGS Technicolor conversion. The provided aspect
                # ratio is 1 for the official undistorted sequences.
                "fx": focal / float(downscale),
                "fy": focal / float(downscale),
                "cx": center_x / float(downscale),
                "cy": center_y / float(downscale),
                "qvec": qvec,
                "tvec": tvec,
            }
        )
    return cameras


def discover_source_images(raw_scene, scene, source_start_frame, duration):
    expression = re.compile(
        rf"^{re.escape(scene)}_undist_(\d{{5}})_(\d{{2}})\.png$"
    )
    requested_frames = set(
        range(source_start_frame, source_start_frame + duration)
    )
    images = {}
    for path in raw_scene.glob(f"{scene}_undist_*.png"):
        match = expression.match(path.name)
        if match is None:
            continue
        frame = int(match.group(1))
        if frame not in requested_frames:
            continue
        camera = int(match.group(2))
        key = (frame, camera)
        if key in images:
            raise RuntimeError(
                f"Duplicate Technicolor image for frame/camera {key}: "
                f"{images[key]} and {path}"
            )
        images[key] = path
    if not images:
        raise FileNotFoundError(
            f"No selected Technicolor PNG files found in {raw_scene}"
        )

    cameras_by_frame = {
        frame: sorted(
            camera
            for (image_frame, camera) in images
            if image_frame == frame
        )
        for frame in sorted(requested_frames)
    }
    first_frame = source_start_frame
    expected_cameras = cameras_by_frame[first_frame]
    if not expected_cameras:
        raise FileNotFoundError(
            f"No images found for first selected frame {first_frame}"
        )
    for frame, camera_indices in cameras_by_frame.items():
        if camera_indices != expected_cameras:
            missing = sorted(set(expected_cameras) - set(camera_indices))
            extra = sorted(set(camera_indices) - set(expected_cameras))
            raise RuntimeError(
                f"Inconsistent cameras at source frame {frame}: "
                f"missing={missing}, extra={extra}"
            )
    return images, expected_cameras


def discover_source_image_repairs(source_images, scene):
    from PIL import Image

    repairs = {}
    failures = []
    for (frame, camera), path in sorted(source_images.items()):
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, SyntaxError) as error:
            repair_key = (scene, frame, camera)
            reference = TECHNICOLOR_IMAGE_REPAIRS.get(repair_key)
            if reference is None:
                failures.append((path, error))
                continue
            reference_path = source_images.get(reference)
            if reference_path is None:
                raise RuntimeError(
                    f"Repair reference {reference} is unavailable for "
                    f"corrupt source image: {path}"
                ) from error
            try:
                with Image.open(reference_path) as reference_image:
                    reference_image.verify()
            except (OSError, SyntaxError) as reference_error:
                raise RuntimeError(
                    f"Repair reference is also invalid: {reference_path}"
                ) from reference_error
            repairs[(frame, camera)] = reference_path
            print(
                f"[STEGF][Preprocess] Corrupt source image scheduled for "
                f"repair: {path.name} <- {reference_path.name}",
                flush=True,
            )
    if failures:
        details = ", ".join(
            f"{path.name}: {error}" for path, error in failures
        )
        raise RuntimeError(
            "Corrupt Technicolor source images have no registered repair: "
            f"{details}"
        )
    return repairs


def read_source_size(path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read source image: {path}")
    return int(image.shape[1]), int(image.shape[0])


def repair_source_image(
    source,
    reference,
    destination,
    expected_size,
    downscale,
):
    import cv2
    import numpy as np
    from PIL import Image, ImageFile

    previous_truncated_setting = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(source) as image:
            image.load()
            recovered = np.asarray(image.convert("RGB")).copy()
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_truncated_setting

    reference_bgr = cv2.imread(str(reference), cv2.IMREAD_COLOR)
    if reference_bgr is None:
        raise RuntimeError(f"Could not read repair reference: {reference}")
    reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
    if recovered.shape != reference_rgb.shape:
        raise RuntimeError(
            f"Repair image shape mismatch: source={recovered.shape}, "
            f"reference={reference_rgb.shape}"
        )

    missing = recovered == 0
    repaired = np.where(missing, reference_rgb, recovered).astype(np.uint8)
    output_width, output_height = expected_size
    if downscale != 1:
        repaired = cv2.resize(
            repaired,
            (output_width, output_height),
            interpolation=cv2.INTER_AREA,
        )
    repaired_bgr = cv2.cvtColor(repaired, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(destination), repaired_bgr):
        raise RuntimeError(f"Failed to write repaired image: {destination}")
    try:
        with Image.open(destination) as image:
            image.verify()
    except (OSError, SyntaxError) as error:
        raise RuntimeError(
            f"Repaired image failed validation: {destination}"
        ) from error
    print(
        f"[STEGF][Preprocess] Repaired source image: {source.name} <- "
        f"{reference.name}; replaced_channels={int(missing.sum())}",
        flush=True,
    )


def stage_source_image(
    source,
    destination,
    expected_size,
    downscale,
    repair_reference=None,
):
    if repair_reference is not None:
        repair_source_image(
            source,
            repair_reference,
            destination,
            expected_size,
            downscale,
        )
        return
    if downscale == 1:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return

    import cv2

    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read source image: {source}")
    output_width, output_height = expected_size
    resized = cv2.resize(
        image,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )
    if not cv2.imwrite(str(destination), resized):
        raise RuntimeError(f"Failed to write staged image: {destination}")


def prepare_build_directory(scene_root, time_index, restart_incomplete):
    build_dir = scene_root / f".stegf_colmap_{time_index}.building"
    if build_dir.exists():
        if not restart_incomplete:
            raise RuntimeError(
                f"Incomplete preprocessing workspace exists: {build_dir}. "
                "Rerun with --restart_incomplete to replace generated "
                "workspaces."
            )
        shutil.rmtree(build_dir)
    (build_dir / "input").mkdir(parents=True)
    return build_dir


def build_sparse_sequence(
    scene_root,
    source_images,
    cameras,
    duration,
    source_start_frame,
    source_image_repairs,
    downscale,
    colmap_binary,
    restart_incomplete,
    dry_run,
):
    missing = []
    completed = 0
    for time_index in range(duration):
        final_dir = scene_root / f"colmap_{time_index}"
        if final_dir.exists():
            if not time_slice_complete(final_dir, len(cameras)):
                raise RuntimeError(
                    f"Published time slice is incomplete: {final_dir}. "
                    "Move it aside before rerunning preprocessing."
                )
            completed += 1
        else:
            missing.append(time_index)
    if completed:
        print(
            f"[STEGF][Preprocess] Sparse time slices already complete: "
            f"{completed}/{duration}",
            flush=True,
        )
    if not missing:
        return
    print(
        f"[STEGF][Preprocess] Sparse time slices to build: "
        f"{len(missing)}/{duration}",
        flush=True,
    )
    if dry_run:
        print(
            f"[STEGF][Preprocess] Would build time indices "
            f"{missing[0]}..{missing[-1]}",
            flush=True,
        )
        return

    expected_size = (cameras[0]["width"], cameras[0]["height"])
    for completed_index, time_index in enumerate(missing, start=1):
        source_frame = source_start_frame + time_index
        build_dir = prepare_build_directory(
            scene_root, time_index, restart_incomplete
        )
        input_dir = build_dir / "input"
        try:
            for camera in cameras:
                source = source_images[
                    (source_frame, camera["camera_index"])
                ]
                destination = input_dir / camera["name"]
                stage_source_image(
                    source,
                    destination,
                    expected_size,
                    downscale,
                    repair_reference=source_image_repairs.get(
                        (source_frame, camera["camera_index"])
                    ),
                )
            print(
                f"[STEGF][Preprocess] Sparse COLMAP "
                f"{completed_index}/{len(missing)}: time={time_index}, "
                f"source_frame={source_frame}",
                flush=True,
            )
            build_sparse_time_slice(
                colmap_binary,
                build_dir,
                scene_root / f"colmap_{time_index}",
                cameras,
                dry_run=False,
            )
        except Exception:
            print(
                f"[STEGF][Preprocess] Incomplete workspace retained for "
                f"inspection: {build_dir}",
                flush=True,
            )
            raise


def copy_camera_parameters(source, scene_root, dry_run):
    destination = scene_root / "cameras_parameters.txt"
    if destination.exists():
        if source.resolve() != destination.resolve():
            if source.read_bytes() != destination.read_bytes():
                raise RuntimeError(
                    f"Existing camera parameters differ from raw input: "
                    f"{destination}"
                )
        return destination
    print(
        f"[STEGF][Preprocess] Calibration: {source} -> {destination}",
        flush=True,
    )
    if not dry_run:
        scene_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return source if dry_run else destination


def enter_virtual_display_if_available(colmap_needed, dry_run):
    if not colmap_needed or os.environ.get("DISPLAY"):
        return
    if os.environ.get("STEGF_XVFB_ACTIVE"):
        raise RuntimeError(
            "xvfb-run did not provide DISPLAY to the preprocessing process"
        )
    xvfb_run = shutil.which("xvfb-run")
    if xvfb_run is None:
        print(
            "[STEGF][Preprocess] Headless host detected without xvfb-run; "
            "continuing with the COLMAP CLI directly. Install xvfb and xauth "
            "if the local COLMAP build requires a display.",
            flush=True,
        )
        return
    command = (
        xvfb_run,
        "-a",
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    )
    print(
        "[STEGF][Preprocess] Headless host detected; entering a virtual "
        f"display: {shlex.join(command)}",
        flush=True,
    )
    if dry_run:
        return
    environment = os.environ.copy()
    environment["STEGF_XVFB_ACTIVE"] = "1"
    os.execvpe(xvfb_run, command, environment)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert one extracted Technicolor scene into the complete "
            "STEGF scene layout."
        )
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--inputpath",
        "--rawpath",
        dest="input_path",
        default="dataset/Tech",
        help="Raw-data root containing <scene>, or the extracted scene itself.",
    )
    parser.add_argument(
        "--datapath",
        default="dataset/Tech",
        help="Output data root; the script writes <datapath>/<scene>.",
    )
    parser.add_argument(
        "--configpath",
        help=(
            "Scene configuration. By default, use "
            "configs/tech_ours/<scene>.json and then default.json."
        ),
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Number of time steps; defaults to the model config or 50.",
    )
    parser.add_argument(
        "--source_start_frame",
        type=int,
        help="First raw frame; defaults to the official STGS scene split.",
    )
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument(
        "--dense",
        choices=("auto", "on", "off"),
        default="auto",
        help="Frame-0 dense reconstruction; auto follows the model config.",
    )
    parser.add_argument(
        "--midas",
        choices=("auto", "on", "off"),
        default="auto",
        help="MiDaS depth prior generation; auto follows the model config.",
    )
    parser.add_argument("--colmap_binary", default="colmap")
    parser.add_argument("--gpu_index", default="0")
    parser.add_argument("--dense_max_points", type=int, default=100000)
    parser.add_argument("--dense_voxel_start", type=float, default=0.001)
    parser.add_argument("--dense_voxel_step", type=float, default=0.005)
    parser.add_argument(
        "--midas_repo",
        help="MiDaS checkout (default: <repo>/thirdparty/MiDaS).",
    )
    parser.add_argument("--midas_weights")
    parser.add_argument(
        "--midas_device", choices=("cuda", "cpu"), default="cuda"
    )
    parser.add_argument("--overwrite_midas", action="store_true")
    parser.add_argument("--restart_incomplete", action="store_true")
    parser.add_argument("--keep_dense_workspace", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.duration is not None and args.duration < 1:
        parser.error("--duration must be >= 1")
    if args.downscale < 1:
        parser.error("--downscale must be >= 1")
    if args.dense_max_points < 1:
        parser.error("--dense_max_points must be >= 1")
    if args.dense_voxel_start <= 0 or args.dense_voxel_step <= 0:
        parser.error("Dense voxel sizes must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    input_path = resolve_path(args.input_path, repo_root)
    data_root = resolve_path(args.datapath, repo_root)
    raw_scene = resolve_raw_scene(input_path, args.scene)
    scene_root = data_root / args.scene
    camera_parameters = raw_scene / "cameras_parameters.txt"
    if not camera_parameters.is_file():
        raise FileNotFoundError(
            f"Required camera parameters not found: {camera_parameters}"
        )

    config_path, uses_default = resolve_config(
        repo_root, args.scene, args.configpath
    )
    config = json.loads(config_path.read_text()) if config_path else {}
    duration = int(args.duration or config.get("duration", 50))
    if duration < 1:
        raise ValueError(f"Configured duration must be >= 1, got {duration}")
    source_start_frame = args.source_start_frame
    if source_start_frame is None:
        if args.scene not in TECHNICOLOR_START_FRAMES:
            parser.error(
                "No official start frame is known for this scene; provide "
                "--source_start_frame"
            )
        source_start_frame = TECHNICOLOR_START_FRAMES[args.scene]
    if source_start_frame < 0:
        parser.error("--source_start_frame must be >= 0")

    source_images, camera_indices = discover_source_images(
        raw_scene,
        args.scene,
        source_start_frame,
        duration,
    )
    source_image_repairs = discover_source_image_repairs(
        source_images,
        args.scene,
    )
    first_source = source_images[(source_start_frame, camera_indices[0])]
    source_width, source_height = read_source_size(first_source)
    cameras = parse_camera_parameters(
        camera_parameters,
        source_width,
        source_height,
        args.downscale,
    )
    if camera_indices != list(range(len(cameras))):
        raise RuntimeError(
            "Image camera indices do not match camera-parameter row order: "
            f"images={camera_indices}, parameters=0..{len(cameras)-1}"
        )

    dense_enabled = feature_enabled(
        args.dense, int(config.get("field_dense_initialization", 0)) != 0
    )
    midas_configured = (
        int(config.get("field_bg_dense_add", 0)) != 0
        and int(config.get("field_bg_dense_beit_filter", 0)) != 0
    )
    midas_enabled = feature_enabled(args.midas, midas_configured)
    midas_times = str(config.get("field_bg_dense_add_time_indices", "0"))
    midas_repo = resolve_path(
        args.midas_repo or "thirdparty/MiDaS", repo_root
    )
    midas_weights = (
        str(resolve_path(args.midas_weights, repo_root))
        if args.midas_weights
        else None
    )

    print(f"[STEGF][Preprocess] Dataset: Technicolor", flush=True)
    print(f"[STEGF][Preprocess] Raw scene: {raw_scene}", flush=True)
    print(f"[STEGF][Preprocess] Output scene: {scene_root}", flush=True)
    if config_path:
        print(
            f"[STEGF][Preprocess] Config: {config_path}"
            f"{' (default fallback)' if uses_default else ''}",
            flush=True,
        )
    else:
        print(
            "[STEGF][Preprocess] Config: none; optional stages follow CLI "
            "flags and default to off in auto mode",
            flush=True,
        )
    print(
        f"[STEGF][Preprocess] Cameras={len(cameras)}, "
        f"duration={duration}, source_frames={source_start_frame}.."
        f"{source_start_frame + duration - 1}, "
        f"source_size={source_width}x{source_height}, "
        f"downscale={args.downscale}",
        flush=True,
    )
    print(
        f"[STEGF][Preprocess] Optional stages: "
        f"dense={int(dense_enabled)}, midas={int(midas_enabled)}",
        flush=True,
    )

    if not args.dry_run:
        scene_root.mkdir(parents=True, exist_ok=True)
    output_calibration = copy_camera_parameters(
        camera_parameters, scene_root, args.dry_run
    )
    sparse_needs_colmap = any(
        not time_slice_complete(
            scene_root / f"colmap_{time_index}", len(cameras)
        )
        for time_index in range(duration)
    )
    dense_needs_colmap = dense_enabled and not (
        scene_root / "dense_points.ply"
    ).is_file()
    enter_virtual_display_if_available(
        sparse_needs_colmap or dense_needs_colmap,
        args.dry_run,
    )
    if (
        not args.dry_run
        and (sparse_needs_colmap or dense_needs_colmap)
        and shutil.which(args.colmap_binary) is None
    ):
        raise FileNotFoundError(
            f"COLMAP executable not found: {args.colmap_binary}. "
            "Run bash script/setup_preprocess.sh first."
        )

    build_sparse_sequence(
        scene_root,
        source_images,
        cameras,
        duration,
        source_start_frame,
        source_image_repairs,
        args.downscale,
        args.colmap_binary,
        args.restart_incomplete,
        args.dry_run,
    )
    if dense_enabled:
        build_dense_points(
            scene_root,
            args.colmap_binary,
            args.gpu_index,
            args.dense_max_points,
            args.dense_voxel_start,
            args.dense_voxel_step,
            args.keep_dense_workspace,
            args.restart_incomplete,
            args.dry_run,
        )
    if midas_enabled:
        if not args.dry_run and not midas_repo.is_dir():
            raise FileNotFoundError(
                f"MiDaS checkout not found: {midas_repo}. "
                "Run bash script/setup_preprocess.sh first."
            )
        build_midas_depth(
            repo_root,
            args.scene,
            data_root,
            midas_times,
            midas_repo,
            midas_weights,
            args.midas_device,
            args.overwrite_midas,
            args.dry_run,
        )

    if not args.dry_run:
        write_manifest(
            scene_root / "preprocess_manifest.json",
            {
                "dataset": "Technicolor",
                "scene": args.scene,
                "raw_scene": str(raw_scene),
                "scene_root": str(scene_root),
                "config": str(config_path) if config_path else "",
                "uses_default_config": uses_default,
                "duration": duration,
                "source_start_frame": source_start_frame,
                "source_end_frame": source_start_frame + duration - 1,
                "downscale": args.downscale,
                "source_image_size": [source_width, source_height],
                "camera_names": [camera["name"] for camera in cameras],
                "repaired_source_images": [
                    {
                        "source": str(source_images[key]),
                        "reference": str(reference),
                        "time_index": key[0] - source_start_frame,
                        "camera_index": key[1],
                    }
                    for key, reference in sorted(
                        source_image_repairs.items()
                    )
                ],
                "test_camera": "cam10",
                "camera_parameters": str(output_calibration),
                "dense_enabled": dense_enabled,
                "dense_max_points": args.dense_max_points,
                "midas_enabled": midas_enabled,
                "midas_time_indices": midas_times if midas_enabled else "",
                "midas_repo": str(midas_repo) if midas_enabled else "",
                "midas_reference_commit": MIDAS_DEFAULT_COMMIT,
            },
        )
    print("[STEGF][Preprocess] Scene preprocessing complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
