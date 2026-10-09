#!/usr/bin/env python3
"""Build one STEGF scene from a Google Immersive video sequence.

The full SpacetimeGaussians Immersive path uses two image domains.  COLMAP
triangulation and Gaussian rasterization use a wide pinhole projection, while
the loss is evaluated against the original fisheye frames.  This preprocessor
publishes both representations together with one fixed differentiable sampling
grid per calibrated camera.

Expected raw input::

    <inputpath>/<official-scene-name>/
      models.json
      camera_0001.mp4
      camera_....mp4

Published output::

    <datapath>/<canonical-scene-name>/
      models.json
      camera_0001/0.png ... <duration-1>.png
      camera_0001.npy
      colmap_0/ ... colmap_<duration-1>/
      dense_points.ply                 # optional
      midas_beit_large_512/            # optional

Only cameras present in ``models.json`` are used.  Official archives that
contain an extra uncalibrated MP4 are therefore handled without a manual
deletion step.  Completed published time slices and decoded fisheye frames are
reused; interrupted hidden COLMAP workspaces are replaced only when requested.
"""

# The fisheye projection and calibrated COLMAP conversion follow
# SpacetimeGaussians, Copyright (c) 2023 OPPO, distributed under the MIT
# License.  Resumable reconstruction, optional dense initialization, and
# optional MiDaS stages reuse STEGF's integrated preprocessing path.

import argparse
import json
import os
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
    prepare_build_directories,
    resolve_path,
    rotation_matrix_to_qvec,
    time_slice_complete,
    write_manifest,
)


IMMERSIVE_SCENES = {
    "01_Welder": {
        "official_name": "01_Welder",
        "focal_scale": 0.36,
    },
    "02_Flames": {
        "official_name": "02_Flames",
        "focal_scale": 0.35,
    },
    "04_Truck": {
        "official_name": "04_Truck",
        "focal_scale": 0.36,
    },
    "09_Alexa": {
        "official_name": "09_Alexa_Meade_Exhibit",
        "focal_scale": 0.36,
    },
    "10_Alexa": {
        "official_name": "10_Alexa_Meade_Face_Paint_1",
        "focal_scale": 0.36,
    },
    "11_Alexa": {
        "official_name": "11_Alexa_Meade_Face_Paint_2",
        "focal_scale": 0.36,
    },
    "12_Cave": {
        "official_name": "12_Cave",
        "focal_scale": 0.36,
    },
}


def canonical_scene_name(name):
    if name in IMMERSIVE_SCENES:
        return name
    for canonical, metadata in IMMERSIVE_SCENES.items():
        if name == metadata["official_name"]:
            return canonical
    supported = sorted(
        set(IMMERSIVE_SCENES)
        | {entry["official_name"] for entry in IMMERSIVE_SCENES.values()}
    )
    raise ValueError(
        f"Unsupported Immersive scene {name!r}; expected one of {supported}"
    )


def resolve_raw_scene(input_path, requested_scene, canonical_scene):
    official_name = IMMERSIVE_SCENES[canonical_scene]["official_name"]
    candidates = []
    for name in (requested_scene, official_name, canonical_scene):
        candidate = input_path / name
        if candidate not in candidates:
            candidates.append(candidate)
    if input_path.is_dir() and (input_path / "models.json").is_file():
        candidates.insert(0, input_path)
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "models.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Extracted Immersive scene was not found. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def resolve_config(repo_root, scene, explicit_path):
    if explicit_path:
        path = resolve_path(explicit_path, repo_root)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration not found: {path}")
        return path, False
    config_dir = repo_root / "configs" / "immersive_ours"
    scene_path = config_dir / f"{scene}.json"
    if scene_path.is_file():
        return scene_path, False
    default_path = config_dir / "default.json"
    if default_path.is_file():
        return default_path, True
    return None, False


def rotation_vector_to_matrix(rotation_vector):
    import cv2
    import numpy as np

    vector = np.asarray(rotation_vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Invalid orientation rotation vector: {vector}")
    matrix, _ = cv2.Rodrigues(vector)
    return matrix


def load_camera_records(models_path, downscale, focal_scale):
    import numpy as np

    metadata = json.loads(models_path.read_text())
    if not isinstance(metadata, list) or not metadata:
        raise ValueError(f"Expected a non-empty camera list in {models_path}")
    cameras = []
    names = set()
    # COLMAP 4.x creates one implicit rig per image during feature extraction
    # in image-name order.  Assign camera/image IDs in the same order so the
    # implicit rigs in the text reconstruction and database have identical
    # reference sensors.  Older COLMAP releases did not expose this ordering
    # requirement, and the official JSON order is not camera-name order.
    ordered_metadata = sorted(metadata, key=lambda model: str(model["name"]))
    for index, model in enumerate(ordered_metadata):
        required = {
            "name",
            "width",
            "height",
            "focal_length",
            "principal_point",
            "position",
            "orientation",
            "radial_distortion",
        }
        missing = sorted(required - set(model))
        if missing:
            raise ValueError(
                f"Camera record {index} in {models_path} is missing {missing}"
            )
        name = str(model["name"])
        if name in names:
            raise ValueError(f"Duplicate camera name in models.json: {name}")
        names.add(name)
        projection_type = str(model.get("projection_type", "fisheye"))
        if projection_type != "fisheye":
            raise ValueError(
                f"Unsupported projection for {name}: {projection_type}"
            )
        source_width = int(round(float(model["width"])))
        source_height = int(round(float(model["height"])))
        width = int(source_width / float(downscale))
        height = int(source_height / float(downscale))
        if width < 1 or height < 1:
            raise ValueError(
                f"Invalid downscaled dimensions for {name}: {width}x{height}"
            )
        principal = np.asarray(model["principal_point"], dtype=np.float64)
        position = np.asarray(model["position"], dtype=np.float64)
        radial = np.asarray(model["radial_distortion"], dtype=np.float64)
        if principal.shape != (2,) or position.shape != (3,):
            raise ValueError(f"Invalid calibration vector shape for {name}")
        if radial.size < 2:
            raise ValueError(
                f"At least two fisheye radial coefficients are required: {name}"
            )
        rotation = rotation_vector_to_matrix(model["orientation"])
        translation = -rotation @ position
        focal = float(model["focal_length"]) / float(downscale)
        center_x = float(principal[0]) / float(downscale)
        center_y = float(principal[1]) / float(downscale)
        distortion = np.zeros(4, dtype=np.float64)
        # Match the STGS full-model preprocessing exactly: the public files
        # contain a third zero coefficient, but only k1 and k2 are consumed.
        distortion[:2] = radial[:2]
        cameras.append(
            {
                "id": index + 1,
                "name": f"{name}.png",
                "camera_name": name,
                "source_width": source_width,
                "source_height": source_height,
                "width": width,
                "height": height,
                "source_focal": focal,
                "fx": focal * float(focal_scale),
                "fy": focal * float(focal_scale),
                "cx": center_x,
                "cy": center_y,
                "distortion": distortion,
                "qvec": rotation_matrix_to_qvec(rotation),
                "tvec": translation,
            }
        )
    return metadata, cameras


def discover_videos(raw_scene, cameras):
    calibrated = {camera["camera_name"] for camera in cameras}
    videos = {path.stem: path for path in raw_scene.glob("camera_*.mp4")}
    missing = sorted(calibrated - set(videos))
    if missing:
        raise FileNotFoundError(
            "Calibrated camera videos are missing: " + ", ".join(missing)
        )
    ignored = sorted(set(videos) - calibrated)
    return {name: videos[name] for name in sorted(calibrated)}, ignored


def copy_models_json(source, destination, dry_run):
    if destination.exists():
        if source.resolve() != destination.resolve():
            source_models = json.loads(source.read_text())
            destination_models = json.loads(destination.read_text())
            if source_models != destination_models:
                raise RuntimeError(
                    f"Existing models.json differs from raw input: {destination}"
                )
        return destination
    print(
        f"[STEGF][Preprocess] Calibration: {source} -> {destination}",
        flush=True,
    )
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return source if dry_run else destination


def write_png_atomic(path, image, compression):
    import cv2

    temporary = path.with_name(f".{path.stem}.writing.png")
    if not cv2.imwrite(
        str(temporary),
        image,
        (cv2.IMWRITE_PNG_COMPRESSION, int(compression)),
    ):
        raise RuntimeError(f"Failed to write image: {temporary}")
    temporary.replace(path)


def ensure_fisheye_frames(
    scene_root,
    cameras,
    videos,
    duration,
    source_start_frame,
    downscale,
    png_compression,
    dry_run,
):
    import cv2

    missing_counts = {}
    for camera in cameras:
        camera_dir = scene_root / camera["camera_name"]
        missing_counts[camera["camera_name"]] = sum(
            not (camera_dir / f"{time_index}.png").is_file()
            for time_index in range(duration)
        )
    total_missing = sum(missing_counts.values())
    total_expected = len(cameras) * duration
    print(
        f"[STEGF][Preprocess] Distorted supervision frames: "
        f"{total_expected - total_missing}/{total_expected} complete",
        flush=True,
    )
    if total_missing == 0 or dry_run:
        return

    for camera_index, camera in enumerate(cameras, start=1):
        if missing_counts[camera["camera_name"]] == 0:
            continue
        video_path = videos[camera["camera_name"]]
        camera_dir = scene_root / camera["camera_name"]
        camera_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_start_frame)
        try:
            for time_index in range(duration):
                success, frame = capture.read()
                if not success or frame is None:
                    raise RuntimeError(
                        f"Could not read source frame "
                        f"{source_start_frame + time_index} from {video_path}"
                    )
                decoded_size = (int(frame.shape[1]), int(frame.shape[0]))
                calibrated_size = (
                    camera["source_width"],
                    camera["source_height"],
                )
                if decoded_size != calibrated_size:
                    raise RuntimeError(
                        f"Video/calibration size mismatch for {video_path}: "
                        f"decoded={decoded_size[0]}x{decoded_size[1]}, "
                        f"models.json={calibrated_size[0]}x{calibrated_size[1]}"
                    )
                destination = camera_dir / f"{time_index}.png"
                if destination.is_file():
                    continue
                if downscale != 1:
                    frame = cv2.resize(
                        frame,
                        (camera["width"], camera["height"]),
                        interpolation=cv2.INTER_AREA,
                    )
                write_png_atomic(destination, frame, png_compression)
        finally:
            capture.release()
        print(
            f"[STEGF][Preprocess] Decoded calibrated camera "
            f"{camera_index}/{len(cameras)}: {camera['camera_name']}",
            flush=True,
        )


def camera_matrices(camera):
    import numpy as np

    intrinsic = np.array(
        [
            [camera["source_focal"], 0.0, camera["cx"]],
            [0.0, camera["source_focal"], camera["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pinhole = np.array(
        [
            [camera["fx"], 0.0, camera["cx"]],
            [0.0, camera["fy"], camera["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return intrinsic, pinhole


def compute_fisheye_mapper(camera, row_chunk=128):
    import cv2
    import numpy as np

    width, height = camera["width"], camera["height"]
    intrinsic, pinhole = camera_matrices(camera)
    mapper = np.empty((height, width, 2), dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    for row_start in range(0, height, row_chunk):
        row_end = min(row_start + row_chunk, height)
        ys = np.arange(row_start, row_end, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        distorted_pixels = np.stack((grid_x, grid_y), axis=-1).reshape(
            -1, 1, 2
        )
        pinhole_pixels = cv2.fisheye.undistortPoints(
            distorted_pixels,
            intrinsic,
            camera["distortion"],
            R=None,
            P=pinhole,
        ).reshape(row_end - row_start, width, 2)
        mapper[row_start:row_end, :, 0] = (
            2.0 * pinhole_pixels[:, :, 0] / float(width - 1) - 1.0
        )
        mapper[row_start:row_end, :, 1] = (
            2.0 * pinhole_pixels[:, :, 1] / float(height - 1) - 1.0
        )
    return mapper


def ensure_fisheye_mappers(scene_root, cameras, dry_run):
    import numpy as np

    missing = [
        camera
        for camera in cameras
        if not (scene_root / f"{camera['camera_name']}.npy").is_file()
    ]
    print(
        f"[STEGF][Preprocess] Fisheye sampling grids: "
        f"{len(cameras) - len(missing)}/{len(cameras)} complete",
        flush=True,
    )
    if dry_run:
        return
    for index, camera in enumerate(missing, start=1):
        output_path = scene_root / f"{camera['camera_name']}.npy"
        temporary = output_path.with_name(f".{output_path.stem}.writing.npy")
        mapper = compute_fisheye_mapper(camera)
        with temporary.open("wb") as output_file:
            np.save(output_file, mapper)
        temporary.replace(output_path)
        print(
            f"[STEGF][Preprocess] Fisheye grid {index}/{len(missing)}: "
            f"{camera['camera_name']} shape={list(mapper.shape)}",
            flush=True,
        )


def prepare_pinhole_inputs(
    scene_root,
    build_dirs,
    cameras,
    duration,
    png_compression,
):
    import cv2

    missing_times = set(build_dirs)
    for camera_index, camera in enumerate(cameras, start=1):
        intrinsic, pinhole = camera_matrices(camera)
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            intrinsic,
            camera["distortion"],
            R=None,
            P=pinhole,
            size=(camera["width"], camera["height"]),
            m1type=cv2.CV_32FC1,
        )
        for time_index in range(duration):
            if time_index not in missing_times:
                continue
            source_path = (
                scene_root / camera["camera_name"] / f"{time_index}.png"
            )
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read fisheye frame: {source_path}")
            undistorted = cv2.remap(
                image,
                map_x,
                map_y,
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
            )
            destination = build_dirs[time_index] / "input" / camera["name"]
            write_png_atomic(destination, undistorted, png_compression)
        print(
            f"[STEGF][Preprocess] Pinhole staging "
            f"{camera_index}/{len(cameras)}: {camera['camera_name']}",
            flush=True,
        )


def build_sparse_sequence(
    scene_root,
    cameras,
    duration,
    colmap_binary,
    restart_incomplete,
    png_compression,
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
    print(
        f"[STEGF][Preprocess] Sparse time slices: "
        f"{completed}/{duration} complete",
        flush=True,
    )
    if not missing or dry_run:
        return
    build_dirs = prepare_build_directories(
        scene_root, missing, restart_incomplete
    )
    try:
        prepare_pinhole_inputs(
            scene_root,
            build_dirs,
            cameras,
            duration,
            png_compression,
        )
        for completed_index, time_index in enumerate(missing, start=1):
            print(
                f"[STEGF][Preprocess] Sparse COLMAP "
                f"{completed_index}/{len(missing)}: time={time_index}",
                flush=True,
            )
            build_sparse_time_slice(
                colmap_binary,
                build_dirs[time_index],
                scene_root / f"colmap_{time_index}",
                cameras,
                dry_run=False,
            )
    except Exception:
        print(
            "[STEGF][Preprocess] Incomplete hidden workspaces retained "
            "for inspection; rerun with --restart_incomplete to replace them.",
            flush=True,
        )
        raise


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
            "Convert one calibrated Google Immersive fisheye scene into "
            "the complete STEGF scene layout."
        )
    )
    parser.add_argument(
        "--scene",
        required=True,
        help=(
            "Canonical STGS name (for example 04_Truck or 09_Alexa) or the "
            "official extracted scene name."
        ),
    )
    parser.add_argument(
        "--inputpath",
        "--rawpath",
        dest="input_path",
        default="dataset/Immersive",
        help="Raw-data root containing the extracted official scene.",
    )
    parser.add_argument(
        "--datapath",
        default="dataset/Immersive-pre",
        help="Output root; the script writes <datapath>/<canonical-scene>.",
    )
    parser.add_argument(
        "--output_scene",
        help="Override the canonical output directory name.",
    )
    parser.add_argument(
        "--configpath",
        help=(
            "Scene configuration. By default, use "
            "configs/immersive_ours/<scene>.json and then default.json when "
            "either exists."
        ),
    )
    parser.add_argument("--duration", type=int)
    parser.add_argument("--source_start_frame", type=int, default=0)
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument(
        "--focal_scale",
        type=float,
        help="Wide-pinhole focal multiplier; defaults to the STGS scene value.",
    )
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
    parser.add_argument("--png_compression", type=int, default=3)
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
    if args.source_start_frame < 0:
        parser.error("--source_start_frame must be >= 0")
    if args.downscale < 1:
        parser.error("--downscale must be >= 1")
    if args.focal_scale is not None and args.focal_scale <= 0:
        parser.error("--focal_scale must be positive")
    if not 0 <= args.png_compression <= 9:
        parser.error("--png_compression must be between 0 and 9")
    if args.dense_max_points < 1:
        parser.error("--dense_max_points must be >= 1")
    if args.dense_voxel_start <= 0 or args.dense_voxel_step <= 0:
        parser.error("Dense voxel sizes must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    input_path = resolve_path(args.input_path, repo_root)
    data_root = resolve_path(args.datapath, repo_root)
    try:
        canonical_scene = canonical_scene_name(args.scene)
    except ValueError as error:
        parser.error(str(error))
    output_scene = args.output_scene or canonical_scene
    raw_scene = resolve_raw_scene(input_path, args.scene, canonical_scene)
    scene_root = data_root / output_scene
    models_path = raw_scene / "models.json"

    config_path, uses_default = resolve_config(
        repo_root, canonical_scene, args.configpath
    )
    config = json.loads(config_path.read_text()) if config_path else {}
    duration = int(args.duration or config.get("duration", 50))
    if duration < 1:
        raise ValueError(f"Configured duration must be >= 1, got {duration}")
    focal_scale = float(
        args.focal_scale
        if args.focal_scale is not None
        else IMMERSIVE_SCENES[canonical_scene]["focal_scale"]
    )
    _, cameras = load_camera_records(
        models_path,
        args.downscale,
        focal_scale,
    )
    videos, ignored_videos = discover_videos(raw_scene, cameras)

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

    print("[STEGF][Preprocess] Dataset: Google Immersive", flush=True)
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
        f"[STEGF][Preprocess] Cameras={len(cameras)}, duration={duration}, "
        f"source_frames={args.source_start_frame}.."
        f"{args.source_start_frame + duration - 1}, "
        f"source_size={cameras[0]['source_width']}x"
        f"{cameras[0]['source_height']}, downscale={args.downscale}, "
        f"focal_scale={focal_scale:g}",
        flush=True,
    )
    if ignored_videos:
        print(
            "[STEGF][Preprocess] Ignoring videos absent from models.json: "
            + ",".join(ignored_videos),
            flush=True,
        )
    print(
        f"[STEGF][Preprocess] Optional stages: "
        f"dense={int(dense_enabled)}, midas={int(midas_enabled)}",
        flush=True,
    )

    if not args.dry_run:
        scene_root.mkdir(parents=True, exist_ok=True)
    output_models = copy_models_json(
        models_path,
        scene_root / "models.json",
        args.dry_run,
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

    ensure_fisheye_frames(
        scene_root,
        cameras,
        videos,
        duration,
        args.source_start_frame,
        args.downscale,
        args.png_compression,
        args.dry_run,
    )
    ensure_fisheye_mappers(scene_root, cameras, args.dry_run)
    build_sparse_sequence(
        scene_root,
        cameras,
        duration,
        args.colmap_binary,
        args.restart_incomplete,
        args.png_compression,
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
            output_scene,
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
                "dataset": "Google Immersive",
                "scene": canonical_scene,
                "official_scene": IMMERSIVE_SCENES[canonical_scene][
                    "official_name"
                ],
                "output_scene": output_scene,
                "raw_scene": str(raw_scene),
                "scene_root": str(scene_root),
                "config": str(config_path) if config_path else "",
                "uses_default_config": uses_default,
                "duration": duration,
                "source_start_frame": args.source_start_frame,
                "source_end_frame": args.source_start_frame + duration - 1,
                "source_image_size": [
                    cameras[0]["source_width"],
                    cameras[0]["source_height"],
                ],
                "published_image_size": [
                    cameras[0]["width"],
                    cameras[0]["height"],
                ],
                "downscale": args.downscale,
                "focal_scale": focal_scale,
                "camera_names": [
                    camera["camera_name"] for camera in cameras
                ],
                "ignored_uncalibrated_videos": ignored_videos,
                "test_camera": sorted(
                    camera["camera_name"] for camera in cameras
                )[0],
                "models_json": str(output_models),
                "distorted_supervision": True,
                "fisheye_mapper_format": "grid_sample_xy_normalized",
                "fisheye_radial_coefficients_used": 2,
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
