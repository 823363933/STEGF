#!/usr/bin/env python3
"""Build one STEGF scene from calibrated N3D/DyNeRF-style videos.

The sparse COLMAP preparation follows the data conversion used by
SpacetimeGaussians.  Frame-0 dense reconstruction and iterative voxel
downsampling follow the E-D3DGS preprocessing path.  Optional MiDaS-BEiT
inference is delegated to ``precompute_midas_beit_depth_like.py``.

Raw input is expected to contain one ``cam*.mp4`` per calibrated camera and a
matching ``poses_bounds.npy`` in LLFF/N3D format.  The script is resumable at
completed time slices and never removes a published ``colmap_<time>`` result.
"""

# SpacetimeGaussians and the E-D3DGS preprocessing scripts are Copyright
# (c) 2023 OPPO and distributed under the MIT License. The COLMAP database
# layout follows COLMAP's scripts/python/database.py helper.

import argparse
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path


np = None


MAX_IMAGE_ID = 2**31 - 1
MIDAS_DEFAULT_COMMIT = "454597711a62eabcbf7d1e89f3fb9f569051ac9b"

CREATE_DATABASE = """
CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < %d),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB);
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB);
CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);
""" % MAX_IMAGE_ID


def require_numpy():
    global np
    if np is None:
        import numpy as numpy_module

        np = numpy_module
    return np


def resolve_path(value, base):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def resolve_scene_input(input_path, scene):
    candidate = input_path / scene
    if candidate.is_dir():
        return candidate
    if input_path.is_dir() and input_path.name == scene:
        return input_path
    raise FileNotFoundError(
        f"Raw scene not found: expected {candidate} or a direct scene path"
    )


def resolve_config(repo_root, scene, explicit_path):
    if explicit_path:
        path = resolve_path(explicit_path, repo_root)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration not found: {path}")
        return path, False
    scene_path = repo_root / "configs" / "n3d_ours" / f"{scene}.json"
    if scene_path.is_file():
        return scene_path, False
    default_path = repo_root / "configs" / "n3d_ours" / "default.json"
    if not default_path.is_file():
        raise FileNotFoundError(
            f"No scene configuration or default configuration found for {scene}"
        )
    return default_path, True


def feature_enabled(mode, configured):
    if mode == "on":
        return True
    if mode == "off":
        return False
    return bool(configured)


def llff_poses_to_world_to_camera(llff_poses):
    require_numpy()
    reordered = np.concatenate(
        (
            llff_poses[:, 1:2, :],
            llff_poses[:, 0:1, :],
            -llff_poses[:, 2:3, :],
            llff_poses[:, 3:4, :],
            llff_poses[:, 4:5, :],
        ),
        axis=1,
    )
    matrices = reordered[:, 0:4, :].transpose(2, 0, 1)
    bottom = np.zeros((matrices.shape[0], 1, 4), dtype=matrices.dtype)
    bottom[:, 0, 3] = 1
    return list(np.linalg.inv(np.concatenate((matrices, bottom), axis=1)))


def rotation_matrix_to_qvec(rotation):
    require_numpy()
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    matrix = np.array(
        [
            [rxx - ryy - rzz, 0, 0, 0],
            [ryx + rxy, ryy - rxx - rzz, 0, 0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ]
    ) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    qvec = eigenvectors[[3, 0, 1, 2], np.argmax(eigenvalues)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def load_camera_records(poses_path, video_paths, downscale):
    require_numpy()
    poses_bounds = np.load(poses_path)
    if poses_bounds.ndim != 2 or poses_bounds.shape[1] < 15:
        raise ValueError(
            f"Expected poses_bounds.npy with shape [cameras, >=15], got "
            f"{poses_bounds.shape}"
        )
    if poses_bounds.shape[0] != len(video_paths):
        raise ValueError(
            "Camera count mismatch: poses_bounds.npy has {} rows, but {} "
            "cam*.mp4 files were found".format(
                poses_bounds.shape[0], len(video_paths)
            )
        )
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    world_to_camera = llff_poses_to_world_to_camera(
        poses.copy().transpose(1, 2, 0)
    )
    records = []
    for index, (pose, matrix, video_path) in enumerate(
        zip(poses, world_to_camera, video_paths)
    ):
        height, width, focal = pose[:, -1] / float(downscale)
        records.append(
            {
                "id": index + 1,
                "name": f"{video_path.stem}.png",
                "width": int(float(width)),
                "height": int(float(height)),
                "fx": float(focal),
                "fy": float(focal),
                "cx": float(width // 2),
                "cy": float(height // 2),
                "qvec": rotation_matrix_to_qvec(matrix[:3, :3]),
                "tvec": matrix[:3, 3],
            }
        )
    return records


def array_blob(values):
    require_numpy()
    return np.asarray(values, dtype=np.float64).tobytes()


def write_manual_model_and_database(work_dir, cameras):
    manual_dir = work_dir / "manual"
    manual_dir.mkdir(parents=True, exist_ok=True)
    image_lines = []
    camera_lines = []
    for camera in cameras:
        image_lines.append(
            "{} {} {} {} {} {} {} {} {} {}\n\n".format(
                camera["id"],
                *camera["qvec"],
                *camera["tvec"],
                camera["id"],
                camera["name"],
            )
        )
        camera_lines.append(
            "{} PINHOLE {} {} {} {} {} {}\n".format(
                camera["id"],
                camera["width"],
                camera["height"],
                camera["fx"],
                camera["fy"],
                camera["cx"],
                camera["cy"],
            )
        )
    (manual_dir / "images.txt").write_text("".join(image_lines))
    (manual_dir / "cameras.txt").write_text("".join(camera_lines))
    (manual_dir / "points3D.txt").write_text("")

    database_path = work_dir / "input.db"
    if database_path.exists():
        database_path.unlink()
    database = sqlite3.connect(str(database_path))
    try:
        database.executescript(CREATE_DATABASE)
        for camera in cameras:
            database.execute(
                "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
                (
                    camera["id"],
                    1,
                    camera["width"],
                    camera["height"],
                    array_blob(
                        (
                            camera["fx"],
                            camera["fy"],
                            camera["cx"],
                            camera["cy"],
                        )
                    ),
                    0,
                ),
            )
            database.execute(
                "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    camera["id"],
                    camera["name"],
                    camera["id"],
                    *camera["qvec"],
                    *camera["tvec"],
                ),
            )
        database.commit()
    finally:
        database.close()


def run_command(command, dry_run=False):
    print(f"[STEGF][Preprocess] {shlex.join(map(str, command))}", flush=True)
    if dry_run:
        return
    subprocess.run([str(item) for item in command], check=True)


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


def time_slice_complete(path, camera_count):
    image_count = len(list((path / "images").glob("*.png")))
    sparse_dir = path / "sparse" / "0"
    required = (
        sparse_dir / "cameras.bin",
        sparse_dir / "images.bin",
        sparse_dir / "points3D.bin",
    )
    return image_count == camera_count and all(item.is_file() for item in required)


def prepare_build_directories(scene_root, time_indices, restart_incomplete):
    build_dirs = {}
    for time_index in time_indices:
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
        build_dirs[time_index] = build_dir
    return build_dirs


def extract_requested_frames(
    video_paths,
    build_dirs,
    duration,
    source_start_frame,
    downscale,
    cameras,
):
    import cv2
    from tqdm import tqdm

    expected_sizes = {
        Path(camera["name"]).stem: (
            camera["height"],
            camera["width"],
        )
        for camera in cameras
    }
    for video_path in tqdm(video_paths, desc="Extracting calibrated video frames"):
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
                if time_index not in build_dirs:
                    continue
                if downscale > 1:
                    new_size = (
                        int(frame.shape[1] / float(downscale)),
                        int(frame.shape[0] / float(downscale)),
                    )
                    frame = cv2.resize(
                        frame, new_size, interpolation=cv2.INTER_AREA
                    )
                expected_size = expected_sizes[video_path.stem]
                if frame.shape[:2] != expected_size:
                    raise RuntimeError(
                        f"Video/calibration size mismatch for {video_path}: "
                        f"decoded={frame.shape[1]}x{frame.shape[0]}, "
                        f"poses_bounds={expected_size[1]}x{expected_size[0]}"
                    )
                output_path = build_dirs[time_index] / "input" / (
                    f"{video_path.stem}.png"
                )
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Failed to write image: {output_path}")
        finally:
            capture.release()


def nest_sparse_model(work_dir):
    sparse_dir = work_dir / "sparse"
    nested_dir = sparse_dir / "0"
    nested_dir.mkdir(exist_ok=True)
    for child in list(sparse_dir.iterdir()):
        if child == nested_dir:
            continue
        shutil.move(str(child), str(nested_dir / child.name))


def build_sparse_time_slice(
    colmap_binary,
    work_dir,
    final_dir,
    cameras,
    dry_run,
):
    write_manual_model_and_database(work_dir, cameras)
    database_path = work_dir / "input.db"
    input_dir = work_dir / "input"
    distorted_dir = work_dir / "distorted" / "sparse"
    distorted_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        (
            colmap_binary,
            "feature_extractor",
            "--database_path",
            database_path,
            "--image_path",
            input_dir,
        ),
        dry_run,
    )
    run_command(
        (colmap_binary, "exhaustive_matcher", "--database_path", database_path),
        dry_run,
    )
    run_command(
        (
            colmap_binary,
            "point_triangulator",
            "--database_path",
            database_path,
            "--image_path",
            input_dir,
            "--output_path",
            distorted_dir,
            "--input_path",
            work_dir / "manual",
            "--Mapper.ba_global_function_tolerance",
            "0.000001",
        ),
        dry_run,
    )
    run_command(
        (
            colmap_binary,
            "image_undistorter",
            "--image_path",
            input_dir,
            "--input_path",
            distorted_dir,
            "--output_path",
            work_dir,
            "--output_type",
            "COLMAP",
        ),
        dry_run,
    )
    if dry_run:
        return
    nest_sparse_model(work_dir)
    if not time_slice_complete(work_dir, len(cameras)):
        raise RuntimeError(f"COLMAP time slice is incomplete: {work_dir}")
    shutil.rmtree(input_dir)
    work_dir.rename(final_dir)


def build_sparse_sequence(
    scene_root,
    video_paths,
    cameras,
    duration,
    source_start_frame,
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
    if dry_run:
        print(
            f"[STEGF][Preprocess] Would build {len(missing)} sparse time "
            f"slices: {missing[0]}..{missing[-1]}",
            flush=True,
        )
        return

    build_dirs = prepare_build_directories(
        scene_root, missing, restart_incomplete
    )
    extract_requested_frames(
        video_paths,
        build_dirs,
        duration,
        source_start_frame,
        downscale,
        cameras,
    )
    for completed, time_index in enumerate(missing, start=1):
        print(
            f"[STEGF][Preprocess] Sparse COLMAP {completed}/{len(missing)}: "
            f"time={time_index}",
            flush=True,
        )
        build_sparse_time_slice(
            colmap_binary,
            build_dirs[time_index],
            scene_root / f"colmap_{time_index}",
            cameras,
            dry_run=False,
        )


def prepare_dense_workspace(scene_root, restart_incomplete):
    source_workspace = scene_root / "colmap_0"
    dense_workspace = scene_root / ".stegf_dense_workspace"
    if dense_workspace.exists() and restart_incomplete:
        shutil.rmtree(dense_workspace)
    dense_workspace.mkdir(parents=True, exist_ok=True)

    images_link = dense_workspace / "images"
    source_images = source_workspace / "images"
    if not images_link.exists():
        images_link.symlink_to(source_images.resolve(), target_is_directory=True)
    elif not images_link.is_dir():
        raise RuntimeError(
            f"Invalid dense image workspace entry: {images_link}"
        )

    dense_sparse = dense_workspace / "sparse"
    dense_sparse.mkdir(exist_ok=True)
    source_sparse = source_workspace / "sparse" / "0"
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        source_path = source_sparse / filename
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Frame-0 sparse model is incomplete: {source_path}"
            )
        destination = dense_sparse / filename
        if not destination.exists():
            destination.symlink_to(source_path.resolve())
        elif not destination.is_file():
            raise RuntimeError(
                f"Invalid dense sparse-model workspace entry: {destination}"
            )

    dense_stereo = dense_workspace / "stereo"
    dense_stereo.mkdir(exist_ok=True)
    for dirname in ("depth_maps", "normal_maps", "consistency_graphs"):
        (dense_stereo / dirname).mkdir(exist_ok=True)
    source_stereo = source_workspace / "stereo"
    for filename in ("patch-match.cfg", "fusion.cfg"):
        source_path = source_stereo / filename
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Frame-0 COLMAP stereo configuration is missing: {source_path}"
            )
        destination = dense_stereo / filename
        if not destination.exists():
            shutil.copy2(source_path, destination)
    return dense_workspace


def build_dense_points(
    scene_root,
    colmap_binary,
    gpu_index,
    max_points,
    voxel_start,
    voxel_step,
    keep_workspace,
    restart_incomplete,
    dry_run,
):
    output_path = scene_root / "dense_points.ply"
    if output_path.is_file():
        print(
            f"[STEGF][Preprocess] Dense point cloud already exists: "
            f"{output_path}",
            flush=True,
        )
        return
    workspace = scene_root / ".stegf_dense_workspace"
    if not dry_run:
        workspace = prepare_dense_workspace(
            scene_root, restart_incomplete
        )
    fused_path = workspace / "fused.ply"
    if not dry_run:
        # A failed dense run may leave valid-looking but incomplete maps behind.
        # COLMAP skips outputs that already exist, so always rebuild this hidden
        # scratch stage as one complete geometric-consistency pass.
        for dirname in ("depth_maps", "normal_maps", "consistency_graphs"):
            output_dir = workspace / "stereo" / dirname
            if output_dir.is_dir():
                shutil.rmtree(output_dir)
            output_dir.mkdir()
        if fused_path.exists():
            fused_path.unlink()
    run_command(
        (
            colmap_binary,
            "patch_match_stereo",
            "--workspace_path",
            workspace,
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.gpu_index",
            gpu_index,
            "--PatchMatchStereo.geom_consistency",
            "true",
            "--PatchMatchStereo.filter",
            "true",
        ),
        dry_run,
    )
    run_command(
        (
            colmap_binary,
            "stereo_fusion",
            "--workspace_path",
            workspace,
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            fused_path,
        ),
        dry_run,
    )
    if dry_run:
        return

    import open3d as o3d

    point_cloud = o3d.io.read_point_cloud(str(fused_path))
    initial_count = len(point_cloud.points)
    if initial_count == 0:
        raise RuntimeError(f"COLMAP dense fusion produced no points: {fused_path}")
    voxel_size = float(voxel_start)
    while len(point_cloud.points) > max_points:
        point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
        print(
            f"[STEGF][Preprocess] Dense downsample: voxel={voxel_size:g}, "
            f"points={len(point_cloud.points)}",
            flush=True,
        )
        voxel_size += float(voxel_step)
    temporary_output = output_path.parent / ".dense_points.writing.ply"
    if not o3d.io.write_point_cloud(str(temporary_output), point_cloud):
        raise RuntimeError(f"Failed to write dense point cloud: {temporary_output}")
    temporary_output.replace(output_path)
    print(
        f"[STEGF][Preprocess] Dense frame-0 point cloud: "
        f"{initial_count} -> {len(point_cloud.points)} points",
        flush=True,
    )
    if not keep_workspace:
        shutil.rmtree(workspace)
        source_workspace = scene_root / "colmap_0"
        for generated_dir in (
            source_workspace / "stereo" / "depth_maps",
            source_workspace / "stereo" / "normal_maps",
        ):
            if generated_dir.is_dir():
                shutil.rmtree(generated_dir)


def build_midas_depth(
    repo_root,
    scene,
    data_root,
    time_indices,
    midas_repo,
    model_weights,
    device,
    overwrite,
    dry_run,
):
    helper = repo_root / "script" / "precompute_midas_beit_depth_like.py"
    command = [
        sys.executable,
        helper,
        "--scene",
        scene,
        "--datapath",
        data_root,
        "--time_indices",
        time_indices,
        "--midas_repo",
        midas_repo,
        "--device",
        device,
    ]
    if model_weights:
        command.extend(("--model_weights", model_weights))
    if overwrite:
        command.append("--overwrite")
    if dry_run:
        command.append("--dry_run")
    run_command(command, dry_run=dry_run)


def copy_calibration(raw_scene, scene_root, dry_run):
    require_numpy()
    source = raw_scene / "poses_bounds.npy"
    if not source.is_file():
        raise FileNotFoundError(f"Required calibration not found: {source}")
    destination = scene_root / "poses_bounds.npy"
    if destination.exists():
        if source.resolve() != destination.resolve():
            source_array = np.load(source)
            destination_array = np.load(destination)
            if not np.array_equal(source_array, destination_array):
                raise RuntimeError(
                    f"Existing calibration differs from raw input: {destination}"
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


def write_manifest(path, payload):
    temporary = path.with_suffix(".json.writing")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert one calibrated N3D/DyNeRF-style video scene into the "
            "complete STEGF scene layout."
        )
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--inputpath",
        "--rawpath",
        dest="input_path",
        default="dataset/DyNeRF",
        help="Raw-data root containing <scene>, or the raw scene itself.",
    )
    parser.add_argument(
        "--datapath",
        default="/root/autodl-tmp",
        help="Output data root; the script writes <datapath>/<scene>.",
    )
    parser.add_argument(
        "--configpath",
        help=(
            "Scene configuration. By default, use <scene>.json and then "
            "default.json."
        ),
    )
    parser.add_argument("--duration", type=int)
    parser.add_argument("--source_start_frame", type=int, default=0)
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
    if args.source_start_frame < 0:
        parser.error("--source_start_frame must be >= 0")
    if args.downscale < 1:
        parser.error("--downscale must be >= 1")
    if args.dense_max_points < 1:
        parser.error("--dense_max_points must be >= 1")
    if args.dense_voxel_start <= 0 or args.dense_voxel_step <= 0:
        parser.error("Dense voxel sizes must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    input_path = resolve_path(args.input_path, repo_root)
    data_root = resolve_path(args.datapath, repo_root)
    raw_scene = resolve_scene_input(input_path, args.scene)
    scene_root = data_root / args.scene
    config_path, uses_default = resolve_config(
        repo_root, args.scene, args.configpath
    )
    config = json.loads(config_path.read_text())
    duration = int(args.duration or config.get("duration", 50))
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

    video_paths = sorted(raw_scene.glob("cam*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"No cam*.mp4 files found in {raw_scene}")
    calibration_path = raw_scene / "poses_bounds.npy"
    cameras = load_camera_records(calibration_path, video_paths, args.downscale)

    print(f"[STEGF][Preprocess] Raw scene: {raw_scene}", flush=True)
    print(f"[STEGF][Preprocess] Output scene: {scene_root}", flush=True)
    print(
        f"[STEGF][Preprocess] Config: {config_path}"
        f"{' (default fallback)' if uses_default else ''}",
        flush=True,
    )
    print(
        f"[STEGF][Preprocess] Cameras={len(cameras)}, duration={duration}, "
        f"source_frames={args.source_start_frame}.."
        f"{args.source_start_frame + duration - 1}, downscale={args.downscale}",
        flush=True,
    )
    print(
        f"[STEGF][Preprocess] Optional stages: "
        f"dense={int(dense_enabled)}, midas={int(midas_enabled)}",
        flush=True,
    )

    output_calibration = copy_calibration(
        raw_scene, scene_root, args.dry_run
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
        video_paths,
        cameras,
        duration,
        args.source_start_frame,
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
                "scene": args.scene,
                "raw_scene": str(raw_scene),
                "scene_root": str(scene_root),
                "config": str(config_path),
                "uses_default_config": uses_default,
                "duration": duration,
                "source_start_frame": args.source_start_frame,
                "downscale": args.downscale,
                "camera_names": [path.stem for path in video_paths],
                "poses_bounds": str(output_calibration),
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
