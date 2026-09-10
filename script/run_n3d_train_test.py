#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SCENES = ("coffee_martini",)


def get_model_path(args, scene, repeat_idx=None):
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else Path(args.data_root) / "output"
    )
    scene_output = scene
    if repeat_idx is not None:
        scene_output = f"{scene_output}-re{repeat_idx}"
    return output_root / scene_output


def build_train_command(args, scene, repo_root, model_path, config_path):
    cmd = [
        sys.executable,
        str(repo_root / "train.py"),
        "--quiet",
        "--eval",
        "--configpath",
        str(config_path),
        "--model_path",
        str(model_path),
        "--source_path",
        str(Path(args.data_root) / scene / args.colmap_subdir),
        "--save_iterations",
    ]
    cmd.extend(str(iteration) for iteration in args.save_iterations)
    if args.iterations is not None:
        cmd.extend(["--iterations", str(args.iterations)])
    return cmd


def build_test_command(args, scene, repo_root, model_path, config_path):
    cmd = [
        sys.executable,
        str(repo_root / "script" / "test_all_iterations.py"),
        "--iterations",
        ",".join(str(iteration) for iteration in args.test_iterations),
        "--quiet",
        "--eval",
        "--skip_train",
        "--valloader",
        args.valloader,
        "--configpath",
        str(config_path),
        "--model_path",
        str(model_path),
        "--source_path",
        str(Path(args.data_root) / scene / args.colmap_subdir),
    ]
    return cmd


def resolve_scene_config(args, scene, repo_root):
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = repo_root / config_dir
    scene_config = config_dir / f"{scene}.json"
    if scene_config.is_file():
        return scene_config, False
    default_config = config_dir / "default.json"
    if default_config.is_file():
        return default_config, True
    raise FileNotFoundError(
        "No configuration found for scene {!r}: expected {} or {}"
        .format(scene, scene_config, default_config)
    )


def get_configured_couptest_mode(config_path):
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    configured_mode = str(config.get("field_couptest_mode", "none"))
    if configured_mode != "coupled_detached":
        raise ValueError(
            f"The final runner requires field_couptest_mode="
            f"'coupled_detached' in {config_path}, got "
            f"{configured_mode!r}"
        )
    return configured_mode


def get_scene_config(config_path):
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def resolve_dense_path(source_path, configured_path):
    path = Path(str(configured_path).strip())
    if path.is_absolute():
        return path
    return (source_path / path).resolve()


def parse_time_indices(value, duration):
    indices = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            indices.append(max(0, min(int(item), duration - 1)))
    return sorted(set(indices or [0]))


def validate_scene_data(scene, source_path, config, repo_root):
    missing = []
    scene_root = source_path.parent
    duration = int(config.get("duration", 50))
    poses_path = scene_root / "poses_bounds.npy"
    if not poses_path.is_file():
        missing.append(str(poses_path))
    for time_index in range(duration):
        colmap_path = scene_root / f"colmap_{time_index}"
        image_path = colmap_path / "images"
        points_path = colmap_path / "sparse" / "0" / "points3D.bin"
        if not image_path.is_dir():
            missing.append(str(image_path))
        if not points_path.is_file():
            missing.append(str(points_path))
    for filename in ("cameras.bin", "images.bin"):
        path = source_path / "sparse" / "0" / filename
        if not path.is_file():
            missing.append(str(path))

    if int(config.get("field_dense_initialization", 0)) != 0:
        configured_path = config.get("field_dense_initialization_path", "")
        if not str(configured_path).strip():
            missing.append(
                f"{scene}: field_dense_initialization_path is empty"
            )
        else:
            dense_path = resolve_dense_path(source_path, configured_path)
            if not dense_path.is_file():
                missing.append(str(dense_path))

    needs_midas = (
        int(config.get("field_bg_dense_add", 0)) != 0
        and int(config.get("field_bg_dense_beit_filter", 0)) != 0
    )
    if needs_midas:
        configured_path = str(config.get("field_bg_dense_beit_path", "")).strip()
        candidates = []
        if configured_path:
            candidate = Path(configured_path)
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            candidates.append(candidate)
        candidates.extend(
            (
                scene_root / "midas_beit_large_512",
                source_path / "midas_beit_large_512",
            )
        )
        midas_root = next(
            (path for path in candidates if path.is_dir()), None
        )
        if midas_root is None:
            missing.append(
                "{} (required by field_bg_dense_add=1 and "
                "field_bg_dense_beit_filter=1)".format(candidates[0])
            )
        else:
            time_indices = parse_time_indices(
                config.get("field_bg_dense_add_time_indices", "0"),
                duration,
            )
            for time_index in time_indices:
                image_dir = scene_root / f"colmap_{time_index}" / "images"
                for image_path in sorted(image_dir.glob("*.png")):
                    nested = (
                        midas_root
                        / f"colmap_{time_index}"
                        / image_path.stem
                        / "raw_depth_like.npy"
                    )
                    legacy = (
                        midas_root / image_path.stem / "raw_depth_like.npy"
                    )
                    if not nested.is_file() and not (
                        time_index == 0 and legacy.is_file()
                    ):
                        missing.append(str(nested))
    return missing


def run_command(cmd, repo_root, dry_run):
    print("\n[STEGF] " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=repo_root).returncode


def validate_paths(args, scenes, repo_root, dry_run=False):
    missing = []
    for scene in scenes:
        source_path = Path(args.data_root) / scene / args.colmap_subdir
        try:
            config_path, _ = resolve_scene_config(args, scene, repo_root)
        except FileNotFoundError as error:
            missing.append(str(error))
            continue
        if not dry_run:
            if not source_path.is_dir():
                missing.append(str(source_path))
            else:
                missing.extend(
                    validate_scene_data(
                        scene,
                        source_path,
                        get_scene_config(config_path),
                        repo_root,
                    )
                )
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def main():
    parser = argparse.ArgumentParser(
        description="Train and test DyNeRF/N3D STEGF scenes with one command."
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=list(DEFAULT_SCENES),
        help="Scene names to run sequentially.",
    )
    parser.add_argument(
        "--scene",
        help=(
            "Run one scene. Uses <scene>.json when present, otherwise "
            "default.json."
        ),
    )
    parser.add_argument(
        "--datapath",
        "--data_root",
        dest="data_root",
        default="/root/autodl-tmp",
        help="Root containing scene folders (default: /root/autodl-tmp).",
    )
    parser.add_argument(
        "--savepath",
        "--output_root",
        dest="output_root",
        default=None,
        help=(
            "Root containing per-scene outputs. Defaults to "
            "<datapath>/output."
        ),
    )
    parser.add_argument("--config_dir", default="configs/n3d_ours", help="Directory containing <scene>.json configs.")
    parser.add_argument("--colmap_subdir", default="colmap_0", help="Scene subdirectory used as --source_path.")
    parser.add_argument(
        "--iterations",
        type=int,
        help="Total training iterations passed to train.py. Defaults to train.py/config value.",
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[30000],
        help="Checkpoint iterations to save during training.",
    )
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[30000],
        help="Checkpoint iterations to test after training.",
    )
    parser.add_argument("--valloader", default="colmapvalid", help="Validation loader passed to test_all_iterations.py.")
    parser.add_argument("--skip_train_stage", action="store_true", help="Only run testing.")
    parser.add_argument("--skip_test_stage", action="store_true", help="Only run training.")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue with later stages after a failure.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--re",
        type=int,
        default=1,
        help="Repeat each requested scene N times. N=1 keeps the original output path; N>1 writes <scene>-re1, <scene>-re2, ...",
    )
    args = parser.parse_args()
    if args.re < 1:
        parser.error("--re must be >= 1")

    repo_root = Path(__file__).resolve().parents[1]
    scenes = [args.scene] if args.scene else list(args.scenes)
    validate_paths(args, scenes, repo_root, dry_run=args.dry_run)

    failures = []
    for scene in scenes:
        config_path, uses_default_config = resolve_scene_config(
            args, scene, repo_root
        )
        if uses_default_config:
            print(
                f"[STEGF] Scene config not found for {scene}; "
                f"using defaults from {config_path}",
                flush=True,
            )
        couptest_mode = get_configured_couptest_mode(config_path)
        args.configured_couptest_mode = couptest_mode
        for repeat_idx in range(1, args.re + 1):
            repeat_suffix = None if args.re == 1 else repeat_idx
            model_path = get_model_path(args, scene, repeat_suffix)
            if args.re == 1:
                variant = (
                    f" | couptest {couptest_mode}"
                    if couptest_mode != "none"
                    else ""
                )
                print(
                    f"\n[STEGF] ===== Scene: {scene}{variant} =====",
                    flush=True,
                )
            else:
                print(
                    f"\n[STEGF] ===== Scene: {scene} | "
                    f"couptest {couptest_mode} | repeat "
                    f"{repeat_idx}/{args.re} =====",
                    flush=True,
                )
                print(f"[STEGF] Repeat output: {model_path}", flush=True)

            failure_label = model_path.name
            if not args.skip_train_stage:
                code = run_command(
                    build_train_command(
                        args,
                        scene,
                        repo_root,
                        model_path,
                        config_path,
                    ),
                    repo_root,
                    args.dry_run,
                )
                if code != 0:
                    failures.append((failure_label, "train", code))
                    if not args.continue_on_error:
                        break

            if not args.skip_test_stage:
                code = run_command(
                    build_test_command(
                        args,
                        scene,
                        repo_root,
                        model_path,
                        config_path,
                    ),
                    repo_root,
                    args.dry_run,
                )
                if code != 0:
                    failures.append((failure_label, "test", code))
                    if not args.continue_on_error:
                        break
        if failures and not args.continue_on_error:
            break

    if failures:
        print("\n[STEGF] Failed stages:", flush=True)
        for scene, stage, code in failures:
            print(f"  {scene} {stage}: exit code {code}", flush=True)
        return 1

    print("\n[STEGF] All requested stages completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
