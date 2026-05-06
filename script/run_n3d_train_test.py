#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SCENES = ("coffee_martini", "cook_spinach")


def build_train_command(args, scene, repo_root):
    return [
        sys.executable,
        str(repo_root / "train.py"),
        "--quiet",
        "--eval",
        "--configpath",
        str(Path(args.config_dir) / f"{scene}.json"),
        "--model_path",
        str(Path(args.output_root) / scene),
        "--source_path",
        str(Path(args.data_root) / scene / args.colmap_subdir),
        "--save_iterations",
        str(args.save_iteration),
    ]


def build_test_command(args, scene, repo_root):
    return [
        sys.executable,
        str(repo_root / "script" / "test_all_iterations.py"),
        "--quiet",
        "--eval",
        "--skip_train",
        "--valloader",
        args.valloader,
        "--configpath",
        str(Path(args.config_dir) / f"{scene}.json"),
        "--model_path",
        str(Path(args.output_root) / scene),
        "--source_path",
        str(Path(args.data_root) / scene / args.colmap_subdir),
    ]


def run_command(cmd, repo_root, dry_run):
    print("\n[STEGF] " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=repo_root).returncode


def validate_paths(args, scenes, repo_root, dry_run=False):
    missing = []
    for scene in scenes:
        config_path = repo_root / args.config_dir / f"{scene}.json"
        source_path = Path(args.data_root) / scene / args.colmap_subdir
        if not config_path.is_file():
            missing.append(str(config_path))
        if (not dry_run) and (not source_path.is_dir()):
            missing.append(str(source_path))
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def main():
    parser = argparse.ArgumentParser(
        description="Train and test the default DyNeRF/N3D STEGF scenes with one command."
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=list(DEFAULT_SCENES),
        help="Scene names to run sequentially.",
    )
    parser.add_argument(
        "--scene",
        choices=DEFAULT_SCENES,
        help="Run a single default scene. This is a convenience alias for --scenes <scene>.",
    )
    parser.add_argument("--data_root", default="/root/autodl-tmp", help="Root containing scene folders.")
    parser.add_argument("--output_root", default="/root/autodl-tmp/output", help="Root for scene outputs.")
    parser.add_argument("--config_dir", default="configs/n3d_ours", help="Directory containing <scene>.json configs.")
    parser.add_argument("--colmap_subdir", default="colmap_0", help="Scene subdirectory used as --source_path.")
    parser.add_argument("--save_iteration", type=int, default=30000, help="Single checkpoint iteration to save.")
    parser.add_argument("--valloader", default="colmapvalid", help="Validation loader passed to test_all_iterations.py.")
    parser.add_argument("--skip_train_stage", action="store_true", help="Only run testing.")
    parser.add_argument("--skip_test_stage", action="store_true", help="Only run training.")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue with later stages after a failure.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    scenes = [args.scene] if args.scene else list(args.scenes)
    validate_paths(args, scenes, repo_root, dry_run=args.dry_run)

    failures = []
    for scene in scenes:
        print(f"\n[STEGF] ===== Scene: {scene} =====", flush=True)

        if not args.skip_train_stage:
            code = run_command(build_train_command(args, scene, repo_root), repo_root, args.dry_run)
            if code != 0:
                failures.append((scene, "train", code))
                if not args.continue_on_error:
                    break

        if not args.skip_test_stage:
            code = run_command(build_test_command(args, scene, repo_root), repo_root, args.dry_run)
            if code != 0:
                failures.append((scene, "test", code))
                if not args.continue_on_error:
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
