#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SCENES = ("coffee_martini",)


def get_model_path(args, scene, repeat_idx=None):
    couptest_mode = getattr(args, "configured_couptest_mode", "none")
    couptest_suffix = couptest_mode.replace("_", "-")
    scene_output = (
        f"{scene}-{couptest_suffix}"
        if couptest_mode != "none"
        else scene
    )
    if repeat_idx is not None:
        scene_output = f"{scene_output}-re{repeat_idx}"
    return Path(args.output_root) / scene_output


def build_train_command(args, scene, repo_root, model_path):
    cmd = [
        sys.executable,
        str(repo_root / "train.py"),
        "--quiet",
        "--eval",
        "--configpath",
        str(Path(args.config_dir) / f"{scene}.json"),
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


def build_test_command(args, scene, repo_root, model_path):
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
        str(Path(args.config_dir) / f"{scene}.json"),
        "--model_path",
        str(model_path),
        "--source_path",
        str(Path(args.data_root) / scene / args.colmap_subdir),
    ]
    return cmd


def get_configured_couptest_mode(args, scene, repo_root):
    config_path = repo_root / args.config_dir / f"{scene}.json"
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
        couptest_mode = get_configured_couptest_mode(
            args, scene, repo_root
        )
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
