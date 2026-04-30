#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


ITERATION_PATTERN = re.compile(r"^iteration_(\d+)$")


def parse_iteration_filter(spec):
    if not spec:
        return None

    selected = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid iteration range: {chunk}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(chunk))
    return selected


def discover_iterations(model_path):
    point_cloud_dir = Path(model_path) / "point_cloud"
    if not point_cloud_dir.is_dir():
        raise FileNotFoundError(f"Point cloud directory not found: {point_cloud_dir}")

    iterations = []
    for child in point_cloud_dir.iterdir():
        if not child.is_dir():
            continue
        match = ITERATION_PATTERN.match(child.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        ply_path = child / "point_cloud.ply"
        if ply_path.is_file():
            iterations.append(iteration)

    iterations.sort()
    return iterations


def build_test_command(python_exe, repo_root, forwarded_args, iteration):
    cmd = [python_exe, str(repo_root / "test.py")]
    cmd.extend(forwarded_args)
    cmd.extend(["--test_iteration", str(iteration)])
    return cmd


def has_existing_results(model_path, iteration):
    result_path = Path(model_path) / f"{iteration}_runtimeresults.json"
    perview_path = Path(model_path) / f"{iteration}_runtimeperview.json"
    return result_path.is_file() and perview_path.is_file()


def main():
    parser = argparse.ArgumentParser(description="Run test.py for every saved checkpoint iteration.")
    parser.add_argument("--model_path", required=True, help="Model output directory containing point_cloud/iteration_*.")
    parser.add_argument(
        "--iterations",
        default="",
        help="Optional iteration filter. Examples: 7000,10000,30000 or 7000-12000,30000",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip iterations that already have both *_runtimeresults.json and *_runtimeperview.json.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately if any iteration test command fails.",
    )

    args, forwarded_args = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable
    sanitized_forwarded_args = []
    skip_next = False
    for idx, arg in enumerate(forwarded_args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--test_iteration":
            skip_next = True
            continue
        if arg.startswith("--test_iteration="):
            continue
        sanitized_forwarded_args.append(arg)
    forwarded_args = ["--model_path", args.model_path] + sanitized_forwarded_args

    selected = parse_iteration_filter(args.iterations)
    iterations = discover_iterations(args.model_path)
    if selected is not None:
        iterations = [iteration for iteration in iterations if iteration in selected]

    if not iterations:
        raise RuntimeError("No checkpoint iterations matched the requested filter.")

    exit_code = 0
    print(f"Discovered iterations: {iterations}")

    for iteration in iterations:
        if args.skip_existing and has_existing_results(args.model_path, iteration):
            print(f"[skip] iteration {iteration}: results already exist")
            continue

        cmd = build_test_command(python_exe, repo_root, forwarded_args, iteration)
        print(f"[run] iteration {iteration}: {' '.join(cmd)}")

        if args.dry_run:
            continue

        completed = subprocess.run(cmd, cwd=repo_root)
        if completed.returncode != 0:
            exit_code = completed.returncode
            print(f"[fail] iteration {iteration} exited with code {completed.returncode}")
            if args.fail_fast:
                break
        else:
            print(f"[done] iteration {iteration}")

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
