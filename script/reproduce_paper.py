#!/usr/bin/env python3
"""Train and evaluate a selected paper checkpoint without packaging any data."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from run_n3d_train_test import DATASET_PROFILES


ROOT = Path(__file__).resolve().parents[1]


def build_commands(profile_name, profile, scene, datapath, savepath):
    entry = profile["scenes"][scene]
    dataset = profile["dataset"]
    valloader = DATASET_PROFILES[dataset]["valloader"]
    config = ROOT / entry["config"]
    budget, selected = entry["train_steps"], entry["checkpoint"]
    if not 0 < selected <= budget:
        raise ValueError(f"{profile_name}/{scene}: checkpoint must be within the training budget")
    if config.name != f"{scene}.json":
        raise ValueError(f"{profile_name}/{scene}: configuration must match the scene name")
    scene_config = json.loads(config.read_text(encoding="utf-8"))
    if scene_config["duration"] != profile["duration"]:
        raise ValueError(f"{profile_name}/{scene}: duration does not match the paper protocol")

    output_root = savepath / profile_name
    output = output_root / scene
    source = datapath / scene / "colmap_0"
    compact = output / "compact"
    return output, [
        [
            sys.executable, str(ROOT / "script" / "run_n3d_train_test.py"),
            "--dataset", dataset, "--scene", scene, "--config_dir", str(config.parent),
            "--datapath", str(datapath), "--savepath", str(output_root),
            "--iterations", str(budget), "--save_iterations", str(selected),
            "--test_iterations", str(selected), "--skip_test_stage", "--skip_compact_stage",
        ],
        [
            sys.executable, str(ROOT / "script" / "export_compact_model.py"),
            "--quiet", "--eval", "--test_iteration", str(selected), "--valloader", valloader,
            "--source_path", str(source), "--model_path", str(output), "--configpath", str(config),
        ],
        [
            sys.executable, str(ROOT / "test.py"),
            "--quiet", "--eval", "--skip_train", "--test_iteration", str(selected),
            "--valloader", valloader, "--source_path", str(source), "--model_path", str(compact),
            "--configpath", str(compact / "compact_config.json"), "--test_photometric_fit", "0",
        ],
    ]


def main():
    protocols = json.loads((ROOT / "configs" / "paper_runs.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=protocols, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--datapath", type=Path, required=True, help="Root containing the prepared scene directories.")
    parser.add_argument("--savepath", type=Path, required=True, help="New runs go in <savepath>/<profile>/<scene>.")
    parser.add_argument("--dry_run", action="store_true", help="Show all commands without reading data or writing output.")
    args = parser.parse_args()
    profile = protocols[args.profile]
    if args.scene not in profile["scenes"]:
        parser.error(f"unknown scene {args.scene!r} for {args.profile}; choose from {', '.join(profile['scenes'])}")
    output, commands = build_commands(
        args.profile, profile, args.scene, args.datapath.resolve(), args.savepath.resolve()
    )
    if not args.dry_run and output.exists():
        parser.error(f"output already exists: {output}; choose a new --savepath")
    for command in commands:
        print("[STEGF][Paper] " + shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
