import argparse
import json
import shutil
import subprocess
from pathlib import Path


def parse_indices(spec, max_index):
    if spec.lower() == "all":
        return list(range(max_index + 1))

    indices = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(item))
    return sorted(i for i in indices if 0 <= i <= max_index)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute Depth Anything 3 depth maps for DyNeRF COLMAP time slices."
    )
    parser.add_argument("--scene", default="coffee_martini")
    parser.add_argument("--dataset_root", default="dataset")
    parser.add_argument("--output_root", default="output/da3_depth")
    parser.add_argument("--model_dir", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--time_indices", default="all", help="all, comma list, or range like 0-49")
    parser.add_argument("--da3_bin", default="da3")
    parser.add_argument("--with_depth_vis", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root
    scene_root = dataset_root / args.scene
    if not scene_root.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_root}")

    colmap_dirs = sorted(
        [p for p in scene_root.glob("colmap_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not colmap_dirs:
        raise FileNotFoundError(f"No colmap_* directories found under {scene_root}")

    max_index = max(int(p.name.split("_")[-1]) for p in colmap_dirs)
    requested = set(parse_indices(args.time_indices, max_index))
    colmap_dirs = [p for p in colmap_dirs if int(p.name.split("_")[-1]) in requested]
    if not colmap_dirs:
        raise ValueError(f"No valid time indices selected by: {args.time_indices}")

    da3_bin = shutil.which(args.da3_bin)
    if da3_bin is None:
        raise RuntimeError(
            f"Cannot find '{args.da3_bin}'. Activate the DA3 environment before running this script."
        )

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    scene_output_root = output_root / args.scene
    scene_output_root.mkdir(parents=True, exist_ok=True)

    export_format = "mini_npz-depth_vis" if args.with_depth_vis else "mini_npz"
    manifest = {
        "scene": args.scene,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "model_dir": args.model_dir,
        "process_res": args.process_res,
        "export_format": export_format,
        "items": [],
    }

    for colmap_dir in colmap_dirs:
        time_index = int(colmap_dir.name.split("_")[-1])
        out_dir = scene_output_root / colmap_dir.name
        result_path = out_dir / "exports" / "mini_npz" / "results.npz"

        if result_path.exists() and not args.overwrite:
            print(f"[skip] {colmap_dir.name}: {result_path}")
            manifest["items"].append(
                {
                    "time_index": time_index,
                    "colmap_dir": str(colmap_dir),
                    "output_dir": str(out_dir),
                    "result_npz": str(result_path),
                    "status": "skipped_existing",
                }
            )
            continue

        cmd = [
            da3_bin,
            "colmap",
            str(colmap_dir),
            "--sparse-subdir",
            "0",
            "--model-dir",
            args.model_dir,
            "--export-format",
            export_format,
            "--export-dir",
            str(out_dir),
            "--process-res",
            str(args.process_res),
            "--auto-cleanup",
        ]
        print("[run]", " ".join(cmd))
        subprocess.run(cmd, check=True)
        if not result_path.exists():
            raise FileNotFoundError(f"DA3 finished but result file is missing: {result_path}")

        manifest["items"].append(
            {
                "time_index": time_index,
                "colmap_dir": str(colmap_dir),
                "output_dir": str(out_dir),
                "result_npz": str(result_path),
                "status": "generated",
            }
        )

    manifest_path = scene_output_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
