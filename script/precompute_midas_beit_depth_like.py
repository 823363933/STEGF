import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


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


def resolve_path(path, base):
    path = Path(path)
    if path.is_absolute():
        return path
    return base / path


def camera_name(path):
    return path.stem


def write_stats(path, depth_like):
    valid = np.isfinite(depth_like)
    with open(path, "w") as f:
        f.write(f"shape: {list(depth_like.shape)}\n")
        f.write(f"dtype: {depth_like.dtype}\n")
        if not np.any(valid):
            f.write("valid: 0\n")
            return
        values = depth_like[valid].astype(np.float64)
        for p in [0, 1, 5, 10, 25, 30, 50, 70, 75, 80, 85, 90, 95, 99, 100]:
            f.write(f"p{p}: {np.percentile(values, p):.6f}\n")


def write_band(path, depth_like, low_q=10.0, high_q=30.0):
    valid = np.isfinite(depth_like)
    if not np.any(valid):
        band = np.zeros(depth_like.shape, dtype=np.uint8)
    else:
        values = depth_like[valid].astype(np.float32)
        lo = np.percentile(values, low_q)
        hi = np.percentile(values, high_q)
        denom = max(float(hi - lo), 1e-6)
        band = np.clip((depth_like.astype(np.float32) - lo) / denom, 0.0, 1.0)
        band = (band * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), band)


@torch.no_grad()
def infer_depth_like(device, model, transform, image_rgb):
    image = transform({"image": image_rgb})["image"]
    sample = torch.from_numpy(image).to(device).unsqueeze(0)
    prediction = model.forward(sample)
    prediction = F.interpolate(
        prediction.unsqueeze(1),
        size=image_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    )
    return prediction.squeeze().detach().cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute MiDaS dpt_beit_large_512 raw depth-like arrays for DyNeRF COLMAP time slices."
    )
    parser.add_argument("--scene", default="coffee_martini")
    parser.add_argument("--dataset_root", default="dataset")
    parser.add_argument(
        "--output_dir",
        default="dataset/MiDaS-BEiT-pre/coffee_martini/midas_beit_large_512",
        help="Directory that contains or will contain colmap_<time>/camXX/raw_depth_like.npy.",
    )
    parser.add_argument("--time_indices", default="12,25,37,49")
    parser.add_argument(
        "--midas_repo",
        default=None,
        help="Path to OriginalRepository/MiDaS. Defaults to ../OriginalRepository/MiDaS from STEGF root.",
    )
    parser.add_argument("--model_weights", default=None)
    parser.add_argument("--model_type", default="dpt_beit_large_512")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    dataset_root = resolve_path(args.dataset_root, repo_root)
    scene_root = dataset_root / args.scene
    output_dir = resolve_path(args.output_dir, repo_root)
    midas_repo = Path(args.midas_repo) if args.midas_repo else workspace_root / "OriginalRepository" / "MiDaS"
    midas_repo = resolve_path(midas_repo, repo_root)

    if not scene_root.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_root}")
    if not midas_repo.exists():
        raise FileNotFoundError(f"MiDaS repository not found: {midas_repo}")

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

    model_weights = Path(args.model_weights) if args.model_weights else midas_repo / "weights" / "dpt_beit_large_512.pt"
    model_weights = resolve_path(model_weights, repo_root)
    if not model_weights.exists():
        raise FileNotFoundError(f"Model weights not found: {model_weights}")

    items = []
    for colmap_dir in colmap_dirs:
        image_dir = colmap_dir / "images"
        images = sorted(image_dir.glob("*.png"))
        if not images:
            raise FileNotFoundError(f"No PNG images found: {image_dir}")
        for image_path in images:
            out_cam_dir = output_dir / colmap_dir.name / camera_name(image_path)
            raw_path = out_cam_dir / "raw_depth_like.npy"
            items.append((colmap_dir, image_path, raw_path))

    print(f"[STEGF] MiDaS repo: {midas_repo}")
    print(f"[STEGF] Model weights: {model_weights}")
    print(f"[STEGF] Scene root: {scene_root}")
    print(f"[STEGF] Output dir: {output_dir}")
    print(f"[STEGF] Time indices: {[int(p.name.split('_')[-1]) for p in colmap_dirs]}")
    print(f"[STEGF] Images: {len(items)}")

    if args.dry_run:
        for _, _, raw_path in items[:10]:
            print(f"[dry-run] {raw_path}")
        return

    sys.path.insert(0, str(midas_repo))
    import utils as midas_utils
    from midas.model_loader import load_model

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu or run on the GPU server.")
    device = torch.device(args.device)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model, transform, _, _ = load_model(
        device,
        str(model_weights),
        model_type=args.model_type,
        optimize=False,
        height=None,
        square=False,
    )
    model.eval()

    manifest = {
        "scene": args.scene,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "midas_repo": str(midas_repo),
        "model_weights": str(model_weights),
        "model_type": args.model_type,
        "time_indices": [int(p.name.split("_")[-1]) for p in colmap_dirs],
        "items": [],
    }

    for colmap_dir, image_path, raw_path in tqdm(items, desc="Precomputing MiDaS BEiT"):
        if raw_path.exists() and not args.overwrite:
            status = "skipped_existing"
        else:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            image_rgb = midas_utils.read_image(str(image_path))
            depth_like = infer_depth_like(device, model, transform, image_rgb)
            np.save(raw_path, depth_like)
            write_stats(raw_path.parent / "stats.txt", depth_like)
            write_band(raw_path.parent / "band_p10_p030.png", depth_like)
            status = "generated"
        manifest["items"].append(
            {
                "time_index": int(colmap_dir.name.split("_")[-1]),
                "camera": camera_name(image_path),
                "image": str(image_path),
                "raw_depth_like": str(raw_path),
                "status": status,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest_multitime.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[STEGF] Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
