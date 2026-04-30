#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_frames(gt_dir: Path) -> tuple[np.ndarray, list[str]]:
    frame_paths = sorted(gt_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"no png frames found under {gt_dir}")
    frames = []
    for path in frame_paths:
        frame = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        frames.append(frame)
    return np.stack(frames, axis=0), [path.name for path in frame_paths]


def window_candidates(
    score_map: np.ndarray,
    luminance: np.ndarray,
    patch_size: int,
    step: int,
    top_k: int,
) -> list[tuple[int, int]]:
    h, w = score_map.shape
    candidates: list[tuple[float, int, int]] = []
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            lum_patch = luminance[y : y + patch_size, x : x + patch_size]
            valid_ratio = np.mean((lum_patch > 0.05) & (lum_patch < 0.95))
            if valid_ratio < 0.9:
                continue
            score = float(score_map[y : y + patch_size, x : x + patch_size].mean())
            candidates.append((score, y, x))
    candidates.sort()
    selected: list[tuple[int, int]] = []
    for _, y, x in candidates:
        overlaps = False
        for sy, sx in selected:
            if not (
                y + patch_size <= sy
                or sy + patch_size <= y
                or x + patch_size <= sx
                or sx + patch_size <= x
            ):
                overlaps = True
                break
        if overlaps:
            continue
        selected.append((y, x))
        if len(selected) >= top_k:
            break
    return selected


def patch_summary(frames: np.ndarray, y: int, x: int, patch_size: int) -> dict:
    patch = frames[:, y : y + patch_size, x : x + patch_size, :]
    temporal_std = patch.std(axis=0).mean(axis=2)
    temporal_range = (patch.max(axis=0) - patch.min(axis=0)).mean(axis=2)
    center = patch[:, patch_size // 2, patch_size // 2, :]
    return {
        "y": y,
        "x": x,
        "patch_size": patch_size,
        "mean_temporal_std": float(temporal_std.mean()),
        "mean_temporal_range": float(temporal_range.mean()),
        "ratio_range_gt_1_255": float((temporal_range > (1.0 / 255.0)).mean()),
        "ratio_range_gt_2_255": float((temporal_range > (2.0 / 255.0)).mean()),
        "ratio_range_gt_5_255": float((temporal_range > (5.0 / 255.0)).mean()),
        "center_pixel_std_rgb": [float(v) for v in center.std(axis=0)],
        "center_pixel_range_rgb": [float(v) for v in (center.max(axis=0) - center.min(axis=0))],
    }


def quantile_summary(score_map: np.ndarray, range_map: np.ndarray, quantile: float) -> dict:
    threshold = float(np.quantile(score_map, quantile))
    mask = score_map <= threshold
    masked_range = range_map[mask]
    return {
        "quantile": quantile,
        "score_threshold": threshold,
        "pixel_count": int(mask.sum()),
        "mean_temporal_std": float(score_map[mask].mean()),
        "mean_temporal_range": float(masked_range.mean()),
        "ratio_range_gt_1_255": float((masked_range > (1.0 / 255.0)).mean()),
        "ratio_range_gt_2_255": float((masked_range > (2.0 / 255.0)).mean()),
        "ratio_range_gt_5_255": float((masked_range > (5.0 / 255.0)).mean()),
    }


def analyze_sequence(gt_dir: Path, patch_size: int, step: int, top_k: int) -> dict:
    frames, frame_names = load_frames(gt_dir)
    luminance = 0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]
    temporal_std = frames.std(axis=0).mean(axis=2)
    temporal_range = (frames.max(axis=0) - frames.min(axis=0)).mean(axis=2)

    summary = {
        "gt_dir": str(gt_dir),
        "num_frames": int(frames.shape[0]),
        "image_shape": [int(v) for v in frames.shape[1:3]],
        "global_mean_temporal_std": float(temporal_std.mean()),
        "global_mean_temporal_range": float(temporal_range.mean()),
        "low_variation_quantiles": [
            quantile_summary(temporal_std, temporal_range, quantile=q) for q in (0.05, 0.10, 0.20)
        ],
        "candidate_static_patches": [],
        "frame_names": frame_names,
    }

    for y, x in window_candidates(temporal_std, luminance.mean(axis=0), patch_size, step, top_k):
        summary["candidate_static_patches"].append(patch_summary(frames, y, x, patch_size))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze temporal pixel fluctuation in GT frames.")
    parser.add_argument("gt_dirs", nargs="+", help="Directories that contain gt/*.png frames")
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--step", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    results = [
        analyze_sequence(Path(gt_dir), patch_size=args.patch_size, step=args.step, top_k=args.top_k)
        for gt_dir in args.gt_dirs
    ]

    payload = {"results": results}
    text = json.dumps(payload, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
