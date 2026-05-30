#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from depth_pro import create_model_and_transforms, load_rgb


def norm01(x, lo=None, hi=None):
    x = x.astype(np.float32)
    if lo is None:
        lo = float(np.nanpercentile(x, 1))
    if hi is None:
        hi = float(np.nanpercentile(x, 99))
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def save_gray(path, arr):
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def resize_to_h(im, h=260):
    w = round(im.width * h / im.height)
    return im.resize((w, h), Image.Resampling.BILINEAR)


def export_depthpro(image_dir, output_dir, precision="half"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.half if precision == "half" and device.type == "cuda" else torch.float32
    print(f"device: {device}, precision: {dtype}")
    model, transform = create_model_and_transforms(device=device, precision=dtype)
    model.eval()

    raw_dir = output_dir / "raw"
    diag_root = output_dir / "diag"
    raw_dir.mkdir(parents=True, exist_ok=True)
    diag_root.mkdir(parents=True, exist_ok=True)

    percentiles = [0, 1, 5, 10, 25, 30, 50, 70, 75, 80, 85, 90, 95, 99, 100]
    summary_rows = []
    image_paths = sorted(image_dir.glob("*.png"))

    for idx, image_path in enumerate(image_paths, 1):
        name = image_path.stem
        print(f"[{idx}/{len(image_paths)}] {name}")
        image, _, f_px = load_rgb(image_path)
        with torch.no_grad():
            pred = model.infer(transform(image), f_px=f_px)
        depth = pred["depth"].detach().float().cpu().numpy().squeeze()
        focal = pred["focallength_px"].detach().float().cpu().item()

        np.save(raw_dir / f"{name}_depth_m.npy", depth)
        np.savez_compressed(raw_dir / f"{name}.npz", depth=depth, focal_px=focal)

        out = diag_root / name
        out.mkdir(exist_ok=True)
        np.save(out / "raw_depth_m.npy", depth)
        vals = np.nanpercentile(depth, percentiles)
        (out / "stats.txt").write_text(
            "\n".join(
                [
                    "shape: " + str(list(depth.shape)),
                    "dtype: float32",
                    f"focal_px: {focal:.6f}",
                ]
                + [f"p{p:g}: {v:.6f}" for p, v in zip(percentiles, vals)]
            )
            + "\n"
        )
        summary_rows.append([name, f"{focal:.6f}"] + [f"{v:.6f}" for v in vals])

        save_gray(out / "raw_norm_p01_p99.png", norm01(depth))
        inv = 1.0 / np.maximum(depth, 1e-6)
        save_gray(out / "inverse_norm_p01_p99.png", norm01(inv))
        for lo, hi in [(0, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 10), (10, 20)]:
            save_gray(out / f"depth_band_{lo:g}_{hi:g}m.png", norm01(depth, lo, hi))
        for thr in [1.0, 1.5, 2.0, 2.5, 3.0]:
            save_gray(out / f"depth_lt_{thr:g}m.png", (depth < thr).astype(np.float32))

        gt = resize_to_h(Image.open(image_path).convert("RGB"))
        panels = [gt]
        labels = ["gt"]
        for fname in [
            "raw_norm_p01_p99.png",
            "inverse_norm_p01_p99.png",
            "depth_band_1_1.5m.png",
            "depth_band_1.5_2m.png",
            "depth_band_2_3m.png",
            "depth_lt_2m.png",
        ]:
            panels.append(resize_to_h(Image.open(out / fname).convert("RGB")))
            labels.append(fname.replace(".png", ""))
        width = sum(im.width for im in panels)
        height = max(im.height for im in panels) + 24
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)
        x = 0
        for im, label in zip(panels, labels):
            sheet.paste(im, (x, 24))
            draw.text((x + 4, 4), label, fill=(0, 0, 0))
            x += im.width
        sheet.save(out / "contact_sheet.png")

    thumbs = []
    for image_path in image_paths:
        name = image_path.stem
        out = diag_root / name
        gt = resize_to_h(Image.open(image_path).convert("RGB"), 150)
        raw = resize_to_h(Image.open(out / "raw_norm_p01_p99.png").convert("RGB"), 150)
        lt2 = resize_to_h(Image.open(out / "depth_lt_2m.png").convert("RGB"), 150)
        row = Image.new("RGB", (gt.width + raw.width + lt2.width, 174), "white")
        draw = ImageDraw.Draw(row)
        draw.text((4, 4), name, fill=(255, 0, 0))
        x = 0
        for im in [gt, raw, lt2]:
            row.paste(im, (x, 24))
            x += im.width
        thumbs.append(row)

    index = Image.new("RGB", (max(t.width for t in thumbs), sum(t.height for t in thumbs)), "white")
    y = 0
    for thumb in thumbs:
        index.paste(thumb, (0, y))
        y += thumb.height
    index.save(diag_root / "all_views_index_gt_raw_lt2m.png")

    with (diag_root / "stats_summary.csv").open("w") as f:
        f.write(",".join(["view", "focal_px"] + [f"p{p:g}" for p in percentiles]) + "\n")
        for row in summary_rows:
            f.write(",".join(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--precision", default="half", choices=["half", "float32"])
    args = parser.parse_args()
    export_depthpro(Path(args.image_dir), Path(args.output_dir), args.precision)


if __name__ == "__main__":
    main()
