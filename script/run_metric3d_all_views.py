#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


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


def prepare_metric3d_input(rgb_origin, device, input_size=(616, 1064)):
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(
        rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
    )

    padding = [123.675, 116.28, 103.53]
    h2, w2 = rgb.shape[:2]
    pad_h = input_size[0] - h2
    pad_w = input_size[1] - w2
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(
        rgb,
        pad_h_half,
        pad_h - pad_h_half,
        pad_w_half,
        pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=padding,
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    mean = torch.tensor([123.675, 116.28, 103.53], device=device).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375], device=device).float()[:, None, None]
    rgb_t = torch.from_numpy(rgb.transpose((2, 0, 1))).float().to(device)
    rgb_t = torch.div((rgb_t - mean), std)[None, :, :, :]
    return rgb_t, pad_info


def run_metric3d(image_dir, output_dir, model_name, metric3d_repo):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"loading {model_name} from {metric3d_repo}")
    model = torch.hub.load(str(metric3d_repo), model_name, pretrain=True, source="local")
    model = model.to(device).eval()

    raw_dir = output_dir / "raw"
    diag_root = output_dir / "diag"
    raw_dir.mkdir(parents=True, exist_ok=True)
    diag_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.png"))
    summary_rows = []
    percentiles = [0, 1, 5, 10, 25, 30, 50, 70, 75, 80, 85, 90, 95, 99, 100]

    for idx, image_path in enumerate(image_paths, 1):
        name = image_path.stem
        print(f"[{idx}/{len(image_paths)}] {name}")
        rgb_origin = cv2.imread(str(image_path))[:, :, ::-1]
        rgb_t, pad_info = prepare_metric3d_input(rgb_origin, device)

        with torch.no_grad():
            pred_depth, confidence, output_dict = model.inference({"input": rgb_t})

        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[
            pad_info[0] : pred_depth.shape[0] - pad_info[1],
            pad_info[2] : pred_depth.shape[1] - pad_info[3],
        ]
        pred_depth = torch.nn.functional.interpolate(
            pred_depth[None, None, :, :],
            rgb_origin.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        depth = torch.clamp(pred_depth, 0, 300).detach().float().cpu().numpy()

        np.save(raw_dir / f"{name}_depth.npy", depth)
        np.savez_compressed(raw_dir / f"{name}.npz", depth=depth)

        out = diag_root / name
        out.mkdir(exist_ok=True)
        np.save(out / "raw_depth_m.npy", depth)
        vals = np.nanpercentile(depth, percentiles)
        (out / "stats.txt").write_text(
            "\n".join(
                ["shape: " + str(list(depth.shape)), "dtype: float32"]
                + [f"p{p:g}: {v:.6f}" for p, v in zip(percentiles, vals)]
            )
            + "\n"
        )
        summary_rows.append([name] + [f"{v:.6f}" for v in vals])

        save_gray(out / "raw_norm_p01_p99.png", norm01(depth))
        inv = 1.0 / np.maximum(depth - np.nanmin(depth) + 1e-6, 1e-6)
        save_gray(out / "inverse_norm_p01_p99.png", norm01(inv))
        for a, b in [(0, 10), (10, 30), (30, 50), (50, 70), (70, 90), (90, 100)]:
            lo, hi = np.nanpercentile(depth, [a, b])
            save_gray(out / f"band_p{a:02d}_p{b:03d}.png", norm01(depth, lo, hi))
        for thr in [50, 60, 70, 75, 80, 85, 90]:
            v = np.nanpercentile(depth, thr)
            save_gray(out / f"ge_p{thr:02d}.png", (depth >= v).astype(np.float32))
            save_gray(out / f"lt_p{thr:02d}.png", (depth < v).astype(np.float32))

        gt = resize_to_h(Image.open(image_path).convert("RGB"))
        panels = [gt]
        labels = ["gt"]
        for fname in [
            "raw_norm_p01_p99.png",
            "inverse_norm_p01_p99.png",
            "band_p70_p090.png",
            "band_p90_p100.png",
            "ge_p70.png",
            "ge_p80.png",
            "ge_p90.png",
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
            draw.text((x + 4, 4), label, fill=(255, 0, 0))
            x += im.width
        sheet.save(out / "contact_sheet.png")

    thumbs = []
    for image_path in image_paths:
        name = image_path.stem
        out = diag_root / name
        gt = resize_to_h(Image.open(image_path).convert("RGB"), 150)
        raw = resize_to_h(Image.open(out / "raw_norm_p01_p99.png").convert("RGB"), 150)
        ge80 = resize_to_h(Image.open(out / "ge_p80.png").convert("RGB"), 150)
        row = Image.new("RGB", (gt.width + raw.width + ge80.width, 174), "white")
        draw = ImageDraw.Draw(row)
        draw.text((4, 4), name, fill=(255, 0, 0))
        x = 0
        for im in [gt, raw, ge80]:
            row.paste(im, (x, 24))
            x += im.width
        thumbs.append(row)

    index = Image.new("RGB", (max(t.width for t in thumbs), sum(t.height for t in thumbs)), "white")
    y = 0
    for thumb in thumbs:
        index.paste(thumb, (0, y))
        y += thumb.height
    index.save(diag_root / "all_views_index_gt_raw_ge80.png")

    with (diag_root / "stats_summary.csv").open("w") as f:
        f.write(",".join(["view"] + [f"p{p:g}" for p in percentiles]) + "\n")
        for row in summary_rows:
            f.write(",".join(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="metric3d_vit_small")
    parser.add_argument(
        "--metric3d_repo",
        default="/home/wendy/code/workspace/OriginalRepository/Metric3D",
    )
    args = parser.parse_args()
    run_metric3d(
        Path(args.image_dir),
        Path(args.output_dir),
        args.model_name,
        Path(args.metric3d_repo),
    )


if __name__ == "__main__":
    main()
