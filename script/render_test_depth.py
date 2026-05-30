import json
import os
import sys
from os import makedirs

import numpy as np
import torch
import torchvision
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
THIRDPARTY_ROOT = os.path.join(REPO_ROOT, "thirdparty", "gaussian_splatting")
if THIRDPARTY_ROOT not in sys.path:
    sys.path.append(THIRDPARTY_ROOT)

from helper_train import getmodel, getrenderpip, trbfunction
from thirdparty.gaussian_splatting.helper3dg import gettestparse
from thirdparty.gaussian_splatting.scene import Scene


def normalize_depth_quantile(depth, valid_mask):
    depth = depth.detach().float()
    valid = torch.isfinite(depth) & valid_mask.bool()
    values = depth[valid]
    if values.numel() == 0:
        return torch.zeros_like(depth)
    lo = torch.quantile(values, 0.02)
    hi = torch.quantile(values, 0.98)
    denom = torch.clamp(hi - lo, min=1e-6)
    return torch.nan_to_num((depth - lo) / denom, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def normalize_depth_quantile_all(depth):
    depth = depth.detach().float()
    finite = torch.isfinite(depth) & (depth > 0.0)
    values = depth[finite]
    if values.numel() == 0:
        return torch.zeros_like(depth)
    lo = torch.quantile(values, 0.02)
    hi = torch.quantile(values, 0.98)
    denom = torch.clamp(hi - lo, min=1e-6)
    return torch.nan_to_num((depth - lo) / denom, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def normalize_depth_ge_threshold(depth, threshold):
    depth = depth.detach().float()
    threshold = float(threshold)
    valid = torch.isfinite(depth) & (depth >= threshold)
    values = depth[valid]
    if values.numel() == 0:
        return torch.zeros_like(depth)
    lo = torch.quantile(values, 0.02)
    hi = torch.quantile(values, 0.98)
    lo = torch.clamp(lo, min=threshold)
    denom = torch.clamp(hi - lo, min=1e-6)
    normalized = torch.nan_to_num((depth - lo) / denom, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return torch.where(valid, normalized, torch.zeros_like(normalized))


def normalize_depth_fixed(depth, depth_max):
    depth_max = max(float(depth_max), 1e-6)
    depth = depth.detach().float()
    return torch.nan_to_num(depth / depth_max, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def set_eval_mode(gaussians):
    if gaussians.rgbdecoder is not None:
        gaussians.rgbdecoder.cuda()
        gaussians.rgbdecoder.eval()
    if getattr(gaussians, "use_euler_field", False):
        if gaussians.euler_field is not None:
            gaussians.euler_field.cuda()
            gaussians.euler_field.eval()
        if getattr(gaussians, "field_router", None) is not None:
            gaussians.field_router.cuda()
            gaussians.field_router.eval()
        if getattr(gaussians, "field_query_gate", None) is not None:
            gaussians.field_query_gate.cuda()
            gaussians.field_query_gate.eval()
        if gaussians.field_decoder is not None:
            gaussians.field_decoder.cuda()
            gaussians.field_decoder.eval()


def render_test_depth(dataset, iteration, pipeline, multiview, duration, rgbfunction="rgbv1", rdpip="v2", loader="colmap"):
    with torch.no_grad():
        print("use model {}".format(dataset.model))
        GaussianModel = getmodel(dataset.model)
        gaussians = GaussianModel(dataset.sh_degree, rgbfunction)
        if hasattr(gaussians, "configure_euler_field"):
            gaussians.configure_euler_field(dataset)

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, multiview=multiview, duration=duration, loader=loader)
        views = scene.getTestCameras()
        if len(views) == 0:
            print("[STEGF] No test cameras found.")
            return

        if gaussians.ts is None:
            height, width = views[0].image_height, views[0].image_width
            gaussians.ts = torch.ones(1, 1, height, width, device="cuda")

        set_eval_mode(gaussians)
        render_name = "test_ours_full" if rdpip == "train_ours_full" else rdpip
        render, gr_setting, gr_zero = getrenderpip(render_name)
        background = torch.zeros(9, dtype=torch.float32, device="cuda")
        rbfbasefunction = trbfunction

        out_root = os.path.join(dataset.model_path, "test", "ours_{}".format(scene.loaded_iter), "depth")
        depth_path = os.path.join(out_root, "render_depth")
        depth_all_path = os.path.join(out_root, "render_depth_all")
        depth_ge15_path = os.path.join(out_root, "render_depth_ge15")
        fixed_path = os.path.join(out_root, "render_depth_fixed")
        saturated_path = os.path.join(out_root, "depth_saturated_mask")
        raw_path = os.path.join(out_root, "raw_npy")
        meta_path = os.path.join(out_root, "meta")
        makedirs(depth_path, exist_ok=True)
        makedirs(depth_all_path, exist_ok=True)
        makedirs(depth_ge15_path, exist_ok=True)
        makedirs(fixed_path, exist_ok=True)
        makedirs(saturated_path, exist_ok=True)
        makedirs(raw_path, exist_ok=True)
        makedirs(meta_path, exist_ok=True)
        depth_max = float(getattr(gaussians, "field_bg_prior_depth_max", 15.0))
        manifest = []

        for idx, view in enumerate(tqdm(views, desc="Rendering test depth")):
            render_pkg = render(
                view,
                gaussians,
                pipeline,
                background,
                scaling_modifier=1.0,
                basicfunction=rbfbasefunction,
                GRsetting=gr_setting,
                GRzer=gr_zero,
            )
            depth = render_pkg["depth"].detach().squeeze(0).float()
            valid = torch.isfinite(depth) & (depth > 0.0) & (depth < depth_max - 1e-4)
            depth_quantile = normalize_depth_quantile(depth, valid)
            depth_quantile_all = normalize_depth_quantile_all(depth)
            depth_ge15 = normalize_depth_ge_threshold(depth, depth_max)
            depth_fixed = normalize_depth_fixed(depth, depth_max)
            saturated = torch.isfinite(depth) & (depth >= depth_max)

            timestamp = float(getattr(view, "timestamp", 0.0))
            file_stem = "{0:05d}".format(idx)

            torchvision.utils.save_image(depth_quantile.unsqueeze(0), os.path.join(depth_path, file_stem + ".png"))
            torchvision.utils.save_image(depth_quantile_all.unsqueeze(0), os.path.join(depth_all_path, file_stem + ".png"))
            torchvision.utils.save_image(depth_ge15.unsqueeze(0), os.path.join(depth_ge15_path, file_stem + ".png"))
            torchvision.utils.save_image(depth_fixed.unsqueeze(0), os.path.join(fixed_path, file_stem + ".png"))
            torchvision.utils.save_image(saturated.float().unsqueeze(0), os.path.join(saturated_path, file_stem + ".png"))
            np.save(os.path.join(raw_path, file_stem + ".npy"), depth.detach().cpu().numpy())

            valid_values = depth[valid]
            meta = {
                "index": int(idx),
                "file": file_stem + ".png",
                "camera": str(getattr(view, "image_name", "")),
                "timestamp": timestamp,
                "depth_max_for_fixed": depth_max,
                "valid_pixels": int(torch.count_nonzero(valid).item()),
                "saturated_pixels": int(torch.count_nonzero(saturated).item()),
                "depth_min": float(valid_values.min().cpu()) if valid_values.numel() > 0 else 0.0,
                "depth_median": float(torch.median(valid_values).cpu()) if valid_values.numel() > 0 else 0.0,
                "depth_max": float(valid_values.max().cpu()) if valid_values.numel() > 0 else 0.0,
            }
            with open(os.path.join(meta_path, file_stem + ".json"), "w") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
            manifest.append(meta)

        with open(os.path.join(out_root, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"[STEGF] Saved {len(manifest)} test depth maps to {out_root}")


if __name__ == "__main__":
    args, model_extract, pp_extract, multiview = gettestparse()
    render_test_depth(
        model_extract,
        args.test_iteration,
        pp_extract,
        multiview,
        args.duration,
        rgbfunction=args.rgbfunction,
        rdpip=args.rdpip,
        loader=args.valloader,
    )
