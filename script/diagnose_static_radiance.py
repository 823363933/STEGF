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
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams
from thirdparty.gaussian_splatting.helper3dg import gettestparse
from thirdparty.gaussian_splatting.lpipsPyTorch import lpips
from thirdparty.gaussian_splatting.scene import Scene
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.utils.loss_utils import ssim


def normalize_map(values):
    values = values.detach().float()
    finite = torch.isfinite(values)
    if torch.count_nonzero(finite) == 0:
        return torch.zeros_like(values)
    valid = values[finite]
    vmin = torch.amin(valid)
    vmax = torch.amax(valid)
    if float((vmax - vmin).detach().cpu()) < 1e-6:
        return torch.zeros_like(values)
    return torch.clamp((values - vmin) / (vmax - vmin), 0.0, 1.0)


def make_static_radiance_mask(gaussians, rendering, gt):
    error_map = torch.abs(rendering.detach().float() - gt.detach().float()).mean(dim=0)
    score = normalize_map(error_map)
    min_threshold = max(float(getattr(gaussians, "field_obs_reliability_min_error", 0.03)), 0.0)
    error_threshold = float(getattr(gaussians, "field_obs_reliability_error_threshold", 0.0))
    if error_threshold > 0.0:
        threshold = max(error_threshold, min_threshold)
    else:
        quantile = float(getattr(gaussians, "field_obs_reliability_error_quantile", 0.90))
        quantile = max(0.0, min(quantile, 1.0))
        values = score[torch.isfinite(score)]
        if values.numel() == 0:
            return None, score, float("nan")
        threshold = max(float(torch.quantile(values.float(), quantile).item()), min_threshold)
    mask = score >= threshold
    if torch.count_nonzero(mask) == 0:
        return None, score, float(threshold)
    return mask.detach(), score.detach(), float(threshold)


def set_eval_mode(gaussians):
    if gaussians.rgbdecoder is not None:
        gaussians.rgbdecoder.cuda()
        gaussians.rgbdecoder.eval()
    if getattr(gaussians, "use_euler_field", False):
        for name in (
            "euler_field",
            "field_router",
            "field_query_gate",
            "field_decoder",
            "field_temporal_opacity_head",
            "field_static_view_mapper",
            "field_static_app_head",
        ):
            module = getattr(gaussians, name, None)
            if module is not None:
                module.cuda()
                module.eval()


def metric_pack(rendering, gt):
    rendering = torch.clamp(rendering, 0.0, 1.0)
    gt = torch.clamp(gt, 0.0, 1.0)
    return {
        "SSIM": float(ssim(rendering.unsqueeze(0), gt.unsqueeze(0)).detach().cpu()),
        "PSNR": float(psnr(rendering.unsqueeze(0), gt.unsqueeze(0)).detach().cpu()),
        "LPIPS": float(lpips(rendering.unsqueeze(0), gt.unsqueeze(0), net_type="alex").detach().cpu()),
    }


def summarize(values):
    if not values:
        return {}
    keys = values[0].keys()
    return {k: float(np.mean([v[k] for v in values])) for k in keys}


def diagnose(dataset: ModelParams, iteration: int, pipeline: PipelineParams, multiview: bool, duration: int, rgbfunction="rgbv1", rdpip="v2", loader="colmap"):
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

        out_root = os.path.join(dataset.model_path, "test", "ours_{}".format(scene.loaded_iter), "static_radiance_diagnostic")
        paths = {
            "off": os.path.join(out_root, "branch_off", "renders"),
            "on": os.path.join(out_root, "branch_on", "renders"),
            "gt": os.path.join(out_root, "gt"),
            "mask": os.path.join(out_root, "debug", "static_radiance_mask"),
            "score": os.path.join(out_root, "debug", "static_radiance_error_score"),
            "delta": os.path.join(out_root, "debug", "branch_on_minus_off"),
            "delta_abs": os.path.join(out_root, "debug", "branch_on_minus_off_abs"),
        }
        for path in paths.values():
            makedirs(path, exist_ok=True)

        off_metrics = []
        on_metrics = []
        per_view = []

        old_branch_state = bool(getattr(gaussians, "field_static_radiance_branch", False))
        for idx, view in enumerate(tqdm(views, desc="Diagnosing static radiance")):
            gt = view.original_image[0:3, :, :].cuda().float()
            file_stem = "{0:05d}".format(idx)

            gaussians.field_static_radiance_branch = False
            off_pkg = render(
                view,
                gaussians,
                pipeline,
                background,
                scaling_modifier=1.0,
                basicfunction=trbfunction,
                GRsetting=gr_setting,
                GRzer=gr_zero,
            )
            render_off = torch.clamp(off_pkg["render"], 0.0, 1.0)

            mask, score, threshold = make_static_radiance_mask(gaussians, render_off, gt)

            gaussians.field_static_radiance_branch = old_branch_state
            if mask is not None and old_branch_state:
                on_pkg = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    scaling_modifier=1.0,
                    basicfunction=trbfunction,
                    GRsetting=gr_setting,
                    GRzer=gr_zero,
                    static_radiance_mask=mask,
                    iteration=scene.loaded_iter,
                )
                render_on = torch.clamp(on_pkg["render"], 0.0, 1.0)
            else:
                render_on = render_off

            off_metric = metric_pack(render_off, gt)
            on_metric = metric_pack(render_on, gt)
            off_metrics.append(off_metric)
            on_metrics.append(on_metric)

            delta = render_on - render_off
            delta_abs = torch.mean(torch.abs(delta), dim=0, keepdim=True)
            mask_pixels = int(torch.count_nonzero(mask).item()) if mask is not None else 0
            per_view.append(
                {
                    "index": idx,
                    "file": file_stem + ".png",
                    "camera": str(getattr(view, "image_name", "")),
                    "timestamp": float(getattr(view, "timestamp", 0.0)),
                    "mask_pixels": mask_pixels,
                    "mask_ratio": float(mask_pixels) / float(mask.numel()) if mask is not None else 0.0,
                    "mask_threshold": threshold,
                    "off": off_metric,
                    "on": on_metric,
                    "delta_abs_mean": float(delta_abs.mean().detach().cpu()),
                    "delta_abs_max": float(delta_abs.max().detach().cpu()),
                }
            )

            torchvision.utils.save_image(render_off, os.path.join(paths["off"], file_stem + ".png"))
            torchvision.utils.save_image(render_on, os.path.join(paths["on"], file_stem + ".png"))
            torchvision.utils.save_image(gt, os.path.join(paths["gt"], file_stem + ".png"))
            torchvision.utils.save_image(score.unsqueeze(0), os.path.join(paths["score"], file_stem + ".png"))
            torchvision.utils.save_image((mask.float() if mask is not None else torch.zeros_like(score)).unsqueeze(0), os.path.join(paths["mask"], file_stem + ".png"))
            torchvision.utils.save_image(torch.clamp(0.5 + 4.0 * delta, 0.0, 1.0), os.path.join(paths["delta"], file_stem + ".png"))
            torchvision.utils.save_image(normalize_map(delta_abs.squeeze(0)).unsqueeze(0), os.path.join(paths["delta_abs"], file_stem + ".png"))

        gaussians.field_static_radiance_branch = old_branch_state
        summary = {
            "branch_enabled": old_branch_state,
            "iteration": int(scene.loaded_iter),
            "off": summarize(off_metrics),
            "on": summarize(on_metrics),
            "on_minus_off": {
                k: summarize(on_metrics).get(k, 0.0) - summarize(off_metrics).get(k, 0.0)
                for k in summarize(off_metrics).keys()
            },
            "static_radiance_level_logits": None,
            "static_radiance_level_weights": None,
            "views": per_view,
        }
        logits = getattr(gaussians, "_static_radiance_level_logits", None)
        if logits is not None and getattr(logits, "numel", lambda: 0)() > 0:
            values = logits.detach().cpu().float().view(-1)
            summary["static_radiance_level_logits"] = values.tolist()
            summary["static_radiance_level_weights"] = torch.softmax(values, dim=0).tolist()
        with open(os.path.join(out_root, "metrics.json"), "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print("[STEGF] Static radiance diagnostic saved to {}".format(out_root))
        print(json.dumps({k: summary[k] for k in ("off", "on", "on_minus_off", "static_radiance_level_weights")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    args, model_extract, pp_extract, multiview = gettestparse()
    diagnose(
        model_extract,
        args.test_iteration,
        pp_extract,
        multiview,
        args.duration,
        rgbfunction=args.rgbfunction,
        rdpip=args.rdpip,
        loader=args.valloader,
    )
