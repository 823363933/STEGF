# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import torch
from random import randint
import random 
import sys 
import uuid
import time 
import json
import math

import torchvision
import numpy as np 
import torch.nn.functional as F
import cv2
from tqdm import tqdm


sys.path.append("./thirdparty/gaussian_splatting")

from thirdparty.gaussian_splatting.utils.loss_utils import l1_loss, ssim, l2_loss, rel_loss
from helper_train import getrenderpip, getmodel, getloss, controlgaussians, reloadhelper, trbfunction, setgtisint8, getgtisint8
from thirdparty.gaussian_splatting.scene import Scene
from argparse import Namespace
from thirdparty.gaussian_splatting.helper3dg import getparser, getrenderparts
from thirdparty.gaussian_splatting.renderer import observation_contribution_ours_full
from thirdparty.gaussian_splatting.utils.graphics_utils import geom_transform_points


def _init_status(message):
    print(f"[STEGF][Init] {message}", file=sys.stderr, flush=True)


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_stegf_config(args, gaussians):
    all_args = {key: _jsonable(value) for key, value in vars(args).items()}
    euler_keys = {
        key
        for key in all_args.keys()
        if key.startswith("field_") or key in {"use_euler_field", "grid_logits_lr"}
    }
    euler_grid = {key: all_args[key] for key in sorted(euler_keys)}
    baseline_gaussian = {
        key: all_args[key]
        for key in sorted(all_args.keys())
        if key not in euler_keys
    }
    resolved_euler_grid = {}
    if hasattr(gaussians, "_checkpoint_field_config"):
        resolved_euler_grid = _jsonable(gaussians._checkpoint_field_config())

    payload = {
        "meta": {
            "command": " ".join(sys.argv),
            "configpath": all_args.get("configpath"),
            "model_path": all_args.get("model_path"),
            "source_path": all_args.get("source_path"),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "baseline_gaussian": baseline_gaussian,
        "euler_grid": euler_grid,
        "resolved_euler_grid": resolved_euler_grid,
    }
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "stegf_config.json"), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _gaussian_blur_2d(x, sigma):
    if sigma <= 0:
        return x
    radius = max(int(round(3.0 * sigma)), 1)
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    kernel_h = kernel.view(1, 1, 1, -1)
    kernel_v = kernel.view(1, 1, -1, 1)
    x = F.conv2d(x, kernel_h, padding=(0, radius))
    x = F.conv2d(x, kernel_v, padding=(radius, 0))
    return x


def compute_highfreq_densify_gate(image, gt_image, viewpoint_camera, means3D, gaussians):
    if not bool(getattr(gaussians, "field_highfreq_densify", 0)):
        return None
    if means3D is None or means3D.numel() == 0:
        return None

    eps = max(float(getattr(gaussians, "field_highfreq_densify_eps", 1e-3)), 1e-8)
    y_min = float(getattr(gaussians, "field_highfreq_densify_y_min", 0.03))
    y_max = float(getattr(gaussians, "field_highfreq_densify_y_max", 0.97))
    min_pixels = max(int(getattr(gaussians, "field_highfreq_densify_min_pixels", 64)), 1)

    pred = image.clamp(0.0, 1.0)
    gt = gt_image.clamp(0.0, 1.0)
    coeff = pred.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    y_pred = (pred * coeff).sum(dim=0, keepdim=True)
    y_gt = (gt * coeff).sum(dim=0, keepdim=True)
    valid = ((y_gt > y_min) & (y_gt < y_max)).to(dtype=pred.dtype)
    valid_count = valid.sum()
    if valid_count.item() < min_pixels:
        return None

    log_residual = torch.log(y_pred + eps) - torch.log(y_gt + eps)
    _, h, w = log_residual.shape
    sigma_divisor = max(float(getattr(gaussians, "field_highfreq_densify_sigma_divisor", 64.0)), 1.0)
    sigma = min(h, w) / sigma_divisor
    numerator = _gaussian_blur_2d((valid * log_residual).unsqueeze(0), sigma)
    denominator = _gaussian_blur_2d(valid.unsqueeze(0), sigma).clamp_min(1e-6)
    low_residual = (numerator / denominator).squeeze(0)
    high_residual = log_residual - low_residual
    q = torch.abs(high_residual) / (torch.abs(high_residual) + torch.abs(low_residual) + eps)
    q_start = float(getattr(gaussians, "field_highfreq_densify_gate_start", 0.3))
    q_width = max(float(getattr(gaussians, "field_highfreq_densify_gate_width", 0.4)), 1e-6)
    gate_map = ((q - q_start) / q_width).clamp(0.0, 1.0)
    gate_map = gate_map * valid + (1.0 - valid)

    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)
    projected = geom_transform_points(means3D.detach(), viewpoint_camera.full_proj_transform)
    ndc = projected[:, :2]
    finite = torch.isfinite(ndc[:, 0]) & torch.isfinite(ndc[:, 1])
    inside = finite & (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
    if torch.count_nonzero(inside) == 0:
        return None

    x = (((ndc[:, 0] + 1.0) * float(width)) - 1.0) * 0.5
    y = (((ndc[:, 1] + 1.0) * float(height)) - 1.0) * 0.5
    grid_x = (2.0 * (x / max(width - 1, 1))) - 1.0
    grid_y = (2.0 * (y / max(height - 1, 1))) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        gate_map.detach().unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).view(-1, 1)
    point_gate = torch.ones((means3D.shape[0], 1), device=means3D.device, dtype=means3D.dtype)
    point_gate[inside] = sampled[inside].to(dtype=means3D.dtype)
    return point_gate


def train(dataset, opt, pipe, saving_iterations, debug_from, densify=0, duration=50, rgbfunction="rgbv1", rdpip="v2"):
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    first_iter = 0
    render, GRsetting, GRzer = getrenderpip(rdpip)

    init_started = time.perf_counter()
    _init_status(f"1/5 Constructing Gaussian model: {dataset.model}")
    GaussianModel = getmodel(dataset.model) # gmodel, gmodelrgbonly
    
    gaussians = GaussianModel(dataset.sh_degree, rgbfunction)
    if hasattr(gaussians, "configure_euler_field"):
        gaussians.configure_euler_field(dataset)
    gaussians.trbfslinit = -1*opt.trbfslinit # 
    gaussians.preprocesspoints = opt.preprocesspoints 
    gaussians.addsphpointsscale = opt.addsphpointsscale 
    gaussians.raystart = opt.raystart





    rbfbasefunction = trbfunction
    _init_status("2/5 Loading scene metadata, cameras, rays, and point cloud")
    scene = Scene(dataset, gaussians, duration=duration, loader=dataset.loader)
    _init_status(
        "3/5 Scene and Gaussian initialization complete: "
        f"train_views={len(scene.getTrainCameras())}, "
        f"test_views={len(scene.getTestCameras())}, "
        f"gaussians={int(gaussians.get_xyz.shape[0])}"
    )
    write_stegf_config(args, gaussians)
    

    currentxyz = gaussians._xyz 
    maxx, maxy, maxz = torch.amax(currentxyz[:,0]), torch.amax(currentxyz[:,1]), torch.amax(currentxyz[:,2])# z wrong...
    minx, miny, minz = torch.amin(currentxyz[:,0]), torch.amin(currentxyz[:,1]), torch.amin(currentxyz[:,2])
     

    if os.path.exists(opt.prevpath):
        print("load from " + opt.prevpath)
        reloadhelper(gaussians, opt, maxx, maxy, maxz,  minx, miny, minz)
   


    maxbounds = [maxx, maxy, maxz]
    minbounds = [minx, miny, minz]


    _init_status("4/5 Building optimizer and gradient caches")
    gaussians.training_setup(opt)
    
    numchannel = 9 

    bg_color = [1, 1, 1] if dataset.white_background else [0 for i in range(numchannel)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    #if freeze != 1:
    first_iter = 0
    _init_status(
        "5/5 Training ready after {:.2f}s; starting {} iterations".format(
            time.perf_counter() - init_started,
            int(opt.iterations),
        )
    )
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    flag = 0
    flagtwo = 0
    depthdict = {}

    if opt.batch > 1:
        traincameralist = scene.getTrainCameras().copy()
        traincamdict = {}
        traincamlookup = {}
        for i in range(duration): # 0 to 4, -> (0.0, to 0.8)
            traincamdict[i] = [cam for cam in traincameralist if cam.timestamp == i/duration]
            traincamlookup[i] = {cam.image_name: cam for cam in traincamdict[i]}
    
    
    if gaussians.ts is None :
        H,W = traincameralist[0].image_height, traincameralist[0].image_width
        gaussians.ts = torch.ones(1,1,H,W).cuda()

    scene.recordpoints(0, "start training")

                                                            
    ems_main_enabled = not bool(getattr(gaussians, "field_disable_ems_main", 0))
    flagems = 0 if ems_main_enabled else -1
    emscnt = 0
    lossdiect = {}
    ssimdict = {}
    depthdict = {}
    validdepthdict = {}
    emsstartfromiterations = opt.emsstart   
    gtisint8 = getgtisint8()

    if getattr(gaussians, "field_staged_training", False) and (not getattr(gaussians, "field_v23_compat", False)):
        gaussians.field_warmup_iters = max(int(gaussians.field_warmup_iters), int(opt.densify_until_iter))
        gaussians.field_problem_mining_start = max(int(getattr(gaussians, "field_problem_mining_start", gaussians.field_warmup_iters)), gaussians.field_warmup_iters)
        gaussians.field_category_activate_iter = max(int(getattr(gaussians, "field_category_activate_iter", gaussians.field_problem_mining_start + 2000)), gaussians.field_problem_mining_start + 1)
        gaussians.field_activate_iter = gaussians.field_category_activate_iter
        gaussians.field_fast_activate_iter = max(int(gaussians.field_fast_activate_iter), gaussians.field_category_activate_iter + 2000)

    def get_gt_image(camera):
        if gtisint8:
            return camera.original_image.float().cuda() / 255.0
        return camera.original_image.float().cuda()

    def get_temporal_motion_map(time_idx, camera):
        current_gt = get_gt_image(camera)
        if duration <= 1:
            return torch.zeros((current_gt.shape[1], current_gt.shape[2]), device=current_gt.device, dtype=current_gt.dtype)
        prev_idx = max(time_idx - 1, 0)
        next_idx = min(time_idx + 1, duration - 1)
        prev_cam = traincamlookup[prev_idx].get(camera.image_name, camera)
        next_cam = traincamlookup[next_idx].get(camera.image_name, camera)
        prev_gt = get_gt_image(prev_cam)
        next_gt = get_gt_image(next_cam)
        return torch.abs(current_gt - prev_gt).mean(dim=0) + torch.abs(next_gt - current_gt).mean(dim=0)

    def observation_reliability_enabled():
        return bool(getattr(gaussians, "field_obs_reliability", 0))

    def build_background_priors():
        if not (bool(getattr(gaussians, "field_bg_prior", 0)) or observation_reliability_enabled()):
            return {}
        priors = {}
        cameras_by_name = {}
        for cam in traincameralist:
            cameras_by_name.setdefault(cam.image_name, []).append(cam)
        for image_name, cameras in cameras_by_name.items():
            ordered = sorted(cameras, key=lambda cam: float(cam.timestamp))
            frames = [get_gt_image(cam).detach().float().cpu() for cam in ordered]
            stack = torch.stack(frames, dim=0)
            median = torch.median(stack, dim=0).values
            mad = torch.median(torch.abs(stack - median.unsqueeze(0)), dim=0).values.mean(dim=0)
            stable = mad < float(getattr(gaussians, "field_bg_prior_stability_threshold", 0.12))
            priors[image_name] = {
                "median": median.contiguous(),
                "mad": mad.contiguous(),
                "stable": stable.contiguous(),
            }
            del stack
        print(f"[STEGF] Built temporal-median observation priors for {len(priors)} cameras.")
        return priors

    background_priors = build_background_priors()
    background_prior_gpu_cache = {}
    observation_reliability_debug_cameras = set()
    observation_error_ema = {}
    obs_reset_debug_events = 0
    obs_reset_debug_dir = os.path.join(args.model_path, "obs_reset_debug")
    obs_reset_log_path = os.path.join(obs_reset_debug_dir, "events.jsonl")
    layer_resp_debug_events = 0
    layer_resp_debug_dir = os.path.join(args.model_path, "layer_responsibility_debug")

    def obs_reset_debug_enabled():
        return bool(getattr(gaussians, "field_obs_reset_debug", 0))

    def obs_reset_log_zero_enabled():
        return bool(getattr(gaussians, "field_obs_reset_log_zero", 1))

    def obs_reset_due(iteration):
        if not bool(getattr(gaussians, "field_obs_reset", 0)):
            return False
        schedule_raw = str(getattr(gaussians, "field_obs_reset_schedule", "")).strip()
        if schedule_raw:
            scheduled_iters = set()
            for item in schedule_raw.split(","):
                item = item.strip()
                if item:
                    scheduled_iters.add(int(item))
            return int(iteration) in scheduled_iters
        start = int(getattr(gaussians, "field_obs_reset_start", 1500))
        until = int(getattr(gaussians, "field_obs_reset_until", 9000))
        interval = max(int(getattr(gaussians, "field_obs_reset_interval", 500)), 1)
        return int(iteration) >= start and int(iteration) <= until and (int(iteration) - start) % interval == 0

    def global_reset_due(iteration):
        if not bool(getattr(gaussians, "field_global_reset", 0)):
            return False
        schedule_raw = str(getattr(gaussians, "field_global_reset_schedule", "")).strip()
        if not schedule_raw:
            return False
        scheduled_iters = set()
        for item in schedule_raw.split(","):
            item = item.strip()
            if item:
                scheduled_iters.add(int(item))
        return int(iteration) in scheduled_iters

    def bg_prior_due(iteration):
        if not bool(getattr(gaussians, "field_bg_prior", 0)):
            return False
        start = int(getattr(gaussians, "field_bg_prior_start", 3200))
        until = int(getattr(gaussians, "field_bg_prior_until", 9000))
        interval = max(int(getattr(gaussians, "field_bg_prior_interval", 500)), 1)
        return int(iteration) >= start and int(iteration) <= until and (int(iteration) - start) % interval == 0

    def freq_prior_due(iteration):
        if not bool(getattr(gaussians, "field_freq_prior_on_reset_only", 0)):
            return True
        if str(getattr(gaussians, "field_bg_prior_source", "background")).lower() == "obs_reset":
            return bg_prior_due(iteration)
        return obs_reset_due(iteration)

    def obs_reset_uses_contribution():
        return str(getattr(gaussians, "field_obs_reset_selection_mode", "center")).lower() == "contribution"

    def strip_debug_tensors(payload):
        return {
            key: _jsonable(value)
            for key, value in payload.items()
            if not key.startswith("_")
        }

    def save_observation_reset_debug(iteration, camera, gt_image, image, unreliable_mask, stats):
        nonlocal obs_reset_debug_events
        if not bool(getattr(gaussians, "field_obs_reset", 0)) or not isinstance(stats, dict):
            return
        if stats.get("reason") == "outside_schedule":
            return
        selected_count = int(stats.get("selected_points", 0))
        if selected_count <= 0 and not obs_reset_log_zero_enabled():
            return

        os.makedirs(obs_reset_debug_dir, exist_ok=True)
        serializable = strip_debug_tensors(stats)
        serializable["iteration"] = int(iteration)
        serializable["camera"] = str(getattr(camera, "image_name", ""))
        serializable["timestamp"] = float(getattr(camera, "timestamp", 0.0))
        with open(obs_reset_log_path, "a") as f:
            f.write(json.dumps(serializable, sort_keys=True) + "\n")

        if not obs_reset_debug_enabled():
            return
        max_events = int(getattr(gaussians, "field_obs_reset_debug_max_events", 32))
        if max_events > 0 and obs_reset_debug_events >= max_events:
            return
        if selected_count <= 0:
            return
        obs_reset_debug_events += 1

        safe_name = str(getattr(camera, "image_name", "camera")).replace("/", "_").replace("\\", "_")
        event_dir = os.path.join(
            obs_reset_debug_dir,
            f"{int(iteration):06d}_{safe_name}_t{float(getattr(camera, 'timestamp', 0.0)):.4f}_{selected_count}",
        )
        os.makedirs(event_dir, exist_ok=True)
        save_debug_image(os.path.join(event_dir, "gt.png"), gt_image.detach().float().clamp(0.0, 1.0))
        save_debug_image(os.path.join(event_dir, "render.png"), image.detach().float().clamp(0.0, 1.0))
        save_debug_image(os.path.join(event_dir, "unreliable_mask.png"), unreliable_mask.float())
        save_debug_image(os.path.join(event_dir, "unreliable_overlay.png"), overlay_debug_mask(gt_image, unreliable_mask, (1.0, 0.0, 0.0), alpha=0.50))

        masks = stats.get("_debug_masks", {})
        for name, mask in masks.items():
            save_debug_image(os.path.join(event_dir, f"{name}.png"), mask.float())
        if "in_region_points" in masks:
            save_debug_image(os.path.join(event_dir, "in_region_overlay.png"), overlay_debug_mask(gt_image, masks["in_region_points"], (0.0, 0.4, 1.0), alpha=0.70))
        if "candidate_points" in masks:
            save_debug_image(os.path.join(event_dir, "candidate_overlay.png"), overlay_debug_mask(gt_image, masks["candidate_points"], (1.0, 1.0, 0.0), alpha=0.75))
        if "reset_points" in masks:
            save_debug_image(os.path.join(event_dir, "reset_overlay.png"), overlay_debug_mask(gt_image, masks["reset_points"], (1.0, 0.0, 0.0), alpha=0.85))
        with open(os.path.join(event_dir, "meta.json"), "w") as f:
            json.dump(serializable, f, indent=2, sort_keys=True)

    def get_cached_background_prior(camera):
        image_name = str(camera.image_name)
        prior = background_priors.get(image_name)
        if prior is None:
            return None
        cached = background_prior_gpu_cache.get(image_name)
        if cached is None:
            cached = {
                "median": prior["median"].to(device="cuda", dtype=torch.float32),
                "mad": prior["mad"].to(device="cuda", dtype=torch.float32),
                "stable": prior["stable"].to(device="cuda"),
            }
            background_prior_gpu_cache[image_name] = cached
        return cached

    def get_background_prior(camera):
        prior = get_cached_background_prior(camera)
        if prior is None:
            return None, None
        median = prior["median"]
        stable = prior["stable"]
        return median, stable

    def get_observation_prior(camera):
        prior = get_cached_background_prior(camera)
        if prior is None:
            return None, None, None
        median = prior["median"]
        mad = prior["mad"]
        stable = prior["stable"]
        return median, mad, stable

    def get_background_visible_mask(camera, gt_image):
        median, stable = get_background_prior(camera)
        if median is None:
            return None, None
        current_to_median = torch.abs(gt_image.detach().float() - median).mean(dim=0)
        visible = stable & (current_to_median < float(getattr(gaussians, "field_bg_prior_visible_threshold", 0.08)))
        return median, visible

    def get_background_prior_loss(camera, image, gt_image):
        if not bool(getattr(gaussians, "field_bg_prior", 0)):
            return None
        weight = float(getattr(gaussians, "field_bg_prior_loss_weight", 0.0))
        if weight <= 0.0:
            return None
        median, visible = get_background_visible_mask(camera, gt_image)
        if median is None or visible is None or torch.count_nonzero(visible) == 0:
            return None
        bg_error = torch.abs(image.float() - median).mean(dim=0)
        return weight * bg_error[visible].mean()

    def save_observation_reliability_debug(camera, gt_image, median, mad, current_to_median,
                                           temporal_motion_map, dynamic_mask, error_map,
                                           error_ema, unreliable_mask, reliability):
        if not bool(getattr(gaussians, "field_obs_reliability_debug", 0)):
            return
        safe_name = str(camera.image_name).replace("/", "_").replace("\\", "_")
        if safe_name in observation_reliability_debug_cameras:
            return
        observation_reliability_debug_cameras.add(safe_name)

        event_dir = os.path.join(args.model_path, "observation_reliability_debug", safe_name)
        os.makedirs(event_dir, exist_ok=True)
        save_debug_image(os.path.join(event_dir, "gt.png"), gt_image.detach().float().clamp(0.0, 1.0))
        save_debug_image(os.path.join(event_dir, "temporal_median.png"), median.detach().float().clamp(0.0, 1.0))
        save_debug_image(os.path.join(event_dir, "mad.png"), normalize_debug_map(mad))
        save_debug_image(os.path.join(event_dir, "current_to_median.png"), normalize_debug_map(current_to_median))
        save_debug_image(os.path.join(event_dir, "temporal_motion.png"), normalize_debug_map(temporal_motion_map))
        save_debug_image(os.path.join(event_dir, "dynamic_mask.png"), dynamic_mask.float())
        save_debug_image(os.path.join(event_dir, "render_error.png"), normalize_debug_map(error_map))
        save_debug_image(os.path.join(event_dir, "error_ema.png"), normalize_debug_map(error_ema))
        save_debug_image(os.path.join(event_dir, "low_reliability_mask.png"), unreliable_mask.float())
        save_debug_image(os.path.join(event_dir, "reliability.png"), reliability.detach().float().clamp(0.0, 1.0))
        save_debug_image(os.path.join(event_dir, "unreliable_overlay.png"), overlay_debug_mask(gt_image, unreliable_mask, (1.0, 0.0, 0.0), alpha=0.55))

    def get_observation_reliability(
        iteration,
        camera,
        image,
        gt_image,
        temporal_motion_map,
        update_ema=True,
        use_accumulated=False,
    ):
        if not observation_reliability_enabled():
            return None, None
        median, mad, _ = get_observation_prior(camera)
        if median is None:
            return None, None

        eps = 1e-6
        start_iter = int(getattr(gaussians, "field_obs_reliability_start", 1500))
        if int(iteration) < start_iter:
            return None, None
        until_iter = int(getattr(gaussians, "field_obs_reliability_until", -1))
        if update_ema and until_iter >= 0 and int(iteration) > until_iter:
            update_ema = False

        floor = max(0.0, min(float(getattr(gaussians, "field_obs_reliability_floor", 0.35)), 1.0))
        ema_decay = max(0.0, min(float(getattr(gaussians, "field_obs_reliability_ema", 0.05)), 1.0))
        motion_threshold = max(float(getattr(gaussians, "field_obs_reliability_motion_threshold", 0.12)), eps)
        dynamic_dilate = max(int(getattr(gaussians, "field_obs_reliability_dynamic_dilate", 5)), 0)
        quantile = float(getattr(gaussians, "field_obs_reliability_error_quantile", 0.90))
        error_threshold = float(getattr(gaussians, "field_obs_reliability_error_threshold", 0.0))
        min_threshold = max(float(getattr(gaussians, "field_obs_reliability_min_error", 0.03)), 0.0)

        current_to_median = torch.abs(gt_image.detach().float() - median).mean(dim=0)
        dynamic_mask = (temporal_motion_map.detach().float() > motion_threshold) | (current_to_median > float(getattr(gaussians, "field_obs_reliability_diff_threshold", 0.12)))
        dynamic_mask = dilate_mask(dynamic_mask, dynamic_dilate)

        color_error = torch.abs(image.detach().float() - gt_image.detach().float()).mean(dim=0)
        window = int(getattr(gaussians, "field_obs_reliability_local_window", 31))
        render_norm = local_normalize_gray(rgb_to_luma(image.detach().float()), window)
        gt_norm = local_normalize_gray(rgb_to_luma(gt_image.detach().float()), window)
        structural_error = torch.abs(render_norm - gt_norm)
        structural_weight = max(0.0, min(float(getattr(gaussians, "field_obs_reliability_structural_weight", 0.5)), 1.0))
        color_score = normalize_debug_map(color_error)
        structural_score = normalize_debug_map(structural_error)
        error_map = (1.0 - structural_weight) * color_score + structural_weight * structural_score
        error_map = error_map.detach()

        key = str(camera.image_name)
        prev = observation_error_ema.get(key)
        non_dynamic = ~dynamic_mask
        if update_ema:
            if prev is None or prev.shape != error_map.shape:
                ema = error_map.clone()
            else:
                ema = (1.0 - ema_decay) * prev + ema_decay * error_map
            if prev is not None and prev.shape == error_map.shape:
                ema = torch.where(non_dynamic, ema, prev)
            observation_error_ema[key] = ema.detach()
        elif use_accumulated:
            if prev is None or prev.shape != error_map.shape:
                reliability = torch.ones_like(error_map, dtype=image.dtype).detach()
                unreliable_mask = torch.zeros_like(error_map, dtype=torch.bool)
                return reliability, unreliable_mask
            ema = prev.to(device=error_map.device, dtype=error_map.dtype)
        else:
            ema = error_map.clone()

        values = ema[non_dynamic & torch.isfinite(ema)]
        unreliable_mask = torch.zeros_like(ema, dtype=torch.bool)
        if values.numel() > 0:
            if error_threshold > 0.0:
                threshold = torch.tensor(
                    max(error_threshold, min_threshold),
                    device=values.device,
                    dtype=values.dtype,
                )
            else:
                quantile = max(0.0, min(quantile, 1.0))
                threshold = torch.quantile(values.float(), quantile)
                threshold = torch.maximum(threshold, torch.tensor(min_threshold, device=threshold.device, dtype=threshold.dtype))
            unreliable_mask = non_dynamic & (ema >= threshold)

        reliability = torch.ones_like(ema, dtype=image.dtype)
        reliability = torch.where(unreliable_mask, torch.full_like(reliability, floor), reliability).detach()

        save_observation_reliability_debug(
            camera,
            gt_image,
            median,
            mad,
            current_to_median,
            temporal_motion_map,
            dynamic_mask,
            error_map,
            ema,
            unreliable_mask,
            reliability,
        )
        return reliability, unreliable_mask.detach()

    def get_persistent_need_mask_from_ema(camera, target_shape, device="cuda"):
        key = str(camera.image_name)
        prev = observation_error_ema.get(key)
        height, width = int(target_shape[0]), int(target_shape[1])
        if prev is None or prev.shape != (height, width):
            return torch.zeros((height, width), device=device, dtype=torch.bool), {
                "enabled": True,
                "reason": "missing_ema",
                "pixels": 0,
            }

        ema = prev.to(device=device, dtype=torch.float32)
        finite = torch.isfinite(ema)
        values = ema[finite]
        need_mask = torch.zeros_like(ema, dtype=torch.bool)
        stats = {
            "enabled": True,
            "reason": "ok",
            "finite_pixels": int(torch.count_nonzero(finite).item()),
            "pixels": 0,
        }
        if values.numel() == 0:
            stats["reason"] = "empty_finite"
            return need_mask, stats

        quantile = float(getattr(gaussians, "field_obs_reliability_error_quantile", 0.90))
        error_threshold = float(getattr(gaussians, "field_obs_reliability_error_threshold", 0.0))
        min_threshold = max(float(getattr(gaussians, "field_obs_reliability_min_error", 0.03)), 0.0)
        if error_threshold > 0.0:
            threshold = torch.tensor(
                max(error_threshold, min_threshold),
                device=values.device,
                dtype=values.dtype,
            )
            stats["threshold_mode"] = "fixed"
        else:
            quantile = max(0.0, min(quantile, 1.0))
            threshold = torch.quantile(values.float(), quantile)
            threshold = torch.maximum(
                threshold,
                torch.tensor(min_threshold, device=threshold.device, dtype=threshold.dtype),
            )
            stats["threshold_mode"] = "quantile"
            stats["quantile"] = float(quantile)
        need_mask = finite & (ema >= threshold)
        stats.update({
            "threshold": float(threshold.detach().cpu()),
            "min_threshold": float(min_threshold),
            "error_threshold": float(error_threshold),
            "pixels": int(torch.count_nonzero(need_mask).item()),
        })
        return need_mask.detach(), stats

    def apply_observation_reliability_to_loss_image(image, gt_image, reliability):
        if reliability is None:
            return image
        weight = reliability.to(device=image.device, dtype=image.dtype).unsqueeze(0)
        return image * weight + gt_image.detach() * (1.0 - weight)

    def get_unreliable_rgb_boost_loss(image, gt_image, unreliable_mask):
        if not bool(getattr(gaussians, "field_obs_boost_unreliable_loss", 0)):
            return None
        weight = float(getattr(gaussians, "field_obs_boost_weight", 0.0))
        if weight <= 0.0:
            return None
        if unreliable_mask is None or torch.count_nonzero(unreliable_mask) == 0:
            return None
        mask = unreliable_mask.to(device=image.device, dtype=torch.bool)
        rgb_error = torch.abs(image - gt_image).mean(dim=0)
        if rgb_error.shape != mask.shape:
            return None
        return weight * rgb_error[mask].mean()

    def static_radiance_enabled(iteration):
        return (
            bool(getattr(gaussians, "field_static_radiance_branch", 0))
            and int(iteration) >= int(getattr(gaussians, "field_static_radiance_start", 3000))
            and observation_reliability_enabled()
        )

    def get_static_radiance_mask(iteration, camera, gt_image, temporal_motion_map):
        if not static_radiance_enabled(iteration):
            return None
        _, unreliable_mask = get_observation_reliability(
            iteration,
            camera,
            gt_image,
            gt_image,
            temporal_motion_map,
            update_ema=False,
            use_accumulated=True,
        )
        return unreliable_mask

    def get_background_median_loss(camera, image, unreliable_mask):
        if not bool(getattr(gaussians, "field_bg_median_loss", 0)):
            return None
        weight = float(getattr(gaussians, "field_bg_median_loss_weight", 0.0))
        if weight <= 0.0:
            return None
        if unreliable_mask is None or torch.count_nonzero(unreliable_mask) == 0:
            return None
        prior = get_cached_background_prior(camera)
        if prior is None or prior.get("median") is None:
            return None
        reference = prior["median"].to(device=image.device, dtype=image.dtype)
        mask = unreliable_mask.to(device=image.device, dtype=torch.bool)
        median_error = torch.abs(image - reference).mean(dim=0)
        if median_error.shape != mask.shape:
            return None
        return weight * median_error[mask].mean()

    def get_depthpro_foreground_depth_loss(iteration, camera, render_depth, unreliable_mask):
        if not bool(getattr(gaussians, "field_depthpro_supervision", 0)):
            return None
        weight = float(getattr(gaussians, "field_depthpro_loss_weight", 0.0))
        if weight <= 0.0:
            return None
        start_iter = int(getattr(gaussians, "field_depthpro_start", 0))
        until_iter = int(getattr(gaussians, "field_depthpro_until", -1))
        if int(iteration) < start_iter:
            return None
        if until_iter >= 0 and int(iteration) > until_iter:
            return None
        if render_depth is None:
            return None

        if render_depth.dim() == 3:
            depth_pred = render_depth.squeeze(0)
        else:
            depth_pred = render_depth
        target_depth, _ = get_depthpro_depth(camera, depth_pred.shape)
        if target_depth is None:
            return None

        valid = torch.isfinite(target_depth) & (target_depth > 1e-4) & torch.isfinite(depth_pred) & (depth_pred > 1e-4)
        max_depth = float(getattr(gaussians, "field_depthpro_max_depth", 2.0))
        if max_depth > 0.0:
            valid = valid & (target_depth <= max_depth)
        if bool(getattr(gaussians, "field_depthpro_use_beit_mask", 1)):
            _, beit_foreground, _ = get_dense_beit_foreground_mask(camera, depth_pred.shape)
            if beit_foreground is None:
                return None
            valid = valid & beit_foreground.to(device=valid.device, dtype=torch.bool)
        if bool(getattr(gaussians, "field_depthpro_exclude_unreliable", 1)) and unreliable_mask is not None:
            valid = valid & (~unreliable_mask.to(device=valid.device, dtype=torch.bool))

        min_pixels = int(getattr(gaussians, "field_depthpro_min_pixels", 256))
        if torch.count_nonzero(valid) < max(min_pixels, 1):
            return None

        error = torch.abs(depth_pred - target_depth)
        clamp = float(getattr(gaussians, "field_depthpro_error_clamp", 1.0))
        if clamp > 0.0:
            error = torch.clamp(error, max=clamp)
        return weight * error[valid].mean()

    def get_depth_compensated_scale_loss(iteration, camera, means3D):
        if not bool(getattr(gaussians, "field_scale_reg", 0)):
            return None
        weight = float(getattr(gaussians, "field_scale_reg_weight", 0.0))
        if weight <= 0.0:
            return None
        start_iter = int(getattr(gaussians, "field_scale_reg_start", 9000))
        until_iter = int(getattr(gaussians, "field_scale_reg_until", -1))
        if int(iteration) < start_iter:
            return None
        if until_iter >= 0 and int(iteration) > until_iter:
            return None
        if means3D is None or means3D.numel() == 0:
            return None
        if not hasattr(gaussians, "get_scaling"):
            return None

        scales = gaussians.get_scaling
        if scales is None or scales.numel() == 0:
            return None
        if scales.shape[0] != means3D.shape[0]:
            return None

        depth_mode = str(getattr(gaussians, "field_scale_reg_depth_mode", "euclidean")).lower()
        if depth_mode in ("euclidean", "distance", "camera_distance", "camera_center"):
            camera_center = getattr(camera, "camera_center", None)
            if camera_center is None:
                return None
            camera_center = camera_center.to(device=means3D.device, dtype=means3D.dtype).view(1, 3)
            depth = torch.linalg.norm(means3D - camera_center, dim=1).detach()
        else:
            view_transform = getattr(camera, "world_view_transform", None)
            if view_transform is not None:
                ones = torch.ones((means3D.shape[0], 1), device=means3D.device, dtype=means3D.dtype)
                means_h = torch.cat((means3D, ones), dim=1)
                view_transform = view_transform.to(device=means3D.device, dtype=means3D.dtype)
                depth = torch.abs((means_h @ view_transform)[:, 2]).detach()
                if not torch.all(torch.isfinite(depth)):
                    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                camera_center = getattr(camera, "camera_center", None)
                if camera_center is None:
                    return None
                camera_center = camera_center.to(device=means3D.device, dtype=means3D.dtype).view(1, 3)
                depth = torch.linalg.norm(means3D - camera_center, dim=1).detach()
        depth = depth.clamp_min(1e-6)

        base_limit = max(float(getattr(gaussians, "field_scale_reg_base_limit", 0.3)), 1e-8)
        depth_ref = max(float(getattr(gaussians, "field_scale_reg_depth_ref", 8.0)), 1e-6)
        gamma = max(float(getattr(gaussians, "field_scale_reg_depth_gamma", 0.75)), 0.0)
        max_boost = max(float(getattr(gaussians, "field_scale_reg_max_boost", 8.0)), 1.0)
        depth_boost = torch.clamp((depth / depth_ref).pow(gamma), min=1.0, max=max_boost)
        scale_limit = (base_limit * depth_boost).to(device=scales.device, dtype=scales.dtype)

        scale_max = torch.max(scales, dim=1).values
        excess = torch.relu(scale_max - scale_limit)
        if torch.count_nonzero(excess > 0) == 0:
            return None
        return weight * torch.mean(excess * excess)

    freq_prior_cache = {}

    def get_frequency_reference(camera):
        reference_mode = str(getattr(gaussians, "field_freq_prior_reference", "median")).lower()
        if reference_mode != "median":
            return get_gt_image(camera).detach().float()
        image_name = str(camera.image_name)
        cached = freq_prior_cache.get(image_name)
        if cached is not None:
            return cached
        prior = get_cached_background_prior(camera)
        if prior is None:
            return None
        cached = prior["median"].detach().float()
        freq_prior_cache[image_name] = cached
        return cached

    def highpass_frequency_mask(size, highpass, device, dtype):
        freq = torch.fft.fftfreq(size, d=1.0, device=device)
        fy, fx = torch.meshgrid(freq, freq, indexing="ij")
        radius = torch.sqrt(fx * fx + fy * fy)
        threshold = max(float(highpass), 0.0) * 0.5
        return (radius >= threshold).to(dtype=dtype)

    def patch_frequency_loss(render_patch, ref_patch, highpass):
        render_gray = rgb_to_luma(render_patch)
        ref_gray = rgb_to_luma(ref_patch)
        render_fft = torch.fft.fft2(render_gray.float())
        ref_fft = torch.fft.fft2(ref_gray.float())
        render_mag = torch.log1p(torch.abs(render_fft))
        ref_mag = torch.log1p(torch.abs(ref_fft))
        mask = highpass_frequency_mask(render_gray.shape[-1], highpass, render_gray.device, render_gray.dtype)
        denom = torch.clamp(mask.sum(), min=1.0)
        return torch.sum(torch.abs(render_mag - ref_mag) * mask) / denom

    freq_prior_debug_events = 0
    freq_prior_debug_camera_names = set()

    def save_frequency_prior_debug(iteration, camera, image, gt_image, reference, unreliable_mask, error_map,
                                   selected_blocks, selected_patch_mask, patch_size, highpass, losses):
        nonlocal freq_prior_debug_events
        nonlocal freq_prior_debug_camera_names
        if not bool(getattr(gaussians, "field_freq_prior_debug", 0)):
            return
        safe_name = str(camera.image_name).replace("/", "_").replace("\\", "_")
        debug_mode = str(getattr(gaussians, "field_freq_prior_debug_mode", "first_per_camera")).lower()
        if debug_mode == "first_per_camera":
            if safe_name in freq_prior_debug_camera_names:
                return
            freq_prior_debug_camera_names.add(safe_name)
        max_events = int(getattr(gaussians, "field_freq_prior_debug_max_events", 32))
        if max_events > 0 and freq_prior_debug_events >= max_events:
            return
        freq_prior_debug_events += 1

        timestamp = float(getattr(camera, "timestamp", 0.0))
        event_name = f"{int(iteration):06d}_{safe_name}_t{timestamp:.4f}"
        event_dir = os.path.join(args.model_path, "freq_prior_debug", event_name)
        os.makedirs(event_dir, exist_ok=True)

        gt_vis = gt_image.detach().float().clamp(0.0, 1.0)
        render_vis = image.detach().float().clamp(0.0, 1.0)
        ref_vis = reference.detach().float().clamp(0.0, 1.0)
        selected_patch_mask = selected_patch_mask.detach().bool()
        unreliable_mask = unreliable_mask.detach().bool()
        error_vis = normalize_debug_map(error_map.detach())

        save_debug_image(os.path.join(event_dir, "gt.png"), gt_vis)
        save_debug_image(os.path.join(event_dir, "render.png"), render_vis)
        save_debug_image(os.path.join(event_dir, "reference.png"), ref_vis)
        save_debug_image(os.path.join(event_dir, "unreliable_mask.png"), unreliable_mask.float())
        save_debug_image(os.path.join(event_dir, "selected_patch_mask.png"), selected_patch_mask.float())
        save_debug_image(os.path.join(event_dir, "render_to_reference_error.png"), error_vis)
        save_debug_image(os.path.join(event_dir, "selected_patch_overlay.png"), overlay_debug_mask(gt_vis, selected_patch_mask, (1.0, 0.0, 0.0), alpha=0.55))

        first_block = selected_blocks[0] if len(selected_blocks) > 0 else None
        if first_block is not None:
            y0, x0, score, mask_ratio = first_block
            render_patch = image[:, y0:y0 + patch_size, x0:x0 + patch_size]
            ref_patch = reference[:, y0:y0 + patch_size, x0:x0 + patch_size].to(device=image.device, dtype=image.dtype)
            gt_patch = gt_image[:, y0:y0 + patch_size, x0:x0 + patch_size]
            save_debug_image(os.path.join(event_dir, "patch_render.png"), render_patch)
            save_debug_image(os.path.join(event_dir, "patch_reference.png"), ref_patch)
            save_debug_image(os.path.join(event_dir, "patch_gt.png"), gt_patch)
            patch_mask = selected_patch_mask[y0:y0 + patch_size, x0:x0 + patch_size].float()
            save_debug_image(os.path.join(event_dir, "patch_mask.png"), patch_mask)

            render_gray = rgb_to_luma(render_patch).float()
            ref_gray = rgb_to_luma(ref_patch).float()
            render_mag = torch.log1p(torch.abs(torch.fft.fftshift(torch.fft.fft2(render_gray))))
            ref_mag = torch.log1p(torch.abs(torch.fft.fftshift(torch.fft.fft2(ref_gray))))
            freq_diff = torch.abs(render_mag - ref_mag)
            highpass_mask = torch.fft.fftshift(highpass_frequency_mask(patch_size, highpass, image.device, render_gray.dtype))
            save_debug_image(os.path.join(event_dir, "fft_render_mag.png"), normalize_debug_map(render_mag))
            save_debug_image(os.path.join(event_dir, "fft_reference_mag.png"), normalize_debug_map(ref_mag))
            save_debug_image(os.path.join(event_dir, "fft_abs_diff.png"), normalize_debug_map(freq_diff))
            save_debug_image(os.path.join(event_dir, "fft_highpass_mask.png"), highpass_mask.float())

        meta = {
            "iteration": int(iteration),
            "camera": str(camera.image_name),
            "timestamp": timestamp,
            "patch_size": int(patch_size),
            "highpass": float(highpass),
            "selected_patch_count": int(len(selected_blocks)),
            "unreliable_pixels": int(torch.count_nonzero(unreliable_mask).item()),
            "selected_patch_pixels": int(torch.count_nonzero(selected_patch_mask).item()),
            "loss_values": [float(item.detach().cpu()) for item in losses],
            "selected_blocks": [
                {
                    "y0": int(y0),
                    "x0": int(x0),
                    "error_score": float(score),
                    "mask_ratio": float(mask_ratio),
                }
                for y0, x0, score, mask_ratio in selected_blocks
            ],
        }
        with open(os.path.join(event_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def get_frequency_prior_loss(iteration, camera, image, gt_image, unreliable_mask):
        if not bool(getattr(gaussians, "field_freq_prior", 0)):
            return None
        if not freq_prior_due(iteration):
            return None
        weight = float(getattr(gaussians, "field_freq_prior_weight", 0.0))
        if weight <= 0.0:
            return None
        if unreliable_mask is None or torch.count_nonzero(unreliable_mask) == 0:
            return None
        if int(iteration) < int(getattr(gaussians, "field_freq_prior_start", 3500)):
            return None
        if int(iteration) > int(getattr(gaussians, "field_freq_prior_until", 12000)):
            return None
        reference = get_frequency_reference(camera)
        if reference is None:
            return None

        patch_size = max(int(getattr(gaussians, "field_freq_prior_patch_size", 32)), 8)
        max_patches = max(int(getattr(gaussians, "field_freq_prior_max_patches", 16)), 1)
        min_mask_ratio = max(float(getattr(gaussians, "field_freq_prior_min_mask_ratio", 0.05)), 0.0)
        highpass = float(getattr(gaussians, "field_freq_prior_highpass", 0.25))

        _, height, width = image.shape
        mask = unreliable_mask.to(device=image.device, dtype=torch.bool)
        valid_height = (height // patch_size) * patch_size
        valid_width = (width // patch_size) * patch_size
        if valid_height < patch_size or valid_width < patch_size:
            return None

        mask_crop = mask[:valid_height, :valid_width].float().unsqueeze(0).unsqueeze(0)
        error_map = torch.abs(image.detach() - gt_image.detach()).mean(dim=0)
        error_crop = error_map[:valid_height, :valid_width].unsqueeze(0).unsqueeze(0)
        mask_mean = F.avg_pool2d(mask_crop, kernel_size=patch_size, stride=patch_size).squeeze()
        error_mean = F.avg_pool2d(error_crop * mask_crop, kernel_size=patch_size, stride=patch_size).squeeze()
        error_mean = error_mean / torch.clamp(mask_mean, min=1.0 / float(patch_size * patch_size))

        valid_blocks = mask_mean >= min_mask_ratio
        if torch.count_nonzero(valid_blocks) == 0:
            return None
        flat_scores = error_mean.flatten()
        flat_valid = valid_blocks.flatten()
        valid_indices = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
        valid_scores = flat_scores[valid_indices]
        topk = min(max_patches, valid_scores.numel())
        selected = valid_indices[torch.topk(valid_scores, k=topk, largest=True).indices]
        blocks_x = valid_width // patch_size

        losses = []
        selected_blocks = []
        selected_patch_mask = torch.zeros((height, width), device=image.device, dtype=torch.bool)
        for flat_index in selected.tolist():
            y0 = int(flat_index // blocks_x) * patch_size
            x0 = int(flat_index % blocks_x) * patch_size
            render_patch = image[:, y0:y0 + patch_size, x0:x0 + patch_size]
            ref_patch = reference[:, y0:y0 + patch_size, x0:x0 + patch_size].to(device=image.device, dtype=image.dtype)
            losses.append(patch_frequency_loss(render_patch, ref_patch, highpass))
            selected_patch_mask[y0:y0 + patch_size, x0:x0 + patch_size] = True
            selected_blocks.append(
                (
                    y0,
                    x0,
                    float(flat_scores[flat_index].detach().cpu()),
                    float(mask_mean.flatten()[flat_index].detach().cpu()),
                )
            )
        if len(losses) == 0:
            return None
        save_frequency_prior_debug(
            iteration,
            camera,
            image,
            gt_image,
            reference.to(device=image.device, dtype=image.dtype),
            mask,
            torch.abs(image.detach() - reference.to(device=image.device, dtype=image.dtype).detach()).mean(dim=0),
            selected_blocks,
            selected_patch_mask,
            patch_size,
            highpass,
            losses,
        )
        return weight * torch.stack(losses).mean()

    def bg_only_train_due(iteration):
        if not bool(getattr(gaussians, "field_bg_only_train", 0)):
            return False
        iteration = int(iteration)
        start = int(getattr(gaussians, "field_bg_only_start", 3500))
        until = int(getattr(gaussians, "field_bg_only_until", 6000))
        interval = max(int(getattr(gaussians, "field_bg_only_interval", 1)), 1)
        return iteration >= start and iteration <= until and iteration % interval == 0

    def get_bg_only_supervision_mask(camera, unreliable_mask):
        if unreliable_mask is None:
            return None
        mask = unreliable_mask.detach().bool()
        if bool(getattr(gaussians, "field_bg_only_da3_filter", 1)):
            _, da3_foreground_mask, _ = get_dense_da3_foreground_mask(camera, mask.shape)
            if da3_foreground_mask is not None:
                mask = mask & (~da3_foreground_mask.to(device=mask.device, dtype=torch.bool))
        min_pixels = max(int(getattr(gaussians, "field_bg_only_min_pixels", 128)), 0)
        if int(torch.count_nonzero(mask).item()) < min_pixels:
            return None
        return mask

    def get_bg_only_render_loss(iteration, camera, gt_image, unreliable_mask):
        if not bg_only_train_due(iteration):
            return None, 0
        if not hasattr(gaussians, "get_background_candidate_mask"):
            return None, 0
        point_mask = gaussians.get_background_candidate_mask()
        if torch.count_nonzero(point_mask) == 0:
            return None, 0
        supervision_mask = get_bg_only_supervision_mask(camera, unreliable_mask)
        if supervision_mask is None:
            return None, 0
        bg_render_pkg = render(
            camera,
            gaussians,
            pipe,
            background,
            override_color=None,
            basicfunction=rbfbasefunction,
            GRsetting=GRsetting,
            GRzer=GRzer,
            time_conditioned=None,
            render_point_mask=point_mask,
        )
        bg_image = bg_render_pkg["render"]
        pixel_error = torch.abs(bg_image - gt_image.detach()).mean(dim=0)
        mask = supervision_mask.to(device=pixel_error.device, dtype=torch.bool)
        if pixel_error.shape != mask.shape or torch.count_nonzero(mask) == 0:
            return None, 0
        weight = float(getattr(gaussians, "field_bg_only_loss_weight", 1.0))
        return weight * pixel_error[mask].mean(), int(torch.count_nonzero(mask).item())

    bg_prior_debug_events = 0
    bg_prior_debug_camera_names = set()
    bg_dense_add_done = False
    bg_dense_add_debug_events = 0
    bg_dense_da3_cache = {
        "loaded": False,
        "path": None,
        "depth": None,
        "name_to_idx": {},
        "warned": False,
    }
    bg_dense_beit_cache = {
        "loaded": False,
        "path": None,
        "depth_like": {},
        "warned": False,
    }
    depthpro_cache = {
        "loaded": False,
        "path": None,
        "depth": {},
        "warned": False,
    }

    def parse_bg_scan_time_indices():
        raw = str(getattr(gaussians, "field_bg_prior_scan_time_indices", "")).strip()
        if raw:
            indices = []
            for item in raw.split(","):
                item = item.strip()
                if item:
                    indices.append(max(0, min(int(item), duration - 1)))
            if indices:
                return sorted(set(indices))
        if duration <= 1:
            return [0]
        defaults = [0, duration // 4, duration // 2, (duration * 3) // 4, duration - 1]
        return sorted(set(max(0, min(int(idx), duration - 1)) for idx in defaults))

    def build_background_prior_schedule():
        if not bool(getattr(gaussians, "field_bg_prior", 0)):
            return []
        if str(getattr(gaussians, "field_bg_prior_schedule_mode", "scan")).lower() != "scan":
            return []
        time_indices = parse_bg_scan_time_indices()
        image_names = sorted({cam.image_name for cam in traincameralist})
        scheduled = []
        for time_idx in time_indices:
            lookup = traincamlookup.get(time_idx, {})
            for image_name in image_names:
                cam = lookup.get(image_name)
                if cam is not None and cam.image_name in background_priors:
                    scheduled.append(cam)
        rng = random.Random(20240510)
        rng.shuffle(scheduled)
        print(f"[STEGF] Built background prior scan schedule with {len(scheduled)} view-time frames.")
        return scheduled

    background_prior_schedule = build_background_prior_schedule()

    def background_prior_key(time_idx, camera):
        return (int(time_idx), str(camera.image_name))

    def build_background_prior_pending():
        if not bool(getattr(gaussians, "field_bg_prior", 0)):
            return set()
        pending = set()
        for time_idx, cameras in traincamdict.items():
            for cam in cameras:
                if cam.image_name in background_priors:
                    pending.add(background_prior_key(time_idx, cam))
        if str(getattr(gaussians, "field_bg_prior_schedule_mode", "scan")).lower() == "on_sample":
            print(f"[STEGF] Built background prior pending table with {len(pending)} view-time frames.")
        return pending

    background_prior_pending = build_background_prior_pending()

    def get_scheduled_background_cameras(iteration, fallback_camera):
        if str(getattr(gaussians, "field_bg_prior_schedule_mode", "scan")).lower() != "scan":
            return [fallback_camera]
        if len(background_prior_schedule) == 0:
            return [fallback_camera]
        start_iter = int(getattr(gaussians, "field_bg_prior_start", 3200))
        interval = max(int(getattr(gaussians, "field_bg_prior_interval", 500)), 1)
        event_index = max((int(iteration) - start_iter) // interval, 0)
        views_per_event = max(int(getattr(gaussians, "field_bg_prior_scan_views_per_event", 1)), 1)
        cameras = []
        for offset in range(views_per_event):
            schedule_index = (event_index * views_per_event + offset) % len(background_prior_schedule)
            cameras.append(background_prior_schedule[schedule_index])
        return cameras

    def dilate_mask(mask, radius):
        radius = int(radius)
        if radius <= 0:
            return mask.bool()
        kernel = radius * 2 + 1
        return F.max_pool2d(mask.float().unsqueeze(0).unsqueeze(0), kernel, stride=1, padding=radius)[0, 0] > 0.5

    def erode_mask(mask, radius):
        radius = int(radius)
        if radius <= 0:
            return mask.bool()
        return ~dilate_mask(~mask.bool(), radius)

    def rgb_to_luma(image):
        weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype).view(3, 1, 1)
        return torch.sum(image * weights, dim=0)

    def local_normalize_gray(gray, window):
        window = max(int(window), 3)
        if window % 2 == 0:
            window += 1
        gray4d = gray.unsqueeze(0).unsqueeze(0)
        mean = F.avg_pool2d(gray4d, window, stride=1, padding=window // 2)
        mean_sq = F.avg_pool2d(gray4d * gray4d, window, stride=1, padding=window // 2)
        var = torch.clamp(mean_sq - mean * mean, min=1e-6)
        return ((gray4d - mean) / torch.sqrt(var))[0, 0]

    def compute_background_error_map(image, median, valid_mask):
        color_error = torch.abs(image.detach().float() - median).mean(dim=0)
        if not bool(getattr(gaussians, "field_bg_prior_exposure_robust", 1)):
            return color_error
        window = int(getattr(gaussians, "field_bg_prior_local_window", 31))
        render_norm = local_normalize_gray(rgb_to_luma(image.detach().float()), window)
        median_norm = local_normalize_gray(rgb_to_luma(median), window)
        structural_error = torch.abs(render_norm - median_norm)
        color_weight = max(0.0, min(1.0, 1.0 - float(getattr(gaussians, "field_bg_prior_structural_weight", 0.5))))
        structural_weight = max(0.0, min(1.0, float(getattr(gaussians, "field_bg_prior_structural_weight", 0.5))))
        color_score = normalize_debug_map(color_error, valid_mask)
        structural_score = normalize_debug_map(structural_error, valid_mask)
        return color_weight * color_score + structural_weight * structural_score

    def pool_blocks(values, block_size, pad_value=0.0):
        height, width = values.shape
        pad_h = (block_size - height % block_size) % block_size
        pad_w = (block_size - width % block_size) % block_size
        padded = F.pad(values.unsqueeze(0).unsqueeze(0), (0, pad_w, 0, pad_h), value=pad_value)
        pooled = F.avg_pool2d(padded, kernel_size=block_size, stride=block_size)[0, 0]
        return pooled, height, width

    def expand_block_mask(block_mask, block_size, height, width):
        expanded = F.interpolate(
            block_mask.float().unsqueeze(0).unsqueeze(0),
            scale_factor=block_size,
            mode="nearest",
        )[0, 0]
        return expanded[:height, :width] > 0.5

    def select_background_prior_pool(base_mask, stable, occlusion_mask, render_to_bg, max_pixels, pixels_per_block,
                                     min_visible_ratio, min_stable_ratio, max_occlusion_ratio, error_quantile):
        block_size = max(int(getattr(gaussians, "field_bg_prior_block_size", 32)), 4)
        pixels_per_block = max(int(pixels_per_block), 1)
        max_pixels = max(int(max_pixels), 1)
        base_mask = base_mask & torch.isfinite(render_to_bg)
        if torch.count_nonzero(base_mask) == 0:
            empty = torch.zeros_like(base_mask, dtype=torch.bool)
            return empty, empty, torch.tensor(0.0, device=render_to_bg.device)

        block_area = float(block_size * block_size)
        base_mean, height, width = pool_blocks(base_mask.float(), block_size)
        stable_mean, _, _ = pool_blocks(stable.float(), block_size)
        occlusion_mean, _, _ = pool_blocks(occlusion_mask.float(), block_size)
        error_sum, _, _ = pool_blocks(render_to_bg * base_mask.float(), block_size)
        error_mean = error_sum / torch.clamp(base_mean, min=1.0 / block_area)

        block_ok = (
            (base_mean >= float(min_visible_ratio))
            & (stable_mean >= float(min_stable_ratio))
            & (occlusion_mean <= float(max_occlusion_ratio))
            & torch.isfinite(error_mean)
        )
        block_scores = error_mean[block_ok]
        if block_scores.numel() == 0:
            empty = torch.zeros_like(base_mask, dtype=torch.bool)
            return empty, empty, torch.tensor(0.0, device=render_to_bg.device)

        threshold = torch.quantile(block_scores.float(), max(0.0, min(float(error_quantile), 1.0)))
        selected_blocks = block_ok & (error_mean >= threshold)
        selected_block_indices = torch.nonzero(selected_blocks.reshape(-1), as_tuple=False).squeeze(1)
        if selected_block_indices.numel() == 0:
            empty = torch.zeros_like(base_mask, dtype=torch.bool)
            return empty, empty, threshold

        max_blocks = max(int(np.ceil(max_pixels / float(pixels_per_block))), 1)
        if selected_block_indices.numel() > max_blocks:
            values = error_mean.reshape(-1)[selected_block_indices]
            _, topk_idx = torch.topk(values, k=max_blocks, largest=True)
            selected_block_indices = selected_block_indices[topk_idx]
            selected_blocks = torch.zeros_like(selected_blocks, dtype=torch.bool)
            selected_blocks.reshape(-1)[selected_block_indices] = True

        candidate = expand_block_mask(selected_blocks, block_size, height, width) & base_mask
        selected = torch.zeros_like(candidate, dtype=torch.bool)
        block_cols = selected_blocks.shape[1]
        remaining = max_pixels
        for block_index in selected_block_indices:
            if remaining <= 0:
                break
            by = int((block_index // block_cols).item())
            bx = int((block_index % block_cols).item())
            y0 = by * block_size
            x0 = bx * block_size
            y1 = min(y0 + block_size, height)
            x1 = min(x0 + block_size, width)
            local_valid = base_mask[y0:y1, x0:x1]
            local_flat = torch.nonzero(local_valid.reshape(-1), as_tuple=False).squeeze(1)
            if local_flat.numel() == 0:
                continue
            pick_count = min(pixels_per_block, int(local_flat.numel()), remaining)
            pick_positions = torch.linspace(0, local_flat.numel() - 1, pick_count, device=local_flat.device).round().long()
            chosen = local_flat[pick_positions]
            local_y = torch.div(chosen, (x1 - x0), rounding_mode="floor") + y0
            local_x = chosen % (x1 - x0) + x0
            selected[local_y, local_x] = True
            remaining -= pick_count
        return candidate, selected, threshold

    def select_background_prior_pixels(stable, visible, occlusion, occlusion_dilated, render_to_bg):
        strict_base = visible & (~occlusion_dilated)
        strict_candidate, strict_selected, strict_threshold = select_background_prior_pool(
            strict_base,
            stable,
            occlusion_dilated,
            render_to_bg,
            max_pixels=int(getattr(gaussians, "field_bg_prior_strict_max_pixels", getattr(gaussians, "field_bg_prior_max_pixels", 256))),
            pixels_per_block=int(getattr(gaussians, "field_bg_prior_strict_pixels_per_block", getattr(gaussians, "field_bg_prior_pixels_per_block", 8))),
            min_visible_ratio=float(getattr(gaussians, "field_bg_prior_min_visible_ratio", 0.35)),
            min_stable_ratio=float(getattr(gaussians, "field_bg_prior_min_stable_ratio", 0.65)),
            max_occlusion_ratio=float(getattr(gaussians, "field_bg_prior_max_occlusion_ratio", 0.05)),
            error_quantile=float(getattr(gaussians, "field_bg_prior_error_quantile", 0.99)),
        )

        recall_base = visible & (~occlusion)
        recall_candidate, recall_selected, recall_threshold = select_background_prior_pool(
            recall_base,
            stable,
            occlusion,
            render_to_bg,
            max_pixels=int(getattr(gaussians, "field_bg_prior_recall_max_pixels", 32)),
            pixels_per_block=int(getattr(gaussians, "field_bg_prior_recall_pixels_per_block", 2)),
            min_visible_ratio=float(getattr(gaussians, "field_bg_prior_recall_min_visible_ratio", 0.15)),
            min_stable_ratio=float(getattr(gaussians, "field_bg_prior_recall_min_stable_ratio", 0.35)),
            max_occlusion_ratio=float(getattr(gaussians, "field_bg_prior_recall_max_occlusion_ratio", 0.25)),
            error_quantile=float(getattr(gaussians, "field_bg_prior_recall_error_quantile", 0.95)),
        )
        recall_selected = recall_selected & (~strict_selected)

        candidate = strict_candidate | recall_candidate
        selected = strict_selected | recall_selected
        threshold = torch.maximum(strict_threshold, recall_threshold)
        aux = {
            "strict_candidate": strict_candidate,
            "strict_selected": strict_selected,
            "strict_threshold": strict_threshold,
            "recall_candidate": recall_candidate,
            "recall_selected": recall_selected,
            "recall_threshold": recall_threshold,
        }
        return candidate, selected, threshold, aux

    def select_unreliable_background_prior_pixels(unreliable_mask, render_error):
        base = unreliable_mask.bool()
        full = torch.ones_like(base, dtype=torch.bool)
        empty = torch.zeros_like(base, dtype=torch.bool)
        candidate, selected, threshold = select_background_prior_pool(
            base,
            full,
            empty,
            render_error,
            max_pixels=int(getattr(gaussians, "field_bg_prior_unreliable_max_pixels", getattr(gaussians, "field_bg_prior_max_pixels", 32))),
            pixels_per_block=int(getattr(gaussians, "field_bg_prior_unreliable_pixels_per_block", getattr(gaussians, "field_bg_prior_pixels_per_block", 2))),
            min_visible_ratio=1e-6,
            min_stable_ratio=0.0,
            max_occlusion_ratio=1.0,
            error_quantile=float(getattr(gaussians, "field_bg_prior_unreliable_error_quantile", 0.95)),
        )
        aux = {
            "unreliable_candidate": candidate,
            "unreliable_selected": selected,
            "unreliable_threshold": threshold,
        }
        return candidate, selected, threshold, aux

    def normalize_debug_map(values, mask=None):
        values = values.detach().float()
        finite = torch.isfinite(values)
        if mask is not None:
            finite = finite & mask.bool()
        valid = values[finite]
        if valid.numel() == 0:
            return torch.zeros_like(values)
        lo = torch.quantile(valid, 0.02)
        hi = torch.quantile(valid, 0.98)
        denom = torch.clamp(hi - lo, min=1e-6)
        normalized = (values - lo) / denom
        return torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def save_debug_image(path, tensor):
        tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        torchvision.utils.save_image(tensor, path)

    def source_time_map_to_rgb(source_time_map, source_time_indices):
        source_time_map = source_time_map.detach().long()
        output = torch.zeros((3, source_time_map.shape[0], source_time_map.shape[1]), device=source_time_map.device, dtype=torch.float32)
        palette = [
            (0.10, 0.10, 0.10),
            (0.10, 0.45, 1.00),
            (0.10, 0.80, 0.25),
            (1.00, 0.65, 0.05),
            (1.00, 0.10, 0.10),
            (0.65, 0.20, 1.00),
            (0.00, 0.80, 0.80),
            (0.90, 0.90, 0.20),
        ]
        for idx, time_idx in enumerate(source_time_indices):
            mask = source_time_map == int(time_idx)
            if torch.count_nonzero(mask) == 0:
                continue
            color = torch.tensor(palette[idx % len(palette)], device=output.device, dtype=output.dtype).view(3, 1)
            output[:, mask] = color
        return output

    def overlay_debug_mask(base, mask, color, alpha=0.65):
        base = base.detach().float().clamp(0.0, 1.0)
        mask = mask.detach().bool().unsqueeze(0)
        color_tensor = torch.tensor(color, device=base.device, dtype=base.dtype).view(3, 1, 1)
        return torch.where(mask, base * (1.0 - alpha) + color_tensor * alpha, base)

    def overlay_opacity_map(base, opacity, color=(0.0, 0.75, 1.0), alpha_scale=0.65):
        base = base.detach().float().clamp(0.0, 1.0)
        opacity = opacity.detach().float().clamp(0.0, 1.0).unsqueeze(0)
        alpha = alpha_scale * opacity
        color_tensor = torch.tensor(color, device=base.device, dtype=base.dtype).view(3, 1, 1)
        return base * (1.0 - alpha) + color_tensor * alpha

    def save_layer_responsibility_debug(
        iteration,
        time_idx,
        camera,
        gt_image,
        normal_image,
        bg_mask,
        bg_stats,
        far_image,
        far_image_exposed,
        near_opacity,
        far_points,
        near_points,
        far_loss_value,
        front_loss_value,
    ):
        nonlocal layer_resp_debug_events
        if not bool(getattr(gaussians, "field_layer_debug", 0)):
            return
        max_events = int(getattr(gaussians, "field_layer_debug_max_events", 4))
        if max_events > 0 and layer_resp_debug_events >= max_events:
            return
        layer_resp_debug_events += 1

        safe_name = str(getattr(camera, "image_name", "camera")).replace("/", "_").replace("\\", "_")
        event_dir = os.path.join(
            layer_resp_debug_dir,
            f"{int(iteration):06d}_{safe_name}_t{int(time_idx):02d}_px{int(torch.count_nonzero(bg_mask).item())}",
        )
        os.makedirs(event_dir, exist_ok=True)
        save_debug_image(os.path.join(event_dir, "gt.png"), gt_image.detach().float().clamp(0.0, 1.0))
        save_debug_image(os.path.join(event_dir, "normal_render.png"), normal_image.detach().float().clamp(0.0, 1.0))
        save_debug_image(os.path.join(event_dir, "M_bg.png"), bg_mask.detach().float())
        save_debug_image(os.path.join(event_dir, "M_bg_overlay.png"), overlay_debug_mask(gt_image, bg_mask, (1.0, 0.0, 0.0), alpha=0.60))
        if far_image is not None:
            save_debug_image(os.path.join(event_dir, "far_only_raw.png"), far_image.detach().float().clamp(0.0, 1.0))
        if far_image_exposed is not None:
            save_debug_image(os.path.join(event_dir, "far_only_exposed.png"), far_image_exposed.detach().float().clamp(0.0, 1.0))
        if near_opacity is not None:
            budget = float(getattr(gaussians, "field_layer_front_opacity_budget", 0.15))
            over_budget = near_opacity.detach().float() > budget
            save_debug_image(os.path.join(event_dir, "near_opacity.png"), near_opacity.detach().float().clamp(0.0, 1.0))
            save_debug_image(os.path.join(event_dir, "near_opacity_overlay.png"), overlay_opacity_map(gt_image, near_opacity))
            save_debug_image(os.path.join(event_dir, "near_opacity_over_budget_overlay.png"), overlay_debug_mask(gt_image, bg_mask & over_budget, (0.0, 0.75, 1.0), alpha=0.65))
        for name, mask in bg_stats.get("_debug_masks", {}).items():
            save_debug_image(os.path.join(event_dir, f"{name}.png"), mask.detach().float())
        for name, value in bg_stats.get("_debug_maps", {}).items():
            save_debug_image(os.path.join(event_dir, f"{name}.png"), normalize_debug_map(value.detach()))
        meta = strip_debug_tensors(bg_stats)
        meta.update({
            "iteration": int(iteration),
            "time_idx": int(time_idx),
            "camera": str(getattr(camera, "image_name", "")),
            "timestamp": float(getattr(camera, "timestamp", 0.0)),
            "far_points": int(far_points),
            "near_points": int(near_points),
            "far_loss": float(far_loss_value),
            "front_loss": float(front_loss_value),
            "exposure_domain": "main_render_params_detached",
        })
        with open(os.path.join(event_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    def save_background_prior_debug(iteration, camera, gt_image, image, median, stable, visible, occlusion, occlusion_dilated,
                                    candidate, selected, render_to_bg, depth, anchor_depth, bg_depth, threshold,
                                    selection_aux=None, suppressed=None, suppressed_count=0):
        nonlocal bg_prior_debug_events
        nonlocal bg_prior_debug_camera_names
        if not bool(getattr(gaussians, "field_bg_prior_debug", 0)):
            return
        safe_name = str(camera.image_name).replace("/", "_").replace("\\", "_")
        debug_mode = str(getattr(gaussians, "field_bg_prior_debug_mode", "first_per_camera")).lower()
        if debug_mode == "first_per_camera":
            if safe_name in bg_prior_debug_camera_names:
                return
            bg_prior_debug_camera_names.add(safe_name)
        max_events = int(getattr(gaussians, "field_bg_prior_debug_max_events", 0))
        if max_events > 0 and bg_prior_debug_events >= max_events:
            return
        bg_prior_debug_events += 1

        timestamp = float(getattr(camera, "timestamp", 0.0))
        event_name = f"{iteration:06d}_{safe_name}_t{timestamp:.4f}"
        event_dir = os.path.join(args.model_path, "bg_prior_debug", event_name)
        os.makedirs(event_dir, exist_ok=True)

        gt_vis = gt_image.detach().float().clamp(0.0, 1.0)
        render_vis = image.detach().float().clamp(0.0, 1.0)
        median_vis = median.detach().float().clamp(0.0, 1.0)
        error_vis = normalize_debug_map(render_to_bg)
        depth_vis = normalize_debug_map(depth, mask=torch.isfinite(depth) & (depth > 0.0))
        anchor_vis = normalize_debug_map(anchor_depth, mask=torch.isfinite(anchor_depth) & (anchor_depth > 0.0))
        depth_max = max(float(getattr(gaussians, "field_bg_prior_depth_max", 15.0)), 1e-6)
        depth_fixed = torch.nan_to_num(depth / depth_max, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        anchor_fixed = torch.nan_to_num(anchor_depth / depth_max, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        save_debug_image(os.path.join(event_dir, "gt.png"), gt_vis)
        save_debug_image(os.path.join(event_dir, "render.png"), render_vis)
        save_debug_image(os.path.join(event_dir, "temporal_median.png"), median_vis)
        save_debug_image(os.path.join(event_dir, "render_to_median_error.png"), error_vis)
        save_debug_image(os.path.join(event_dir, "render_depth.png"), depth_vis)
        save_debug_image(os.path.join(event_dir, "anchor_depth.png"), anchor_vis)
        save_debug_image(os.path.join(event_dir, "render_depth_fixed.png"), depth_fixed)
        save_debug_image(os.path.join(event_dir, "anchor_depth_fixed.png"), anchor_fixed)
        save_debug_image(os.path.join(event_dir, "stable_mask.png"), stable.float())
        save_debug_image(os.path.join(event_dir, "visible_mask.png"), visible.float())
        save_debug_image(os.path.join(event_dir, "occlusion_mask.png"), occlusion.float())
        save_debug_image(os.path.join(event_dir, "occlusion_dilated.png"), occlusion_dilated.float())
        save_debug_image(os.path.join(event_dir, "candidate_mask.png"), candidate.float())
        save_debug_image(os.path.join(event_dir, "selected_pixels.png"), selected.float())
        save_debug_image(os.path.join(event_dir, "selected_overlay.png"), overlay_debug_mask(gt_vis, selected, (1.0, 0.0, 0.0)))
        save_debug_image(os.path.join(event_dir, "candidate_overlay.png"), overlay_debug_mask(gt_vis, candidate, (1.0, 1.0, 0.0), alpha=0.45))
        if selection_aux is None:
            selection_aux = {}
        for mask_name in (
            "strict_candidate",
            "strict_selected",
            "recall_candidate",
            "recall_selected",
            "unreliable_candidate",
            "unreliable_selected",
            "obs_reset_candidate",
            "obs_reset_selected",
        ):
            mask = selection_aux.get(mask_name)
            if mask is not None:
                save_debug_image(os.path.join(event_dir, f"{mask_name}.png"), mask.float())
        if suppressed is not None:
            save_debug_image(os.path.join(event_dir, "suppressed_explainers.png"), suppressed.float())
            save_debug_image(os.path.join(event_dir, "suppressed_overlay.png"), overlay_debug_mask(gt_vis, suppressed, (0.0, 0.5, 1.0), alpha=0.65))

        stable_count = int(torch.count_nonzero(stable).item())
        visible_count = int(torch.count_nonzero(visible).item())
        candidate_count = int(torch.count_nonzero(candidate).item())
        selected_count = int(torch.count_nonzero(selected).item())
        meta = {
            "iteration": int(iteration),
            "camera": str(camera.image_name),
            "timestamp": timestamp,
            "bg_depth": float(bg_depth.detach().cpu()),
            "source_mode": str(getattr(gaussians, "field_bg_prior_source", "background")),
            "color_source": str(getattr(gaussians, "field_bg_prior_color_source", "median")),
            "depth_values": parse_float_list(getattr(gaussians, "field_bg_prior_depth_values", ""), default=[]),
            "error_threshold": float(threshold.detach().cpu()),
            "stable_pixels": stable_count,
            "visible_pixels": visible_count,
            "occlusion_pixels": int(torch.count_nonzero(occlusion).item()),
            "occlusion_dilated_pixels": int(torch.count_nonzero(occlusion_dilated).item()),
            "candidate_pixels": candidate_count,
            "selected_pixels": selected_count,
            "strict_candidate_pixels": int(torch.count_nonzero(selection_aux["strict_candidate"]).item()) if "strict_candidate" in selection_aux else 0,
            "strict_selected_pixels": int(torch.count_nonzero(selection_aux["strict_selected"]).item()) if "strict_selected" in selection_aux else 0,
            "recall_candidate_pixels": int(torch.count_nonzero(selection_aux["recall_candidate"]).item()) if "recall_candidate" in selection_aux else 0,
            "recall_selected_pixels": int(torch.count_nonzero(selection_aux["recall_selected"]).item()) if "recall_selected" in selection_aux else 0,
            "unreliable_candidate_pixels": int(torch.count_nonzero(selection_aux["unreliable_candidate"]).item()) if "unreliable_candidate" in selection_aux else 0,
            "unreliable_selected_pixels": int(torch.count_nonzero(selection_aux["unreliable_selected"]).item()) if "unreliable_selected" in selection_aux else 0,
            "obs_reset_candidate_pixels": int(torch.count_nonzero(selection_aux["obs_reset_candidate"]).item()) if "obs_reset_candidate" in selection_aux else 0,
            "obs_reset_selected_pixels": int(torch.count_nonzero(selection_aux["obs_reset_selected"]).item()) if "obs_reset_selected" in selection_aux else 0,
            "suppressed_explainers": int(suppressed_count),
            "render_error_mean": float(render_to_bg.detach().float().mean().cpu()),
            "render_error_selected_mean": float(render_to_bg[selected].detach().float().mean().cpu()) if selected_count > 0 else 0.0,
        }
        with open(os.path.join(event_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    def sample_pixels_from_obs_reset_mask(mask):
        mask = mask.to(dtype=torch.bool)
        selected = torch.zeros_like(mask, dtype=torch.bool)
        if torch.count_nonzero(mask) == 0:
            return selected

        height, width = mask.shape
        block_size = max(int(getattr(gaussians, "field_bg_prior_block_size", 32)), 4)
        max_pixels = int(getattr(gaussians, "field_bg_prior_max_pixels", 256))
        pixels_per_block = max(int(getattr(gaussians, "field_bg_prior_pixels_per_block", 4)), 1)

        for y0 in range(0, height, block_size):
            y1 = min(y0 + block_size, height)
            for x0 in range(0, width, block_size):
                x1 = min(x0 + block_size, width)
                local = torch.nonzero(mask[y0:y1, x0:x1], as_tuple=False)
                if local.numel() == 0:
                    continue
                take = min(pixels_per_block, local.shape[0])
                if local.shape[0] > take:
                    pick = torch.linspace(0, local.shape[0] - 1, steps=take, device=local.device).round().long()
                    local = local[pick]
                selected[y0 + local[:, 0], x0 + local[:, 1]] = True

        selected_coords = torch.nonzero(selected, as_tuple=False)
        if max_pixels > 0 and selected_coords.shape[0] > max_pixels:
            pick = torch.linspace(0, selected_coords.shape[0] - 1, steps=max_pixels, device=selected_coords.device).round().long()
            reduced = torch.zeros_like(selected, dtype=torch.bool)
            selected_coords = selected_coords[pick]
            reduced[selected_coords[:, 0], selected_coords[:, 1]] = True
            selected = reduced
        return selected

    def maybe_add_background_prior_points_for_camera(iteration, camera, image, gt_image, render_pkg, unreliable_mask=None):
        median, stable = get_background_prior(camera)
        if median is None or stable is None:
            return 0
        current_to_median = torch.abs(gt_image.detach().float() - median).mean(dim=0)
        visible = stable & (current_to_median < float(getattr(gaussians, "field_bg_prior_visible_threshold", 0.08)))
        occlusion = current_to_median > float(getattr(gaussians, "field_bg_prior_occlusion_threshold", 0.12))
        occlusion_dilated = dilate_mask(occlusion, int(getattr(gaussians, "field_bg_prior_occlusion_dilate", 7)))

        depth = render_pkg["depth"].detach().squeeze(0).float()
        valid_depth = torch.isfinite(depth) & (depth > 0.0) & (depth < 14.99)
        depth_values = parse_float_list(getattr(gaussians, "field_bg_prior_depth_values", ""), default=[])
        if len(depth_values) > 0:
            first_depth = max(float(depth_values[0]), 1e-6)
            bg_depth = torch.tensor(first_depth, device=depth.device, dtype=depth.dtype)
            anchor_depth = torch.full_like(depth, first_depth)
        elif bool(getattr(gaussians, "field_bg_prior_fixed_depth", 1)):
            depth_max = max(float(getattr(gaussians, "field_bg_prior_depth_max", 15.0)), 1e-6)
            fixed_depth = depth_max * float(getattr(gaussians, "field_bg_prior_fixed_depth_ratio", 0.95))
            bg_depth = torch.tensor(fixed_depth, device=depth.device, dtype=depth.dtype)
            anchor_depth = torch.full_like(depth, fixed_depth)
        else:
            depth_pool = depth[visible & (~occlusion_dilated) & valid_depth]
            if depth_pool.numel() < 16:
                depth_pool = depth[valid_depth]
            if depth_pool.numel() == 0:
                return 0
            depth_q = float(getattr(gaussians, "field_bg_prior_depth_quantile", 0.80))
            bg_depth = torch.quantile(depth_pool.float(), max(0.0, min(depth_q, 1.0)))
            anchor_depth = torch.where(valid_depth, torch.maximum(depth, bg_depth), bg_depth.expand_as(depth))

        source_mode = str(getattr(gaussians, "field_bg_prior_source", "background")).lower()
        if source_mode == "obs_reset":
            if unreliable_mask is None:
                return 0
            unreliable_mask = unreliable_mask.to(device=depth.device, dtype=torch.bool)
            if torch.count_nonzero(unreliable_mask) == 0:
                return 0
            render_to_bg = compute_background_error_map(image, gt_image, unreliable_mask)
            candidate = unreliable_mask
            selected = sample_pixels_from_obs_reset_mask(candidate)
            threshold = torch.tensor(0.0, device=depth.device, dtype=depth.dtype)
            selection_aux = {
                "obs_reset_candidate": candidate,
                "obs_reset_selected": selected,
            }
        elif source_mode == "unreliable":
            if unreliable_mask is None:
                return 0
            unreliable_mask = unreliable_mask.to(device=depth.device, dtype=torch.bool)
            if torch.count_nonzero(unreliable_mask) == 0:
                return 0
            render_to_bg = compute_background_error_map(image, gt_image, unreliable_mask)
            candidate, selected, threshold, selection_aux = select_unreliable_background_prior_pixels(
                unreliable_mask,
                render_to_bg,
            )
        else:
            render_to_bg = compute_background_error_map(image, median, visible & (~occlusion))
            candidate, selected, threshold, selection_aux = select_background_prior_pixels(
                stable, visible, occlusion, occlusion_dilated, render_to_bg
            )
        suppressed_count = 0
        suppressed_pixels = torch.zeros_like(candidate, dtype=torch.bool)
        if hasattr(gaussians, "suppress_background_explainers"):
            suppressed_count, suppressed_pixels = gaussians.suppress_background_explainers(
                candidate,
                camera,
                bg_depth,
                iteration,
            )
        flat_candidates = torch.nonzero(selected.reshape(-1), as_tuple=False).squeeze(1)
        if flat_candidates.numel() == 0:
            save_background_prior_debug(
                iteration,
                camera,
                gt_image,
                image,
                median,
                stable,
                visible,
                occlusion,
                occlusion_dilated,
                candidate,
                selected,
                render_to_bg,
                depth,
                anchor_depth,
                bg_depth,
                threshold,
                selection_aux,
                suppressed_pixels,
                suppressed_count,
            )
            return 0

        height, width = candidate.shape
        ys = torch.div(flat_candidates, width, rounding_mode="floor")
        xs = flat_candidates % width
        pixel_indices = torch.stack((ys, xs), dim=1)
        color_source = str(getattr(gaussians, "field_bg_prior_color_source", "median")).lower()
        color_image = gt_image if color_source == "gt" else median
        bg_added = gaussians.add_static_background_gaussians(
            pixel_indices,
            camera,
            anchor_depth.unsqueeze(0),
            color_image,
            iteration,
            numperay=int(getattr(gaussians, "field_bg_prior_num_per_ray", 1)),
            depth_scale=float(getattr(gaussians, "field_bg_prior_depth_scale", 1.02)),
            depth_values=depth_values,
        )
        save_background_prior_debug(
            iteration,
            camera,
            gt_image,
            image,
            median,
            stable,
            visible,
            occlusion,
            occlusion_dilated,
            candidate,
            selected,
            render_to_bg,
            depth,
            anchor_depth,
            bg_depth,
            threshold,
            selection_aux,
            suppressed_pixels,
            suppressed_count,
        )
        return bg_added

    def maybe_add_background_prior_points(iteration, time_idx, camera, image, gt_image, render_pkg, unreliable_mask=None):
        if not bool(getattr(gaussians, "field_bg_prior", 0)):
            return 0
        start_iter = int(getattr(gaussians, "field_bg_prior_start", 3200))
        if iteration < start_iter:
            return 0
        if iteration > int(getattr(gaussians, "field_bg_prior_until", 9000)):
            return 0
        if not hasattr(gaussians, "add_static_background_gaussians"):
            return 0

        schedule_mode = str(getattr(gaussians, "field_bg_prior_schedule_mode", "scan")).lower()
        if schedule_mode == "on_sample":
            key = background_prior_key(time_idx, camera)
            if key not in background_prior_pending:
                return 0
            background_prior_pending.remove(key)
            return maybe_add_background_prior_points_for_camera(
                iteration,
                camera,
                image,
                gt_image,
                render_pkg,
                unreliable_mask,
            )

        interval = max(int(getattr(gaussians, "field_bg_prior_interval", 500)), 1)
        if (iteration - start_iter) % interval != 0:
            return 0

        total_added = 0
        scheduled_cameras = get_scheduled_background_cameras(iteration, camera)
        for target_camera in scheduled_cameras:
            if schedule_mode != "scan" and target_camera is camera:
                target_render_pkg = render_pkg
                target_image = image
                target_gt = gt_image
            else:
                target_render_pkg = render(
                    target_camera,
                    gaussians,
                    pipe,
                    background,
                    override_color=None,
                    basicfunction=rbfbasefunction,
                    GRsetting=GRsetting,
                    GRzer=GRzer,
                )
                target_image = target_render_pkg["render"]
                target_gt = get_gt_image(target_camera)
            total_added += maybe_add_background_prior_points_for_camera(
                iteration,
                target_camera,
                target_image,
                target_gt,
                target_render_pkg,
                None,
            )
        return total_added

    def parse_int_list(value, default=None):
        if value is None:
            return list(default or [])
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        text = str(value).strip()
        if text == "":
            return list(default or [])
        return [int(part.strip()) for part in text.split(",") if part.strip() != ""]

    def parse_float_list(value, default=None):
        if value is None:
            return list(default or [])
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        text = str(value).strip()
        if text == "":
            return list(default or [])
        return [float(part.strip()) for part in text.split(",") if part.strip() != ""]

    def parse_dense_add_time_indices():
        raw = str(getattr(gaussians, "field_bg_dense_add_time_indices", "")).strip()
        if raw == "":
            raw = str(getattr(gaussians, "field_bg_prior_scan_time_indices", "")).strip()
        if raw:
            indices = []
            for item in raw.split(","):
                item = item.strip()
                if item:
                    indices.append(max(0, min(int(item), duration - 1)))
            if indices:
                return sorted(set(indices))
        return parse_bg_scan_time_indices()

    def parse_dense_source_time_indices():
        raw = str(getattr(gaussians, "field_bg_dense_source_time_indices", "")).strip()
        if raw:
            indices = []
            for item in raw.split(","):
                item = item.strip()
                if item:
                    indices.append(max(0, min(int(item), duration - 1)))
            if indices:
                return sorted(set(indices))
        return parse_dense_add_time_indices()

    def resolve_dense_da3_npz_path():
        raw = str(getattr(gaussians, "field_bg_dense_da3_path", "")).strip()
        candidates = []
        if raw:
            if os.path.isdir(raw):
                candidates.extend([
                    os.path.join(raw, "results.npz"),
                    os.path.join(raw, "exports", "mini_npz", "results.npz"),
                ])
            else:
                candidates.append(raw)

        source_root = os.path.abspath(getattr(args, "source_path", ""))
        scene_root = os.path.dirname(source_root)
        candidates.extend([
            os.path.join(scene_root, "results.npz"),
            os.path.join(source_root, "results.npz"),
            os.path.join(scene_root, "exports", "mini_npz", "results.npz"),
        ])
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    def camera_basename(camera):
        return os.path.splitext(os.path.basename(str(camera.image_name)))[0]

    def camera_time_index(camera):
        timestamp = getattr(camera, "timestamp", 0.0)
        if torch.is_tensor(timestamp):
            timestamp = float(timestamp.detach().reshape(-1)[0].cpu())
        else:
            timestamp = float(timestamp)

        source_root = os.path.abspath(getattr(args, "source_path", ""))
        source_name = os.path.basename(source_root)
        source_start = 0
        if source_name.startswith("colmap_"):
            maybe_index = source_name.split("_", 1)[1].split("_", 1)[0]
            if maybe_index.isdigit():
                source_start = int(maybe_index)

        return int(round(timestamp * float(duration) + float(source_start)))

    def resolve_scene_aux_dir(config_value, default_dirname):
        raw = str(config_value).strip()
        candidates = []
        if raw:
            candidates.append(raw)
        source_root = os.path.abspath(getattr(args, "source_path", ""))
        scene_root = os.path.dirname(source_root)
        candidates.extend([
            os.path.join(scene_root, default_dirname),
            os.path.join(source_root, default_dirname),
        ])
        for path in candidates:
            if path and os.path.isdir(path):
                return path
        return None

    def read_colmap_image_names_binary(images_bin_path):
        import struct

        image_names = []
        if not os.path.exists(images_bin_path):
            return image_names
        with open(images_bin_path, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                f.read(4 + 8 * 7 + 4)
                name_bytes = bytearray()
                while True:
                    char = f.read(1)
                    if char == b"" or char == b"\x00":
                        break
                    name_bytes.extend(char)
                image_name = name_bytes.decode("utf-8")
                image_names.append(os.path.splitext(image_name)[0])
                num_points2d = struct.unpack("<Q", f.read(8))[0]
                f.seek(num_points2d * (8 + 8 + 8), os.SEEK_CUR)
        return image_names

    def read_colmap_image_names_text(images_txt_path):
        image_names = []
        if not os.path.exists(images_txt_path):
            return image_names
        with open(images_txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 10:
                    image_names.append(os.path.splitext(parts[9])[0])
                    next(f, None)
        return image_names

    def get_da3_colmap_image_order():
        source_root = os.path.abspath(getattr(args, "source_path", ""))
        sparse_candidates = [
            os.path.join(source_root, "sparse", "0"),
            os.path.join(source_root, "sparse"),
            os.path.join(source_root, "0"),
        ]
        for sparse_dir in sparse_candidates:
            names = read_colmap_image_names_binary(os.path.join(sparse_dir, "images.bin"))
            if names:
                return names
            names = read_colmap_image_names_text(os.path.join(sparse_dir, "images.txt"))
            if names:
                return names

        image_dir = os.path.join(source_root, "images")
        if os.path.isdir(image_dir):
            return [
                os.path.splitext(filename)[0]
                for filename in sorted(os.listdir(image_dir))
                if os.path.splitext(filename)[1].lower() in (".png", ".jpg", ".jpeg")
            ]
        return []

    def load_dense_da3_depths():
        if bg_dense_da3_cache["loaded"]:
            return bg_dense_da3_cache
        bg_dense_da3_cache["loaded"] = True

        npz_path = resolve_dense_da3_npz_path()
        if npz_path is None:
            print("[STEGF] Dense DA3 foreground filter disabled: results.npz not found.")
            return bg_dense_da3_cache

        payload = np.load(npz_path)
        if "depth" not in payload:
            print(f"[STEGF] Dense DA3 foreground filter disabled: no 'depth' key in {npz_path}.")
            return bg_dense_da3_cache

        depth = payload["depth"].astype(np.float32)
        image_names = get_da3_colmap_image_order()
        name_to_idx = {name: idx for idx, name in enumerate(image_names[: depth.shape[0]])}
        if len(name_to_idx) != int(depth.shape[0]):
            print(
                f"[STEGF] Dense DA3 warning: depth views={int(depth.shape[0])}, "
                f"image names={len(image_names)}. Mapping uses COLMAP image read order."
            )

        bg_dense_da3_cache.update({
            "path": npz_path,
            "depth": depth,
            "name_to_idx": name_to_idx,
        })
        print(
            f"[STEGF] Loaded DA3 depths for dense add: {npz_path}, views={int(depth.shape[0])}, "
            f"first_names={image_names[:3]}, last_names={image_names[-3:]}."
        )
        return bg_dense_da3_cache

    def get_dense_da3_foreground_mask(camera, target_shape):
        if not bool(getattr(gaussians, "field_bg_dense_da3_filter", 0)):
            return None, None, {"enabled": False}
        cache = load_dense_da3_depths()
        depth_np = cache.get("depth")
        name_to_idx = cache.get("name_to_idx", {})
        if depth_np is None:
            return None, None, {"enabled": True, "reason": "missing_depth"}

        image_name = str(camera.image_name)
        idx = name_to_idx.get(image_name)
        if idx is None:
            if not bg_dense_da3_cache["warned"]:
                print(f"[STEGF] Dense DA3 warning: no depth index for camera {image_name}.")
                bg_dense_da3_cache["warned"] = True
            return None, None, {"enabled": True, "reason": "missing_camera"}

        depth = torch.from_numpy(depth_np[int(idx)]).to(device="cuda", dtype=torch.float32)
        height, width = int(target_shape[0]), int(target_shape[1])
        if depth.shape != (height, width):
            depth = F.interpolate(
                depth.view(1, 1, depth.shape[0], depth.shape[1]),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).view(height, width)

        valid = torch.isfinite(depth) & (depth > 0.0)
        if torch.count_nonzero(valid) == 0:
            return depth, torch.zeros((height, width), device="cuda", dtype=torch.bool), {
                "enabled": True,
                "reason": "empty_valid_depth",
            }

        quantile = max(0.0, min(float(getattr(gaussians, "field_bg_dense_da3_foreground_quantile", 0.45)), 1.0))
        threshold = torch.quantile(depth[valid].float(), quantile)
        foreground = valid & (depth <= threshold)
        return depth, foreground.detach(), {
            "enabled": True,
            "path": cache.get("path"),
            "index": int(idx),
            "quantile": float(quantile),
            "threshold": float(threshold.detach().cpu()),
            "foreground_pixels": int(torch.count_nonzero(foreground).item()),
        }

    def load_dense_beit_depth_like():
        if bg_dense_beit_cache["loaded"]:
            return bg_dense_beit_cache
        bg_dense_beit_cache["loaded"] = True

        root = resolve_scene_aux_dir(
            getattr(gaussians, "field_bg_dense_beit_path", ""),
            "midas_beit_large_512",
        )
        if root is None:
            print("[STEGF] Dense BEiT foreground filter disabled: midas_beit_large_512 directory not found.")
            return bg_dense_beit_cache

        depth_like = {}
        legacy_views = 0
        nested_views = 0

        def load_raw(time_index, cam_name, raw_path):
            if not os.path.exists(raw_path):
                return
            depth_like[(int(time_index), str(cam_name))] = np.load(raw_path).astype(np.float32)

        for entry in sorted(os.listdir(root)):
            entry_dir = os.path.join(root, entry)
            if not os.path.isdir(entry_dir):
                continue
            if entry.startswith("colmap_"):
                maybe_index = entry.split("_", 1)[1].split("_", 1)[0]
                if not maybe_index.isdigit():
                    continue
                time_index = int(maybe_index)
                for cam_entry in sorted(os.listdir(entry_dir)):
                    cam_dir = os.path.join(entry_dir, cam_entry)
                    if not os.path.isdir(cam_dir):
                        continue
                    before = len(depth_like)
                    load_raw(time_index, cam_entry, os.path.join(cam_dir, "raw_depth_like.npy"))
                    nested_views += int(len(depth_like) > before)
                continue

            before = len(depth_like)
            load_raw(0, entry, os.path.join(entry_dir, "raw_depth_like.npy"))
            legacy_views += int(len(depth_like) > before)

        bg_dense_beit_cache.update({
            "path": root,
            "depth_like": depth_like,
        })
        times = sorted({int(key[0]) for key in depth_like.keys()})
        cameras = sorted({str(key[1]) for key in depth_like.keys()})
        print(
            f"[STEGF] Loaded BEiT depth-like foreground masks: {root}, "
            f"entries={len(depth_like)}, times={times}, cameras={len(cameras)}, "
            f"legacy_views={legacy_views}, nested_views={nested_views}."
        )
        return bg_dense_beit_cache

    def get_dense_beit_foreground_mask(camera, target_shape, time_index_override=None, allow_fallback=True):
        if not bool(getattr(gaussians, "field_bg_dense_beit_filter", 0)):
            return None, None, {"enabled": False}
        cache = load_dense_beit_depth_like()
        cam_name = camera_basename(camera)
        time_index = camera_time_index(camera) if time_index_override is None else int(time_index_override)
        depth_np = cache.get("depth_like", {}).get((time_index, cam_name))
        used_time_index = time_index
        if depth_np is None and allow_fallback and time_index != 0:
            depth_np = cache.get("depth_like", {}).get((0, cam_name))
            used_time_index = 0
        if depth_np is None:
            if not bg_dense_beit_cache["warned"]:
                print(f"[STEGF] Dense BEiT warning: no raw_depth_like.npy for time={time_index}, camera={cam_name}.")
                bg_dense_beit_cache["warned"] = True
            return None, None, {"enabled": True, "reason": "missing_camera"}

        depth_like = torch.from_numpy(depth_np).to(device="cuda", dtype=torch.float32)
        height, width = int(target_shape[0]), int(target_shape[1])
        if depth_like.shape != (height, width):
            depth_like = F.interpolate(
                depth_like.view(1, 1, depth_like.shape[0], depth_like.shape[1]),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).view(height, width)

        valid = torch.isfinite(depth_like)
        if torch.count_nonzero(valid) == 0:
            return depth_like, torch.zeros((height, width), device="cuda", dtype=torch.bool), {
                "enabled": True,
                "reason": "empty_valid_depth_like",
            }

        low_q = max(0.0, min(float(getattr(gaussians, "field_bg_dense_beit_band_low", 0.10)), 1.0))
        high_q = max(0.0, min(float(getattr(gaussians, "field_bg_dense_beit_band_high", 0.30)), 1.0))
        if high_q <= low_q:
            high_q = min(low_q + 0.01, 1.0)
        lo = torch.quantile(depth_like[valid].float(), low_q)
        hi = torch.quantile(depth_like[valid].float(), high_q)
        denom = torch.clamp(hi - lo, min=1e-6)
        band = ((depth_like - lo) / denom).clamp(0.0, 1.0)
        threshold = max(0.0, min(float(getattr(gaussians, "field_bg_dense_beit_threshold", 0.5)), 1.0))
        foreground = valid & (band >= threshold)
        return band.detach(), foreground.detach(), {
            "enabled": True,
            "path": cache.get("path"),
            "camera": cam_name,
            "time_index": int(time_index),
            "used_time_index": int(used_time_index),
            "band_low": float(low_q),
            "band_high": float(high_q),
            "threshold": float(threshold),
            "foreground_pixels": int(torch.count_nonzero(foreground).item()),
        }

    def layer_responsibility_due(iteration):
        if not bool(getattr(gaussians, "field_layer_responsibility", 0)):
            return False
        start = int(getattr(gaussians, "field_layer_responsibility_start", 3000))
        until = int(getattr(gaussians, "field_layer_responsibility_until", 18000))
        if iteration < start or iteration > until:
            return False
        interval = max(int(getattr(gaussians, "field_layer_responsibility_interval", 5)), 1)
        return (iteration - start) % interval == 0

    def nearest_layer_beit_time_index(time_idx):
        candidates = parse_int_list(
            getattr(gaussians, "field_layer_beit_time_indices", "0,12,25,37,49"),
            default=[0, 12, 25, 37, 49],
        )
        candidates = [max(0, min(int(value), duration - 1)) for value in candidates]
        if not candidates:
            return int(time_idx)
        return min(candidates, key=lambda value: (abs(int(value) - int(time_idx)), int(value)))

    def detach_content_exposure_params(params):
        if not isinstance(params, dict):
            return None
        detached = {"mode": params.get("mode", "affine")}
        for key in ("log_scale", "bias", "delta_r", "delta_b"):
            value = params.get(key)
            if torch.is_tensor(value):
                detached[key] = value.detach()
        if "log_scale" not in detached or "bias" not in detached:
            return None
        return detached

    def apply_content_exposure_params(image, params):
        if not isinstance(params, dict):
            return image
        log_scale = params["log_scale"].to(device=image.device, dtype=image.dtype).view(1, 1, 1)
        bias = params["bias"].to(device=image.device, dtype=image.dtype).view(1, 1, 1)
        if params.get("mode", "affine") != "luma_wb":
            return torch.exp(log_scale) * image + bias
        delta_r = params.get("delta_r")
        delta_b = params.get("delta_b")
        if delta_r is None or delta_b is None:
            return torch.exp(log_scale) * image + bias
        delta_r = delta_r.to(device=image.device, dtype=image.dtype).view(1, 1, 1)
        delta_b = delta_b.to(device=image.device, dtype=image.dtype).view(1, 1, 1)
        wb = torch.cat(
            [
                torch.exp(delta_r),
                torch.ones_like(delta_r),
                torch.exp(delta_b),
            ],
            dim=0,
        )
        return torch.exp(log_scale) * wb * image + bias

    def build_layer_responsibility_mask(time_idx, camera, gt_image, temporal_motion_map):
        height, width = int(gt_image.shape[1]), int(gt_image.shape[2])
        median, _, stable = get_observation_prior(camera)
        if median is None or stable is None:
            return None, {"reason": "missing_temporal_prior", "pixels": 0}

        rep_time_idx = nearest_layer_beit_time_index(time_idx)
        beit_band, beit_foreground, beit_stats = get_dense_beit_foreground_mask(
            camera,
            (height, width),
            time_index_override=rep_time_idx,
            allow_fallback=False,
        )
        if beit_band is None or beit_foreground is None:
            return None, {
                "reason": str((beit_stats or {}).get("reason", "missing_beit")),
                "pixels": 0,
                "beit_time": int(rep_time_idx),
            }

        median = median.to(device=gt_image.device, dtype=gt_image.dtype)
        stable = stable.to(device=gt_image.device, dtype=torch.bool)
        beit_band = beit_band.to(device=gt_image.device, dtype=torch.float32)
        beit_foreground = beit_foreground.to(device=gt_image.device, dtype=torch.bool)
        temporal_motion_map = temporal_motion_map.to(device=gt_image.device, dtype=torch.float32)

        erode_radius = max(int(getattr(gaussians, "field_layer_mask_erode", 3)), 0)
        beit_bg_threshold = float(getattr(gaussians, "field_layer_beit_background_threshold", 0.35))
        motion_threshold = max(float(getattr(gaussians, "field_layer_motion_threshold", 0.12)), 1e-6)
        median_threshold = max(float(getattr(gaussians, "field_layer_median_threshold", 0.12)), 1e-6)

        current_to_median = torch.abs(gt_image - median).mean(dim=0)
        strict_background = (beit_band < beit_bg_threshold) & (~dilate_mask(beit_foreground, erode_radius))
        stable_current = stable & (current_to_median <= median_threshold) & (temporal_motion_map <= motion_threshold)
        mask = strict_background & stable_current
        if erode_radius > 0:
            mask = erode_mask(mask, erode_radius)

        pixel_count = int(torch.count_nonzero(mask).item())
        min_pixels = int(getattr(gaussians, "field_layer_min_pixels", 256))
        stats = {
            "reason": "ok" if pixel_count >= min_pixels else "too_few_pixels",
            "pixels": pixel_count,
            "beit_time": int(rep_time_idx),
            "strict_bg_pixels": int(torch.count_nonzero(strict_background).item()),
            "stable_pixels": int(torch.count_nonzero(stable_current).item()),
            "_debug_masks": {
                "M_bg": mask.detach(),
                "strict_background": strict_background.detach(),
                "stable_current": stable_current.detach(),
                "beit_foreground": beit_foreground.detach(),
            },
            "_debug_maps": {
                "beit_band": beit_band.detach(),
                "temporal_motion": temporal_motion_map.detach(),
                "current_to_median": current_to_median.detach(),
            },
        }
        if pixel_count < min_pixels:
            return None, stats
        return mask.detach(), stats

    def layer_depth_masks(camera):
        with torch.no_grad():
            means3D, _, _, _ = gaussians.compose_time_conditioned_attributes(
                camera.timestamp,
                rbfbasefunction,
                camera_center=camera.camera_center,
            )
            view_depth = geom_transform_points(
                means3D.detach().float(),
                camera.world_view_transform,
            )[:, 2]
            valid_depth = torch.isfinite(view_depth) & (view_depth > 0.0)
            near_depth = float(getattr(gaussians, "field_layer_near_depth", 30.0))
            far_depth = float(getattr(gaussians, "field_layer_far_depth", 80.0))
            near_mask = valid_depth & (view_depth < near_depth)
            far_mask = valid_depth & (view_depth >= far_depth)
        return near_mask.detach(), far_mask.detach()

    def rasterize_layer_opacity(camera, point_mask):
        point_mask = point_mask.to(device=gaussians.get_xyz.device, dtype=torch.bool).reshape(-1)
        if point_mask.shape[0] != gaussians.get_xyz.shape[0] or torch.count_nonzero(point_mask) == 0:
            return None
        means3D, opacity, rotations, _ = gaussians.compose_time_conditioned_attributes(
            camera.timestamp,
            rbfbasefunction,
            camera_center=camera.camera_center,
        )
        scales = gaussians.get_scaling
        means3D = means3D[point_mask]
        opacity = opacity[point_mask]
        rotations = rotations[point_mask]
        scales = scales[point_mask]
        values = torch.ones(
            (means3D.shape[0], int(background.shape[0])),
            device=means3D.device,
            dtype=means3D.dtype,
        )
        means2D = torch.zeros_like(means3D, dtype=means3D.dtype, device=means3D.device)
        zero_background = torch.zeros_like(background)
        tanfovx = math.tan(camera.FoVx * 0.5)
        tanfovy = math.tan(camera.FoVy * 0.5)
        raster_settings = GRsetting(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=zero_background,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=gaussians.active_sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
        )
        rasterizer = GRzer(raster_settings=raster_settings)
        rendered, _, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=values,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        return rendered[0].float().clamp(0.0, 1.0)

    def ensure_core_grads_for_cache():
        for param in (
            gaussians._xyz,
            gaussians._features_dc,
            gaussians._features_t,
            gaussians._scaling,
            gaussians._rotation,
            gaussians._opacity,
            gaussians._trbf_center,
            gaussians._trbf_scale,
            gaussians._motion,
            gaussians._omega,
        ):
            if param.grad is None:
                param.grad = torch.zeros_like(param)

    def filter_layer_responsibility_gradients(allowed_point_masks):
        core_groups = {
            "xyz",
            "f_dc",
            "f_t",
            "opacity",
            "scaling",
            "rotation",
            "omega",
            "trbf_center",
            "trbf_scale",
            "motion",
        }
        num_points = gaussians.get_xyz.shape[0]
        for group in gaussians.optimizer.param_groups:
            name = group.get("name", "")
            for param in group.get("params", []):
                if param is None:
                    continue
                if name in core_groups:
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    if name not in allowed_point_masks:
                        param.grad.zero_()
                        continue
                    mask = allowed_point_masks[name].to(device=param.grad.device, dtype=torch.bool).reshape(-1)
                    if mask.shape[0] != num_points or param.grad.shape[0] != num_points:
                        param.grad.zero_()
                        continue
                    while mask.dim() < param.grad.dim():
                        mask = mask.unsqueeze(-1)
                    param.grad.mul_(mask.to(dtype=param.grad.dtype))
                else:
                    param.grad = None
        ensure_core_grads_for_cache()

    def cache_layer_auxiliary_loss(aux_loss, allowed_point_masks, batch_scale):
        if aux_loss is None:
            return False
        (aux_loss * float(batch_scale)).backward()
        filter_layer_responsibility_gradients(allowed_point_masks)
        gaussians.cache_gradient()
        gaussians.optimizer.zero_grad(set_to_none=True)
        return True

    def apply_layer_responsibility_losses(
        iteration,
        time_idx,
        camera,
        gt_image,
        temporal_motion_map,
        batch_scale,
        normal_image=None,
        content_exposure_params=None,
    ):
        if not layer_responsibility_due(iteration):
            return None
        bg_mask, bg_stats = build_layer_responsibility_mask(time_idx, camera, gt_image, temporal_motion_map)
        if bg_mask is None:
            return {
                "pixels": 0,
                "reason": bg_stats.get("reason", "missing_mask"),
                "far_loss": 0.0,
                "front_loss": 0.0,
            }

        near_mask, far_mask = layer_depth_masks(camera)
        far_points = int(torch.count_nonzero(far_mask).item())
        near_points = int(torch.count_nonzero(near_mask).item())
        far_image_debug = None
        far_image_exposed_debug = None
        near_opacity_debug = None
        if far_points > 0:
            far_pkg = render(
                camera,
                gaussians,
                pipe,
                background,
                override_color=None,
                basicfunction=rbfbasefunction,
                GRsetting=GRsetting,
                GRzer=GRzer,
                time_conditioned=None,
                static_radiance_mask=None,
                iteration=iteration,
                render_point_mask=far_mask,
            )
            far_image = far_pkg["render"]
            far_image_exposed = apply_content_exposure_params(far_image, content_exposure_params)
            far_l1 = torch.abs(far_image_exposed - gt_image)[:, bg_mask].mean()
            far_weight = float(getattr(gaussians, "field_layer_far_loss_weight", 0.10))
            far_image_debug = far_image.detach()
            far_image_exposed_debug = far_image_exposed.detach()
            cache_layer_auxiliary_loss(
                far_weight * far_l1,
                {
                    "xyz": far_mask,
                    "f_dc": far_mask,
                    "opacity": far_mask,
                    "scaling": far_mask,
                    "rotation": far_mask,
                },
                batch_scale,
            )
            far_loss_value = float(far_l1.detach().cpu())
            del far_pkg, far_image, far_image_exposed, far_l1
        else:
            far_loss_value = 0.0

        if near_points > 0:
            near_opacity = rasterize_layer_opacity(camera, near_mask)
            if near_opacity is not None:
                budget = float(getattr(gaussians, "field_layer_front_opacity_budget", 0.15))
                front_loss = torch.relu(near_opacity - budget)[bg_mask].pow(2).mean()
                front_weight = float(getattr(gaussians, "field_layer_front_opacity_weight", 0.01))
                near_opacity_debug = near_opacity.detach()
                cache_layer_auxiliary_loss(
                    front_weight * front_loss,
                    {"opacity": near_mask},
                    batch_scale,
                )
                front_loss_value = float(front_loss.detach().cpu())
                del near_opacity, front_loss
            else:
                front_loss_value = 0.0
        else:
            front_loss_value = 0.0

        if normal_image is not None:
            save_layer_responsibility_debug(
                iteration,
                time_idx,
                camera,
                gt_image,
                normal_image,
                bg_mask,
                bg_stats,
                far_image_debug,
                far_image_exposed_debug,
                near_opacity_debug,
                far_points,
                near_points,
                far_loss_value,
                front_loss_value,
            )

        return {
            "pixels": int(bg_stats.get("pixels", 0)),
            "reason": bg_stats.get("reason", "ok"),
            "beit_time": int(bg_stats.get("beit_time", -1)),
            "far_points": far_points,
            "near_points": near_points,
            "far_loss": far_loss_value,
            "front_loss": front_loss_value,
        }

    def load_depthpro_depths():
        if depthpro_cache["loaded"]:
            return depthpro_cache
        depthpro_cache["loaded"] = True

        root = resolve_scene_aux_dir(
            getattr(gaussians, "field_depthpro_path", ""),
            "depth_pro_colmap_0",
        )
        if root is not None and os.path.basename(os.path.normpath(root)) == "raw":
            raw_root = root
        else:
            raw_root = os.path.join(root, "raw") if root is not None else None
        if raw_root is None or not os.path.isdir(raw_root):
            print("[STEGF] Depth Pro supervision disabled: depth_pro_colmap_0/raw directory not found.")
            return depthpro_cache

        depth = {}
        for filename in sorted(os.listdir(raw_root)):
            if not filename.endswith(".npz"):
                continue
            payload = np.load(os.path.join(raw_root, filename))
            if "depth" not in payload:
                continue
            depth[os.path.splitext(filename)[0]] = payload["depth"].astype(np.float32)

        depthpro_cache.update({
            "path": raw_root,
            "depth": depth,
        })
        print(f"[STEGF] Loaded Depth Pro foreground depths: {raw_root}, views={len(depth)}.")
        return depthpro_cache

    def get_depthpro_depth(camera, target_shape):
        if not bool(getattr(gaussians, "field_depthpro_supervision", 0)):
            return None, {"enabled": False}
        cache = load_depthpro_depths()
        depth_np = cache.get("depth", {}).get(camera_basename(camera))
        if depth_np is None:
            if not depthpro_cache["warned"]:
                print(f"[STEGF] Depth Pro warning: no depth npz for camera {camera_basename(camera)}.")
                depthpro_cache["warned"] = True
            return None, {"enabled": True, "reason": "missing_camera"}

        depth = torch.from_numpy(depth_np).to(device="cuda", dtype=torch.float32)
        height, width = int(target_shape[0]), int(target_shape[1])
        if depth.shape != (height, width):
            depth = F.interpolate(
                depth.view(1, 1, depth.shape[0], depth.shape[1]),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).view(height, width)
        return depth, {
            "enabled": True,
            "path": cache.get("path"),
            "camera": camera_basename(camera),
        }

    def sample_dense_union_mask(union_mask):
        block_size = max(int(getattr(gaussians, "field_bg_dense_sample_block_size", 3)), 1)
        pixels_per_block = max(int(getattr(gaussians, "field_bg_dense_pixels_per_block", 1)), 1)
        union_mask = union_mask.bool()
        selected_mask = torch.zeros_like(union_mask, dtype=torch.bool)
        coords = torch.nonzero(union_mask, as_tuple=False)
        if coords.numel() == 0:
            return selected_mask
        if block_size <= 1:
            return union_mask

        _, width = union_mask.shape
        blocks_x = (int(width) + block_size - 1) // block_size
        block_y = torch.div(coords[:, 0], block_size, rounding_mode="floor")
        block_x = torch.div(coords[:, 1], block_size, rounding_mode="floor")
        block_ids = block_y * blocks_x + block_x
        local_y = coords[:, 0] - block_y * block_size
        local_x = coords[:, 1] - block_x * block_size
        center = (block_size - 1) * 0.5
        priority = (local_y.float() - center).pow(2) + (local_x.float() - center).pow(2)

        order = torch.argsort(block_ids)
        coords = coords[order]
        block_ids = block_ids[order]
        priority = priority[order]
        _, counts = torch.unique_consecutive(block_ids, return_counts=True)
        start = 0
        for count in counts.tolist():
            end = start + int(count)
            if count <= pixels_per_block:
                chosen = torch.arange(start, end, device=coords.device)
            else:
                _, local_order = torch.topk(priority[start:end], k=pixels_per_block, largest=False)
                chosen = start + local_order
            selected_mask[coords[chosen, 0], coords[chosen, 1]] = True
            start = end
        max_pixels = int(getattr(gaussians, "field_bg_dense_max_pixels_per_camera", 0))
        selected_coords = torch.nonzero(selected_mask, as_tuple=False)
        if max_pixels > 0 and selected_coords.shape[0] > max_pixels:
            indices = torch.linspace(
                0,
                selected_coords.shape[0] - 1,
                steps=max_pixels,
                device=selected_coords.device,
            ).round().long()
            capped_mask = torch.zeros_like(selected_mask, dtype=torch.bool)
            capped_coords = selected_coords[indices]
            capped_mask[capped_coords[:, 0], capped_coords[:, 1]] = True
            selected_mask = capped_mask
        return selected_mask

    def select_dense_source_times(iteration, image_name, need_mask, reference_camera, reference_gt, fallback_depth, source_time_indices, collect_debug=False):
        need_mask = need_mask.bool()
        device = need_mask.device
        height, width = need_mask.shape
        if fallback_depth is None:
            fallback_depth = torch.full((height, width), max(float(getattr(gaussians, "field_bg_prior_depth_max", 15.0)), 1e-4), device=device, dtype=torch.float32)
        fallback_depth = fallback_depth.to(device=device, dtype=torch.float32)
        fallback_depth = torch.where(
            torch.isfinite(fallback_depth) & (fallback_depth > 1e-4),
            fallback_depth,
            torch.full_like(fallback_depth, max(float(getattr(gaussians, "field_bg_prior_depth_max", 15.0)), 1e-4)),
        )

        source_time_indices = [int(t) for t in source_time_indices]
        min_support = max(int(getattr(gaussians, "field_bg_dense_source_min_support", 2)), 1)
        beit_bg_threshold = float(getattr(gaussians, "field_bg_dense_source_beit_background_threshold", 0.35))
        dilate_radius = max(int(getattr(gaussians, "field_bg_dense_source_dilate", 5)), 0)
        motion_threshold = max(float(getattr(gaussians, "field_bg_dense_source_motion_threshold", 0.12)), 1e-6)
        median_threshold = max(float(getattr(gaussians, "field_bg_dense_source_median_threshold", 0.12)), 1e-6)
        beit_weight = float(getattr(gaussians, "field_bg_dense_source_score_beit_weight", 0.60))
        motion_weight = float(getattr(gaussians, "field_bg_dense_source_score_motion_weight", 0.25))
        median_weight = float(getattr(gaussians, "field_bg_dense_source_score_median_weight", 0.15))

        support_count = torch.zeros_like(need_mask, dtype=torch.int32)
        best_score = torch.full_like(fallback_depth, float("inf"))
        source_time_map = torch.full_like(support_count, -1)
        source_timestamp_map = torch.full_like(
            fallback_depth,
            -1.0,
            dtype=torch.float32,
        )
        selected_depth = fallback_depth.clone()
        selected_color = reference_gt.detach().float().to(device=device).clone()

        per_time_stats = []
        debug_per_time = []
        fallback_not_counted = 0

        for time_idx in source_time_indices:
            source_camera = traincamlookup.get(int(time_idx), {}).get(image_name)
            if source_camera is None:
                per_time_stats.append({
                    "time_idx": int(time_idx),
                    "reason": "missing_camera",
                    "valid_pixels": 0,
                })
                continue

            source_render_pkg = render(
                source_camera,
                gaussians,
                pipe,
                background,
                override_color=None,
                basicfunction=rbfbasefunction,
                GRsetting=GRsetting,
                GRzer=GRzer,
                time_conditioned=None,
            )
            source_depth = source_render_pkg["depth"].detach().squeeze(0).float()
            source_gt = get_gt_image(source_camera).detach().float()
            temporal_motion = get_temporal_motion_map(int(time_idx), source_camera).detach().float()
            median, _, _ = get_observation_prior(source_camera)
            if median is None:
                current_to_median = torch.full_like(temporal_motion, float("inf"))
            else:
                current_to_median = torch.abs(source_gt - median.to(device=source_gt.device, dtype=source_gt.dtype)).mean(dim=0)

            beit_band, beit_foreground, beit_stats = get_dense_beit_foreground_mask(source_camera, need_mask.shape)
            if beit_band is None or beit_foreground is None:
                per_time_stats.append({
                    "time_idx": int(time_idx),
                    "reason": str((beit_stats or {}).get("reason", "missing_beit")),
                    "valid_pixels": 0,
                })
                del source_render_pkg, source_depth, source_gt, temporal_motion, current_to_median
                torch.cuda.empty_cache()
                continue

            beit_band = beit_band.to(device=device, dtype=torch.float32)
            beit_foreground = beit_foreground.to(device=device, dtype=torch.bool)
            beit_used_time_index = int((beit_stats or {}).get("used_time_index", time_idx))
            if beit_used_time_index != int(time_idx):
                fallback_not_counted += 1
                per_time_stats.append({
                    "time_idx": int(time_idx),
                    "reason": "fallback_not_counted",
                    "valid_pixels": 0,
                    "beit_used_time_index": int(beit_used_time_index),
                })
                if collect_debug:
                    debug_per_time.append({
                        "time_idx": int(time_idx),
                        "beit_band": beit_band.detach().cpu(),
                        "strict_background": torch.zeros_like(need_mask, dtype=torch.bool).detach().cpu(),
                        "dynamic_mask": torch.zeros_like(need_mask, dtype=torch.bool).detach().cpu(),
                        "valid_source": torch.zeros_like(need_mask, dtype=torch.bool).detach().cpu(),
                    })
                del source_render_pkg, source_depth, source_gt, temporal_motion, current_to_median, beit_band, beit_foreground
                torch.cuda.empty_cache()
                continue

            strict_background = (beit_band < beit_bg_threshold) & (~dilate_mask(beit_foreground, dilate_radius))
            dynamic_mask = (temporal_motion > motion_threshold) | (current_to_median > median_threshold)
            dynamic_mask = dilate_mask(dynamic_mask, dilate_radius)
            valid_source = need_mask & strict_background & (~dynamic_mask)
            support_count = support_count + valid_source.to(dtype=torch.int32)

            motion_score = torch.clamp(temporal_motion / motion_threshold, 0.0, 1.0)
            median_score = torch.clamp(current_to_median / median_threshold, 0.0, 1.0)
            source_score = beit_weight * beit_band + motion_weight * motion_score + median_weight * median_score
            better = valid_source & (source_score < best_score)
            if torch.count_nonzero(better) > 0:
                source_time_map = torch.where(better, torch.full_like(source_time_map, int(time_idx)), source_time_map)
                source_timestamp_map = torch.where(
                    better,
                    torch.full_like(
                        source_timestamp_map,
                        float(source_camera.timestamp),
                    ),
                    source_timestamp_map,
                )
                best_score = torch.where(better, source_score, best_score)
                valid_depth = torch.isfinite(source_depth) & (source_depth > 1e-4)
                depth_for_source = torch.where(valid_depth, source_depth, fallback_depth)
                selected_depth = torch.where(better, depth_for_source, selected_depth)
                selected_color = torch.where(better.unsqueeze(0), source_gt.to(device=device), selected_color)

            valid_pixels = int(torch.count_nonzero(valid_source).item())
            per_time_stats.append({
                "time_idx": int(time_idx),
                "reason": "ok",
                "valid_pixels": valid_pixels,
                "strict_background_pixels": int(torch.count_nonzero(strict_background & need_mask).item()),
                "dynamic_pixels": int(torch.count_nonzero(dynamic_mask & need_mask).item()),
                "foreground_pixels": int(torch.count_nonzero(beit_foreground & need_mask).item()),
                "beit_used_time_index": int(beit_used_time_index),
            })
            if collect_debug:
                debug_per_time.append({
                    "time_idx": int(time_idx),
                    "beit_band": beit_band.detach().cpu(),
                    "strict_background": strict_background.detach().cpu(),
                    "dynamic_mask": dynamic_mask.detach().cpu(),
                    "valid_source": valid_source.detach().cpu(),
                })
            del source_render_pkg, source_depth, source_gt, temporal_motion, current_to_median, beit_band, beit_foreground
            torch.cuda.empty_cache()

        supported = need_mask & (support_count >= min_support) & (source_time_map >= 0)
        source_time_map = torch.where(supported, source_time_map, torch.full_like(source_time_map, -1))
        source_timestamp_map = torch.where(
            supported,
            source_timestamp_map,
            torch.full_like(source_timestamp_map, -1.0),
        )
        selected_depth = torch.where(
            torch.isfinite(selected_depth) & (selected_depth > 1e-4),
            selected_depth,
            fallback_depth,
        ).clamp_min(1e-4)

        source_counts = {
            str(int(time_idx)): int(torch.count_nonzero(source_time_map == int(time_idx)).item())
            for time_idx in source_time_indices
        }
        stats = {
            "enabled": 1,
            "source_time_indices": source_time_indices,
            "min_support": int(min_support),
            "beit_background_threshold": float(beit_bg_threshold),
            "dilate": int(dilate_radius),
            "motion_threshold": float(motion_threshold),
            "median_threshold": float(median_threshold),
            "need_pixels": int(torch.count_nonzero(need_mask).item()),
            "supported_pixels": int(torch.count_nonzero(supported).item()),
            "fallback_not_counted": int(fallback_not_counted),
            "source_counts": source_counts,
            "per_time": per_time_stats,
        }
        debug = None
        if collect_debug:
            debug = {
                "source_time_indices": source_time_indices,
                "support_count": support_count.detach().cpu(),
                "source_time_map": source_time_map.detach().cpu(),
                "per_time": debug_per_time,
            }
        return (
            supported.detach(),
            selected_color.detach(),
            selected_depth.detach(),
            source_timestamp_map.detach(),
            stats,
            debug,
        )

    def dense_candidate_worldpoints(pixel_indices, viewpoint_cam, depth_base, depth_value=None, depth_scale=None):
        if pixel_indices is None or pixel_indices.numel() == 0:
            return torch.empty((0, 3), device=depth_base.device, dtype=depth_base.dtype)

        def pix2ndc(v, S):
            return (v * 2.0 + 1.0) / S - 1.0

        if depth_base.dim() == 3:
            depth_base = depth_base.squeeze(0)
        pixel_indices = pixel_indices.to(device=depth_base.device)
        u = pixel_indices[:, 0].long()
        v = pixel_indices[:, 1].long()

        if depth_value is not None and float(depth_value) > 0.0:
            target_depth = torch.full((pixel_indices.shape[0], 1), float(depth_value), device=depth_base.device, dtype=depth_base.dtype)
        else:
            scale = float(depth_scale) if depth_scale is not None and float(depth_scale) > 0.0 else 1.0
            target_depth = depth_base[u, v].view(-1, 1).clamp_min(1e-4) * scale

        camera2wold = viewpoint_cam.world_view_transform.T.inverse().to(device=depth_base.device, dtype=depth_base.dtype)
        projectinverse = viewpoint_cam.projection_matrix.T.inverse().to(device=depth_base.device, dtype=depth_base.dtype)
        ndcu = pix2ndc(u, viewpoint_cam.image_height).unsqueeze(1).to(dtype=depth_base.dtype)
        ndcv = pix2ndc(v, viewpoint_cam.image_width).unsqueeze(1).to(dtype=depth_base.dtype)
        ndccamera = torch.cat((ndcv, ndcu, torch.ones_like(ndcu), torch.ones_like(ndcu)), dim=1)
        localpointuv = ndccamera @ projectinverse.T
        direction_local = localpointuv / localpointuv[:, 3:]
        denom = direction_local[:, 2:3]
        denom = torch.where(torch.abs(denom) < 1e-6, torch.full_like(denom, 1e-6), denom)
        rate = target_depth / denom
        localpoint = direction_local * rate
        localpoint[:, -1] = 1.0
        worldpoint_h = localpoint @ camera2wold.T
        worldpoint = worldpoint_h / worldpoint_h[:, 3:]
        xyz = worldpoint[:, :3]
        return clip_dense_worldpoints_to_bbox(xyz, viewpoint_cam)

    def clip_dense_worldpoints_to_bbox(xyz, viewpoint_cam):
        if not bool(getattr(gaussians, "field_bg_dense_clip_to_bbox", 0)):
            return xyz
        euler_field = getattr(gaussians, "euler_field", None)
        if euler_field is None or not hasattr(euler_field, "bbox_min") or not hasattr(euler_field, "bbox_max"):
            return xyz
        if xyz.numel() == 0:
            return xyz

        bbox_min = euler_field.bbox_min.to(device=xyz.device, dtype=xyz.dtype).view(1, 3)
        bbox_max = euler_field.bbox_max.to(device=xyz.device, dtype=xyz.dtype).view(1, 3)
        inside = torch.all((xyz >= bbox_min) & (xyz <= bbox_max), dim=1)
        if torch.all(inside):
            return xyz

        origin = viewpoint_cam.camera_center.to(device=xyz.device, dtype=xyz.dtype).view(1, 3)
        direction = xyz - origin
        eps = torch.tensor(1e-6, device=xyz.device, dtype=xyz.dtype)
        inv_dir = torch.where(torch.abs(direction) > eps, 1.0 / direction, torch.full_like(direction, float("inf")))
        t0 = (bbox_min - origin) * inv_dir
        t1 = (bbox_max - origin) * inv_dir
        t_min_axis = torch.minimum(t0, t1)
        t_max_axis = torch.maximum(t0, t1)

        parallel = torch.abs(direction) <= eps
        origin_inside_axis = (origin >= bbox_min) & (origin <= bbox_max)
        t_min_axis = torch.where(parallel & origin_inside_axis, torch.full_like(t_min_axis, -float("inf")), t_min_axis)
        t_max_axis = torch.where(parallel & origin_inside_axis, torch.full_like(t_max_axis, float("inf")), t_max_axis)

        t_enter = torch.max(t_min_axis, dim=1).values
        t_exit = torch.min(t_max_axis, dim=1).values
        intersects = (t_exit >= torch.clamp_min(t_enter, 0.0)) & torch.isfinite(t_exit) & (t_exit > 0.0)
        margin = max(0.0, min(float(getattr(gaussians, "field_bg_dense_bbox_clip_margin", 0.999)), 1.0))
        t_clip = torch.clamp(t_exit * margin, min=0.0, max=1.0).view(-1, 1)
        clipped = origin + direction * t_clip
        should_clip = (~inside) & intersects & (t_exit < 1.0)
        return torch.where(should_clip.view(-1, 1), clipped, xyz)

    def apply_dense_cell_dedup(dense_payloads, depth_values, depth_scales):
        absolute_depths = [float(d) for d in depth_values if float(d) > 0.0]
        if len(absolute_depths) > 0:
            depth_candidates = [("value", float(v)) for v in absolute_depths]
        else:
            valid_scales = [float(s) for s in depth_scales if float(s) > 0.0]
            if len(valid_scales) == 0:
                valid_scales = [1.0]
            depth_candidates = [("scale", float(v)) for v in valid_scales]

        stats = {
            "enabled": int(bool(getattr(gaussians, "field_bg_dense_cell_dedup", 0))),
            "reason": "",
            "pre_pixels": int(sum(int(payload["pixel_indices"].shape[0]) for payload in dense_payloads)),
            "post_pixels": int(sum(int(payload["pixel_indices"].shape[0]) for payload in dense_payloads)),
            "pre_candidates": int(sum(int(payload["pixel_indices"].shape[0]) for payload in dense_payloads) * len(depth_candidates)),
            "post_candidates": int(sum(int(payload["pixel_indices"].shape[0]) for payload in dense_payloads) * len(depth_candidates)),
            "removed_pixels": 0,
            "removed_candidates": 0,
            "cells": 0,
            "existing_occupied_cells": 0,
            "outside_bbox_candidates": 0,
            "multi_view_cells": 0,
            "mean_candidates_per_cell": 0.0,
            "level": int(getattr(gaussians, "field_bg_dense_dedup_level", 3)),
            "max_per_cell": int(getattr(gaussians, "field_bg_dense_max_per_cell", 1)),
            "priority": str(getattr(gaussians, "field_bg_dense_dedup_priority", "center")),
        }
        if not bool(getattr(gaussians, "field_bg_dense_cell_dedup", 0)):
            stats["reason"] = "disabled"
            for payload in dense_payloads:
                payload["pre_dedup_sample_mask"] = payload["sample_mask"].clone()
                payload["selected_depth_values"] = {
                    float(v): payload["pixel_indices"].clone()
                    for kind, v in depth_candidates
                    if kind == "value"
                }
                payload["selected_depth_scales"] = {
                    float(v): payload["pixel_indices"].clone()
                    for kind, v in depth_candidates
                    if kind == "scale"
                }
                payload["dedup_stats"] = {"enabled": 0}
            return stats

        euler_field = getattr(gaussians, "euler_field", None)
        if euler_field is None or not hasattr(euler_field, "level_resolutions"):
            stats["reason"] = "missing_euler_field"
            for payload in dense_payloads:
                payload["pre_dedup_sample_mask"] = payload["sample_mask"].clone()
                payload["selected_depth_values"] = {}
                payload["selected_depth_scales"] = {}
                payload["dedup_stats"] = {"enabled": 1, "reason": stats["reason"]}
            return stats

        level_resolutions = list(getattr(euler_field, "level_resolutions", []))
        if len(level_resolutions) == 0:
            stats["reason"] = "missing_level_resolutions"
            for payload in dense_payloads:
                payload["pre_dedup_sample_mask"] = payload["sample_mask"].clone()
                payload["selected_depth_values"] = {}
                payload["selected_depth_scales"] = {}
                payload["dedup_stats"] = {"enabled": 1, "reason": stats["reason"]}
            return stats

        level = max(0, min(int(getattr(gaussians, "field_bg_dense_dedup_level", 3)), len(level_resolutions) - 1))
        max_per_cell = max(int(getattr(gaussians, "field_bg_dense_max_per_cell", 1)), 1)
        dedup_priority = str(getattr(gaussians, "field_bg_dense_dedup_priority", "center")).lower()
        resolution = tuple(int(v) for v in level_resolutions[level])
        bbox_min = euler_field.bbox_min.detach().view(3).float()
        bbox_span = torch.clamp(euler_field.bbox_span.detach().view(3).float(), min=1e-6)
        cell_size = bbox_span / torch.tensor(resolution, device=bbox_span.device, dtype=bbox_span.dtype)

        candidates_by_cell = {}
        cell_camera_sets = {}
        existing_counts_by_cell = {}
        selected_by_payload = []
        per_payload_pre = []
        per_payload_pre_candidates = []

        with torch.no_grad():
            existing_xyz = gaussians.get_xyz.detach()
            if existing_xyz.numel() > 0:
                bbox_min_existing = bbox_min.to(device=existing_xyz.device, dtype=existing_xyz.dtype)
                cell_size_existing = cell_size.to(device=existing_xyz.device, dtype=existing_xyz.dtype)
                resolution_existing = torch.tensor(resolution, device=existing_xyz.device, dtype=torch.long)
                existing_cell_coords = torch.floor((existing_xyz - bbox_min_existing) / cell_size_existing).long()
                valid_existing = torch.all((existing_cell_coords >= 0) & (existing_cell_coords < resolution_existing.view(1, 3)), dim=1)
                existing_cell_coords_cpu = existing_cell_coords[valid_existing].detach().cpu()
                for idx in range(existing_cell_coords_cpu.shape[0]):
                    cell_key = tuple(int(v) for v in existing_cell_coords_cpu[idx].tolist())
                    existing_counts_by_cell[cell_key] = existing_counts_by_cell.get(cell_key, 0) + 1
                del existing_cell_coords, valid_existing, existing_cell_coords_cpu
                torch.cuda.empty_cache()

        for payload_idx, payload in enumerate(dense_payloads):
            pixel_indices = payload["pixel_indices"]
            payload["pre_dedup_sample_mask"] = payload["sample_mask"].clone()
            pre_count = int(pixel_indices.shape[0])
            per_payload_pre.append(pre_count)
            per_payload_pre_candidates.append(pre_count * len(depth_candidates))
            selected_by_payload.append({})
            for kind, value in depth_candidates:
                selected_by_payload[payload_idx][(kind, float(value))] = torch.zeros(pre_count, dtype=torch.bool)
            if pre_count == 0:
                continue

            depth_base = payload["depth_base"].cuda(non_blocking=True)
            pixel_indices_cuda = pixel_indices.cuda(non_blocking=True)
            camera_name = str(payload["image_name"])
            for kind, value in depth_candidates:
                with torch.no_grad():
                    xyz = dense_candidate_worldpoints(
                        pixel_indices_cuda,
                        payload["camera"],
                        depth_base,
                        depth_value=value if kind == "value" else None,
                        depth_scale=value if kind == "scale" else None,
                    )
                    bbox_min_cuda = bbox_min.to(device=xyz.device, dtype=xyz.dtype)
                    cell_size_cuda = cell_size.to(device=xyz.device, dtype=xyz.dtype)
                    resolution_cuda = torch.tensor(resolution, device=xyz.device, dtype=torch.long)
                    cell_coords = torch.floor((xyz - bbox_min_cuda) / cell_size_cuda).long()
                    valid_cell = torch.all((cell_coords >= 0) & (cell_coords < resolution_cuda.view(1, 3)), dim=1)
                    if torch.count_nonzero(valid_cell) == 0:
                        stats["outside_bbox_candidates"] += int(cell_coords.shape[0])
                        del xyz, cell_coords, valid_cell
                        continue
                    outside_count = int(cell_coords.shape[0] - torch.count_nonzero(valid_cell).item())
                    stats["outside_bbox_candidates"] += outside_count
                    valid_indices = torch.nonzero(valid_cell, as_tuple=False).view(-1)
                    valid_cell_coords = cell_coords[valid_indices]
                    valid_xyz = xyz[valid_indices]
                    cell_centers = bbox_min_cuda + (valid_cell_coords.to(dtype=xyz.dtype) + 0.5) * cell_size_cuda
                    center_distance = ((valid_xyz - cell_centers) / cell_size_cuda).pow(2).sum(dim=1)
                    cell_coords_cpu = valid_cell_coords.detach().cpu()
                    center_distance_cpu = center_distance.detach().cpu()
                    valid_indices_cpu = valid_indices.detach().cpu()

                for valid_local_idx in range(valid_indices_cpu.shape[0]):
                    local_idx = int(valid_indices_cpu[valid_local_idx].item())
                    cell_key = tuple(int(v) for v in cell_coords_cpu[valid_local_idx].tolist())
                    candidates_by_cell.setdefault(cell_key, []).append((
                        float(value),
                        float(center_distance_cpu[valid_local_idx].item()),
                        payload_idx,
                        local_idx,
                        kind,
                        float(value),
                    ))
                    cell_camera_sets.setdefault(cell_key, set()).add(camera_name)
                del xyz, cell_coords, valid_cell, valid_indices, valid_cell_coords, valid_xyz, cell_centers, center_distance
                torch.cuda.empty_cache()
            del depth_base, pixel_indices_cuda
            torch.cuda.empty_cache()

        for cell_key, items in candidates_by_cell.items():
            if dedup_priority in ("far", "far_first", "depth_far", "depth_desc"):
                items.sort(key=lambda item: (-item[0], item[1]))
            elif dedup_priority in ("near", "near_first", "depth_near", "depth_asc"):
                items.sort(key=lambda item: (item[0], item[1]))
            else:
                items.sort(key=lambda item: item[1])
            existing_count = int(existing_counts_by_cell.get(cell_key, 0))
            remaining = max_per_cell - existing_count
            if remaining <= 0:
                continue
            for _, _, payload_idx, local_idx, kind, value in items[:remaining]:
                selected_by_payload[payload_idx][(kind, float(value))][local_idx] = True

        post_pixel_total = 0
        post_candidate_total = 0
        for payload_idx, payload in enumerate(dense_payloads):
            pixel_indices = payload["pixel_indices"]
            selected_depth_values = {}
            selected_depth_scales = {}
            selected_pixel_mask = torch.zeros(int(pixel_indices.shape[0]), dtype=torch.bool)
            for (kind, value), selected in selected_by_payload[payload_idx].items():
                if selected.numel() != int(pixel_indices.shape[0]):
                    selected = torch.zeros(int(pixel_indices.shape[0]), dtype=torch.bool)
                chosen_pixels = pixel_indices[selected]
                if chosen_pixels.numel() == 0:
                    continue
                selected_pixel_mask = selected_pixel_mask | selected
                post_candidate_total += int(chosen_pixels.shape[0])
                if kind == "value":
                    selected_depth_values[float(value)] = chosen_pixels
                else:
                    selected_depth_scales[float(value)] = chosen_pixels
            new_pixel_indices = pixel_indices[selected_pixel_mask]
            new_sample_mask = torch.zeros_like(payload["sample_mask"], dtype=torch.bool)
            if new_pixel_indices.numel() > 0:
                new_sample_mask[new_pixel_indices[:, 0].long(), new_pixel_indices[:, 1].long()] = True
            payload["pixel_indices"] = new_pixel_indices
            payload["sample_mask"] = new_sample_mask
            payload["selected_depth_values"] = selected_depth_values
            payload["selected_depth_scales"] = selected_depth_scales
            post_count = int(new_pixel_indices.shape[0])
            post_pixel_total += post_count
            payload["dedup_stats"] = {
                "enabled": 1,
                "level": int(level),
                "resolution": resolution,
                "max_per_cell": int(max_per_cell),
                "pre_pixels": int(per_payload_pre[payload_idx]),
                "post_pixels": int(post_count),
                "removed_pixels": int(per_payload_pre[payload_idx] - post_count),
                "pre_candidates": int(per_payload_pre_candidates[payload_idx]),
                "post_candidates": int(sum(int(v.shape[0]) for v in selected_depth_values.values()) + sum(int(v.shape[0]) for v in selected_depth_scales.values())),
                "removed_candidates": int(per_payload_pre_candidates[payload_idx] - (sum(int(v.shape[0]) for v in selected_depth_values.values()) + sum(int(v.shape[0]) for v in selected_depth_scales.values()))),
            }

        cell_count = len(candidates_by_cell)
        stats.update({
            "reason": "ok",
            "level": int(level),
            "resolution": resolution,
            "max_per_cell": int(max_per_cell),
            "priority": dedup_priority,
            "post_pixels": int(post_pixel_total),
            "post_candidates": int(post_candidate_total),
            "removed_pixels": int(stats["pre_pixels"] - post_pixel_total),
            "removed_candidates": int(stats["pre_candidates"] - post_candidate_total),
            "cells": int(cell_count),
            "existing_occupied_cells": int(len(existing_counts_by_cell)),
            "multi_view_cells": int(sum(1 for cameras in cell_camera_sets.values() if len(cameras) > 1)),
            "mean_candidates_per_cell": float(stats["pre_candidates"] / max(cell_count, 1)),
        })
        print(
            f"[STEGF] Dense cell dedup: level={level}, resolution={resolution}, "
            f"max_per_cell={max_per_cell}, priority={dedup_priority}, "
            f"candidates {stats['pre_candidates']} -> {stats['post_candidates']}, "
            f"pixels {stats['pre_pixels']} -> {stats['post_pixels']}, cells={stats['cells']}, "
            f"existing_cells={stats['existing_occupied_cells']}, multi_view_cells={stats['multi_view_cells']}"
        )
        return stats

    def save_dense_background_add_debug(iteration, camera, gt_image, image, color_image,
                                        union_mask, sample_mask, depth_base, depth_scales, added_points, frame_stats,
                                        filtered_mask=None, da3_depth=None, da3_foreground_mask=None, da3_stats=None,
                                        beit_band=None, beit_foreground_mask=None, beit_stats=None,
                                        pre_dedup_sample_mask=None, dedup_stats=None,
                                        source_selection_stats=None, source_selection_debug=None,
                                        persistent_need_mask=None, time0_unreliable_mask=None):
        if not bool(getattr(gaussians, "field_bg_dense_debug", 0)):
            return
        safe_name = str(camera.image_name).replace("/", "_").replace("\\", "_")
        filtered_mask = union_mask if filtered_mask is None else filtered_mask
        pre_dedup_sample_mask = sample_mask if pre_dedup_sample_mask is None else pre_dedup_sample_mask

        union_dir = os.path.join(args.model_path, "bg_dense_add_debug", "unreliable_union_by_camera")
        os.makedirs(union_dir, exist_ok=True)
        sample_dir = os.path.join(args.model_path, "bg_dense_add_debug", "sampled_pixels_by_camera")
        os.makedirs(sample_dir, exist_ok=True)
        pre_dedup_sample_dir = os.path.join(args.model_path, "bg_dense_add_debug", "sampled_pixels_before_cell_dedup_by_camera")
        os.makedirs(pre_dedup_sample_dir, exist_ok=True)
        gt_vis = gt_image.detach().float().clamp(0.0, 1.0)
        save_debug_image(os.path.join(union_dir, f"{safe_name}.png"), union_mask.float())
        save_debug_image(
            os.path.join(union_dir, f"{safe_name}_overlay.png"),
            overlay_debug_mask(gt_vis, union_mask, (1.0, 0.0, 0.0), alpha=0.65),
        )
        save_debug_image(os.path.join(union_dir, f"{safe_name}_candidate_after_da3_filter.png"), filtered_mask.float())
        save_debug_image(
            os.path.join(union_dir, f"{safe_name}_candidate_after_da3_filter_overlay.png"),
            overlay_debug_mask(gt_vis, filtered_mask, (0.0, 1.0, 0.0), alpha=0.65),
        )
        save_debug_image(os.path.join(union_dir, f"{safe_name}_candidate_after_foreground_filter.png"), filtered_mask.float())
        save_debug_image(
            os.path.join(union_dir, f"{safe_name}_candidate_after_foreground_filter_overlay.png"),
            overlay_debug_mask(gt_vis, filtered_mask, (0.0, 1.0, 0.0), alpha=0.65),
        )
        save_debug_image(os.path.join(sample_dir, f"{safe_name}.png"), sample_mask.float())
        save_debug_image(
            os.path.join(sample_dir, f"{safe_name}_overlay.png"),
            overlay_debug_mask(gt_vis, sample_mask, (1.0, 0.0, 0.0), alpha=0.65),
        )
        save_debug_image(os.path.join(pre_dedup_sample_dir, f"{safe_name}.png"), pre_dedup_sample_mask.float())
        save_debug_image(
            os.path.join(pre_dedup_sample_dir, f"{safe_name}_overlay.png"),
            overlay_debug_mask(gt_vis, pre_dedup_sample_mask, (1.0, 0.6, 0.0), alpha=0.65),
        )

        nonlocal bg_dense_add_debug_events
        max_events = int(getattr(gaussians, "field_bg_dense_debug_max_events", 0))
        if max_events > 0 and bg_dense_add_debug_events >= max_events:
            return
        bg_dense_add_debug_events += 1

        event_dir = os.path.join(args.model_path, "bg_dense_add_debug", f"{int(iteration):06d}_{safe_name}")
        os.makedirs(event_dir, exist_ok=True)

        render_vis = image.detach().float().clamp(0.0, 1.0)
        color_vis = color_image.detach().float().clamp(0.0, 1.0)
        depth_vis = normalize_debug_map(depth_base, mask=torch.isfinite(depth_base) & (depth_base > 0.0))
        depth_max = max(float(getattr(gaussians, "field_bg_prior_depth_max", 15.0)), 1e-6)
        depth_fixed = torch.nan_to_num(depth_base / depth_max, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        save_debug_image(os.path.join(event_dir, "gt_ref.png"), gt_vis)
        save_debug_image(os.path.join(event_dir, "render_ref.png"), render_vis)
        save_debug_image(os.path.join(event_dir, "color_image.png"), color_vis)
        save_debug_image(os.path.join(event_dir, "unreliable_union.png"), union_mask.float())
        save_debug_image(os.path.join(event_dir, "unreliable_union_overlay.png"), overlay_debug_mask(gt_vis, union_mask, (1.0, 0.0, 0.0), alpha=0.65))
        save_debug_image(os.path.join(event_dir, "candidate_after_da3_filter.png"), filtered_mask.float())
        save_debug_image(os.path.join(event_dir, "candidate_after_da3_filter_overlay.png"), overlay_debug_mask(gt_vis, filtered_mask, (0.0, 1.0, 0.0), alpha=0.65))
        if da3_depth is not None:
            da3_depth = da3_depth.detach().float()
            save_debug_image(os.path.join(event_dir, "da3_depth.png"), normalize_debug_map(da3_depth, mask=torch.isfinite(da3_depth) & (da3_depth > 0.0)))
        if da3_foreground_mask is not None:
            save_debug_image(os.path.join(event_dir, "da3_foreground_mask.png"), da3_foreground_mask.float())
            save_debug_image(os.path.join(event_dir, "da3_foreground_overlay.png"), overlay_debug_mask(gt_vis, da3_foreground_mask, (0.0, 0.4, 1.0), alpha=0.55))
        if beit_band is not None:
            save_debug_image(os.path.join(event_dir, "beit_band_p10_p30.png"), beit_band.detach().float().clamp(0.0, 1.0))
        if beit_foreground_mask is not None:
            save_debug_image(os.path.join(event_dir, "beit_foreground_mask.png"), beit_foreground_mask.float())
            save_debug_image(os.path.join(event_dir, "beit_foreground_overlay.png"), overlay_debug_mask(gt_vis, beit_foreground_mask, (0.0, 0.4, 1.0), alpha=0.55))
        if source_selection_debug is not None:
            if persistent_need_mask is not None:
                save_debug_image(os.path.join(event_dir, "persistent_need_mask.png"), persistent_need_mask.float())
                save_debug_image(
                    os.path.join(event_dir, "persistent_need_overlay.png"),
                    overlay_debug_mask(gt_vis, persistent_need_mask, (1.0, 0.0, 0.0), alpha=0.65),
                )
            if time0_unreliable_mask is not None:
                save_debug_image(os.path.join(event_dir, "time0_unreliable_mask.png"), time0_unreliable_mask.float())
                save_debug_image(
                    os.path.join(event_dir, "time0_unreliable_overlay.png"),
                    overlay_debug_mask(gt_vis, time0_unreliable_mask, (0.0, 0.4, 1.0), alpha=0.65),
                )
            source_times = source_selection_debug.get("source_time_indices", [])
            support_count = source_selection_debug.get("support_count")
            source_time_map = source_selection_debug.get("source_time_map")
            if support_count is not None:
                denom = max(float(len(source_times)), 1.0)
                save_debug_image(os.path.join(event_dir, "source_support_count.png"), support_count.float() / denom)
            if source_time_map is not None:
                save_debug_image(os.path.join(event_dir, "source_time_map.png"), source_time_map_to_rgb(source_time_map.to(device=gt_vis.device), source_times))
            for item in source_selection_debug.get("per_time", []):
                time_idx = int(item.get("time_idx", -1))
                prefix = f"source_t{time_idx:02d}"
                if item.get("beit_band") is not None:
                    save_debug_image(os.path.join(event_dir, f"{prefix}_beit_band.png"), item["beit_band"].float().clamp(0.0, 1.0))
                if item.get("strict_background") is not None:
                    save_debug_image(os.path.join(event_dir, f"{prefix}_strict_background.png"), item["strict_background"].float())
                if item.get("dynamic_mask") is not None:
                    save_debug_image(os.path.join(event_dir, f"{prefix}_dynamic_mask.png"), item["dynamic_mask"].float())
                if item.get("valid_source") is not None:
                    save_debug_image(os.path.join(event_dir, f"{prefix}_valid_source.png"), item["valid_source"].float())
        save_debug_image(os.path.join(event_dir, "selected_pixels.png"), sample_mask.float())
        save_debug_image(os.path.join(event_dir, "selected_overlay.png"), overlay_debug_mask(gt_vis, sample_mask, (1.0, 0.0, 0.0), alpha=0.65))
        save_debug_image(os.path.join(event_dir, "selected_pixels_before_cell_dedup.png"), pre_dedup_sample_mask.float())
        save_debug_image(
            os.path.join(event_dir, "selected_before_cell_dedup_overlay.png"),
            overlay_debug_mask(gt_vis, pre_dedup_sample_mask, (1.0, 0.6, 0.0), alpha=0.65),
        )
        save_debug_image(os.path.join(event_dir, "base_depth.png"), depth_vis)
        save_debug_image(os.path.join(event_dir, "base_depth_fixed.png"), depth_fixed)

        meta = {
            "iteration": int(iteration),
            "camera": str(camera.image_name),
            "time_indices": parse_dense_add_time_indices(),
            "depth_base": str(getattr(gaussians, "field_bg_dense_depth_base", "render")),
            "depth_values": parse_float_list(getattr(gaussians, "field_bg_dense_depth_values", ""), default=[]),
            "depth_scales": [float(v) for v in depth_scales],
            "mask_source": str(getattr(gaussians, "field_bg_dense_mask_source", "instant")),
            "union_pixels": int(torch.count_nonzero(union_mask).item()),
            "candidate_after_da3_filter_pixels": int(torch.count_nonzero(filtered_mask).item()),
            "candidate_after_foreground_filter_pixels": int(torch.count_nonzero(filtered_mask).item()),
            "sample_pixels": int(torch.count_nonzero(sample_mask).item()),
            "sample_block_size": int(getattr(gaussians, "field_bg_dense_sample_block_size", 3)),
            "pixels_per_block": int(getattr(gaussians, "field_bg_dense_pixels_per_block", 1)),
            "max_pixels_per_camera": int(getattr(gaussians, "field_bg_dense_max_pixels_per_camera", 0)),
            "added_points": int(added_points),
            "da3": da3_stats or {},
            "beit": beit_stats or {},
            "source_selection": source_selection_stats or {},
            "cell_dedup": dedup_stats or {},
            "frame_stats": frame_stats,
        }
        with open(os.path.join(event_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    def run_dense_background_add(iteration):
        nonlocal bg_dense_add_done
        if not bool(getattr(gaussians, "field_bg_dense_add", 0)):
            return 0
        if bg_dense_add_done:
            return 0
        if int(iteration) != int(getattr(gaussians, "field_bg_dense_add_iter", 3000)):
            return 0
        if not observation_reliability_enabled():
            print("[STEGF] Dense background add skipped: field_obs_reliability is disabled.")
            bg_dense_add_done = True
            return 0
        if not hasattr(gaussians, "add_static_background_gaussians"):
            print("[STEGF] Dense background add skipped: Gaussian model has no add_static_background_gaussians.")
            bg_dense_add_done = True
            return 0

        depth_values = parse_float_list(getattr(gaussians, "field_bg_dense_depth_values", ""), default=[])
        depth_values = [float(v) for v in depth_values if float(v) > 0.0]
        depth_scales = parse_float_list(getattr(gaussians, "field_bg_dense_depth_scales", ""), default=[0.75, 1.09, 1.58, 2.29, 3.32, 4.82, 7.0])
        depth_scales = [float(v) for v in depth_scales if float(v) > 0.0]
        if len(depth_scales) == 0:
            depth_scales = [0.75, 1.09, 1.58, 2.29, 3.32, 4.82, 7.0]
        mask_source = str(getattr(gaussians, "field_bg_dense_mask_source", "instant")).lower()
        use_accumulated_mask = mask_source in ("accumulated", "ema", "history")
        time_indices = parse_dense_add_time_indices()
        if use_accumulated_mask and len(time_indices) > 0:
            reference_time_indices = [time_indices[len(time_indices) // 2]]
        else:
            reference_time_indices = time_indices
        source_time_select = bool(getattr(gaussians, "field_bg_dense_source_time_select", 0))
        source_time_indices = parse_dense_source_time_indices() if source_time_select else []
        image_names = sorted({cam.image_name for cam in traincameralist})
        total_added = 0
        total_sampled = 0
        camera_events = 0
        bg_dense_add_done = True
        dense_debug_enabled = bool(getattr(gaussians, "field_bg_dense_debug", 0))
        source_debug_collected = 0
        source_debug_max_events = int(getattr(gaussians, "field_bg_dense_debug_max_events", 0))

        print(
            f"[STEGF] Dense background add at iter {int(iteration)}: "
            f"{len(image_names)} cameras, times={reference_time_indices}, "
            f"source_time_select={int(source_time_select)}, source_times={source_time_indices}, "
            f"depth_values={depth_values}, depth_scales={depth_scales}, "
            f"sample_block={int(getattr(gaussians, 'field_bg_dense_sample_block_size', 3))}x"
            f"{int(getattr(gaussians, 'field_bg_dense_sample_block_size', 3))}, "
            f"pixels_per_block={int(getattr(gaussians, 'field_bg_dense_pixels_per_block', 1))}, "
            f"max_pixels_per_camera={int(getattr(gaussians, 'field_bg_dense_max_pixels_per_camera', 0))}, "
            f"mask_source={mask_source}, da3_filter={int(getattr(gaussians, 'field_bg_dense_da3_filter', 0))}, "
            f"beit_filter={int(getattr(gaussians, 'field_bg_dense_beit_filter', 0))}, "
            f"cell_dedup={int(getattr(gaussians, 'field_bg_dense_cell_dedup', 0))}"
        )

        dense_payloads = []
        for image_name in image_names:
            reference_camera = None
            reference_gt = None
            reference_image = None
            union_mask = None
            filtered_mask = None
            da3_depth = None
            da3_foreground_mask = None
            da3_stats = {}
            beit_band = None
            beit_foreground_mask = None
            beit_stats = {}
            color_accum = None
            color_count = None
            depth_base = None
            fallback_depth = None
            persistent_need_union = None
            time0_unreliable_union = None
            frame_stats = []

            for time_idx in reference_time_indices:
                target_camera = traincamlookup.get(int(time_idx), {}).get(image_name)
                if target_camera is None:
                    continue
                target_render_pkg = render(
                    target_camera,
                    gaussians,
                    pipe,
                    background,
                    override_color=None,
                    basicfunction=rbfbasefunction,
                    GRsetting=GRsetting,
                    GRzer=GRzer,
                    time_conditioned=None,
                )
                target_image = target_render_pkg["render"].detach()
                target_gt = get_gt_image(target_camera).detach()
                temporal_motion_map = get_temporal_motion_map(int(time_idx), target_camera)
                ema_available = (
                    str(target_camera.image_name) in observation_error_ema
                    and observation_error_ema[str(target_camera.image_name)].shape == target_image.shape[-2:]
                )
                _, unreliable_mask = get_observation_reliability(
                    iteration,
                    target_camera,
                    target_image,
                    target_gt,
                    temporal_motion_map,
                    update_ema=False,
                    use_accumulated=use_accumulated_mask,
                )
                depth = target_render_pkg["depth"].detach().squeeze(0).float()
                time0_unreliable_mask = unreliable_mask
                persistent_need_stats = {}
                if source_time_select and use_accumulated_mask:
                    persistent_need_mask, persistent_need_stats = get_persistent_need_mask_from_ema(
                        target_camera,
                        target_image.shape[-2:],
                        device=depth.device,
                    )
                    if time0_unreliable_mask is None:
                        time0_unreliable_mask = torch.zeros_like(persistent_need_mask, dtype=torch.bool)
                    else:
                        time0_unreliable_mask = time0_unreliable_mask.to(device=depth.device, dtype=torch.bool)
                    unreliable_mask = persistent_need_mask
                if unreliable_mask is None:
                    frame_stats.append({
                        "time_idx": int(time_idx),
                        "unreliable_pixels": 0,
                        "reason": "no_mask",
                        "mask_source": mask_source,
                        "ema_available": bool(ema_available),
                    })
                    del target_render_pkg, target_image, target_gt, temporal_motion_map, depth
                    torch.cuda.empty_cache()
                    continue
                unreliable_mask = unreliable_mask.to(device=depth.device, dtype=torch.bool)
                if reference_camera is None:
                    reference_camera = target_camera
                    reference_gt = target_gt
                    reference_image = target_image
                    union_mask = torch.zeros_like(unreliable_mask, dtype=torch.bool)
                    color_accum = torch.zeros_like(target_gt, dtype=torch.float32)
                    color_count = torch.zeros_like(unreliable_mask, dtype=torch.float32)
                    depth_base = torch.full_like(depth, float("inf"))
                    fallback_depth = depth.clone()
                    if source_time_select and use_accumulated_mask:
                        persistent_need_union = torch.zeros_like(unreliable_mask, dtype=torch.bool)
                        time0_unreliable_union = torch.zeros_like(unreliable_mask, dtype=torch.bool)

                unreliable_count = int(torch.count_nonzero(unreliable_mask).item())
                frame_stat = {
                    "time_idx": int(time_idx),
                    "unreliable_pixels": unreliable_count,
                    "mask_source": mask_source,
                    "ema_available": bool(ema_available),
                }
                if source_time_select and use_accumulated_mask:
                    frame_stat.update({
                        "need_mask_mode": "persistent_ema",
                        "persistent_need_pixels": unreliable_count,
                        "time0_unreliable_pixels": int(torch.count_nonzero(time0_unreliable_mask).item()),
                        "persistent_need": persistent_need_stats,
                    })
                frame_stats.append(frame_stat)
                if source_time_select and use_accumulated_mask:
                    persistent_need_union = persistent_need_union | unreliable_mask
                    time0_unreliable_union = time0_unreliable_union | time0_unreliable_mask
                if unreliable_count == 0:
                    del target_render_pkg, target_image, target_gt, temporal_motion_map, unreliable_mask, depth
                    if source_time_select and use_accumulated_mask:
                        del time0_unreliable_mask, persistent_need_mask
                    torch.cuda.empty_cache()
                    continue

                union_mask = union_mask | unreliable_mask
                mask_f = unreliable_mask.float()
                color_accum = color_accum + target_gt.float() * mask_f.unsqueeze(0)
                color_count = color_count + mask_f

                valid_depth = torch.isfinite(depth) & (depth > 1e-4)
                current_depth = torch.where(valid_depth & unreliable_mask, depth, torch.full_like(depth, float("inf")))
                depth_base = torch.minimum(depth_base, current_depth)
                del target_render_pkg, target_image, target_gt, temporal_motion_map, unreliable_mask, depth
                if source_time_select and use_accumulated_mask:
                    del time0_unreliable_mask, persistent_need_mask
                torch.cuda.empty_cache()

            if reference_camera is None or union_mask is None:
                continue

            if fallback_depth is None:
                fallback_depth = torch.full_like(depth_base, max(float(getattr(gaussians, "field_bg_prior_depth_max", 15.0)), 1e-4))
            fallback_depth = torch.where(
                torch.isfinite(fallback_depth) & (fallback_depth > 1e-4),
                fallback_depth,
                torch.full_like(fallback_depth, max(float(getattr(gaussians, "field_bg_prior_depth_max", 15.0)), 1e-4)),
            )

            source_selection_stats = {}
            source_selection_debug = None
            if source_time_select:
                collect_source_debug = dense_debug_enabled and (
                    source_debug_max_events <= 0 or source_debug_collected < source_debug_max_events
                )
                (
                    filtered_mask,
                    color_image,
                    depth_base,
                    source_timestamp_map,
                    source_selection_stats,
                    source_selection_debug,
                ) = select_dense_source_times(
                    iteration,
                    image_name,
                    union_mask,
                    reference_camera,
                    reference_gt,
                    fallback_depth,
                    source_time_indices,
                    collect_debug=collect_source_debug,
                )
                source_debug_collected += int(source_selection_debug is not None)
                da3_stats = {"enabled": False, "reason": "source_time_select"}
                beit_stats = {
                    "enabled": True,
                    "source_time_select": source_selection_stats,
                }
                da3_depth = None
                da3_foreground_mask = None
                beit_band = None
                beit_foreground_mask = None
            else:
                source_timestamp_map = torch.full_like(
                    depth_base,
                    float(reference_camera.timestamp),
                    dtype=torch.float32,
                )
                finite_depth = torch.isfinite(depth_base) & (depth_base > 1e-4)
                depth_base = torch.where(finite_depth, depth_base, fallback_depth).clamp_min(1e-4)

                color_count_safe = torch.clamp(color_count, min=1.0)
                color_image = color_accum / color_count_safe.unsqueeze(0)
                color_image = torch.where((color_count > 0.0).unsqueeze(0), color_image, reference_gt.float())
                filtered_mask = union_mask
                da3_depth, da3_foreground_mask, da3_stats = get_dense_da3_foreground_mask(reference_camera, union_mask.shape)
                if da3_foreground_mask is not None:
                    da3_foreground_mask = da3_foreground_mask.to(device=union_mask.device, dtype=torch.bool)
                    filtered_mask = union_mask & (~da3_foreground_mask)
                beit_band, beit_foreground_mask, beit_stats = get_dense_beit_foreground_mask(reference_camera, union_mask.shape)
                if beit_foreground_mask is not None:
                    beit_foreground_mask = beit_foreground_mask.to(device=union_mask.device, dtype=torch.bool)
                    filtered_mask = filtered_mask & (~beit_foreground_mask)
                depth_base_mode = str(getattr(gaussians, "field_bg_dense_depth_base", "render")).lower()
                if depth_base_mode == "da3" and da3_depth is not None:
                    da3_depth_for_base = da3_depth.to(device=depth_base.device, dtype=depth_base.dtype)
                    valid_da3_depth = torch.isfinite(da3_depth_for_base) & (da3_depth_for_base > 1e-4)
                    depth_base = torch.where(valid_da3_depth, da3_depth_for_base, depth_base)
                    da3_stats = dict(da3_stats or {})
                    da3_stats["depth_base"] = "da3"
                    da3_stats["depth_base_valid_pixels"] = int(torch.count_nonzero(valid_da3_depth).item())
                else:
                    da3_stats = dict(da3_stats or {})
                    da3_stats["depth_base"] = "render"
            sample_mask = sample_dense_union_mask(filtered_mask)
            pixel_indices = torch.nonzero(sample_mask, as_tuple=False)
            payload = {
                "image_name": image_name,
                "camera": reference_camera,
                "color_image": color_image.detach().cpu(),
                "sample_mask": sample_mask.detach().cpu(),
                "depth_base": depth_base.detach().cpu(),
                "source_timestamp_map": source_timestamp_map.detach().cpu(),
                "pixel_indices": pixel_indices.detach().cpu(),
                "union_pixels": int(torch.count_nonzero(union_mask).item()),
                "candidate_pixels": int(torch.count_nonzero(filtered_mask).item()),
                "pre_dedup_pixels": int(torch.count_nonzero(sample_mask).item()),
                "source_selection_stats": source_selection_stats,
            }
            if dense_debug_enabled:
                payload.update({
                    "gt_image": reference_gt.detach().cpu(),
                    "render_image": reference_image.detach().cpu(),
                    "union_mask": union_mask.detach().cpu(),
                    "filtered_mask": filtered_mask.detach().cpu(),
                    "da3_depth": da3_depth.detach().cpu() if da3_depth is not None else None,
                    "da3_foreground_mask": da3_foreground_mask.detach().cpu() if da3_foreground_mask is not None else None,
                    "da3_stats": da3_stats,
                    "beit_band": beit_band.detach().cpu() if beit_band is not None else None,
                    "beit_foreground_mask": beit_foreground_mask.detach().cpu() if beit_foreground_mask is not None else None,
                    "beit_stats": beit_stats,
                    "source_selection_debug": source_selection_debug,
                    "persistent_need_mask": persistent_need_union.detach().cpu() if persistent_need_union is not None else None,
                    "time0_unreliable_mask": time0_unreliable_union.detach().cpu() if time0_unreliable_union is not None else None,
                    "frame_stats": frame_stats,
                })
            dense_payloads.append(payload)
            del reference_gt, reference_image, union_mask, filtered_mask, sample_mask, color_accum, color_count, depth_base, fallback_depth, color_image, pixel_indices
            if persistent_need_union is not None:
                del persistent_need_union
            if time0_unreliable_union is not None:
                del time0_unreliable_union
            if da3_depth is not None:
                del da3_depth
            if da3_foreground_mask is not None:
                del da3_foreground_mask
            if beit_band is not None:
                del beit_band
            if beit_foreground_mask is not None:
                del beit_foreground_mask
            torch.cuda.empty_cache()

        cell_dedup_stats = apply_dense_cell_dedup(dense_payloads, depth_values, depth_scales)

        all_dense_xyz = []
        all_dense_rgb = []
        all_dense_depth_scale_tags = []
        all_dense_source_times = []
        dense_source_time_range = None

        for payload in dense_payloads:
            image_name = payload["image_name"]
            pixel_indices = payload["pixel_indices"]
            total_sampled += int(pixel_indices.shape[0])
            payload["pending_added_count"] = 0
            if pixel_indices.numel() == 0:
                continue

            depth_base_cuda = payload["depth_base"].cuda(non_blocking=True)
            color_image_cuda = payload["color_image"].cuda(non_blocking=True)
            source_timestamp_map_cuda = payload[
                "source_timestamp_map"
            ].cuda(non_blocking=True)
            added = 0

            def append_dense_candidates(selected_pixels, depth_value=None, depth_scale=None):
                if selected_pixels.numel() == 0:
                    return 0
                selected_pixels_cuda = selected_pixels.cuda(non_blocking=True)
                xyz = dense_candidate_worldpoints(
                    selected_pixels_cuda,
                    payload["camera"],
                    depth_base_cuda,
                    depth_value=depth_value,
                    depth_scale=depth_scale,
                )
                u = selected_pixels_cuda[:, 0].long()
                v = selected_pixels_cuda[:, 1].long()
                rgb = color_image_cuda[:, u, v].permute(1, 0).clamp(0.0, 1.0)
                source_times = source_timestamp_map_cuda[u, v].reshape(-1, 1)
                if depth_value is not None:
                    scale_tag_value = float("inf")
                else:
                    scale_tag_value = float(depth_scale) if depth_scale is not None else 1.0
                all_dense_xyz.append(xyz.detach())
                all_dense_rgb.append(rgb.detach())
                all_dense_depth_scale_tags.append(torch.full((xyz.shape[0],), scale_tag_value, device=xyz.device, dtype=xyz.dtype))
                all_dense_source_times.append(source_times.detach())
                count = int(xyz.shape[0])
                del (
                    selected_pixels_cuda,
                    xyz,
                    rgb,
                    source_times,
                )
                return count

            selected_depth_values = payload.get("selected_depth_values", {})
            selected_depth_scales = payload.get("selected_depth_scales", {})
            if selected_depth_values or selected_depth_scales:
                for depth_value, selected_pixels in sorted(selected_depth_values.items(), key=lambda item: float(item[0])):
                    added += append_dense_candidates(selected_pixels, depth_value=float(depth_value), depth_scale=None)
                for depth_scale_value, selected_pixels in sorted(selected_depth_scales.items(), key=lambda item: float(item[0])):
                    added += append_dense_candidates(selected_pixels, depth_value=None, depth_scale=float(depth_scale_value))
            else:
                if len(depth_values) > 0:
                    for depth_value in depth_values:
                        added += append_dense_candidates(pixel_indices, depth_value=float(depth_value), depth_scale=None)
                else:
                    for depth_scale_value in depth_scales:
                        added += append_dense_candidates(pixel_indices, depth_value=None, depth_scale=float(depth_scale_value))
            payload["pending_added_count"] = int(added)
            if added > 0:
                camera_events += 1
            del depth_base_cuda, color_image_cuda, source_timestamp_map_cuda
            torch.cuda.empty_cache()

        if len(all_dense_xyz) > 0:
            if not hasattr(gaussians, "add_static_background_gaussians_xyz"):
                raise RuntimeError("Gaussian model has no add_static_background_gaussians_xyz.")
            dense_xyz = torch.cat(all_dense_xyz, dim=0)
            dense_rgb = torch.cat(all_dense_rgb, dim=0)
            dense_depth_scale_tags = torch.cat(all_dense_depth_scale_tags, dim=0)
            dense_source_times = torch.cat(all_dense_source_times, dim=0)
            dense_source_time_range = (
                float(dense_source_times.min().item()),
                float(dense_source_times.max().item()),
            )
            total_added = int(gaussians.add_static_background_gaussians_xyz(
                dense_xyz,
                dense_rgb,
                iteration,
                depth_scale_tags=dense_depth_scale_tags,
                source_times=dense_source_times,
            ))
            del dense_xyz, dense_rgb, dense_depth_scale_tags, dense_source_times
            torch.cuda.empty_cache()

        if total_added > 0 and dense_source_time_range is not None:
            single_expert = str(
                getattr(
                    gaussians,
                    "field_existence_single_expert",
                    "none",
                )
            ).strip().lower()
            time_init_mode = (
                "source"
                if single_expert in {"persistent", "interval", "transient"}
                else "legacy_fixed"
            )
            print(
                "[STEGF] Dense add temporal init: "
                f"mode={time_init_mode}, single_expert={single_expert}, "
                f"source_time_min={dense_source_time_range[0]:.6f}, "
                f"source_time_max={dense_source_time_range[1]:.6f}"
            )

        for payload in dense_payloads:
            image_name = payload["image_name"]
            pixel_indices = payload["pixel_indices"]
            added = int(payload.get("pending_added_count", 0))
            if dense_debug_enabled:
                save_dense_background_add_debug(
                    iteration,
                    payload["camera"],
                    payload["gt_image"],
                    payload["render_image"],
                    payload["color_image"],
                    payload["union_mask"],
                    payload["sample_mask"],
                    payload["depth_base"],
                    depth_scales,
                    added,
                    payload["frame_stats"],
                    filtered_mask=payload["filtered_mask"],
                    da3_depth=payload["da3_depth"],
                    da3_foreground_mask=payload["da3_foreground_mask"],
                    da3_stats=payload["da3_stats"],
                    beit_band=payload["beit_band"],
                    beit_foreground_mask=payload["beit_foreground_mask"],
                    beit_stats=payload["beit_stats"],
                    pre_dedup_sample_mask=payload.get("pre_dedup_sample_mask"),
                    dedup_stats=payload.get("dedup_stats"),
                    source_selection_stats=payload.get("source_selection_stats"),
                    source_selection_debug=payload.get("source_selection_debug"),
                    persistent_need_mask=payload.get("persistent_need_mask"),
                    time0_unreliable_mask=payload.get("time0_unreliable_mask"),
                )
            source_counts = (payload.get("source_selection_stats") or {}).get("source_counts", {})
            print(
                f"[STEGF] Dense add {image_name}: "
                f"union_pixels={int(payload.get('union_pixels', 0))}, "
                f"candidate_pixels={int(payload.get('candidate_pixels', 0))}, "
                f"sample_pixels={int(pixel_indices.shape[0])}, "
                f"pre_dedup_pixels={int(payload.get('pre_dedup_pixels', int(pixel_indices.shape[0])))}, "
                f"new_points={int(added)}, source_counts={source_counts}"
            )
        del (
            all_dense_xyz,
            all_dense_rgb,
            all_dense_depth_scale_tags,
            all_dense_source_times,
        )

        if dense_debug_enabled:
            dense_log_dir = os.path.join(args.model_path, "bg_dense_add_debug")
            os.makedirs(dense_log_dir, exist_ok=True)
            with open(os.path.join(dense_log_dir, "events.jsonl"), "a") as f:
                f.write(json.dumps({
                    "iteration": int(iteration),
                    "train_cameras": int(len(image_names)),
                    "camera_events": int(camera_events),
                    "time_indices": reference_time_indices,
                    "configured_time_indices": time_indices,
                    "source_time_select": int(source_time_select),
                    "source_time_indices": source_time_indices,
                    "depth_base": str(getattr(gaussians, "field_bg_dense_depth_base", "render")),
                    "depth_values": depth_values,
                    "depth_scales": depth_scales,
                    "mask_source": mask_source,
                    "da3_filter": int(getattr(gaussians, "field_bg_dense_da3_filter", 0)),
                    "da3_path": bg_dense_da3_cache.get("path"),
                    "beit_filter": int(getattr(gaussians, "field_bg_dense_beit_filter", 0)),
                    "beit_path": bg_dense_beit_cache.get("path"),
                    "sample_block_size": int(getattr(gaussians, "field_bg_dense_sample_block_size", 3)),
                    "pixels_per_block": int(getattr(gaussians, "field_bg_dense_pixels_per_block", 1)),
                    "max_pixels_per_camera": int(getattr(gaussians, "field_bg_dense_max_pixels_per_camera", 0)),
                    "sample_pixels": int(total_sampled),
                    "added_points": int(total_added),
                    "cell_dedup": cell_dedup_stats,
                    "union_map_dir": "unreliable_union_by_camera",
                    "sample_map_dir": "sampled_pixels_by_camera",
                    "pre_dedup_sample_map_dir": "sampled_pixels_before_cell_dedup_by_camera",
                }, sort_keys=True) + "\n")
        print(f"[STEGF] Dense background add finished: cameras={camera_events}, sample_pixels={total_sampled}, added_points={total_added}")
        return total_added

    def parse_obs_reset_scan_time_indices():
        raw = str(getattr(gaussians, "field_obs_reset_scan_time_indices", "")).strip()
        if raw == "":
            raw = str(getattr(gaussians, "field_bg_prior_scan_time_indices", "")).strip()
        if raw:
            indices = []
            for item in raw.split(","):
                item = item.strip()
                if item:
                    indices.append(max(0, min(int(item), duration - 1)))
            if indices:
                return sorted(set(indices))
        if duration <= 1:
            return [0]
        defaults = [0, duration // 4, duration // 2, (duration * 3) // 4, duration - 1]
        return sorted(set(max(0, min(int(idx), duration - 1)) for idx in defaults))

    def build_obs_reset_scan_frames():
        time_indices = parse_obs_reset_scan_time_indices()
        frames = []
        views_per_time = int(getattr(gaussians, "field_obs_reset_scan_views_per_time", 0))
        for time_idx in time_indices:
            cameras = sorted(traincamdict.get(time_idx, []), key=lambda cam: str(cam.image_name))
            if views_per_time > 0 and len(cameras) > views_per_time:
                rng = random.Random(20240515 + int(time_idx))
                cameras = sorted(rng.sample(cameras, views_per_time), key=lambda cam: str(cam.image_name))
            for cam in cameras:
                frames.append((time_idx, cam))
        return frames

    def save_observation_scan_reset_debug(iteration, stats):
        nonlocal obs_reset_debug_events
        if not isinstance(stats, dict):
            return
        if int(stats.get("selected_points", 0)) <= 0 and not obs_reset_log_zero_enabled():
            return
        os.makedirs(obs_reset_debug_dir, exist_ok=True)
        serializable = strip_debug_tensors(stats)
        serializable["iteration"] = int(iteration)
        with open(obs_reset_log_path, "a") as f:
            f.write(json.dumps(serializable, sort_keys=True) + "\n")
        if not obs_reset_debug_enabled():
            return
        max_events = int(getattr(gaussians, "field_obs_reset_debug_max_events", 32))
        if max_events > 0 and obs_reset_debug_events >= max_events:
            return
        obs_reset_debug_events += 1
        event_dir = os.path.join(obs_reset_debug_dir, f"{int(iteration):06d}_scan_{int(stats.get('selected_points', 0))}")
        os.makedirs(event_dir, exist_ok=True)
        with open(os.path.join(event_dir, "meta.json"), "w") as f:
            json.dump(serializable, f, indent=2, sort_keys=True)

    def run_multiview_observation_reset(iteration):
        if not bool(getattr(gaussians, "field_obs_reset", 0)):
            return 0
        if str(getattr(gaussians, "field_obs_reset_mode", "batch")).lower() != "scan":
            return 0
        if not obs_reset_due(iteration):
            return 0
        if not hasattr(gaussians, "reset_observation_score_opacity"):
            return 0

        scan_frames = build_obs_reset_scan_frames()
        if len(scan_frames) == 0:
            save_observation_scan_reset_debug(iteration, {"reason": "empty_scan", "selected_points": 0})
            return 0

        device = gaussians.get_xyz.device
        num_points = gaussians.get_xyz.shape[0]
        reset_score = torch.zeros(num_points, device=device, dtype=torch.float32)
        reset_hits = torch.zeros(num_points, device=device, dtype=torch.int32)
        min_opacity = max(float(getattr(gaussians, "field_obs_reset_min_opacity", 0.05)), 0.0)
        min_masked = max(float(getattr(gaussians, "field_obs_reset_min_masked_contrib", 0.0)), 0.0)
        min_ratio = max(float(getattr(gaussians, "field_obs_reset_min_contrib_ratio", 0.05)), 0.0)
        update_ema = bool(getattr(gaussians, "field_obs_reset_scan_update_ema", 0))
        scan_stats = {
            "reason": "ok",
            "selection_mode": "multiview_contribution",
            "scan_frames": int(len(scan_frames)),
            "scan_frames_with_region": 0,
            "scan_frames_with_candidate": 0,
            "scan_region_pixels_total": 0,
            "scan_candidate_hits_total": 0,
            "scan_time_indices": parse_obs_reset_scan_time_indices(),
            "scan_views_per_time": int(getattr(gaussians, "field_obs_reset_scan_views_per_time", 0)),
            "scan_update_ema": int(update_ema),
            "scan_score_formula": "ratio*log1p(masked_contrib)*opacity",
        }

        for time_idx, target_camera in scan_frames:
            target_render_pkg = render(
                target_camera,
                gaussians,
                pipe,
                background,
                override_color=None,
                basicfunction=rbfbasefunction,
                GRsetting=GRsetting,
                GRzer=GRzer,
            )
            target_image = target_render_pkg["render"]
            target_gt = get_gt_image(target_camera)
            temporal_motion_map = get_temporal_motion_map(time_idx, target_camera)
            _, unreliable_mask = get_observation_reliability(
                iteration,
                target_camera,
                target_image,
                target_gt,
                temporal_motion_map,
                update_ema=update_ema,
            )
            if unreliable_mask is None:
                continue
            region_pixels = int(torch.count_nonzero(unreliable_mask).item())
            if region_pixels == 0:
                continue
            scan_stats["scan_frames_with_region"] += 1
            scan_stats["scan_region_pixels_total"] += region_pixels

            contrib_pkg = observation_contribution_ours_full(
                target_camera,
                gaussians,
                background,
                unreliable_mask,
                basicfunction=rbfbasefunction,
                GRsetting=GRsetting,
                GRzer=GRzer,
                time_conditioned=None,
            )
            contrib_total = contrib_pkg["contrib_total"].detach().to(device=device, dtype=torch.float32)
            contrib_masked = contrib_pkg["contrib_masked"].detach().to(device=device, dtype=torch.float32)
            visibility = contrib_pkg["visibility_filter"].detach().to(device=device, dtype=torch.bool)
            if contrib_total.shape[0] != num_points or contrib_masked.shape[0] != num_points or visibility.shape[0] != num_points:
                continue

            opacity = gaussians.get_opacity.detach().squeeze(1)
            contrib_ratio = contrib_masked / (contrib_total + 1e-6)
            candidate = (
                visibility
                & torch.isfinite(contrib_ratio)
                & (opacity > min_opacity)
                & (contrib_masked > min_masked)
                & (contrib_ratio > min_ratio)
            )
            candidate_count = int(torch.count_nonzero(candidate).item())
            if candidate_count == 0:
                continue
            score = contrib_ratio * torch.log1p(torch.clamp(contrib_masked, min=0.0)) * opacity.float()
            reset_score[candidate] += score[candidate]
            reset_hits[candidate] += 1
            scan_stats["scan_frames_with_candidate"] += 1
            scan_stats["scan_candidate_hits_total"] += candidate_count

        reset_stats = gaussians.reset_observation_score_opacity(
            reset_score,
            reset_hits,
            iteration,
            return_stats=True,
        )
        if isinstance(reset_stats, dict):
            reset_stats.update(scan_stats)
            save_observation_scan_reset_debug(iteration, reset_stats)
            return int(reset_stats.get("selected_points", 0))
        return int(reset_stats)

    def normalize_depth_fixed(depth, depth_max):
        depth_max = max(float(depth_max), 1e-6)
        depth = depth.detach().float()
        return torch.nan_to_num(depth / depth_max, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

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

    def make_point_count_map(x, y, height, width):
        count = torch.zeros((height, width), device=x.device, dtype=torch.float32)
        if x.numel() == 0:
            return count
        linear = y.long() * int(width) + x.long()
        count.view(-1).index_add_(0, linear, torch.ones_like(linear, dtype=torch.float32))
        return count

    def save_point_projection_image(out_root, subdir, file_stem, gt_image, count_map, color, dot_radius):
        count_vis = normalize_debug_map(count_map, mask=count_map > 0)
        point_mask = count_map > 0
        if int(dot_radius) > 0:
            point_mask = dilate_mask(point_mask, int(dot_radius))
        overlay = overlay_debug_mask(gt_image, point_mask, color=color, alpha=0.75)
        save_debug_image(os.path.join(out_root, subdir, file_stem + ".png"), overlay)
        save_debug_image(os.path.join(out_root, subdir + "_count", file_stem + ".png"), count_vis)

    def save_initial_point_projection_debug(out_root, viewpoint_cam, gt_image, file_stem, thresholds):
        if not bool(getattr(args, "field_init_point_projection_debug", 0)):
            return {}
        points = gaussians.get_xyz.detach()
        if points.numel() == 0:
            return {"projected_points": 0}
        height = int(viewpoint_cam.image_height)
        width = int(viewpoint_cam.image_width)
        projected = geom_transform_points(points, viewpoint_cam.full_proj_transform)
        ndc = projected[:, :2]
        valid = (
            torch.isfinite(ndc[:, 0])
            & torch.isfinite(ndc[:, 1])
            & (ndc[:, 0] >= -1.0)
            & (ndc[:, 0] <= 1.0)
            & (ndc[:, 1] >= -1.0)
            & (ndc[:, 1] <= 1.0)
        )
        if torch.count_nonzero(valid) == 0:
            return {"projected_points": 0}

        x = (((ndc[:, 0] + 1.0) * float(width)) - 1.0) * 0.5
        y = (((ndc[:, 1] + 1.0) * float(height)) - 1.0) * 0.5
        x = torch.round(x).long().clamp(0, width - 1)
        y = torch.round(y).long().clamp(0, height - 1)
        camera_center = viewpoint_cam.camera_center.to(device=points.device, dtype=points.dtype).view(1, 3)
        distance = torch.linalg.norm(points - camera_center, dim=1)
        valid = valid & torch.isfinite(distance) & (distance > 0.0)
        if torch.count_nonzero(valid) == 0:
            return {"projected_points": 0}

        x_valid = x[valid]
        y_valid = y[valid]
        depth_valid = distance[valid]
        thresholds = sorted([float(t) for t in thresholds if float(t) > 0.0])
        if len(thresholds) == 0:
            thresholds = [15.0, 25.0, 50.0]

        all_count = make_point_count_map(x_valid, y_valid, height, width)
        save_point_projection_image(
            out_root,
            "point_projection_all",
            file_stem,
            gt_image,
            all_count,
            color=(0.0, 1.0, 1.0),
            dot_radius=getattr(args, "field_init_point_projection_dot_radius", 1),
        )

        stats = {
            "projected_points": int(depth_valid.numel()),
            "projected_depth_min": float(depth_valid.min().detach().cpu()),
            "projected_depth_median": float(torch.median(depth_valid).detach().cpu()),
            "projected_depth_max": float(depth_valid.max().detach().cpu()),
        }

        low = 0.0
        colors = [
            (0.1, 0.7, 1.0),
            (1.0, 0.85, 0.1),
            (1.0, 0.35, 0.1),
            (1.0, 0.0, 1.0),
            (0.5, 1.0, 0.2),
        ]
        for idx, high in enumerate(thresholds):
            if idx == 0:
                mask = depth_valid < high
                name = f"point_projection_lt{high:g}"
                stat_key = f"projected_points_lt{high:g}"
            else:
                mask = (depth_valid >= low) & (depth_valid < high)
                name = f"point_projection_{low:g}_{high:g}"
                stat_key = f"projected_points_{low:g}_{high:g}"
            count_map = make_point_count_map(x_valid[mask], y_valid[mask], height, width)
            save_point_projection_image(
                out_root,
                name,
                file_stem,
                gt_image,
                count_map,
                color=colors[idx % len(colors)],
                dot_radius=getattr(args, "field_init_point_projection_dot_radius", 1),
            )
            stats[stat_key] = int(torch.count_nonzero(mask).item())
            low = high

        mask = depth_valid >= thresholds[-1]
        name = f"point_projection_ge{thresholds[-1]:g}"
        count_map = make_point_count_map(x_valid[mask], y_valid[mask], height, width)
        save_point_projection_image(
            out_root,
            name,
            file_stem,
            gt_image,
            count_map,
            color=colors[len(thresholds) % len(colors)],
            dot_radius=getattr(args, "field_init_point_projection_dot_radius", 1),
        )
        stats[f"projected_points_ge{thresholds[-1]:g}"] = int(torch.count_nonzero(mask).item())
        return stats

    def save_initial_depth_debug():
        if not bool(getattr(args, "field_init_depth_debug", 0)):
            return
        out_root = os.path.join(args.model_path, "init_depth_debug")
        subdirs = [
            "gt",
            "render",
            "render_depth",
            "render_depth_all",
            "render_depth_fixed",
            "depth_saturated_mask",
            "raw_npy",
            "meta",
        ]
        thresholds = parse_float_list(getattr(args, "field_init_depth_thresholds", "15,25,50"), default=[15.0, 25.0, 50.0])
        for threshold in thresholds:
            subdirs.append(f"render_depth_ge{threshold:g}")
        if bool(getattr(args, "field_init_point_projection_debug", 0)):
            projection_dirs = ["point_projection_all"]
            sorted_thresholds = sorted([float(t) for t in thresholds if float(t) > 0.0])
            low = 0.0
            for idx, high in enumerate(sorted_thresholds):
                if idx == 0:
                    projection_dirs.append(f"point_projection_lt{high:g}")
                else:
                    projection_dirs.append(f"point_projection_{low:g}_{high:g}")
                low = high
            if len(sorted_thresholds) > 0:
                projection_dirs.append(f"point_projection_ge{sorted_thresholds[-1]:g}")
            for projection_dir in projection_dirs:
                subdirs.append(projection_dir)
                subdirs.append(projection_dir + "_count")
        for subdir in subdirs:
            os.makedirs(os.path.join(out_root, subdir), exist_ok=True)

        time_indices = parse_int_list(getattr(args, "field_init_depth_time_indices", "0,12,25,37,49"), default=[0])
        time_indices = [idx for idx in time_indices if 0 <= idx < duration]
        if len(time_indices) == 0:
            time_indices = [0]
        views_per_time = int(getattr(args, "field_init_depth_views_per_time", 0))
        depth_max = float(getattr(args, "field_init_depth_max_depth", getattr(gaussians, "field_bg_prior_depth_max", 15.0)))
        manifest = []

        print(f"[STEGF] Saving initial depth debug to {out_root}")
        with torch.no_grad():
            for time_idx in time_indices:
                if time_idx in traincamdict:
                    cameras = list(traincamdict[time_idx])
                else:
                    cameras = [cam for cam in traincameralist if abs(float(cam.timestamp) - float(time_idx) / max(float(duration), 1.0)) < 1e-6]
                cameras = sorted(cameras, key=lambda cam: str(getattr(cam, "image_name", "")))
                if views_per_time > 0:
                    cameras = cameras[:views_per_time]
                for cam_idx, viewpoint_cam in enumerate(cameras):
                    render_pkg = render(
                        viewpoint_cam,
                        gaussians,
                        pipe,
                        background,
                        override_color=None,
                        basicfunction=rbfbasefunction,
                        GRsetting=GRsetting,
                        GRzer=GRzer,
                    )
                    image = render_pkg["render"].detach().float().clamp(0.0, 1.0)
                    gt_image = get_gt_image(viewpoint_cam).detach().float().clamp(0.0, 1.0)
                    depth = render_pkg["depth"].detach().squeeze(0).float()
                    valid = torch.isfinite(depth) & (depth > 0.0) & (depth < depth_max - 1e-4)
                    finite = torch.isfinite(depth) & (depth > 0.0)
                    saturated = torch.isfinite(depth) & (depth >= depth_max)
                    depth_vis = normalize_debug_map(depth, mask=valid)
                    depth_all_vis = normalize_debug_map(depth, mask=finite)
                    depth_fixed = normalize_depth_fixed(depth, depth_max)
                    safe_name = str(getattr(viewpoint_cam, "image_name", f"cam{cam_idx:02d}")).replace("/", "_").replace("\\", "_")
                    file_stem = f"t{time_idx:02d}_{safe_name}_{cam_idx:03d}"

                    save_debug_image(os.path.join(out_root, "gt", file_stem + ".png"), gt_image)
                    save_debug_image(os.path.join(out_root, "render", file_stem + ".png"), image)
                    save_debug_image(os.path.join(out_root, "render_depth", file_stem + ".png"), depth_vis)
                    save_debug_image(os.path.join(out_root, "render_depth_all", file_stem + ".png"), depth_all_vis)
                    save_debug_image(os.path.join(out_root, "render_depth_fixed", file_stem + ".png"), depth_fixed)
                    save_debug_image(os.path.join(out_root, "depth_saturated_mask", file_stem + ".png"), saturated.float())
                    for threshold in thresholds:
                        save_debug_image(
                            os.path.join(out_root, f"render_depth_ge{threshold:g}", file_stem + ".png"),
                            normalize_depth_ge_threshold(depth, threshold),
                        )
                    projection_stats = save_initial_point_projection_debug(
                        out_root,
                        viewpoint_cam,
                        gt_image,
                        file_stem,
                        thresholds,
                    )
                    np.save(os.path.join(out_root, "raw_npy", file_stem + ".npy"), depth.detach().cpu().numpy())

                    finite_values = depth[finite]
                    valid_values = depth[valid]
                    meta = {
                        "camera": str(getattr(viewpoint_cam, "image_name", "")),
                        "file_stem": file_stem,
                        "time_index": int(time_idx),
                        "timestamp": float(getattr(viewpoint_cam, "timestamp", 0.0)),
                        "num_initial_gaussians": int(gaussians.get_xyz.shape[0]),
                        "depth_max_for_fixed": float(depth_max),
                        "finite_pixels": int(torch.count_nonzero(finite).item()),
                        "valid_pixels_lt_depth_max": int(torch.count_nonzero(valid).item()),
                        "saturated_pixels_ge_depth_max": int(torch.count_nonzero(saturated).item()),
                        "depth_finite_min": float(finite_values.min().cpu()) if finite_values.numel() > 0 else 0.0,
                        "depth_finite_median": float(torch.median(finite_values).cpu()) if finite_values.numel() > 0 else 0.0,
                        "depth_finite_max": float(finite_values.max().cpu()) if finite_values.numel() > 0 else 0.0,
                        "depth_valid_min": float(valid_values.min().cpu()) if valid_values.numel() > 0 else 0.0,
                        "depth_valid_median": float(torch.median(valid_values).cpu()) if valid_values.numel() > 0 else 0.0,
                        "depth_valid_max": float(valid_values.max().cpu()) if valid_values.numel() > 0 else 0.0,
                    }
                    meta.update(projection_stats)
                    with open(os.path.join(out_root, "meta", file_stem + ".json"), "w") as f:
                        json.dump(meta, f, indent=2, sort_keys=True)
                    manifest.append(meta)
                    del render_pkg
        with open(os.path.join(out_root, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"[STEGF] Saved {len(manifest)} initial depth debug views.")

    save_initial_depth_debug()
    if bool(getattr(args, "field_init_depth_only", 0)):
        print("[STEGF] field_init_depth_only=1, stop after initial depth debug.")
        return

    with torch.no_grad():
        timeindex = 0 # 0 to 49
        viewpointset = traincamdict[timeindex]
        for viewpoint_cam in viewpointset:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background,  override_color=None,  basicfunction=rbfbasefunction, GRsetting=GRsetting, GRzer=GRzer)
            
            _, depthH, depthW = render_pkg["depth"].shape
            borderH = int(depthH/2)
            borderW = int(depthW/2)

            midh =  int(viewpoint_cam.image_height/2)
            midw =  int(viewpoint_cam.image_width/2)
            
            depth = render_pkg["depth"]
            slectemask = depth != 15.0 

            validdepthdict[viewpoint_cam.image_name] = torch.median(depth[slectemask]).item()   
            depthdict[viewpoint_cam.image_name] = torch.amax(depth[slectemask]).item() 
    
    if densify == 1 or  densify == 2: 
        zmask = gaussians._xyz[:,2] < 4.5  
        gaussians.prune_points(zmask) 
        torch.cuda.empty_cache()


    selectedlength = 2
    lasterems = 0 
    appearance_only_logged = False
    soft_geometry_lr_logged = False
    hard_time_loss_ema = {}
    hard_time_event_count = 0

    def hard_time_camera_key(time_idx, camera):
        return (int(time_idx), str(getattr(camera, "image_name", "")))

    def update_hard_time_loss_ema(time_idx, camera, loss_value):
        if not bool(getattr(gaussians, "field_mvstruct_hard_time", 0)):
            return
        if torch.is_tensor(loss_value):
            value = float(loss_value.detach().float().mean().cpu())
        else:
            value = float(loss_value)
        if not np.isfinite(value):
            return
        decay = float(getattr(gaussians, "field_mvstruct_hard_time_ema_decay", 0.05))
        decay = max(0.0, min(decay, 1.0))
        key = hard_time_camera_key(time_idx, camera)
        previous = hard_time_loss_ema.get(key)
        if previous is None:
            hard_time_loss_ema[key] = value
        else:
            hard_time_loss_ema[key] = (1.0 - decay) * previous + decay * value

    def hard_time_scores_by_time():
        if len(hard_time_loss_ema) > 0:
            fallback = float(np.mean(list(hard_time_loss_ema.values())))
        else:
            fallback = 1.0
        scores = []
        for time_idx in range(duration):
            values = []
            for camera in traincamdict.get(time_idx, []):
                value = hard_time_loss_ema.get(hard_time_camera_key(time_idx, camera))
                if value is not None and np.isfinite(value):
                    values.append(float(value))
            if values:
                scores.append(max(float(np.mean(values)), 1e-8))
            else:
                scores.append(max(fallback, 1e-8))
        return scores

    def select_hard_time_index():
        nonlocal hard_time_event_count
        sampling = str(getattr(gaussians, "field_mvstruct_hard_time_sampling", "alternate")).lower()
        use_error_sampling = sampling in {"error", "proportional", "loss", "hard"}
        if sampling == "alternate":
            use_error_sampling = (hard_time_event_count % 2) == 1
        hard_time_event_count += 1
        if not use_error_sampling:
            return randint(0, duration - 1), "uniform", 0.0
        scores = hard_time_scores_by_time()
        weights = np.asarray(scores, dtype=np.float64)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        weights = np.maximum(weights, 1e-8)
        if float(weights.sum()) <= 0.0:
            return randint(0, duration - 1), "uniform_fallback", 0.0
        selected = int(random.choices(range(duration), weights=weights.tolist(), k=1)[0])
        return selected, "error", float(scores[selected])

    def camera_center_cpu(camera):
        center = getattr(camera, "camera_center", None)
        if center is None:
            return torch.zeros(3, dtype=torch.float32)
        if torch.is_tensor(center):
            tensor = center.detach().float().cpu().view(-1)
        else:
            tensor = torch.tensor(center, dtype=torch.float32).view(-1)
        if tensor.numel() < 3:
            return torch.zeros(3, dtype=torch.float32)
        return tensor[:3]

    def select_diverse_hard_time_cameras(time_idx, cameras, count):
        cameras = list(cameras)
        if count >= len(cameras):
            return random.sample(cameras, len(cameras))
        if not bool(getattr(gaussians, "field_mvstruct_hard_time_diverse_views", 1)):
            return random.sample(cameras, count)
        seen_cameras = [
            camera
            for camera in cameras
            if hard_time_camera_key(time_idx, camera) in hard_time_loss_ema
        ]
        if seen_cameras:
            seed_camera = max(
                seen_cameras,
                key=lambda camera: hard_time_loss_ema.get(hard_time_camera_key(time_idx, camera), 0.0),
            )
        else:
            seed_camera = random.choice(cameras)
        centers = {id(camera): camera_center_cpu(camera) for camera in cameras}
        selected = [seed_camera]
        remaining = [camera for camera in cameras if camera is not seed_camera]
        while len(selected) < count and remaining:
            selected_centers = [centers[id(camera)] for camera in selected]
            best_camera = None
            best_distance = -1.0
            for camera in remaining:
                center = centers[id(camera)]
                min_distance = min(float(torch.norm(center - selected_center)) for selected_center in selected_centers)
                if min_distance > best_distance:
                    best_distance = min_distance
                    best_camera = camera
            selected.append(best_camera)
            remaining = [camera for camera in remaining if camera is not best_camera]
        return selected

    existence_stats_path = os.path.join(scene.model_path, "existence_moe_stats.jsonl")
    if first_iter <= 1:
        with open(existence_stats_path, "w", encoding="utf-8"):
            pass

    for iteration in range(first_iter, opt.iterations + 1):
        if ems_main_enabled and iteration ==  opt.emsstart:
            flagems = 1 # start ems

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        existence_activated = False
        if hasattr(gaussians, "set_existence_iteration"):
            existence_activated = gaussians.set_existence_iteration(iteration)
            if existence_activated:
                print(
                    "\n[STEGF] Existence MoE activated at iter "
                    f"{iteration}: material opacity x "
                    "(persistent + interval + transient), motion clock frozen."
                )
        if hasattr(gaussians, "apply_soft_geometry_lr"):
            soft_geometry_lr_active = gaussians.apply_soft_geometry_lr(iteration)
            if soft_geometry_lr_active and not soft_geometry_lr_logged:
                print(
                    "\n[STEGF] Soft geometry LR active at iter "
                    f"{iteration}: scale={getattr(gaussians, 'field_soft_geometry_lr_scale', '')}, "
                    f"full_lr={getattr(gaussians, 'field_soft_geometry_full_lr_groups', '')}"
                )
                soft_geometry_lr_logged = True
        if hasattr(gaussians, "set_field_training_stage"):
            gaussians.set_field_training_stage(iteration)
            if getattr(gaussians, "field_staged_training", False) and iteration in (
                getattr(gaussians, "field_category_activate_iter", -1),
                getattr(gaussians, "field_fast_activate_iter", -1),
            ):
                gaussians.refresh_dynamic_mask(iteration, force=True)
        
        if (iteration - 1) == debug_from:
            pipe.debug = True
        if gaussians.rgbdecoder is not None:
            gaussians.rgbdecoder.train()
        if getattr(gaussians, "content_exposure_head", None) is not None:
            gaussians.content_exposure_head.train()
        if getattr(gaussians, "use_euler_field", False):
            if gaussians.euler_field is not None:
                gaussians.euler_field.train()
            if getattr(gaussians, "h2_velocity_field", None) is not None:
                gaussians.h2_velocity_field.train()
            if getattr(gaussians, "field_router", None) is not None:
                gaussians.field_router.train()
            if getattr(gaussians, "field_query_gate", None) is not None:
                gaussians.field_query_gate.train()
            if gaussians.field_decoder is not None:
                gaussians.field_decoder.train()

        hard_mvstruct_stepped = False
        if opt.batch > 1:
            gaussians.zero_gradient_cache()
            mvstruct_enabled = bool(getattr(gaussians, "field_mvstruct", 0))
            mvstruct_start = int(getattr(gaussians, "field_mvstruct_start", 10000))
            mvstruct_interval = max(int(getattr(gaussians, "field_mvstruct_interval", 50)), 1)
            if mvstruct_enabled and iteration == mvstruct_start and hasattr(gaussians, "initialize_mvstruct_stats"):
                gaussians.initialize_mvstruct_stats()
            mvstruct_event = (
                mvstruct_enabled
                and iteration > mvstruct_start
                and iteration <= int(getattr(gaussians, "field_mvstruct_until", 20000))
                and (iteration - mvstruct_start) % mvstruct_interval == 0
            )
            hard_mvstruct_event = mvstruct_event and bool(getattr(gaussians, "field_mvstruct_hard_time", 0))
            if hard_mvstruct_event:
                timeindex, hard_time_mode, hard_time_score = select_hard_time_index()
            else:
                timeindex = randint(0, duration-1) # 0 to 49
                hard_time_mode, hard_time_score = "", 0.0
            viewpointset = traincamdict[timeindex]
            current_batch = opt.batch
            if mvstruct_event:
                current_batch = min(max(int(getattr(gaussians, "field_mvstruct_views", 5)), opt.batch), len(viewpointset))
                if current_batch < max(int(getattr(gaussians, "field_mvstruct_min_event_views", 3)), 1):
                    mvstruct_event = False
                    hard_mvstruct_event = False
                    current_batch = opt.batch
            if hard_mvstruct_event:
                camindex = select_diverse_hard_time_cameras(timeindex, viewpointset, current_batch)
                if iteration % 500 == 0:
                    scene.recordpoints(
                        iteration,
                        "mvstruct_hard_time_t{}_{}_score{:.4f}".format(
                            int(timeindex),
                            hard_time_mode,
                            float(hard_time_score),
                        ),
                    )
            else:
                camindex = random.sample(viewpointset, current_batch)
            if mvstruct_event and hasattr(gaussians, "mvstruct_begin_event"):
                gaussians.mvstruct_begin_event(expected_views=current_batch)
            static_app_active = (
                getattr(gaussians, "use_euler_field", False)
                and (not getattr(gaussians, "field_v23_compat", False))
                and getattr(gaussians, "field_static_app_scale", 0.0) > 0.0
            )
            shared_time_conditioned = None
            if (not static_app_active) and (not hard_mvstruct_event) and (not mvstruct_event):
                shared_time_conditioned = gaussians.compose_time_conditioned_attributes(
                    camindex[0].timestamp,
                    rbfbasefunction,
                    camera_center=camindex[0].camera_center,
            )
            bg_prior_contexts = []
            obs_reset_contexts = []
            bg_only_pixels = 0
            layer_resp_events = 0
            layer_resp_pixels = 0
            layer_resp_far_loss_sum = 0.0
            layer_resp_front_loss_sum = 0.0
            densify_point_gate = None
            need_highfreq_densify_stats = bool(getattr(gaussians, "field_highfreq_densify", 0)) and (
                iteration < opt.densify_until_iter
                or (
                    hasattr(gaussians, "background_candidate_stats_enabled")
                    and gaussians.background_candidate_stats_enabled(iteration)
                )
            )

            for i in range(current_batch):
                viewpoint_cam = camindex[i]
                gt_image = get_gt_image(viewpoint_cam)
                temporal_motion_map = get_temporal_motion_map(timeindex, viewpoint_cam)
                static_radiance_mask = get_static_radiance_mask(
                    iteration,
                    viewpoint_cam,
                    gt_image,
                    temporal_motion_map,
                )
                render_pkg = render(
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    background,
                    override_color=None,
                    basicfunction=rbfbasefunction,
                    GRsetting=GRsetting,
                    GRzer=GRzer,
                    time_conditioned=shared_time_conditioned,
                    static_radiance_mask=static_radiance_mask,
                    iteration=iteration,
                )
                image, viewspace_point_tensor, visibility_filter, radii = getrenderparts(render_pkg) 
                means3D_for_mvstruct = render_pkg.get("means3D")
                if mvstruct_event and torch.is_tensor(means3D_for_mvstruct) and means3D_for_mvstruct.requires_grad:
                    means3D_for_mvstruct.retain_grad()
                
                if opt.gtmask: # for training with undistorted immerisve image, masking black pixels in undistorted image. 
                    mask = torch.sum(gt_image, dim=0) == 0
                    mask = mask.float()
                    image = image * (1- mask) +  gt_image * (mask)
                content_exposure_params = None
                if hasattr(gaussians, "apply_content_exposure"):
                    image = gaussians.apply_content_exposure(image)
                    content_exposure_params = detach_content_exposure_params(
                        getattr(gaussians, "_last_content_exposure_params", None)
                    )
                if hasattr(gaussians, "update_error_prior"):
                    gaussians.update_error_prior(visibility_filter, image, gt_image, viewpoint_cam, render_pkg["means3D"].detach())

                reliability, unreliable_mask = get_observation_reliability(iteration, viewpoint_cam, image, gt_image, temporal_motion_map)
                bg_prior_source_mode = str(getattr(gaussians, "field_bg_prior_source", "background")).lower()
                if bool(getattr(gaussians, "field_bg_prior", 0)) and bg_prior_source_mode != "obs_reset":
                    bg_prior_contexts.append(
                        (
                            timeindex,
                            viewpoint_cam,
                            image.detach(),
                            gt_image.detach(),
                            {"depth": render_pkg["depth"].detach()},
                            unreliable_mask.detach() if unreliable_mask is not None else None,
                        )
                    )
                loss_image = apply_observation_reliability_to_loss_image(image, gt_image, reliability)
                bg_prior_needs_obs_mask = (
                    bool(getattr(gaussians, "field_bg_prior", 0))
                    and str(getattr(gaussians, "field_bg_prior_source", "background")).lower() == "obs_reset"
                )
                obs_reset_enabled = bool(getattr(gaussians, "field_obs_reset", 0))
                if (
                    unreliable_mask is not None
                    and (obs_reset_enabled or bg_prior_needs_obs_mask)
                    and hasattr(gaussians, "reset_observation_region_opacity")
                ):
                    obs_reset_contexts.append(
                        (
                            viewpoint_cam,
                            unreliable_mask.detach(),
                            image.detach(),
                            gt_image.detach(),
                            {"depth": render_pkg["depth"].detach()},
                        )
                    )

                if opt.reg == 2:
                    Ll1 = l2_loss(loss_image, gt_image)
                    loss = Ll1
                elif opt.reg == 3:
                    Ll1 = rel_loss(loss_image, gt_image)
                    loss = Ll1
                else:
                    Ll1 = l1_loss(loss_image, gt_image)
                    loss = getloss(opt, Ll1, ssim, loss_image, gt_image, gaussians, radii)
                update_hard_time_loss_ema(timeindex, viewpoint_cam, Ll1)
                content_exposure_reg_loss = (
                    gaussians.get_content_exposure_reg_loss()
                    if hasattr(gaussians, "get_content_exposure_reg_loss")
                    else None
                )
                if content_exposure_reg_loss is not None:
                    loss = loss + content_exposure_reg_loss
                existence_reg_loss = (
                    gaussians.get_existence_regularization_loss(iteration, rbfbasefunction)
                    if hasattr(gaussians, "get_existence_regularization_loss")
                    else None
                )
                if existence_reg_loss is not None:
                    loss = loss + existence_reg_loss
                h2_motion_reg_loss = (
                    gaussians.get_h2_motion_regularization_loss()
                    if hasattr(
                        gaussians,
                        "get_h2_motion_regularization_loss",
                    )
                    else None
                )
                if h2_motion_reg_loss is not None:
                    loss = loss + h2_motion_reg_loss
                bg_prior_loss = get_background_prior_loss(viewpoint_cam, image, gt_image)
                if bg_prior_loss is not None:
                    loss = loss + bg_prior_loss
                unreliable_boost_loss = get_unreliable_rgb_boost_loss(image, gt_image, unreliable_mask)
                if unreliable_boost_loss is not None:
                    loss = loss + unreliable_boost_loss
                bg_median_loss = get_background_median_loss(viewpoint_cam, image, unreliable_mask)
                if bg_median_loss is not None:
                    loss = loss + bg_median_loss
                depthpro_loss = get_depthpro_foreground_depth_loss(
                    iteration,
                    viewpoint_cam,
                    render_pkg.get("depth"),
                    unreliable_mask,
                )
                if depthpro_loss is not None:
                    loss = loss + depthpro_loss
                scale_reg_loss = get_depth_compensated_scale_loss(
                    iteration,
                    viewpoint_cam,
                    render_pkg.get("means3D"),
                )
                if scale_reg_loss is not None:
                    loss = loss + scale_reg_loss
                freq_prior_loss = get_frequency_prior_loss(iteration, viewpoint_cam, image, gt_image, unreliable_mask)
                if freq_prior_loss is not None:
                    loss = loss + freq_prior_loss
                if mvstruct_event:
                    mvstruct_dssim_weight = float(getattr(gaussians, "field_mvstruct_dssim_weight", 0.0))
                    if mvstruct_dssim_weight > 0.0:
                        loss = loss + mvstruct_dssim_weight * 0.5 * (1.0 - ssim(loss_image, gt_image))

                if need_highfreq_densify_stats:
                    densify_point_gate = compute_highfreq_densify_gate(
                        loss_image,
                        gt_image,
                        viewpoint_cam,
                        render_pkg.get("means3D"),
                        gaussians,
                    )

                if ems_main_enabled and flagems == 1:
                    if viewpoint_cam.image_name not in lossdiect:
                        lossdiect[viewpoint_cam.image_name] = loss.item()
                        ssimdict[viewpoint_cam.image_name] = ssim(image.clone().detach(), gt_image.clone().detach()).item()
                
                retain_graph = (not hard_mvstruct_event) and i < (current_batch - 1)
                loss.backward(retain_graph=retain_graph)
                if mvstruct_event and hasattr(gaussians, "mvstruct_capture_view"):
                    feature_dc_grad = getattr(gaussians, "_features_dc", None)
                    feature_dc_grad = feature_dc_grad.grad.detach().clone() if feature_dc_grad is not None and feature_dc_grad.grad is not None else None
                    position_grad = None
                    means3D_for_mvstruct = render_pkg.get("means3D")
                    if torch.is_tensor(means3D_for_mvstruct) and means3D_for_mvstruct.grad is not None:
                        position_grad = means3D_for_mvstruct.grad.detach().clone()
                    gaussians.mvstruct_capture_view(
                        viewspace_point_tensor.grad.detach() if viewspace_point_tensor.grad is not None else None,
                        visibility_filter.detach(),
                        radii.detach(),
                        feature_dc_grad=feature_dc_grad,
                        position_grad=position_grad,
                    )
                if hasattr(gaussians, "update_dynamic_scores"):
                    gaussians.update_dynamic_scores(
                        visibility_filter,
                        image,
                        gt_image,
                        viewpoint_cam,
                        render_pkg["means3D"].detach(),
                        viewspace_point_tensor,
                        temporal_motion_map=temporal_motion_map,
                    )
                if hasattr(gaussians, "boost_background_candidate_gradients"):
                    gaussians.boost_background_candidate_gradients(iteration)
                gaussians.cache_gradient()
                gaussians.optimizer.zero_grad(set_to_none = True)# 

                bg_only_loss, bg_only_pixel_count = get_bg_only_render_loss(
                    iteration,
                    viewpoint_cam,
                    gt_image,
                    unreliable_mask,
                )
                if bg_only_loss is not None:
                    bg_only_loss.backward()
                    if hasattr(gaussians, "keep_background_candidate_gradients_only"):
                        gaussians.keep_background_candidate_gradients_only(
                            update_modules=bool(getattr(gaussians, "field_bg_only_update_modules", 0))
                        )
                    gaussians.cache_gradient()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    bg_only_pixels += int(bg_only_pixel_count)
                if i == 0 and not hard_mvstruct_event:
                    layer_resp_stats = apply_layer_responsibility_losses(
                        iteration,
                        timeindex,
                        viewpoint_cam,
                        gt_image,
                        temporal_motion_map,
                        batch_scale=current_batch,
                        normal_image=image.detach(),
                        content_exposure_params=content_exposure_params,
                    )
                    if layer_resp_stats is not None:
                        if int(layer_resp_stats.get("pixels", 0)) > 0:
                            layer_resp_events += 1
                            layer_resp_pixels += int(layer_resp_stats.get("pixels", 0))
                            layer_resp_far_loss_sum += float(layer_resp_stats.get("far_loss", 0.0))
                            layer_resp_front_loss_sum += float(layer_resp_stats.get("front_loss", 0.0))
                        elif iteration % 500 == 0:
                            scene.recordpoints(
                                iteration,
                                "layer_resp_skip_" + str(layer_resp_stats.get("reason", "unknown")),
                            )
                if hard_mvstruct_event:
                    gaussians.set_batch_gradient(1)
                    if hasattr(gaussians, "apply_appearance_only_gradients"):
                        appearance_only_active = gaussians.apply_appearance_only_gradients(iteration)
                        if appearance_only_active and not appearance_only_logged:
                            print(
                                "\n[STEGF] Appearance-only optimization active at iter "
                                f"{iteration}: allow={getattr(gaussians, 'field_appearance_only_allow', '')}"
                            )
                            appearance_only_logged = True
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    gaussians.zero_gradient_cache()
                    hard_mvstruct_stepped = True

            if ems_main_enabled and flagems == 1 and len(lossdiect.keys()) == len(viewpointset):
                # sort dict by value
                orderedlossdiect = sorted(ssimdict.items(), key=lambda item: item[1], reverse=False) # ssimdict lossdiect
                flagems = 2
                selectviewslist = []
                selectviews = {}
                for idx, pair in enumerate(orderedlossdiect):
                    viewname, lossscore = pair
                    ssimscore = ssimdict[viewname]
                    if ssimscore < 0.91: # avoid large ssim
                        selectviewslist.append((viewname, "rk"+ str(idx) + "_ssim" + str(ssimscore)[0:4]))
                if len(selectviewslist) < 2 :
                    selectviews = []
                else:
                    selectviewslist = selectviewslist[:2]
                    for v in selectviewslist:
                        selectviews[v[0]] = v[1]

                selectedlength = len(selectviews)

            iter_end.record()
            if mvstruct_event and hasattr(gaussians, "mvstruct_commit_event"):
                gaussians.mvstruct_commit_event(
                    min_event_views=int(getattr(gaussians, "field_mvstruct_min_event_views", 3))
                )
            if not hard_mvstruct_event:
                gaussians.set_batch_gradient(current_batch)
            if bg_only_pixels > 0 and iteration % 100 == 0:
                scene.recordpoints(iteration, "bg_only_pixels_" + str(bg_only_pixels))
            if layer_resp_events > 0 and iteration % 100 == 0:
                scene.recordpoints(
                    iteration,
                    "layer_resp_px{}_far{:.5f}_front{:.5f}".format(
                        int(layer_resp_pixels),
                        layer_resp_far_loss_sum / max(layer_resp_events, 1),
                        layer_resp_front_loss_sum / max(layer_resp_events, 1),
                    ),
                )
            if iteration % 500 == 0 and hasattr(gaussians, "get_h2_motion_stats"):
                h2_stats = gaussians.get_h2_motion_stats()
                if h2_stats:
                    scene.recordpoints(
                        iteration,
                        "h2_venergy{:.6g}_disp_mean{:.6g}_disp_max{:.6g}".format(
                            h2_stats.get("normalized_velocity_energy", 0.0),
                            h2_stats.get("displacement_mean", 0.0),
                            h2_stats.get("displacement_max", 0.0),
                        ),
                    )
            if iteration % 500 == 0 and hasattr(gaussians, "get_carrier_motion_stats"):
                carrier_stats = gaussians.get_carrier_motion_stats()
                if carrier_stats:
                    scene.recordpoints(
                        iteration,
                        "carrier_fb{fallback_points:.0f}_c{carrier_points:.0f}_"
                        "h0{h0_points:.0f}_h1{h1_points:.0f}_h2{h2_points:.0f}_"
                        "clamp{clamped_points:.0f}_disp_mean{displacement_mean:.6g}_"
                        "disp_max{displacement_max:.6g}".format(**carrier_stats),
                    )
             # note we retrieve the correct gradient except the mask
        else:
            raise NotImplementedError("Batch size 1 is not supported")

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0 or iteration == opt.iterations:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(max(iteration - progress_bar.n, 0))
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration == opt.iterations and hasattr(gaussians, "get_existence_stats"):
                final_existence_stats = gaussians.get_existence_stats(iteration)
                if final_existence_stats:
                    final_existence_stats["iteration"] = int(iteration)
                    with open(existence_stats_path, "a", encoding="utf-8") as stats_file:
                        stats_file.write(json.dumps(final_existence_stats, sort_keys=True) + "\n")

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)




            # Densification and pruning here
            
            collect_bg_candidate_stats = (
                iteration >= opt.densify_until_iter
                and hasattr(gaussians, "background_candidate_stats_enabled")
                and gaussians.background_candidate_stats_enabled(iteration)
                and hasattr(gaussians, "get_background_candidate_mask")
            )
            if iteration < opt.densify_until_iter or collect_bg_candidate_stats:
                if iteration < opt.densify_until_iter:
                    stats_filter = visibility_filter
                else:
                    bg_candidate_mask = gaussians.get_background_candidate_mask()
                    stats_filter = visibility_filter & bg_candidate_mask
                if torch.count_nonzero(stats_filter) > 0:
                    gaussians.max_radii2D[stats_filter] = torch.max(gaussians.max_radii2D[stats_filter], radii[stats_filter])
                    if bool(getattr(gaussians, "field_highfreq_densify", 0)) and densify_point_gate is not None:
                        gaussians.add_densification_stats_with_gate(viewspace_point_tensor, stats_filter, densify_point_gate)
                    else:
                        gaussians.add_densification_stats(viewspace_point_tensor, stats_filter)
            flag = controlgaussians(opt, gaussians, densify, iteration, scene,  visibility_filter, radii, viewspace_point_tensor, flag,  traincamerawithdistance=None, maxbounds=maxbounds,minbounds=minbounds)
            dense_added_post_control = run_dense_background_add(iteration)
            if dense_added_post_control > 0:
                scene.recordpoints(iteration, "bg_dense_add_post_control_" + str(dense_added_post_control))
            if hasattr(gaussians, "refresh_dynamic_mask"):
                gaussians.refresh_dynamic_mask(iteration)
            if hasattr(gaussians, "maybe_temporal_refine_fast"):
                gaussians.maybe_temporal_refine_fast(iteration)
           
            # guided sampling step
            if ems_main_enabled and iteration > emsstartfromiterations and flagems == 2 and emscnt < selectedlength and viewpoint_cam.image_name in selectviews and (iteration - lasterems > 100): #["camera_0002"] :#selectviews :  #["camera_0002"]:
                selectviews.pop(viewpoint_cam.image_name) # remove sampled cameras
                emscnt += 1
                lasterems = iteration
                ssimcurrent = ssim(image.detach(), gt_image.detach()).item()
                scene.recordpoints(iteration, "ssim_" + str(ssimcurrent))
                # some scenes' strcture is already good, no need to add more points
                if ssimcurrent < 0.88:
                    imageadjust = image /(torch.mean(image)+0.01) # 
                    gtadjust = gt_image / (torch.mean(gt_image)+0.01)
                    diff = torch.abs(imageadjust   - gtadjust)
                    diff = torch.sum(diff,        dim=0) # h, w
                    diff_sorted, _ = torch.sort(diff.reshape(-1)) 
                    numpixels = diff.shape[0] * diff.shape[1]
                    threshold = diff_sorted[int(numpixels*opt.emsthr)].item()
                    outmask = diff > threshold#  
                    kh, kw = 16, 16 # kernel size
                    dh, dw = 16, 16 # stride
                    idealh, idealw = int(image.shape[1] / dh  + 1) * kw, int(image.shape[2] / dw + 1) * kw # compute padding  
                    outmask = torch.nn.functional.pad(outmask, (0, idealw - outmask.shape[1], 0, idealh - outmask.shape[0]), mode='constant', value=0)
                    patches = outmask.unfold(0, kh, dh).unfold(1, kw, dw)
                    dummypatch = torch.ones_like(patches)
                    patchessum = patches.sum(dim=(2,3)) 
                    patchesmusk = patchessum  >  kh * kh * 0.85
                    patchesmusk = patchesmusk.unsqueeze(2).unsqueeze(3).repeat(1,1,kh,kh).float()
                    patches = dummypatch * patchesmusk

                    depth = render_pkg["depth"]
                    depth = depth.squeeze(0)
                    idealdepthh, idealdepthw = int(depth.shape[0] / dh  + 1) * kw, int(depth.shape[1] / dw + 1) * kw # compute padding for depth

                    depth = torch.nn.functional.pad(depth, (0, idealdepthw - depth.shape[1], 0, idealdepthh - depth.shape[0]), mode='constant', value=0)

                    depthpaches = depth.unfold(0, kh, dh).unfold(1, kw, dw)
                    dummydepthpatches =  torch.ones_like(depthpaches)
                    a,b,c,d = depthpaches.shape
                    depthpaches = depthpaches.reshape(a,b,c*d)
                    mediandepthpatch = torch.median(depthpaches, dim=(2))[0]
                    depthpaches = dummydepthpatches * (mediandepthpatch.unsqueeze(2).unsqueeze(3))
                    unfold_depth_shape = dummydepthpatches.size()
                    output_depth_h = unfold_depth_shape[0] * unfold_depth_shape[2]
                    output_depth_w = unfold_depth_shape[1] * unfold_depth_shape[3]

                    patches_depth_orig = depthpaches.view(unfold_depth_shape)
                    patches_depth_orig = patches_depth_orig.permute(0, 2, 1, 3).contiguous()
                    patches_depth = patches_depth_orig.view(output_depth_h, output_depth_w).float() # 1 for error, 0 for no error

                    depth = patches_depth[:render_pkg["depth"].shape[1], :render_pkg["depth"].shape[2]]
                    depth = depth.unsqueeze(0)


                    midpatch = torch.ones_like(patches)
      

                    for i in range(0, kh,  2):
                        for j in range(0, kw, 2):
                            midpatch[:,:, i, j] = 0.0  
   
                    centerpatches = patches * midpatch

                    unfold_shape = patches.size()
                    patches_orig = patches.view(unfold_shape)
                    centerpatches_orig = centerpatches.view(unfold_shape)

                    output_h = unfold_shape[0] * unfold_shape[2]
                    output_w = unfold_shape[1] * unfold_shape[3]
                    patches_orig = patches_orig.permute(0, 2, 1, 3).contiguous()
                    centerpatches_orig = centerpatches_orig.permute(0, 2, 1, 3).contiguous()
                    centermask = centerpatches_orig.view(output_h, output_w).float() # H * W  mask, # 1 for error, 0 for no error
                    centermask = centermask[:image.shape[1], :image.shape[2]] # reverse back
                    
                    errormask = patches_orig.view(output_h, output_w).float() # H * W  mask, # 1 for error, 0 for no error
                    errormask = errormask[:image.shape[1], :image.shape[2]] # reverse back

                    H, W = centermask.shape

                    offsetH = int(H/10)
                    offsetW = int(W/10)

                    centermask[0:offsetH, :] = 0.0
                    centermask[:, 0:offsetW] = 0.0

                    centermask[-offsetH:, :] = 0.0
                    centermask[:, -offsetW:] = 0.0


                    depth = render_pkg["depth"]
                    depthmap = torch.cat((depth, depth, depth), dim=0)
                    invaliddepthmask = depth == 15.0

                    pathdir = scene.model_path + "/ems_" + str(emscnt-1)
                    if not os.path.exists(pathdir): 
                        os.makedirs(pathdir)
                    
                    depthmap = depthmap / torch.amax(depthmap)
                    invalideptmap = torch.cat((invaliddepthmask, invaliddepthmask, invaliddepthmask), dim=0).float()  


                    torchvision.utils.save_image(gt_image, os.path.join(pathdir,  "gt" + str(iteration) + ".png"))
                    torchvision.utils.save_image(image, os.path.join(pathdir,  "render" + str(iteration) + ".png"))
                    torchvision.utils.save_image(depthmap, os.path.join(pathdir,  "depth" + str(iteration) + ".png"))
                    torchvision.utils.save_image(invalideptmap, os.path.join(pathdir,  "indepth" + str(iteration) + ".png"))
                    

                    badindices = centermask.nonzero()
                    diff_sorted , _ = torch.sort(depth.reshape(-1)) 
                    N = diff_sorted.shape[0]
                    mediandepth = int(0.7 * N)
                    mediandepth = diff_sorted[mediandepth]

                    depth = torch.where(depth>mediandepth, depth,mediandepth )

                  
                    totalNnewpoints = gaussians.addgaussians(badindices, viewpoint_cam, depth, gt_image, numperay=opt.farray,ratioend=opt.rayends,  depthmax=depthdict[viewpoint_cam.image_name], shuffle=(opt.shuffleems != 0))

                    gt_image = gt_image * errormask
                    image = render_pkg["render"] * errormask

                    scene.recordpoints(iteration, "after addpointsbyuv")

                    torchvision.utils.save_image(gt_image, os.path.join(pathdir,  "maskedudgt" + str(iteration) + ".png"))
                    torchvision.utils.save_image(image, os.path.join(pathdir,  "maskedrender" + str(iteration) + ".png"))
                    visibility_filter = torch.cat((visibility_filter, torch.zeros(totalNnewpoints).cuda(0)), dim=0)
                    visibility_filter = visibility_filter.bool()
                    radii = torch.cat((radii, torch.zeros(totalNnewpoints).cuda(0)), dim=0)
                    viewspace_point_tensor = torch.cat((viewspace_point_tensor, torch.zeros(totalNnewpoints, 3).cuda(0)), dim=0)


                
            # Optimizer step
            if iteration < opt.iterations:
                if not hard_mvstruct_stepped:
                    if hasattr(gaussians, "apply_appearance_only_gradients"):
                        appearance_only_active = gaussians.apply_appearance_only_gradients(iteration)
                        if appearance_only_active and not appearance_only_logged:
                            print(
                                "\n[STEGF] Appearance-only optimization active at iter "
                                f"{iteration}: allow={getattr(gaussians, 'field_appearance_only_allow', '')}"
                            )
                            appearance_only_logged = True
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                existence_log_interval = max(
                    int(getattr(gaussians, "field_existence_log_interval", 500)),
                    1,
                )
                if (
                    hasattr(gaussians, "get_existence_stats")
                    and (
                        existence_activated
                        or iteration % existence_log_interval == 0
                        or iteration == opt.iterations
                    )
                ):
                    existence_stats = gaussians.get_existence_stats(iteration)
                    if existence_stats:
                        existence_stats["iteration"] = int(iteration)
                        with open(existence_stats_path, "a", encoding="utf-8") as stats_file:
                            stats_file.write(json.dumps(existence_stats, sort_keys=True) + "\n")
                        scene.recordpoints(
                            iteration,
                            "existence_p{:.3f}_i{:.3f}_t{:.3f}_hard{:.3f}_{:.3f}_{:.3f}".format(
                                existence_stats["persistent_weighted"],
                                existence_stats["interval_weighted"],
                                existence_stats["transient_weighted"],
                                existence_stats["persistent_hard"],
                                existence_stats["interval_hard"],
                                existence_stats["transient_hard"],
                            ),
                        )
                obs_reset_count = 0
                bg_added = 0
                if bool(getattr(gaussians, "field_obs_reset", 0)):
                    if str(getattr(gaussians, "field_obs_reset_mode", "batch")).lower() == "scan":
                        obs_reset_count = run_multiview_observation_reset(iteration)
                    else:
                        for obs_camera, obs_mask, obs_image, obs_gt, obs_render_pkg in obs_reset_contexts:
                            if not obs_reset_due(iteration):
                                continue
                            contrib_pkg = observation_contribution_ours_full(
                                obs_camera,
                                gaussians,
                                background,
                                obs_mask,
                                basicfunction=rbfbasefunction,
                                GRsetting=GRsetting,
                                GRzer=GRzer,
                                time_conditioned=None,
                            )
                            obs_means3D = contrib_pkg["means3D"].detach()
                            obs_visibility = contrib_pkg["visibility_filter"].detach()
                            obs_radii = contrib_pkg["radii"].detach()
                            if obs_reset_uses_contribution():
                                obs_contrib_total = contrib_pkg["contrib_total"].detach()
                                obs_contrib_masked = contrib_pkg["contrib_masked"].detach()
                            else:
                                obs_contrib_total = None
                                obs_contrib_masked = None
                            obs_stats = gaussians.reset_observation_region_opacity(
                                obs_mask,
                                obs_camera,
                                obs_means3D,
                                obs_visibility,
                                iteration,
                                radii=obs_radii,
                                return_stats=True,
                                contrib_total=obs_contrib_total,
                                contrib_masked=obs_contrib_masked,
                            )
                            if isinstance(obs_stats, dict):
                                obs_reset_count += int(obs_stats.get("selected_points", 0))
                                save_observation_reset_debug(iteration, obs_camera, obs_gt, obs_image, obs_mask, obs_stats)
                            else:
                                obs_reset_count += int(obs_stats)
                if obs_reset_count > 0:
                    scene.recordpoints(iteration, "obs_reset_" + str(obs_reset_count))

                if global_reset_due(iteration):
                    gaussians.reset_opacity()
                    scene.recordpoints(iteration, "global_reset_opacity")

                if (
                    bool(getattr(gaussians, "field_bg_prior", 0))
                    and str(getattr(gaussians, "field_bg_prior_source", "background")).lower() == "obs_reset"
                    and bg_prior_due(iteration)
                ):
                    for obs_camera, obs_mask, obs_image, obs_gt, obs_render_pkg in obs_reset_contexts:
                        bg_added += maybe_add_background_prior_points_for_camera(
                            iteration,
                            obs_camera,
                            obs_image,
                            obs_gt,
                            obs_render_pkg,
                            obs_mask,
                        )

                for bg_timeindex, bg_camera, bg_image, bg_gt_image, bg_render_pkg, bg_unreliable_mask in bg_prior_contexts:
                    bg_added += maybe_add_background_prior_points(
                        iteration,
                        bg_timeindex,
                        bg_camera,
                        bg_image,
                        bg_gt_image,
                        bg_render_pkg,
                        bg_unreliable_mask,
                    )
                if bg_added > 0:
                    scene.recordpoints(iteration, "bg_prior_add_" + str(bg_added))
                if hasattr(gaussians, "densify_background_candidates"):
                    bg_refine_stats = gaussians.densify_background_candidates(
                        iteration,
                        opt.densify_grad_threshold,
                        scene.cameras_extent,
                    )
                    if isinstance(bg_refine_stats, dict) and int(bg_refine_stats.get("new_points", 0)) > 0:
                        scene.recordpoints(
                            iteration,
                            "bg_prior_refine_{}_c{}_s{}".format(
                                int(bg_refine_stats.get("new_points", 0)),
                                int(bg_refine_stats.get("cloned", 0)),
                                int(bg_refine_stats.get("split_parents", 0)),
                            ),
                        )
                if hasattr(gaussians, "prune_mature_background_candidates"):
                    bg_pruned = gaussians.prune_mature_background_candidates(iteration)
                    if bg_pruned > 0:
                        scene.recordpoints(iteration, "bg_prior_prune_" + str(bg_pruned))
                if hasattr(gaussians, "densify_mvstruct_budgeted"):
                    mvstruct_stats = gaussians.densify_mvstruct_budgeted(
                        iteration,
                        scene.cameras_extent,
                    )
                    if isinstance(mvstruct_stats, dict) and int(mvstruct_stats.get("due", 0)) > 0:
                        scene.recordpoints(
                            iteration,
                            (
                                "mvstruct_densify_elig{}_base{}_recentrej{}_recentwould{}_sel{}"
                                "_ovelig{}_ovsel{}_ovr50{:.1f}_ovr90{:.1f}_ovrmax{:.1f}_normsel{}"
                                "_confelig{}_confsel{}_confb{}_confs50{:.3e}_confs90{:.3e}_confsmax{:.3e}"
                                "_confe50{:.1f}_confe90{:.1f}_confr50{:.1f}_confr90{:.1f}"
                                "_specelig{}_specsel{}_specfb{}_axis50{:.3f}_axis90{:.3f}"
                                "_axev50{:.1f}_axev90{:.1f}_spr50{:.1f}_spr90{:.1f}"
                                "_direlig{}_dirsel{}_dirfb{}_dirsig50{:.3e}_dirsig90{:.3e}"
                                "_scr50{:.1f}_scr90{:.1f}_fdn{:.3f}_off{:.2f}_sscale{:.2f}"
                                "_candc{}_cands{}_c{}_s{}_net{}_total{}_eb{}_rb{}"
                                "_p50{:.3e}_p90{:.3e}_p99{:.3e}_pmax{:.3e}"
                            ).format(
                                int(mvstruct_stats.get("eligible", 0)),
                                int(mvstruct_stats.get("base_eligible", 0)),
                                int(mvstruct_stats.get("recent_child_rejected", 0)),
                                int(mvstruct_stats.get("recent_would_select", 0)),
                                int(mvstruct_stats.get("selected", 0)),
                                int(mvstruct_stats.get("oversize_eligible", 0)),
                                int(mvstruct_stats.get("oversize_selected", 0)),
                                float(mvstruct_stats.get("oversize_radius_p50", 0.0)),
                                float(mvstruct_stats.get("oversize_radius_p90", 0.0)),
                                float(mvstruct_stats.get("oversize_radius_max", 0.0)),
                                int(mvstruct_stats.get("normal_selected", 0)),
                                int(mvstruct_stats.get("conflict_eligible", 0)),
                                int(mvstruct_stats.get("conflict_selected", 0)),
                                int(mvstruct_stats.get("conflict_budget", 0)),
                                float(mvstruct_stats.get("conflict_score_p50", 0.0)),
                                float(mvstruct_stats.get("conflict_score_p90", 0.0)),
                                float(mvstruct_stats.get("conflict_score_max", 0.0)),
                                float(mvstruct_stats.get("conflict_event_count_p50", 0.0)),
                                float(mvstruct_stats.get("conflict_event_count_p90", 0.0)),
                                float(mvstruct_stats.get("conflict_selected_radius_p50", 0.0)),
                                float(mvstruct_stats.get("conflict_selected_radius_p90", 0.0)),
                                int(mvstruct_stats.get("specialize_eligible", 0)),
                                int(mvstruct_stats.get("specialize_selected", 0)),
                                int(mvstruct_stats.get("specialize_fallback", 0)),
                                float(mvstruct_stats.get("axis_ratio_p50", 0.0)),
                                float(mvstruct_stats.get("axis_ratio_p90", 0.0)),
                                float(mvstruct_stats.get("axis_event_count_p50", 0.0)),
                                float(mvstruct_stats.get("axis_event_count_p90", 0.0)),
                                float(mvstruct_stats.get("parent_radius_p50", 0.0)),
                                float(mvstruct_stats.get("parent_radius_p90", 0.0)),
                                int(mvstruct_stats.get("directional_eligible", 0)),
                                int(mvstruct_stats.get("directional_selected", 0)),
                                int(mvstruct_stats.get("directional_fallback", 0)),
                                float(mvstruct_stats.get("directional_sigma_p50", 0.0)),
                                float(mvstruct_stats.get("directional_sigma_p90", 0.0)),
                                float(mvstruct_stats.get("estimated_child_radius_p50", 0.0)),
                                float(mvstruct_stats.get("estimated_child_radius_p90", 0.0)),
                                float(mvstruct_stats.get("feature_delta_norm", 0.0)),
                                float(mvstruct_stats.get("child_offset_ratio", 0.0)),
                                float(mvstruct_stats.get("specialize_scale_ratio", 0.0)),
                                int(mvstruct_stats.get("clone_candidates", 0)),
                                int(mvstruct_stats.get("split_candidates", 0)),
                                int(mvstruct_stats.get("cloned", 0)),
                                int(mvstruct_stats.get("split_parents", 0)),
                                int(mvstruct_stats.get("net_points", 0)),
                                int(mvstruct_stats.get("total_added", 0)),
                                int(mvstruct_stats.get("event_budget", 0)),
                                int(mvstruct_stats.get("remaining_budget", 0)),
                                float(mvstruct_stats.get("score_p50", 0.0)),
                                float(mvstruct_stats.get("score_p90", 0.0)),
                                float(mvstruct_stats.get("score_p99", 0.0)),
                                float(mvstruct_stats.get("score_max", 0.0)),
                            ),
                        )
                
                gaussians.optimizer.zero_grad(set_to_none = True)



if __name__ == "__main__":
    

    args, lp_extract, op_extract, pp_extract = getparser()
    setgtisint8(op_extract.gtisint8)
    train(lp_extract, op_extract, pp_extract, args.save_iterations, args.debug_from, densify=args.densify, duration=args.duration, rgbfunction=args.rgbfunction, rdpip=args.rdpip)

    # All done
    print("\nTraining complete.")
