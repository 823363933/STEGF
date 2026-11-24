#
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

# =============================================

# This license is additionally subject to the following restrictions:

# Licensor grants non-exclusive rights to use the Software for research purposes
# to research users (both academic and industrial), free of charge, without right
# to sublicense. The Software may be used "non-commercially", i.e., for research
# and/or evaluation purposes only.

# Subject to the terms and conditions of this License, you are granted a
# non-exclusive, royalty-free, license to reproduce, prepare derivative works of,
# publicly display, publicly perform and distribute its Work and any resulting
# derivative works in any form.
#

import torch
import numpy as np
import torch
from simple_knn._C import distCUDA2
import os 
import json 
import math
import cv2
import os 

SCALEDICT = {}


def getrenderpip(option="train_ours_full"):
    print("render option", option)
    if option == "train_ours_full":
        from thirdparty.gaussian_splatting.renderer import train_ours_full 
        from diff_gaussian_rasterization_ch9 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch9 import GaussianRasterizer  
        return train_ours_full, GaussianRasterizationSettings, GaussianRasterizer


    elif option == "test_ours_full":
        from thirdparty.gaussian_splatting.renderer import test_ours_full
        from diff_gaussian_rasterization_ch9 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch9 import GaussianRasterizer  
        return test_ours_full, GaussianRasterizationSettings, GaussianRasterizer
    elif option == "train_ours_fullss":
        from thirdparty.gaussian_splatting.renderer import train_ours_fullss
        from diff_gaussian_rasterization_ch9 import GaussianRasterizationSettings
        from diff_gaussian_rasterization_ch9 import GaussianRasterizer
        return train_ours_fullss, GaussianRasterizationSettings, GaussianRasterizer
    elif option == "test_ours_fullss":
        from thirdparty.gaussian_splatting.renderer import test_ours_fullss
        from diff_gaussian_rasterization_ch9 import GaussianRasterizationSettings
        from diff_gaussian_rasterization_ch9 import GaussianRasterizer
        return test_ours_fullss, GaussianRasterizationSettings, GaussianRasterizer
    else:
        raise NotImplementedError("Render {} is not included in the trimmed STEGF runtime".format(option))
    
def getmodel(model="oursfull"):
    if model == "ours_full":
        from  thirdparty.gaussian_splatting.scene.oursfull import GaussianModel
    else:
        raise NotImplementedError("Model {} is not included in the trimmed STEGF runtime".format(model))
    return GaussianModel

def getloss(opt, Ll1, ssim, image, gt_image, gaussians, radii):
    if opt.reg == 1: # add optical flow loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians._motion) / gaussians._motion.shape[0]
    elif opt.reg == 0 :
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))   
    elif opt.reg == 9 : #regulizor on the rotation
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians._omega[radii>0]**2)
    elif opt.reg == 10 : #regulizor on the rotation
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians._motion[radii>0]**2)
    elif opt.reg == 4:
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians.get_scaling) / gaussians._motion.shape[0]
    elif opt.reg == 5:
        loss = Ll1  
    elif opt.reg == 6 :
        ratio = torch.mean(gt_image) - 0.5 + opt.lambda_dssim
        ratio = torch.clamp(ratio, 0.0, 1.0)
        loss = (1.0 - ratio) * Ll1 + ratio * (1.0 - ssim(image, gt_image))
    elif opt.reg == 7 :
        Ll1 = Ll1 / (torch.mean(gt_image) * 2.0) # normalize L1 loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    elif opt.reg == 8 :
        N = gaussians._xyz.shape[0]
        mean = torch.mean(gaussians._xyz, dim=0, keepdim=True)
        varaince = (mean - gaussians._xyz)**2 #/ N
        loss = (1.0 - opt.lambda_dssim) * Ll1  + 0.0002*opt.lambda_dssim * torch.sum(varaince) / N
    return loss 


def freezweights(model, screenlist):
    for k in screenlist:
        grad_tensor = getattr(getattr(model, k), 'grad')
        newgrad = torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), 'grad', newgrad)
    return  

def freezweightsbymask(model, screenlist, mask):
    for k in screenlist:
        grad_tensor = getattr(getattr(model, k), 'grad')
        newgrad =  mask.unsqueeze(1)*grad_tensor #torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), 'grad', newgrad)
    return  


def freezweightsbymasknounsqueeze(model, screenlist, mask):
    for k in screenlist:
        grad_tensor = getattr(getattr(model, k), 'grad')
        newgrad =  mask*grad_tensor #torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), 'grad', newgrad)
    return  


def removeminmax(gaussians, maxbounds, minbounds):
    maxx, maxy, maxz = maxbounds
    minx, miny, minz = minbounds
    xyz = gaussians._xyz
    mask0 = xyz[:,0] > maxx.item()
    mask1 = xyz[:,1] > maxy.item()
    mask2 = xyz[:,2] > maxz.item()

    mask3 = xyz[:,0] < minx.item()
    mask4 = xyz[:,1] < miny.item()
    mask5 = xyz[:,2] < minz.item()
    mask =  logicalorlist([mask0, mask1, mask2, mask3, mask4, mask5])
    gaussians.prune_points(mask) 
    torch.cuda.empty_cache()


def n3d_control_schedule(opt):
    """Scale only the formerly hard-coded densify=1 control clock.

    Public optimization/model iteration fields are configured separately. The
    default scale of one preserves every legacy trigger and pruning cadence.
    """
    value = getattr(opt, "field_n3d_control_time_scale", 1)
    try:
        scale = int(value)
        valid = not isinstance(value, bool) and scale >= 1 and float(value) == scale
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("field_n3d_control_time_scale must be a positive integer")
    return {
        "scale": scale,
        "omega_freeze_iter": 8000 * scale + 1,
        "low_opacity_prune_until": 7000 * scale,
        "opacity_reset_interval": 3000 * scale,
        "spatial_prune_interval": 1000 * scale,
        "spatial_prune_offset": 500 * scale,
        "bounds_prune_iter": 10000 * scale,
    }


def n3d_quality_schedule(opt):
    """Independent long-sequence clocks, opt-in for the 300-frame experiment."""
    if not bool(getattr(opt, "field_n3d_quality_schedule", 0)):
        return None

    def integer(name, minimum=0):
        value = getattr(opt, name)
        try:
            result = int(value)
            valid = not isinstance(value, bool) and result >= minimum and float(value) == result
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("{} must be an integer >= {}".format(name, minimum))
        return result

    reset_steps = tuple(int(value.strip()) for value in str(opt.field_n3d_reset_schedule).split(",") if value.strip())
    if not reset_steps or tuple(sorted(set(reset_steps))) != reset_steps:
        raise ValueError("field_n3d_reset_schedule must contain sorted unique reset steps")
    schedule = {
        "reset_steps": reset_steps,
        "recovery": integer("field_n3d_reset_recovery", 1),
        "prune_start": integer("field_n3d_prune_start", 1),
        "prune_until": integer("field_n3d_prune_until", 1),
        "prune_interval": integer("field_n3d_prune_interval", 1),
        "late_prune_interval": integer("field_n3d_late_prune_interval", 1),
        "bounds_iter": integer("field_n3d_bounds_prune_iter", 1),
        "dense_start": integer("densify_from_iter"),
        "dense_until": integer("densify_until_iter", 1),
        "dense_interval": integer("densification_interval", 1),
        "spatial_clock": n3d_control_schedule(opt),
    }
    # A CLI --iterations 10 smoke run intentionally truncates this schedule.
    if not (schedule["dense_start"] < schedule["prune_start"] < schedule["dense_until"]
            <= schedule["bounds_iter"] <= schedule["prune_until"]):
        raise ValueError("Invalid long-sequence densification/pruning phase order")
    if reset_steps[0] <= schedule["dense_start"] or reset_steps[-1] + schedule["recovery"] >= schedule["dense_until"]:
        raise ValueError("Reset recovery must finish before ordinary densification ends")
    return schedule


def n3d_quality_actions(schedule, iteration):
    last_reset = max((step for step in schedule["reset_steps"] if step <= iteration), default=None)
    recovering = last_reset is not None and iteration - last_reset < schedule["recovery"]
    clock = schedule["spatial_clock"]
    prune_anchor = schedule["prune_start"] if iteration < schedule["dense_until"] else schedule["dense_until"]
    prune_interval = schedule["prune_interval"] if iteration < schedule["dense_until"] else schedule["late_prune_interval"]
    return {
        "reset": iteration in schedule["reset_steps"],
        "recovering": recovering,
        "densify": (schedule["dense_start"] < iteration < schedule["dense_until"]
                    and iteration % schedule["dense_interval"] == 0),
        "prune": (not recovering and schedule["prune_start"] <= iteration <= schedule["prune_until"]
                  and (iteration - prune_anchor) % prune_interval == 0),
        "bounds": iteration == schedule["bounds_iter"],
        "spatial": (schedule["dense_until"] <= iteration <= schedule["prune_until"]
                    and iteration % clock["spatial_prune_interval"] == clock["spatial_prune_offset"]),
    }


def controlgaussians_n3d_quality(opt, gaussians, iteration, scene, flag, maxbounds, minbounds):
    schedule = n3d_quality_schedule(opt)
    actions = n3d_quality_actions(schedule, iteration)
    # This path is deliberately restricted to the final model without the old
    # global omega/rotation hard split; do not silently change another model.
    if not bool(getattr(gaussians, "field_disable_global_omega_split", 0)):
        raise ValueError("The long-sequence topology schedule requires disabled global omega split")
    if actions["densify"]:
        # The time window, not desicnt or a global point cap, determines growth
        # opportunities. Keep the existing gradient and clone/split rules.
        scene.recordpoints(iteration, "before densify")
        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
        gaussians.densify_pruneclone(
            opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold,
        )
        flag += 1
        scene.recordpoints(iteration, "after densify")
    if actions["prune"]:
        # Base opacity, not the current frame's transient opacity. A Gaussian
        # outside its temporal support is not thereby a deletion candidate.
        prune_mask = (gaussians.get_opacity < opt.opthr).reshape(-1)
        if hasattr(gaussians, "protect_background_candidates_from_prune"):
            prune_mask = gaussians.protect_background_candidates_from_prune(prune_mask, iteration)
        gaussians.prune_points(prune_mask)
        scene.recordpoints(iteration, "quality_low_opacity_prune")
    if actions["reset"]:
        gaussians.reset_opacity()
        scene.recordpoints(iteration, "quality_opacity_reset")
    if actions["bounds"]:
        removeminmax(gaussians, maxbounds, minbounds)
        scene.recordpoints(iteration, "quality_bounds_prune")
    if actions["spatial"]:
        gaussians.prune_points(gaussians._xyz[:, 2] < 4.5)
        scene.recordpoints(iteration, "quality_spatial_prune")
    if actions["prune"] or actions["spatial"] or actions["bounds"]:
        torch.cuda.empty_cache()
    return flag


def long_sequence_refinement_schedule(opt):
    """Independent, opt-in pruning and fixed-rate LR decay (no model changes)."""
    if bool(getattr(opt, "field_mvstruct_refinement", 0)):
        from helper_mv_refinement import refinement_schedule
        return refinement_schedule(opt)
    prune = bool(getattr(opt, "field_late_prune", 0))
    decay = bool(getattr(opt, "field_tail_lr_decay", 0))
    if not prune and not decay:
        return None

    def integer(name, minimum=1):
        value = getattr(opt, name)
        try:
            number = int(value)
            valid = not isinstance(value, bool) and number >= minimum and float(value) == number
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("{} must be an integer >= {}".format(name, minimum))
        return number

    schedule = {"prune": None, "lr": None}
    if prune:
        start = integer("field_late_prune_start")
        until = integer("field_late_prune_until")
        interval = integer("field_late_prune_interval")
        threshold = float(opt.opthr)
        if start >= until or (until - start) % interval:
            raise ValueError("Late pruning must end on an event after its start")
        if start < int(opt.densify_until_iter):
            raise ValueError("Independent late pruning must start after ordinary control")
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Late pruning requires 0 < opthr < 1")
        schedule["prune"] = dict(start=start, until=until, interval=interval,
                                 min_birth_age=integer("field_late_prune_min_birth_age"),
                                 threshold=threshold)
    if decay:
        start = integer("field_tail_lr_start")
        until = integer("field_tail_lr_until")
        factor = float(opt.field_tail_lr_final_factor)
        if start >= until or not math.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError("Invalid tail learning-rate decay interval/factor")
        schedule["lr"] = dict(start=start, until=until, final_factor=factor)
    if prune and decay and schedule["prune"]["until"] >= schedule["lr"]["until"]:
        raise ValueError("Tail LR decay must leave a fixed-topology interval after pruning")
    # --iterations 10 may intentionally truncate the otherwise complete clocks.
    return schedule


def validate_long_sequence_refinement(schedule, opt, model, densify):
    """Fail before loading images if another mechanism could defeat the tail."""
    if schedule is None:
        return
    if schedule.get("unlimited", False):
        from helper_mv_refinement import validate_refinement
        validate_refinement(schedule, opt, model, densify)
        return
    if (densify != 2 or model.field_motion_model != "couptest_polynomial"
            or model.field_couptest_mode != "coupled_detached"
            or model.field_existence_single_expert != "transient"
            or not model.field_disable_global_omega_split
            or not model.field_disable_legacy_aux or not model.field_disable_ems_main
            or model.field_soft_geometry_lr or model.field_appearance_only_train):
        raise ValueError("Long refinement requires the ordinary detached-polynomial densify=2 model")
    if schedule["prune"] is None:
        return
    stop = schedule["prune"]["until"]
    narrow = model._narrow_birth_config
    if narrow.get("field_narrow_birth", 0):
        if narrow["field_narrow_birth_cleanup_until"] > stop:
            raise ValueError("Narrow-child cleanup cannot extend into the fixed-topology tail")
        if schedule["prune"]["min_birth_age"] < narrow["field_narrow_birth_cleanup_age"]:
            raise ValueError("Late pruning must preserve the narrow-child maturation period")
        if narrow.get("field_narrow_birth_growth", 0):
            if narrow['field_narrow_birth_growth_until'] > stop:
                raise ValueError('Narrow spatial growth extends into the fixed-topology tail')
            if narrow['field_narrow_birth_start'] < int(opt.densify_until_iter):
                raise ValueError('Narrow spatial growth seeds must start after ordinary control/reset')
    if int(opt.densify_until_iter) > stop:
        raise ValueError("Ordinary control extends into the fixed-topology tail")
    if model.field_mvstruct and model.field_mvstruct_densify and model.field_mvstruct_densify_until > stop:
        raise ValueError("Multiview growth extends into the fixed-topology tail")
    if model.field_bg_dense_add and model.field_bg_dense_add_iter > stop:
        raise ValueError("Background addition extends into the fixed-topology tail")
    for name in ("field_bg_prior", "field_bg_prior_clone_split", "field_bg_prior_mature_prune",
                 "field_temporal_refine", "field_carrier_initialization", "field_existence_moe"):
        if bool(getattr(model, name, False)):
            raise ValueError("Long refinement does not support the topology mechanism " + name)


def apply_refinement_learning_rate(schedule, model, iteration):
    """Decay fixed-rate groups from immutable base LRs, never cumulatively.

    XYZ retains its own exponential schedule; frozen/zero-rate groups stay zero.
    """
    if schedule is None or schedule["lr"] is None:
        return 1.0
    if schedule.get("unlimited", False):
        from helper_mv_refinement import apply_learning_rate
        return apply_learning_rate(schedule, model, iteration)
    clock = schedule["lr"]
    progress = min(max((iteration - clock["start"]) / (clock["until"] - clock["start"]), 0.), 1.)
    factor = math.exp(math.log(clock["final_factor"]) * progress)
    for group in model.optimizer.param_groups:
        if group["name"] != "xyz":
            if "_stegf_base_lr" not in group:
                raise RuntimeError("Tail LR decay requires an immutable optimizer base learning rate")
            group["lr"] = group["_stegf_base_lr"] * factor
    return factor


@torch.no_grad()
def prune_late_low_opacity(schedule, model, iteration):
    """Clean all mature low-base-opacity points, independently of frame visibility."""
    if schedule is None or schedule["prune"] is None:
        return None
    if schedule.get("unlimited", False):
        from helper_mv_refinement import prune_mature_points
        return prune_mature_points(schedule, model, iteration)
    clock = schedule["prune"]
    if not (clock["start"] <= iteration <= clock["until"]
            and (iteration - clock["start"]) % clock["interval"] == 0):
        return None
    count = len(model.get_xyz)
    opacity = model.get_opacity.detach().reshape(-1)
    if opacity.shape != (count,):
        raise RuntimeError("Late pruning opacity/topology mismatch")
    mask = torch.isfinite(opacity) & (opacity < clock["threshold"])
    candidates = int(mask.sum().item())
    narrow = getattr(model, "_narrow_birth_config", {})
    state = getattr(model, "_narrow_birth_state", None)
    young_count = 0
    if narrow.get("field_narrow_birth", 0):
        if state is None:
            raise RuntimeError("Late pruning requires initialized narrow-birth identities")
        state.check(count)
        young = (state.born >= 0) & (iteration - state.born < clock["min_birth_age"])
        young_count = int((mask & young).sum().item())
        mask = mask & ~young
    before_background = int(mask.sum().item())
    if hasattr(model, "protect_background_candidates_from_prune"):
        mask = model.protect_background_candidates_from_prune(mask, iteration)
    removed = int(mask.sum().item())
    if removed and removed == count:
        raise RuntimeError("Late pruning would remove the entire Gaussian scene")
    if removed:
        model.prune_points(mask)
    after = len(model.get_xyz)
    if after != count - removed:
        raise RuntimeError("Late pruning produced an unexpected Gaussian count")
    return dict(iteration=int(iteration), threshold=clock["threshold"],
                points_before=count, points_after=after, low_opacity_candidates=candidates,
                young_narrow_protected=young_count,
                background_protected=before_background - removed, pruned=removed)


def check_refinement_fixed_topology(schedule, model, iteration):
    """Count plus monotonic birth ID catches even equal-count replacement."""
    if schedule is not None and schedule.get("unlimited", False):
        from helper_mv_refinement import check_fixed_topology
        return check_fixed_topology(schedule, model, iteration)
    if schedule is None or schedule["prune"] is None or iteration <= schedule["prune"]["until"]:
        return False
    state = getattr(model, "_narrow_birth_state", None)
    signature = (len(model.get_xyz), None if state is None else state.next_id)
    previous = getattr(model, "_refinement_fixed_topology_signature", None)
    if previous is None:
        model._refinement_fixed_topology_signature = signature
        return True
    if previous != signature:
        raise RuntimeError("Gaussian topology changed inside the fixed-topology refinement tail")
    return False


def controlgaussians(opt, gaussians, densify, iteration, scene,  visibility_filter, radii, viewspace_point_tensor, flag, traincamerawithdistance=None, maxbounds=None, minbounds=None): 
    if (bool(getattr(opt, 'field_mvstruct_refinement', 0))
            and iteration > int(opt.field_late_prune_until)):
        return flag  # Also stop legacy spatial pruning in the fixed-topology tail.
    narrow = getattr(gaussians, '_narrow_birth_config', {})
    if narrow.get('field_narrow_birth', 0) and iteration > narrow['field_narrow_birth_cleanup_until']:
        return flag  # Explicit fixed-topology tail, only for the opt-in experiment.
    disable_global_omega_split = bool(getattr(gaussians, "field_disable_global_omega_split", 0))
    if densify == 1: # n3d
        if bool(getattr(opt, "field_n3d_quality_schedule", 0)):
            return controlgaussians_n3d_quality(opt, gaussians, iteration, scene, flag, maxbounds, minbounds)
        clock = n3d_control_schedule(opt)
        if iteration < opt.densify_until_iter :
            if (not disable_global_omega_split) and iteration == clock["omega_freeze_iter"]:
                omegamask = gaussians.zero_omegabymotion() # 1 we keep omega, 0 we freeze omega
                gaussians.omegamask  = omegamask
                scene.recordpoints(iteration, "seperate omega"+str(torch.sum(omegamask).item()))
            elif (not disable_global_omega_split) and iteration > clock["omega_freeze_iter"]:
                freezweightsbymasknounsqueeze(gaussians, ["_omega"], gaussians.omegamask)
                rotationmask = torch.logical_not(gaussians.omegamask)
                freezweightsbymasknounsqueeze(gaussians, ["_rotation"], rotationmask)
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if flag < opt.desicnt:
                    scene.recordpoints(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_pruneclone(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
                    flag+=1
                    scene.recordpoints(iteration, "after densify")
                else:
                    if iteration < clock["low_opacity_prune_until"]:
                        prune_mask =  (gaussians.get_opacity < opt.opthr).squeeze()
                        if hasattr(gaussians, "protect_background_candidates_from_prune"):
                            prune_mask = gaussians.protect_background_candidates_from_prune(prune_mask, iteration)
                        gaussians.prune_points(prune_mask)
                        torch.cuda.empty_cache()
                        scene.recordpoints(iteration, "addionally prune_mask")
            if iteration % clock["opacity_reset_interval"] == 0:
                gaussians.reset_opacity()
        else:
            if (not disable_global_omega_split) and gaussians.omegamask is not None:
                freezweightsbymasknounsqueeze(gaussians, ["_omega"], gaussians.omegamask)
                rotationmask = torch.logical_not(gaussians.omegamask)
                freezweightsbymasknounsqueeze(gaussians, ["_rotation"], rotationmask) #uncomment freezeweight... for fast traning speed.
            if iteration % clock["spatial_prune_interval"] == clock["spatial_prune_offset"]:
                zmask = gaussians._xyz[:,2] < 4.5  # 
                gaussians.prune_points(zmask) 
                torch.cuda.empty_cache()
            if iteration == clock["bounds_prune_iter"]:
                removeminmax(gaussians, maxbounds, minbounds)
        return flag
    
    elif densify == 2: # n3d 
        if iteration < opt.densify_until_iter :
            if (not disable_global_omega_split) and iteration ==  8001 : # 8001
                omegamask = gaussians.zero_omegabymotion() #
                gaussians.omegamask  = omegamask
                scene.recordpoints(iteration, "seperate omega"+str(torch.sum(omegamask).item()))
            elif (not disable_global_omega_split) and iteration > 8001: # 8001
                freezweightsbymasknounsqueeze(gaussians, ["_omega"], gaussians.omegamask)
                rotationmask = torch.logical_not(gaussians.omegamask)
                freezweightsbymasknounsqueeze(gaussians, ["_rotation"], rotationmask)
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if flag < opt.desicnt:
                    scene.recordpoints(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_pruneclone(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
                    flag+=1
                    scene.recordpoints(iteration, "after densify")
                else:
                    prune_mask =  (gaussians.get_opacity < opt.opthr).squeeze()
                    if hasattr(gaussians, "protect_background_candidates_from_prune"):
                        prune_mask = gaussians.protect_background_candidates_from_prune(prune_mask, iteration)
                    gaussians.prune_points(prune_mask)
                    torch.cuda.empty_cache()
                    scene.recordpoints(iteration, "addionally prune_mask")
            if (not bool(getattr(opt, "field_disable_legacy_opacity_reset", 0))
                    and iteration % 3000 == 0):
                gaussians.reset_opacity()
        else:
            pass 
        return flag
    

    elif densify == 3: # techni
        if iteration < opt.densify_until_iter :
            if not bool(getattr(gaussians, "field_temporal_visibility_densify", False)):
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if flag < opt.desicnt:
                    scene.recordpoints(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_pruneclone(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
                    flag+=1
                    scene.recordpoints(iteration, "after densify")
                else:
                    if iteration < 7000 : # defalt 7000. 
                        prune_mask =  (gaussians.get_opacity < opt.opthr).squeeze()
                        if hasattr(gaussians, "protect_background_candidates_from_prune"):
                            prune_mask = gaussians.protect_background_candidates_from_prune(prune_mask, iteration)
                        gaussians.prune_points(prune_mask)
                        torch.cuda.empty_cache()
                        scene.recordpoints(iteration, "addionally prune_mask")
            if iteration % opt.opacity_reset_interval == 0 :
                gaussians.reset_opacity()
        else:
            if iteration == 10000: 
                removeminmax(gaussians, maxbounds, minbounds)
        return flag

    elif densify in (4, 6, 7, 8): # Google Immersive
        if iteration < opt.densify_until_iter:
            if densify == 4:
                if (not disable_global_omega_split) and iteration == 8001:
                    omegamask = gaussians.zero_omegabymotion()
                    gaussians.omegamask = omegamask
                    scene.recordpoints(
                        iteration,
                        "seperate omega" + str(torch.sum(omegamask).item()),
                    )
                elif (
                    (not disable_global_omega_split)
                    and iteration > 8001
                    and gaussians.omegamask is not None
                ):
                    freezweightsbymasknounsqueeze(
                        gaussians, ["_omega"], gaussians.omegamask
                    )
                    rotationmask = torch.logical_not(gaussians.omegamask)
                    freezweightsbymasknounsqueeze(
                        gaussians, ["_rotation"], rotationmask
                    )

            densify_due = (
                iteration > opt.densify_from_iter
                and iteration % opt.densification_interval == 0
            )
            if densify_due:
                if flag < opt.desicnt:
                    scene.recordpoints(iteration, "before densify")
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    if densify == 4:
                        gaussians.densify_prunecloneim(
                            opt.densify_grad_threshold,
                            opt.opthr,
                            scene.cameras_extent,
                            size_threshold,
                        )
                    else:
                        gaussians.densify_prunecloneimgeneral(
                            opt.densify_grad_threshold,
                            opt.opthr,
                            scene.cameras_extent,
                            size_threshold,
                        )
                    flag += 1
                    scene.recordpoints(iteration, "after densify")
                else:
                    should_prune = (
                        (densify == 4 and iteration < 5000)
                        or (densify == 6 and iteration < 9000)
                        or densify in (7, 8)
                    )
                    reset_interval = max(int(opt.opacity_reset_interval), 1)
                    last_reset = (int(iteration) // reset_interval) * reset_interval
                    if densify == 4:
                        should_prune = should_prune and (
                            iteration > 500 + last_reset and last_reset > 1000
                        )
                    if should_prune:
                        prune_mask = (
                            gaussians.get_opacity < opt.opthr
                        ).squeeze()
                        if opt.prunebysize:
                            big_points_vs = gaussians.max_radii2D > 20
                            big_points_ws = (
                                gaussians.get_scaling.max(dim=1).values
                                > 0.1 * scene.cameras_extent
                            )
                            prune_mask = torch.logical_or(
                                torch.logical_or(
                                    prune_mask, big_points_vs
                                ),
                                big_points_ws,
                            )
                        if hasattr(
                            gaussians,
                            "protect_background_candidates_from_prune",
                        ):
                            prune_mask = (
                                gaussians.protect_background_candidates_from_prune(
                                    prune_mask, iteration
                                )
                            )
                        gaussians.prune_points(prune_mask)
                        torch.cuda.empty_cache()
                        scene.recordpoints(iteration, "addionally prune_mask")

            reset_due = iteration % opt.opacity_reset_interval == 0
            if reset_due and (densify != 4 or iteration < 4000):
                gaussians.reset_opacity()

        if densify == 8 and iteration == 10000:
            removeminmax(gaussians, maxbounds, minbounds)
            scene.recordpoints(iteration, "remove outside initialization bounds")
        return flag

    raise NotImplementedError(
        "Densification mode {} is not included in the STEGF runtime".format(
            densify
        )
    )
    

def logicalorlist(listoftensor):
    mask = None 
    for idx, ele in enumerate(listoftensor):
        if idx == 0 :
            mask = ele 
        else:
            mask = torch.logical_or(mask, ele)
    return mask 



def recordpointshelper(model_path, numpoints, iteration, string):
    txtpath = os.path.join(model_path, "exp_log.txt")
    
    with open(txtpath, 'a') as file:
        file.write("iteration at "+ str(iteration) + "\n")
        file.write(string + " pointsnumber " + str(numpoints) + "\n")




def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0




def reloadhelper(gaussians, opt, maxx, maxy, maxz,  minx, miny, minz):
    givenpath = opt.prevpath
    if opt.loadall == 0:
        gaussians.load_plyandminmax(givenpath, maxx, maxy, maxz,  minx, miny, minz)
    elif opt.loadall == 1 :
        gaussians.load_plyandminmaxall(givenpath, maxx, maxy, maxz,  minx, miny, minz)
    elif opt.loadall == 2 :        
        gaussians.load_ply(givenpath)
    elif opt.loadall == 3:
        gaussians.load_plyandminmaxY(givenpath, maxx, maxy, maxz,  minx, miny, minz)

    gaussians.max_radii2D =  torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    return 

def getfisheyemapper(folder, cameraname):
    parentfolder = os.path.dirname(folder)
    distoritonflowpath = os.path.join(parentfolder, cameraname + ".npy")
    distoritonflow = np.load(distoritonflowpath)
    distoritonflow = torch.from_numpy(distoritonflow).unsqueeze(0).float().cuda()
    return distoritonflow











def undistortimage(imagename, datasetpath,data):


    video = os.path.dirname(datasetpath) # upper folder 
    with open(os.path.join(video + "/models.json"), "r") as f:
                meta = json.load(f)

    for idx , camera in enumerate(meta):
        folder = camera['name'] # camera_0001
        view = camera
        intrinsics = np.array([[view['focal_length'], 0.0, view['principal_point'][0]],
                            [0.0, view['focal_length'], view['principal_point'][1]],
                            [0.0, 0.0, 1.0]])
        dis_cef = np.zeros((4))

        dis_cef[:2] = np.array(view['radial_distortion'])[:2]
        if folder != imagename:
             continue
        print("done one camera")
        map1, map2 = None, None
        sequencename = os.path.basename(video)
        focalscale = SCALEDICT[sequencename]
 
        h, w = data.shape[:2]


        image_size = (w, h)
        knew = np.zeros((3, 3), dtype=np.float32)


        knew[0,0] = focalscale * intrinsics[0,0]
        knew[1,1] = focalscale * intrinsics[1,1]
        knew[0,2] =  view['principal_point'][0] # cx fixed half of the width
        knew[1,2] =  view['principal_point'][1] #
        knew[2,2] =  1.0


        
        
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(intrinsics, dis_cef, R=None, P=knew, size=(w, h), m1type=cv2.CV_32FC1)

        undistorted_image = cv2.remap(data, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        undistorted_image = undistorted_image.clip(0,255.0) 
        return undistorted_image


def trbfunction(x): 
    return torch.exp(-1*x.pow(2))

def setgtisint8(value):
    print("1 means set current resized gt image as int8 for memory 0 for not : ", value)
    os.environ['gtisint8'] = str(value)
def getgtisint8():
    #print("get current gt", bool(int(os.getenv('gtisint8'))))
    try:
        return bool(int(os.getenv('gtisint8')))
    except:
        return False 
