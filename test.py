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
# ========================================================================================================
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the thirdparty/gaussian_splatting/LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys 
sys.path.append("./thirdparty/gaussian_splatting")

import torch
from thirdparty.gaussian_splatting.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import torchvision
import time 
import scipy
import numpy as np 
import warnings
import json 
from PIL import Image

from thirdparty.gaussian_splatting.lpipsPyTorch import lpips
from helper_train import getrenderpip, getmodel, trbfunction
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.helper3dg import gettestparse
from skimage.metrics import structural_similarity as sk_ssim
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams

warnings.filterwarnings("ignore")


def compute_skimage_ssim(rendernumpy, gtnumpy):
    try:
        return sk_ssim(rendernumpy, gtnumpy, channel_axis=-1, data_range=1.0)
    except TypeError:
        return sk_ssim(rendernumpy, gtnumpy, multichannel=True, data_range=1.0)


class PhotometricFitAccumulator:
    def __init__(self, mode="rgb_affine", reg=1e-6):
        self.mode = str(mode).lower()
        self.reg = float(reg)
        self.count = torch.zeros((3,), dtype=torch.float64)
        self.sum_x = torch.zeros((3,), dtype=torch.float64)
        self.sum_y = torch.zeros((3,), dtype=torch.float64)
        self.sum_x2 = torch.zeros((3,), dtype=torch.float64)
        self.sum_xy = torch.zeros((3,), dtype=torch.float64)

    def update(self, rendering, gt):
        x = rendering.detach().float().clamp(0.0, 1.0).reshape(3, -1).double()
        y = gt.detach().float().clamp(0.0, 1.0).reshape(3, -1).double()
        self.count += x.shape[1]
        self.sum_x += x.sum(dim=1).cpu()
        self.sum_y += y.sum(dim=1).cpu()
        self.sum_x2 += (x * x).sum(dim=1).cpu()
        self.sum_xy += (x * y).sum(dim=1).cpu()

    def solve(self):
        count = torch.clamp(self.count, min=1.0)
        mode = self.mode
        if mode == "rgb_scale":
            scale = self.sum_xy / torch.clamp(self.sum_x2 + self.reg, min=1e-12)
            bias = torch.zeros_like(scale)
        elif mode == "scalar_affine":
            n = count.sum()
            sx = self.sum_x.sum()
            sy = self.sum_y.sum()
            sx2 = self.sum_x2.sum()
            sxy = self.sum_xy.sum()
            denom = sx2 - sx * sx / torch.clamp(n, min=1.0) + self.reg
            scale_value = (sxy - sx * sy / torch.clamp(n, min=1.0)) / torch.clamp(denom, min=1e-12)
            bias_value = sy / torch.clamp(n, min=1.0) - scale_value * sx / torch.clamp(n, min=1.0)
            scale = scale_value.repeat(3)
            bias = bias_value.repeat(3)
        else:
            mode = "rgb_affine"
            denom = self.sum_x2 - self.sum_x * self.sum_x / count + self.reg
            scale = (self.sum_xy - self.sum_x * self.sum_y / count) / torch.clamp(denom, min=1e-12)
            bias = self.sum_y / count - scale * self.sum_x / count
        scale = torch.nan_to_num(scale.float(), nan=1.0, posinf=1.0, neginf=1.0)
        bias = torch.nan_to_num(bias.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return {"mode": mode, "scale": scale, "bias": bias}


def load_rgb_tensor(path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def apply_photometric_fit(image, fit_params, clamp=True):
    scale = fit_params["scale"].to(device=image.device, dtype=image.dtype).view(3, 1, 1)
    bias = fit_params["bias"].to(device=image.device, dtype=image.dtype).view(3, 1, 1)
    image = scale * image + bias
    if clamp:
        image = image.clamp(0.0, 1.0)
    return image


def evaluate_photometric_fit(
    model_path,
    name,
    iteration,
    image_names,
    fit_params,
    clamp=True,
    save_images=True,
):
    base_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    render_path = os.path.join(base_path, "renders")
    gts_path = os.path.join(base_path, "gt")
    fit_render_path = os.path.join(base_path, "renders_photometric_fit")
    if save_images:
        makedirs(fit_render_path, exist_ok=True)

    psnrs = []
    lpipss = []
    lpipssvggs = []
    ssims = []
    ssimsv2 = []
    per_view_dict = {model_path: {iteration: {}}}
    full_dict = {model_path: {iteration: {}}}

    for image_name in tqdm(image_names, desc="Photometric fit metric progress"):
        rendering = load_rgb_tensor(os.path.join(render_path, image_name)).cuda().contiguous()
        gt = load_rgb_tensor(os.path.join(gts_path, image_name)).cuda().contiguous()
        fitted = apply_photometric_fit(rendering, fit_params, clamp=clamp).contiguous()

        ssims.append(ssim(fitted.unsqueeze(0), gt.unsqueeze(0)))
        psnrs.append(psnr(fitted.unsqueeze(0), gt.unsqueeze(0)))
        lpipss.append(lpips(fitted.unsqueeze(0), gt.unsqueeze(0), net_type='alex'))
        lpipssvggs.append(lpips(fitted.unsqueeze(0), gt.unsqueeze(0), net_type='vgg'))

        rendernumpy = fitted.permute(1, 2, 0).detach().cpu().numpy()
        gtnumpy = gt.permute(1, 2, 0).detach().cpu().numpy()
        ssimsv2.append(compute_skimage_ssim(rendernumpy, gtnumpy))

        if save_images:
            torchvision.utils.save_image(fitted, os.path.join(fit_render_path, image_name))

    per_view_dict[model_path][iteration].update({
        "SSIM": {name: value for value, name in zip(torch.tensor(ssims).tolist(), image_names)},
        "PSNR": {name: value for value, name in zip(torch.tensor(psnrs).tolist(), image_names)},
        "LPIPS": {name: value for value, name in zip(torch.tensor(lpipss).tolist(), image_names)},
        "ssimsv2": {name: value for value, name in zip(torch.tensor(ssimsv2).tolist(), image_names)},
        "LPIPSVGG": {name: value for value, name in zip(torch.tensor(lpipssvggs).tolist(), image_names)},
    })
    full_dict[model_path][iteration].update({
        "SSIM": torch.tensor(ssims).mean().item(),
        "PSNR": torch.tensor(psnrs).mean().item(),
        "LPIPS": torch.tensor(lpipss).mean().item(),
        "ssimsv2": torch.tensor(ssimsv2).mean().item(),
        "LPIPSVGG": torch.tensor(lpipssvggs).mean().item(),
        "fit_mode": fit_params["mode"],
        "fit_scale": fit_params["scale"].tolist(),
        "fit_bias": fit_params["bias"].tolist(),
        "fit_clamp": int(bool(clamp)),
    })

    with open(os.path.join(model_path, str(iteration) + "_runtimeresults_photometric_fit.json"), 'w') as fp:
        json.dump(full_dict, fp, indent=True)
    with open(os.path.join(model_path, str(iteration) + "_runtimeperview_photometric_fit.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)
    with open(os.path.join(base_path, "photometric_fit.json"), 'w') as fp:
        json.dump({
            "mode": fit_params["mode"],
            "scale": fit_params["scale"].tolist(),
            "bias": fit_params["bias"].tolist(),
            "clamp": int(bool(clamp)),
            "save_images": int(bool(save_images)),
        }, fp, indent=True)

# modified from https://github.com/graphdeco-inria/gaussian-splatting/blob/main/render.py and https://github.com/graphdeco-inria/gaussian-splatting/blob/main/metrics.py
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, rbfbasefunction, rdpip, timing_repeats=0, photometric_fit=False, photometric_fit_mode="rgb_affine", photometric_fit_reg=1e-6, photometric_fit_clamp=True, photometric_fit_save_images=True):
    render, GRsetting, GRzer = getrenderpip(rdpip) 
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if gaussians.rgbdecoder is not None:
        gaussians.rgbdecoder.cuda()
        gaussians.rgbdecoder.eval()
    if getattr(gaussians, "content_exposure_head", None) is not None:
        gaussians.content_exposure_head.cuda()
        gaussians.content_exposure_head.eval()
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
    statsdict = {}

    scales = gaussians.get_scaling

    scalemax = torch.amax(scales).item()
    scalesmean = torch.amin(scales).item()
     
    op = gaussians.get_opacity
    opmax = torch.amax(op).item()
    opmean = torch.mean(op).item()

    statsdict["scales_max"] = scalemax
    statsdict["scales_mean"] = scalesmean

    statsdict["op_max"] = opmax
    statsdict["op_mean"] = opmean 


    statspath = os.path.join(model_path, "stat_" + str(iteration) + ".json")
    with open(statspath, 'w') as fp:
            json.dump(statsdict, fp, indent=True)


    psnrs = []
    lpipss = []
    lpipssvggs = []

    full_dict = {}
    per_view_dict = {}
    ssims = []
    ssimsv2 = []
    scene_dir = model_path
    image_names = []
    times = []
    fit_accumulator = PhotometricFitAccumulator(
        mode=photometric_fit_mode,
        reg=photometric_fit_reg,
    ) if photometric_fit else None

    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}

  

    full_dict[scene_dir][iteration] = {}
    per_view_dict[scene_dir][iteration] = {}


    if rdpip == "train_ours_full":
        render, GRsetting, GRzer = getrenderpip("test_ours_full")
    else:
        render, GRsetting, GRzer = getrenderpip(rdpip) 


    for idx, view in enumerate(tqdm(views, desc="Rendering and metric progress")):
        renderingpkg = render(view, gaussians, pipeline, background, scaling_modifier=1.0, basicfunction=rbfbasefunction,  GRsetting=GRsetting, GRzer=GRzer) # C x H x W
        rendering = renderingpkg["render"]
        duration = renderingpkg.get("duration", None)
        if duration is not None and idx > 10:
            times.append(float(duration))
        gt = view.original_image[0:3, :, :].cuda().float()
        if hasattr(gaussians, "apply_content_exposure"):
            rendering = gaussians.apply_content_exposure(rendering)
        rendering = torch.clamp(rendering, 0, 1.0)
        if fit_accumulator is not None:
            fit_accumulator.update(rendering, gt)
        ssims.append(ssim(rendering.unsqueeze(0),gt.unsqueeze(0))) 

        psnrs.append(psnr(rendering.unsqueeze(0), gt.unsqueeze(0)))
        lpipss.append(lpips(rendering.unsqueeze(0), gt.unsqueeze(0), net_type='alex')) #
        lpipssvggs.append( lpips(rendering.unsqueeze(0), gt.unsqueeze(0), net_type='vgg'))

        rendernumpy = rendering.permute(1,2,0).detach().cpu().numpy()
        gtnumpy = gt.permute(1,2,0).detach().cpu().numpy()
        
        ssimv2 = compute_skimage_ssim(rendernumpy, gtnumpy)
        ssimsv2.append(ssimv2)


        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        image_names.append('{0:05d}'.format(idx) + ".png")

    

    timing_repeats = max(int(timing_repeats), 0)
    if timing_repeats > 0:
        times = []
        for idx, view in enumerate(tqdm(views, desc="release gt images cuda memory for timing")):
            view.original_image = None #.detach()
            torch.cuda.empty_cache()

        # Optional dedicated timing pass. Disabled by default because metrics already render every view once.
        for _ in range(timing_repeats):
            for idx, view in enumerate(tqdm(views, desc="timing ")):
                renderpack = render(view, gaussians, pipeline, background, scaling_modifier=1.0, basicfunction=rbfbasefunction,  GRsetting=GRsetting, GRzer=GRzer)#["time"] # C x H x W
                duration = renderpack["duration"]
                if idx > 10: #warm up
                    times.append(float(duration))

    if len(times) == 0:
        times = [0.0]
    print(f"[STEGF] Mean render time: {float(np.mean(np.array(times))):.6f}s")
    if len(views) > 0:
        full_dict[model_path][iteration].update({"SSIM": torch.tensor(ssims).mean().item(),
                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                        "ssimsv2": torch.tensor(ssimsv2).mean().item(),
                                        "LPIPSVGG": torch.tensor(lpipssvggs).mean().item(),
                                        "times": torch.tensor(times).mean().item()})
        
        per_view_dict[model_path][iteration].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                                "ssimsv2": {name: v for v, name in zip(torch.tensor(ssimsv2).tolist(), image_names)},
                                                                "LPIPSVGG": {name: lpipssvgg for lpipssvgg, name in zip(torch.tensor(lpipssvggs).tolist(), image_names)},})
        
            
            
        with open(model_path + "/" + str(iteration) + "_runtimeresults.json", 'w') as fp:
            json.dump(full_dict, fp, indent=True)

        with open(model_path + "/" + str(iteration) + "_runtimeperview.json", 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

        if fit_accumulator is not None:
            fit_params = fit_accumulator.solve()
            print(
                "[STEGF] Photometric fit: mode={}, scale={}, bias={}".format(
                    fit_params["mode"],
                    [round(float(v), 6) for v in fit_params["scale"]],
                    [round(float(v), 6) for v in fit_params["bias"]],
                )
            )
            evaluate_photometric_fit(
                model_path,
                name,
                iteration,
                image_names,
                fit_params,
                clamp=photometric_fit_clamp,
                save_images=photometric_fit_save_images,
            )


# render free view
def render_setnogt(model_path, name, iteration, views, gaussians, pipeline, background, rbfbasefunction, rdpip):
    render, GRsetting, GRzer = getrenderpip(rdpip) 
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)
    if gaussians.rgbdecoder is not None:
        gaussians.rgbdecoder.cuda()
        gaussians.rgbdecoder.eval()
    if getattr(gaussians, "content_exposure_head", None) is not None:
        gaussians.content_exposure_head.cuda()
        gaussians.content_exposure_head.eval()
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
    if rdpip == "train_ours_full" and getattr(gaussians, "use_euler_field", False):
        render, GRsetting, GRzer = getrenderpip("test_ours_full")

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        rendering = render(view, gaussians, pipeline, background,scaling_modifier=1.0, basicfunction=rbfbasefunction,  GRsetting=GRsetting, GRzer=GRzer)["render"] # C x H x W
        if hasattr(gaussians, "apply_content_exposure"):
            rendering = gaussians.apply_content_exposure(rendering)
        rendering = torch.clamp(rendering, 0, 1.0)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


def run_test(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, multiview : bool, duration: int, rgbfunction="rgbv1", rdpip="v2", loader="colmap", timing_repeats=0):
    
    with torch.no_grad():
        print("use model {}".format(dataset.model))
        GaussianModel = getmodel(dataset.model) # default, gmodel, we are tewsting 

        gaussians = GaussianModel(dataset.sh_degree, rgbfunction)
        if hasattr(gaussians, "configure_euler_field"):
            gaussians.configure_euler_field(dataset)

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, multiview=multiview, duration=duration, loader=loader)
        rbfbasefunction = trbfunction
        numchannels = 9
        bg_color =  [0 for _ in range(numchannels)]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if gaussians.ts is None :
            cameraslit = scene.getTestCameras()
            H,W = cameraslit[0].image_height, cameraslit[0].image_width
            gaussians.ts = torch.ones(1,1,H,W).cuda()

        if not skip_test and not multiview:            
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                rbfbasefunction,
                rdpip,
                timing_repeats=timing_repeats,
                photometric_fit=bool(getattr(dataset, "test_photometric_fit", 0)),
                photometric_fit_mode=str(getattr(dataset, "test_photometric_fit_mode", "rgb_affine")),
                photometric_fit_reg=float(getattr(dataset, "test_photometric_fit_reg", 1e-6)),
                photometric_fit_clamp=bool(getattr(dataset, "test_photometric_fit_clamp", 1)),
                photometric_fit_save_images=bool(getattr(dataset, "test_photometric_fit_save_images", 1)),
            )
        if multiview:
            render_setnogt(dataset.model_path, "mv", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, rbfbasefunction, rdpip)

if __name__ == "__main__":
    

    args, model_extract, pp_extract, multiview =gettestparse()
    run_test(model_extract, args.test_iteration, pp_extract, args.skip_train, args.skip_test, multiview, args.duration,  rgbfunction=args.rgbfunction, rdpip=args.rdpip, loader=args.valloader, timing_repeats=args.timing_repeats)
