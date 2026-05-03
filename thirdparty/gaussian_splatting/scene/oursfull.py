#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, update_quaternion
from utils.graphics_utils import geom_transform_points
from helper_model import getcolormodel, interpolate_point, interpolate_partuse, interpolate_pointv3
from scene.euler_field import EulerField, EulerLevelRouter, EulerQueryFusionGate, EulerResidualDecoder
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp #speical set for visual examples 
        self.scaling_inverse_activation = torch.log #special set for vislual examples

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        self.featureact = torch.sigmoid

        


    def __init__(self, sh_degree : int, rgbfuntion="rgbv1"):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        # self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._motion = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._omega = torch.empty(0)
        self._static_level_logits = torch.empty(0)
        self._dynamic_level_logits = torch.empty(0)
        self._dynamic_level_time_coeff = torch.empty(0)
        self._static_route_logits = torch.empty(0)
        self._field_residual_gate = torch.empty(0)
        
        self.rgbdecoder = getcolormodel(rgbfuntion)
        self.euler_field = None
        self.field_router = None
        self.field_query_gate = None
        self.field_decoder = None
        self.field_temporal_opacity_head = None
        self.field_static_view_mapper = None
        self.field_static_app_head = None
    
        self.setup_functions()
        self.delta_t = None
        self.omegamask = None 
        self.maskforems = None 
        self.distancetocamera = None
        self.trbfslinit = None 
        self.ts = None 
        self.trbfoutput = None 
        self.preprocesspoints = False 
        self.addsphpointsscale = 0.8

        
        self.maxz, self.minz =  0.0 , 0.0 
        self.maxy, self.miny =  0.0 , 0.0 
        self.maxx, self.minx =  0.0 , 0.0  
        self.raystart = 0.7
        self.computedtrbfscale = None 
        self.computedopacity = None 
        self.computedscales = None 
        self.use_euler_field = False
        self.field_base_resolution = 4
        self.field_num_levels = 5
        self.field_resolution_mode = "fixed"
        self.field_level_resolutions = ""
        self.field_resolved_level_resolutions = ""
        self.field_resolution_growth = 2.0
        self.field_max_resolution = 96
        self.field_knn_scale_percentile = 25.0
        self.field_gaussian_scale_percentile = 25.0
        self.field_pixel_scale_percentile = 25.0
        self.field_knn_scale_weight = 1.0
        self.field_gaussian_scale_weight = 1.0
        self.field_pixel_scale_weight = 1.0
        self.field_min_cell_scale = 1.5
        self.field_feature_dim = 8
        self.field_fourier_degree = 10
        self.field_level_fourier_degree = 2
        self.field_decoder_hidden = 32
        self.field_residual_mode = "geometry"
        self.field_query_mode = "hybrid"
        self.field_query_detach = True
        self.field_query_gate_bias = -2.0
        self.field_query_motion_scale = 2.0
        self.field_dyn_threshold = 0.08
        self.field_fast_threshold = 0.18
        self.field_dyn_slope = 10.0
        self.field_fast_slope = 15.0
        self.field_fast_temperature = 0.25
        self.field_disable_dynamic_grid = True
        self.field_v23_compat = False
        self.field_static_route_mode = "learned"
        self.field_static_route_init = 0.0
        self.field_static_start_iter = 0
        self.field_static_warmup_iters = 3000
        self.field_static_motion_scale = 0.05
        self.field_static_opacity_scale = 0.02
        self.field_static_app_scale = 0.05
        self.field_static_use_global_gate = False
        self.field_static_prior_floor = 0.25
        self.field_soft_route_slope = 8.0
        self.field_soft_static_threshold = 0.45
        self.field_soft_dynamic_threshold = 0.25
        self.field_staged_training = True
        self.field_disable_legacy_aux = True
        self.field_disable_ems_main = True
        self.field_disable_global_omega_split = True
        self.field_warmup_iters = 9000
        self.field_problem_mining_start = 9000
        self.field_category_activate_iter = 11000
        self.field_activate_iter = 11000
        self.field_fast_activate_iter = 13000
        self.field_mask_update_interval = 500
        self.field_score_ema = 0.05
        self.field_visibility_ema = 0.05
        self.field_problem_error_weight = 0.50
        self.field_problem_temporal_weight = 0.50
        self.field_static_error_boost = 0.50
        self.field_time_center_ema = 0.05
        self.field_responsibility_on_threshold = 0.45
        self.field_responsibility_off_threshold = 0.25
        self.field_visibility_static_threshold = 0.60
        self.field_slow_motion_on_threshold = 0.35
        self.field_slow_motion_off_threshold = 0.20
        self.field_dynamic_on_threshold = 0.50
        self.field_dynamic_off_threshold = 0.30
        self.field_motion_pre_threshold = 0.20
        self.field_motion_accel_pre_threshold = 0.18
        self.field_static_pre_threshold = 0.15
        self.field_static_on_threshold = 0.70
        self.field_static_off_threshold = 0.50
        self.field_static_motion_threshold = 0.12
        self.field_static_accel_threshold = 0.12
        self.field_fast_on_threshold = 0.60
        self.field_fast_off_threshold = 0.40
        self.field_score_motion_weight = 0.35
        self.field_score_accel_weight = 0.15
        self.field_score_error_weight = 0.25
        self.field_score_screen_weight = 0.15
        self.field_score_xyz_weight = 0.10
        self.field_score_static_residual_weight = 0.15
        self.field_fast_score_motion_weight = 0.45
        self.field_fast_score_accel_weight = 0.35
        self.field_fast_score_error_weight = 0.20
        self.field_fast_score_screen_weight = 0.05
        self.field_fast_score_xyz_weight = 0.05
        self.field_fast_score_static_residual_weight = 0.25
        self.field_static_score_motion_weight = 0.45
        self.field_static_score_accel_weight = 0.35
        self.field_static_score_residual_weight = 0.20
        self.field_fast_opacity_scale = 0.75
        self.field_fast_motion_scale = 1.0
        self.field_temporal_refine = True
        self.field_temporal_refine_start = 7000
        self.field_temporal_refine_interval = 1000
        self.field_temporal_split_children = 2
        self.field_temporal_center_offset = 0.08
        self.field_temporal_scale_shrink = 0.5
        self.field_fast_child_motion_scale = 1.0
        self.field_temporal_refine_opacity_threshold = 0.2
        self.field_temporal_refine_score_threshold = 0.65
        self.field_temporal_refine_max_ratio = 0.02
        self.field_stage = "baseline_warmup"
        self.field_current_iteration = 0
        self._dynamic_score_ema = torch.empty(0)
        self._dynamic_active_mask = torch.empty(0)
        self._responsibility_ema = torch.empty(0)
        self._responsibility_time_center_ema = torch.empty(0)
        self._slow_motion_score_ema = torch.empty(0)
        self._slow_motion_mask = torch.empty(0)
        self._fast_score_ema = torch.empty(0)
        self._fast_active_mask = torch.empty(0)
        self._static_support_ema = torch.empty(0)
        self._static_support_mask = torch.empty(0)
        self._visibility_persistence_ema = torch.empty(0)
        self._last_field_aux = {}
        self._field_camera_scale_hints = []
        self._field_resolution_stats = {}
        self.field_grd = {}
        self.field_router_grd = {}
        self.field_query_gate_grd = {}
        self.field_decoder_grd = {}
        self.field_temporal_opacity_head_grd = {}
        self.field_static_view_mapper_grd = {}
        self.field_static_app_head_grd = {}

    def _init_ems_mask(self, num_points, values=None):
        if values is None:
            values = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        else:
            values = values.to(device="cuda", dtype=torch.float32)
        self.maskforems = values

    def _init_dynamic_score_state(
        self,
        num_points,
        score_values=None,
        active_values=None,
        responsibility_values=None,
        responsibility_time_values=None,
        slow_score_values=None,
        slow_mask_values=None,
        fast_score_values=None,
        fast_active_values=None,
        static_score_values=None,
        static_mask_values=None,
        visibility_values=None,
    ):
        if score_values is None:
            score_values = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        else:
            score_values = score_values.to(device="cuda", dtype=torch.float32)
        if active_values is None:
            active_values = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        else:
            active_values = active_values.to(device="cuda", dtype=torch.float32)
        if fast_score_values is None:
            fast_score_values = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        else:
            fast_score_values = fast_score_values.to(device="cuda", dtype=torch.float32)
        if fast_active_values is None:
            fast_active_values = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        else:
            fast_active_values = fast_active_values.to(device="cuda", dtype=torch.float32)
        if responsibility_values is None:
            responsibility_values = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        else:
            responsibility_values = responsibility_values.to(device="cuda", dtype=torch.float32)
        if responsibility_time_values is None:
            responsibility_time_values = torch.full((num_points, 1), -1.0, device="cuda", dtype=torch.float32)
        else:
            responsibility_time_values = responsibility_time_values.to(device="cuda", dtype=torch.float32)
        if slow_score_values is None:
            slow_score_values = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        else:
            slow_score_values = slow_score_values.to(device="cuda", dtype=torch.float32)
        if slow_mask_values is None:
            slow_mask_values = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        else:
            slow_mask_values = slow_mask_values.to(device="cuda", dtype=torch.float32)
        if static_score_values is None:
            static_score_values = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        else:
            static_score_values = static_score_values.to(device="cuda", dtype=torch.float32)
        if static_mask_values is None:
            static_mask_values = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        else:
            static_mask_values = static_mask_values.to(device="cuda", dtype=torch.float32)
        if visibility_values is None:
            visibility_values = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        else:
            visibility_values = visibility_values.to(device="cuda", dtype=torch.float32)
        self._dynamic_score_ema = score_values
        self._dynamic_active_mask = active_values
        self._responsibility_ema = responsibility_values
        self._responsibility_time_center_ema = responsibility_time_values
        self._slow_motion_score_ema = slow_score_values
        self._slow_motion_mask = slow_mask_values
        self._fast_score_ema = fast_score_values
        self._fast_active_mask = fast_active_values
        self._static_support_ema = static_score_values
        self._static_support_mask = static_mask_values
        self._visibility_persistence_ema = visibility_values

    def _normalize_score_component(self, values):
        if values.numel() == 0:
            return values
        scale = torch.quantile(values.detach().reshape(-1), 0.9).clamp_min(1e-6)
        return torch.clamp(values / scale, min=0.0, max=1.0)

    def _point_state_or_zeros(self, values, num_points, device, dtype):
        if values is None or values.numel() == 0 or values.shape[0] != num_points:
            return torch.zeros((num_points, 1), device=device, dtype=dtype)
        return values.to(device=device, dtype=dtype)

    def _get_viewspace_gradient_score(self, viewspace_point_tensor, visible_mask, valid_mask, device, dtype):
        if viewspace_point_tensor is None or viewspace_point_tensor.grad is None:
            return torch.zeros((int(torch.count_nonzero(valid_mask).item()), 1), device=device, dtype=dtype)
        grad = viewspace_point_tensor.grad
        if grad is None or grad.numel() == 0:
            return torch.zeros((int(torch.count_nonzero(valid_mask).item()), 1), device=device, dtype=dtype)
        grad = grad[visible_mask]
        if grad.numel() == 0:
            return torch.zeros((int(torch.count_nonzero(valid_mask).item()), 1), device=device, dtype=dtype)
        grad = grad[valid_mask]
        if grad.numel() == 0:
            return torch.zeros((0, 1), device=device, dtype=dtype)
        screen_grad = torch.norm(grad[:, :2], dim=1, keepdim=True)
        return screen_grad.to(device=device, dtype=dtype)

    def _get_xyz_gradient_score(self, visible_indices, device, dtype):
        if self._xyz.grad is None or self._xyz.grad.numel() == 0:
            return torch.zeros((visible_indices.shape[0], 1), device=device, dtype=dtype)
        xyz_grad = self._xyz.grad.detach()
        if xyz_grad.shape[0] != self.get_xyz.shape[0]:
            return torch.zeros((visible_indices.shape[0], 1), device=device, dtype=dtype)
        xyz_grad = torch.norm(xyz_grad[visible_indices], dim=1, keepdim=True)
        return xyz_grad.to(device=device, dtype=dtype)

    def set_field_training_stage(self, iteration):
        self.field_current_iteration = int(iteration)
        if (not self.use_euler_field) or (not self.field_staged_training):
            self.field_stage = "fast_refine"
            return
        warmup_end = int(self.field_warmup_iters)
        problem_start = max(int(self.field_problem_mining_start), warmup_end)
        category_start = max(int(self.field_category_activate_iter), problem_start + 1)
        fast_start = max(int(self.field_fast_activate_iter), category_start + 1)
        if iteration <= warmup_end:
            self.field_stage = "baseline_warmup"
        elif iteration < category_start:
            self.field_stage = "problem_mining"
        elif iteration < fast_start:
            self.field_stage = "category_activation"
        else:
            self.field_stage = "fast_refine"

    def refresh_dynamic_mask(self, iteration=None, force=False):
        if (not self.use_euler_field) or (not self.field_staged_training):
            return
        if self._dynamic_active_mask.numel() == 0:
            self._init_dynamic_score_state(self.get_xyz.shape[0])
        if iteration is None:
            iteration = self.field_current_iteration
        if (not force) and iteration < self.field_category_activate_iter:
            return
        if (not force) and self.field_mask_update_interval > 0 and iteration % self.field_mask_update_interval != 0:
            return
        if iteration < self.field_category_activate_iter:
            self._dynamic_active_mask.zero_()
            self._slow_motion_mask.zero_()
            self._fast_active_mask.zero_()
            self._static_support_mask.zero_()
            return
        if self.field_v23_compat:
            dynamic_score = self._dynamic_score_ema
            fast_score = self._fast_score_ema
            dynamic_active = self._dynamic_active_mask > 0.5
            fast_active = self._fast_active_mask > 0.5

            dynamic_active = torch.where(
                dynamic_score >= self.field_dynamic_on_threshold,
                torch.ones_like(dynamic_active, dtype=torch.bool),
                dynamic_active,
            )
            dynamic_active = torch.where(
                dynamic_score <= self.field_dynamic_off_threshold,
                torch.zeros_like(dynamic_active, dtype=torch.bool),
                dynamic_active,
            )
            fast_active = torch.where(
                fast_score >= self.field_fast_on_threshold,
                torch.ones_like(fast_active, dtype=torch.bool),
                fast_active,
            )
            fast_active = torch.where(
                fast_score <= self.field_fast_off_threshold,
                torch.zeros_like(fast_active, dtype=torch.bool),
                fast_active,
            )
            if iteration < self.field_fast_activate_iter:
                fast_active = torch.zeros_like(fast_active, dtype=torch.bool)

            self._dynamic_active_mask = torch.logical_or(dynamic_active, fast_active).float()
            self._fast_active_mask = fast_active.float()
            self._slow_motion_mask.zero_()
            self._static_support_mask.zero_()
            return

        static_support = self._static_support_mask > 0.5
        responsibility = self._responsibility_ema
        visibility = self._visibility_persistence_ema
        static_score = self._static_support_ema
        static_support = torch.where(
            (static_score >= self.field_static_on_threshold)
            & (visibility >= self.field_visibility_static_threshold)
            & (responsibility <= self.field_responsibility_off_threshold),
            torch.ones_like(static_support, dtype=torch.bool),
            static_support,
        )
        static_support = torch.where(
            (static_score <= self.field_static_off_threshold)
            | (visibility < 0.5 * self.field_visibility_static_threshold)
            | (responsibility >= self.field_responsibility_on_threshold),
            torch.zeros_like(static_support, dtype=torch.bool),
            static_support,
        )

        candidate = ~static_support
        fast_active = self._fast_active_mask > 0.5
        slow_active = self._slow_motion_mask > 0.5
        fast_score = self._fast_score_ema
        slow_score = self._slow_motion_score_ema
        fast_active = torch.where(
            candidate
            & (responsibility >= self.field_responsibility_on_threshold)
            & (fast_score >= self.field_fast_on_threshold),
            torch.ones_like(fast_active, dtype=torch.bool),
            fast_active,
        )
        fast_active = torch.where(
            (~candidate)
            | (responsibility <= self.field_responsibility_off_threshold)
            | (fast_score <= self.field_fast_off_threshold),
            torch.zeros_like(fast_active, dtype=torch.bool),
            fast_active,
        )
        slow_active = torch.where(
            candidate
            & (~fast_active)
            & (responsibility >= self.field_responsibility_on_threshold)
            & (slow_score >= self.field_slow_motion_on_threshold),
            torch.ones_like(slow_active, dtype=torch.bool),
            slow_active,
        )
        slow_active = torch.where(
            (~candidate)
            | fast_active
            | (responsibility <= self.field_responsibility_off_threshold)
            | (slow_score <= self.field_slow_motion_off_threshold),
            torch.zeros_like(slow_active, dtype=torch.bool),
            slow_active,
        )

        if iteration < self.field_fast_activate_iter:
            fast_active = torch.zeros_like(fast_active, dtype=torch.bool)

        self._static_support_mask = static_support.float()
        self._slow_motion_mask = slow_active.float()
        self._fast_active_mask = fast_active.float()
        self._dynamic_active_mask = (slow_active | fast_active).float()

    def update_dynamic_scores(self, visibility_filter, image, gt_image, viewpoint_camera, means3D, viewspace_point_tensor, temporal_motion_map=None):
        if (not self.use_euler_field) or (not self.field_staged_training):
            return
        if self.field_stage == "baseline_warmup":
            return
        if self._dynamic_score_ema.numel() == 0:
            self._init_dynamic_score_state(self.get_xyz.shape[0])
        if visibility_filter is None or visibility_filter.numel() == 0:
            return
        visible = visibility_filter.bool()
        if torch.count_nonzero(visible) == 0:
            return
        motion_strength = self._last_field_aux.get("motion_strength")
        motion_acceleration = self._last_field_aux.get("motion_acceleration")
        static_residual_motion = self._last_field_aux.get("static_residual_motion")
        if motion_strength is None or motion_acceleration is None or static_residual_motion is None:
            return

        with torch.no_grad():
            if self.field_v23_compat:
                decay = 0.999
                self._dynamic_score_ema.mul_(decay)
                self._fast_score_ema.mul_(decay)
                visible_indices = torch.nonzero(visible, as_tuple=False).squeeze(1)
                projected = geom_transform_points(means3D[visible], viewpoint_camera.full_proj_transform)
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
                    return

                valid_visible_indices = visible_indices[valid]
                ndc_valid = ndc[valid]
                x = (((ndc_valid[:, 0] + 1.0) * viewpoint_camera.image_width) - 1.0) * 0.5
                y = (((ndc_valid[:, 1] + 1.0) * viewpoint_camera.image_height) - 1.0) * 0.5
                x = torch.round(x).long().clamp(0, viewpoint_camera.image_width - 1)
                y = torch.round(y).long().clamp(0, viewpoint_camera.image_height - 1)

                residual_map = torch.abs(image.detach() - gt_image.detach()).mean(dim=0)
                error_score = residual_map[y, x].unsqueeze(1)
                screen_score = self._get_viewspace_gradient_score(
                    viewspace_point_tensor,
                    visible,
                    valid,
                    device=error_score.device,
                    dtype=error_score.dtype,
                )
                xyz_score = self._get_xyz_gradient_score(
                    valid_visible_indices,
                    device=error_score.device,
                    dtype=error_score.dtype,
                )
                motion_score = motion_strength[visible][valid]
                static_score = static_residual_motion[visible][valid]

                norm_motion = self._normalize_score_component(motion_score)
                norm_error = self._normalize_score_component(error_score)
                norm_screen = self._normalize_score_component(screen_score)
                norm_xyz = self._normalize_score_component(xyz_score)
                norm_static = self._normalize_score_component(static_score)

                dynamic_score = (
                    self.field_score_motion_weight * norm_motion
                    + self.field_score_error_weight * norm_error
                    + self.field_score_screen_weight * norm_screen
                    + self.field_score_xyz_weight * norm_xyz
                    + self.field_score_static_residual_weight * norm_static
                )
                dynamic_score = dynamic_score / max(
                    self.field_score_motion_weight
                    + self.field_score_error_weight
                    + self.field_score_screen_weight
                    + self.field_score_xyz_weight
                    + self.field_score_static_residual_weight,
                    1e-6,
                )

                fast_score = (
                    self.field_fast_score_motion_weight * norm_motion
                    + self.field_fast_score_error_weight * norm_error
                    + self.field_fast_score_screen_weight * norm_screen
                    + self.field_fast_score_xyz_weight * norm_xyz
                    + self.field_fast_score_static_residual_weight * norm_static
                )
                fast_score = fast_score / max(
                    self.field_fast_score_motion_weight
                    + self.field_fast_score_error_weight
                    + self.field_fast_score_screen_weight
                    + self.field_fast_score_xyz_weight
                    + self.field_fast_score_static_residual_weight,
                    1e-6,
                )

                dynamic_candidate = torch.logical_or(
                    motion_score >= self.field_motion_pre_threshold,
                    static_score >= self.field_static_pre_threshold,
                ).float()
                dynamic_score = dynamic_score * dynamic_candidate
                fast_score = fast_score * dynamic_candidate

                current_dynamic = self._dynamic_score_ema[valid_visible_indices]
                current_fast = self._fast_score_ema[valid_visible_indices]
                self._dynamic_score_ema[valid_visible_indices] = (
                    (1.0 - self.field_score_ema) * current_dynamic + self.field_score_ema * dynamic_score
                )
                self._fast_score_ema[valid_visible_indices] = (
                    (1.0 - self.field_score_ema) * current_fast + self.field_score_ema * fast_score
                )
                return

            decay = 0.999
            self._dynamic_score_ema.mul_(decay)
            self._responsibility_ema.mul_(decay)
            self._slow_motion_score_ema.mul_(decay)
            self._fast_score_ema.mul_(decay)
            self._static_support_ema.mul_(decay)
            self._visibility_persistence_ema.mul_(1.0 - self.field_visibility_ema)
            visible_indices = torch.nonzero(visible, as_tuple=False).squeeze(1)
            current_visibility = self._visibility_persistence_ema[visible_indices]
            self._visibility_persistence_ema[visible_indices] = (
                current_visibility + self.field_visibility_ema
            )
            residual_map = torch.abs(image.detach() - gt_image.detach()).mean(dim=0)
            if temporal_motion_map is None:
                temporal_motion_map = torch.zeros_like(residual_map)
            projected = geom_transform_points(means3D[visible], viewpoint_camera.full_proj_transform)
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
                return

            valid_visible_indices = visible_indices[valid]
            ndc_valid = ndc[valid]
            x = (((ndc_valid[:, 0] + 1.0) * viewpoint_camera.image_width) - 1.0) * 0.5
            y = (((ndc_valid[:, 1] + 1.0) * viewpoint_camera.image_height) - 1.0) * 0.5
            x = torch.round(x).long().clamp(0, viewpoint_camera.image_width - 1)
            y = torch.round(y).long().clamp(0, viewpoint_camera.image_height - 1)

            error_score = residual_map[y, x].unsqueeze(1)
            temporal_score = temporal_motion_map[y, x].unsqueeze(1)
            opacity_score = self.get_opacity[visible][valid]
            motion_score = motion_strength[visible][valid]
            accel_score = motion_acceleration[visible][valid]
            static_score = static_residual_motion[visible][valid]
            visibility_score = self._visibility_persistence_ema[valid_visible_indices]

            norm_motion = self._normalize_score_component(motion_score)
            norm_accel = self._normalize_score_component(accel_score)
            norm_error = self._normalize_score_component(error_score)
            norm_temporal = self._normalize_score_component(temporal_score)
            norm_opacity = self._normalize_score_component(opacity_score)
            norm_static = self._normalize_score_component(static_score)
            problem_score = (
                self.field_problem_error_weight * norm_error
                + self.field_problem_temporal_weight * norm_temporal
            )
            motion_problem_score = (norm_error * norm_temporal).clamp(0.0, 1.0)
            static_problem_score = (norm_error * (1.0 - norm_temporal)).clamp(0.0, 1.0)
            responsibility_score = motion_problem_score * (0.5 + 0.5 * norm_opacity)
            dynamic_signal = torch.maximum(norm_motion, norm_static)
            fast_signal = (
                self.field_fast_score_motion_weight * norm_motion
                + self.field_fast_score_accel_weight * norm_accel
                + self.field_fast_score_static_residual_weight * norm_static
            ).clamp(0.0, 1.0)
            slow_signal = (dynamic_signal * (1.0 - fast_signal)).clamp(0.0, 1.0)
            static_support_score = (
                visibility_score
                * (1.0 - norm_temporal)
                * (
                    self.field_static_score_motion_weight * (1.0 - norm_motion)
                    + self.field_static_score_accel_weight * (1.0 - norm_accel)
                    + self.field_static_score_residual_weight * (1.0 - norm_static)
                )
            )
            static_support_score = static_support_score / max(
                self.field_static_score_motion_weight + self.field_static_score_accel_weight + self.field_static_score_residual_weight,
                1e-6,
            )
            static_support_score = torch.clamp(
                static_support_score * (1.0 + self.field_static_error_boost * static_problem_score),
                min=0.0,
                max=1.0,
            )
            current_score = self._dynamic_score_ema[valid_visible_indices]
            current_resp = self._responsibility_ema[valid_visible_indices]
            current_resp_time = self._responsibility_time_center_ema[valid_visible_indices]
            current_slow_score = self._slow_motion_score_ema[valid_visible_indices]
            current_fast_score = self._fast_score_ema[valid_visible_indices]
            current_static_score = self._static_support_ema[valid_visible_indices]
            self._dynamic_score_ema[valid_visible_indices] = (
                (1.0 - self.field_score_ema) * current_score + self.field_score_ema * responsibility_score
            )
            self._responsibility_ema[valid_visible_indices] = (
                (1.0 - self.field_score_ema) * current_resp + self.field_score_ema * responsibility_score
            )
            time_alpha = torch.clamp(self.field_time_center_ema * responsibility_score, min=0.0, max=1.0)
            timestamp_value = torch.full_like(current_resp_time, float(viewpoint_camera.timestamp))
            initialized = current_resp_time >= 0.0
            updated_time = torch.where(
                initialized,
                (1.0 - time_alpha) * current_resp_time + time_alpha * timestamp_value,
                timestamp_value,
            )
            self._responsibility_time_center_ema[valid_visible_indices] = torch.where(
                time_alpha > 1e-5,
                updated_time,
                current_resp_time,
            )
            self._slow_motion_score_ema[valid_visible_indices] = (
                (1.0 - self.field_score_ema) * current_slow_score + self.field_score_ema * slow_signal
            )
            self._fast_score_ema[valid_visible_indices] = (
                (1.0 - self.field_score_ema) * current_fast_score + self.field_score_ema * fast_signal
            )
            self._static_support_ema[valid_visible_indices] = (
                (1.0 - self.field_score_ema) * current_static_score + self.field_score_ema * static_support_score
            )

    def update_error_prior(self, visibility_filter, image, gt_image, viewpoint_camera, means3D):
        if self.field_disable_legacy_aux:
            return
        if self.maskforems is None or self.maskforems.numel() == 0:
            self._init_ems_mask(self.get_xyz.shape[0])
        if visibility_filter is None or visibility_filter.numel() == 0:
            return
        visible = visibility_filter.bool()
        if torch.count_nonzero(visible) == 0:
            return

        with torch.no_grad():
            residual = torch.abs(image.detach() - gt_image.detach()).mean(dim=0)
            flat_residual = residual.reshape(-1)
            error_threshold = torch.quantile(flat_residual, 0.9)
            peak_error = residual.amax()
            if peak_error <= error_threshold + 1e-6:
                self.maskforems.mul_(0.995)
                return

            visible_indices = torch.nonzero(visible, as_tuple=False).squeeze(1)
            projected = geom_transform_points(means3D[visible], viewpoint_camera.full_proj_transform)
            ndc = projected[:, :2]

            valid = (
                torch.isfinite(ndc[:, 0])
                & torch.isfinite(ndc[:, 1])
                & (ndc[:, 0] >= -1.0)
                & (ndc[:, 0] <= 1.0)
                & (ndc[:, 1] >= -1.0)
                & (ndc[:, 1] <= 1.0)
            )
            update_score = torch.zeros((visible_indices.shape[0], 1), device=residual.device, dtype=residual.dtype)
            if torch.count_nonzero(valid) > 0:
                ndc_valid = ndc[valid]
                x = (((ndc_valid[:, 0] + 1.0) * viewpoint_camera.image_width) - 1.0) * 0.5
                y = (((ndc_valid[:, 1] + 1.0) * viewpoint_camera.image_height) - 1.0) * 0.5
                x = torch.round(x).long().clamp(0, viewpoint_camera.image_width - 1)
                y = torch.round(y).long().clamp(0, viewpoint_camera.image_height - 1)
                sampled_residual = residual[y, x]
                sampled_score = torch.clamp(
                    (sampled_residual - error_threshold) / (peak_error - error_threshold + 1e-6),
                    min=0.0,
                    max=1.0,
                )
                update_score[valid] = sampled_score.unsqueeze(1)

            self.maskforems.mul_(0.995)
            current_visible = self.maskforems[visible_indices] * 0.98
            self.maskforems[visible_indices] = torch.maximum(current_visible, update_score)

    def maybe_temporal_refine_fast(self, iteration):
        if (not self.use_euler_field) or (not self.field_temporal_refine):
            return 0
        if iteration < self.field_temporal_refine_start:
            return 0
        if self.field_stage != "fast_refine":
            return 0
        if self.field_temporal_refine_interval <= 0 or iteration % self.field_temporal_refine_interval != 0:
            return 0
        if self._fast_active_mask is None or self._fast_active_mask.numel() == 0:
            return 0

        fast_mask = self._fast_active_mask.squeeze(1) > 0.5
        if torch.count_nonzero(fast_mask) == 0:
            return 0

        opacity = self.get_opacity.detach().squeeze(1)
        fast_score = self._fast_score_ema.detach().squeeze(1)
        if self.field_v23_compat:
            candidate_mask = (
                fast_mask
                & (fast_score >= self.field_temporal_refine_score_threshold)
                & (opacity >= self.field_temporal_refine_opacity_threshold)
            )
        else:
            responsibility = self._responsibility_ema.detach().squeeze(1) if self._responsibility_ema is not None and self._responsibility_ema.numel() > 0 else torch.zeros_like(opacity)
            visibility = self._visibility_persistence_ema.detach().squeeze(1) if self._visibility_persistence_ema is not None and self._visibility_persistence_ema.numel() > 0 else torch.zeros_like(opacity)
            candidate_mask = (
                fast_mask
                & (fast_score >= self.field_temporal_refine_score_threshold)
                & (responsibility >= self.field_responsibility_on_threshold)
                & (opacity >= self.field_temporal_refine_opacity_threshold)
                & (visibility >= self.field_visibility_static_threshold * 0.5)
            )
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
        if candidate_indices.numel() == 0:
            return 0

        max_candidates = max(1, int(self.get_xyz.shape[0] * self.field_temporal_refine_max_ratio))
        if candidate_indices.numel() > max_candidates:
            candidate_scores = fast_score[candidate_indices]
            _, top_idx = torch.topk(candidate_scores, k=max_candidates, largest=True, sorted=False)
            candidate_indices = candidate_indices[top_idx]

        children = max(2, int(self.field_temporal_split_children))
        repeat_count = children
        selected_mask = torch.zeros((self.get_xyz.shape[0],), device=self.get_xyz.device, dtype=torch.bool)
        selected_mask[candidate_indices] = True

        parent_xyz = self._xyz[selected_mask]
        parent_features_dc = self._features_dc[selected_mask]
        parent_opacity_prob = self.get_opacity[selected_mask]
        parent_scaling = self._scaling[selected_mask]
        parent_rotation = self._rotation[selected_mask]
        parent_trbf_center = self._trbf_center[selected_mask]
        parent_trbf_scale = self._trbf_scale[selected_mask]
        parent_motion = self._motion[selected_mask]
        parent_omega = self._omega[selected_mask]
        parent_feature_t = self._features_t[selected_mask]
        parent_static_logits = self._static_level_logits[selected_mask] if self.use_euler_field else None
        parent_dynamic_logits = self._dynamic_level_logits[selected_mask] if self.use_euler_field else None
        parent_dynamic_time = self._dynamic_level_time_coeff[selected_mask] if self.use_euler_field and self._dynamic_level_time_coeff.numel() > 0 else None
        parent_ems_mask = self.maskforems[selected_mask] if self.maskforems is not None and self.maskforems.numel() > 0 else None
        parent_dynamic_score = self._dynamic_score_ema[selected_mask] if self._dynamic_score_ema is not None and self._dynamic_score_ema.numel() > 0 else None
        parent_responsibility = self._responsibility_ema[selected_mask] if self._responsibility_ema is not None and self._responsibility_ema.numel() > 0 else None
        parent_responsibility_time = self._responsibility_time_center_ema[selected_mask] if self._responsibility_time_center_ema is not None and self._responsibility_time_center_ema.numel() > 0 else None
        parent_slow_score = self._slow_motion_score_ema[selected_mask] if self._slow_motion_score_ema is not None and self._slow_motion_score_ema.numel() > 0 else None
        parent_fast_score = self._fast_score_ema[selected_mask] if self._fast_score_ema is not None and self._fast_score_ema.numel() > 0 else None
        parent_visibility = self._visibility_persistence_ema[selected_mask] if self._visibility_persistence_ema is not None and self._visibility_persistence_ema.numel() > 0 else None

        new_xyz = parent_xyz.repeat(repeat_count, 1)
        new_features_dc = parent_features_dc.repeat(repeat_count, 1)
        child_opacity = torch.clamp(parent_opacity_prob / float(repeat_count), min=1e-4, max=0.99)
        new_opacity = inverse_sigmoid(child_opacity).repeat(repeat_count, 1)
        new_scaling = parent_scaling.repeat(repeat_count, 1)
        new_rotation = parent_rotation.repeat(repeat_count, 1)

        parent_sigma = torch.exp(parent_trbf_scale).clamp_min(1e-4)
        center_offset = self.field_temporal_center_offset * parent_sigma
        child_positions = torch.linspace(-1.0, 1.0, steps=repeat_count, device=self.get_xyz.device, dtype=self.get_xyz.dtype).view(repeat_count, 1, 1)
        if parent_responsibility_time is not None:
            base_trbf_center = torch.where(parent_responsibility_time >= 0.0, parent_responsibility_time, parent_trbf_center)
        else:
            base_trbf_center = parent_trbf_center
        new_trbf_center = (base_trbf_center.unsqueeze(0) + child_positions * center_offset.unsqueeze(0)).reshape(-1, 1).clamp(0.0, 1.0)
        shrink = math.log(max(self.field_temporal_scale_shrink, 1e-3))
        new_trbf_scale = (parent_trbf_scale + shrink).repeat(repeat_count, 1)

        new_motion = (parent_motion * self.field_fast_child_motion_scale).repeat(repeat_count, 1)
        new_omega = (parent_omega * self.field_fast_child_motion_scale).repeat(repeat_count, 1)
        new_feature_t = parent_feature_t.repeat(repeat_count, 1)
        new_static_level_logits = parent_static_logits.repeat(repeat_count, 1) if parent_static_logits is not None else None
        new_dynamic_level_logits = parent_dynamic_logits.repeat(repeat_count, 1) if parent_dynamic_logits is not None else None
        new_dynamic_level_time_coeff = parent_dynamic_time.repeat(repeat_count, 1, 1) if parent_dynamic_time is not None else None
        new_ems_mask = parent_ems_mask.repeat(repeat_count, 1) if parent_ems_mask is not None else None

        old_count = self.get_xyz.shape[0]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_opacity,
            new_scaling,
            new_rotation,
            new_trbf_center,
            new_trbf_scale,
            new_motion,
            new_omega,
            new_feature_t,
            new_static_level_logits,
            new_dynamic_level_logits,
            new_dynamic_level_time_coeff,
            new_ems_mask,
        )

        new_count = new_xyz.shape[0]
        if parent_dynamic_score is not None:
            self._dynamic_score_ema[old_count:old_count + new_count] = parent_dynamic_score.repeat(repeat_count, 1)
            self._dynamic_active_mask[old_count:old_count + new_count] = 1.0
        if parent_responsibility is not None:
            self._responsibility_ema[old_count:old_count + new_count] = parent_responsibility.repeat(repeat_count, 1)
        if parent_responsibility_time is not None:
            self._responsibility_time_center_ema[old_count:old_count + new_count] = base_trbf_center.repeat(repeat_count, 1)
        if parent_slow_score is not None:
            self._slow_motion_score_ema[old_count:old_count + new_count] = parent_slow_score.repeat(repeat_count, 1) * 0.0
            self._slow_motion_mask[old_count:old_count + new_count] = 0.0
        if parent_fast_score is not None:
            self._fast_score_ema[old_count:old_count + new_count] = parent_fast_score.repeat(repeat_count, 1)
            self._fast_active_mask[old_count:old_count + new_count] = 1.0
        if parent_visibility is not None:
            self._visibility_persistence_ema[old_count:old_count + new_count] = parent_visibility.repeat(repeat_count, 1)
        if self._static_support_ema is not None and self._static_support_ema.numel() >= old_count + new_count:
            self._static_support_ema[old_count:old_count + new_count] = 0.0
        if self._static_support_mask is not None and self._static_support_mask.numel() >= old_count + new_count:
            self._static_support_mask[old_count:old_count + new_count] = 0.0

        prune_filter = torch.cat((selected_mask, torch.zeros(new_count, device="cuda", dtype=torch.bool)))
        self.prune_points(prune_filter)
        return int(candidate_indices.numel())

        
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    def get_rotation(self, delta_t):
        rotation =  self._rotation + delta_t*self._omega
        self.delta_t = delta_t
        return self.rotation_activation(rotation)
    

    @property
    def get_xyz(self):
        return self._xyz
    @property
    def get_trbfcenter(self):
        return self._trbf_center
    @property
    def get_trbfscale(self):
        return self._trbf_scale
    @property
    def get_level_logits(self):
        return self._dynamic_level_logits
    def get_features(self, deltat):
        return torch.cat((self._features_dc, deltat * self._features_t), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def configure_euler_field(self, args):
        self.use_euler_field = bool(getattr(args, "use_euler_field", False))
        self.field_base_resolution = int(getattr(args, "field_base_resolution", 4))
        self.field_num_levels = int(getattr(args, "field_num_levels", 5))
        self.field_resolution_mode = str(getattr(args, "field_resolution_mode", "fixed"))
        self.field_level_resolutions = getattr(args, "field_level_resolutions", "")
        self.field_resolved_level_resolutions = getattr(args, "field_resolved_level_resolutions", "")
        self.field_resolution_growth = float(getattr(args, "field_resolution_growth", 2.0))
        self.field_max_resolution = int(getattr(args, "field_max_resolution", 96))
        self.field_knn_scale_percentile = float(getattr(args, "field_knn_scale_percentile", 25.0))
        self.field_gaussian_scale_percentile = float(getattr(args, "field_gaussian_scale_percentile", 25.0))
        self.field_pixel_scale_percentile = float(getattr(args, "field_pixel_scale_percentile", 25.0))
        self.field_knn_scale_weight = float(getattr(args, "field_knn_scale_weight", 1.0))
        self.field_gaussian_scale_weight = float(getattr(args, "field_gaussian_scale_weight", 1.0))
        self.field_pixel_scale_weight = float(getattr(args, "field_pixel_scale_weight", 1.0))
        self.field_min_cell_scale = float(getattr(args, "field_min_cell_scale", 1.5))
        self.field_feature_dim = int(getattr(args, "field_feature_dim", 8))
        self.field_fourier_degree = int(getattr(args, "field_fourier_degree", 10))
        self.field_level_fourier_degree = int(getattr(args, "field_level_fourier_degree", 2))
        self.field_decoder_hidden = int(getattr(args, "field_decoder_hidden", 32))
        self.field_residual_mode = str(getattr(args, "field_residual_mode", "geometry"))
        self.field_query_mode = str(getattr(args, "field_query_mode", "hybrid"))
        self.field_query_detach = bool(getattr(args, "field_query_detach", 1))
        self.field_query_gate_bias = float(getattr(args, "field_query_gate_bias", -2.0))
        self.field_query_motion_scale = float(getattr(args, "field_query_motion_scale", 2.0))
        self.field_dyn_threshold = float(getattr(args, "field_dyn_threshold", 0.08))
        self.field_fast_threshold = float(getattr(args, "field_fast_threshold", 0.18))
        self.field_dyn_slope = float(getattr(args, "field_dyn_slope", 10.0))
        self.field_fast_slope = float(getattr(args, "field_fast_slope", 15.0))
        self.field_fast_temperature = float(getattr(args, "field_fast_temperature", 0.25))
        self.field_disable_dynamic_grid = bool(getattr(args, "field_disable_dynamic_grid", 1))
        self.field_v23_compat = bool(getattr(args, "field_v23_compat", 0))
        self.field_static_route_mode = str(getattr(args, "field_static_route_mode", "learned"))
        self.field_static_route_init = float(getattr(args, "field_static_route_init", 0.0))
        self.field_static_start_iter = int(getattr(args, "field_static_start_iter", 0))
        self.field_static_warmup_iters = int(getattr(args, "field_static_warmup_iters", 3000))
        self.field_static_motion_scale = float(getattr(args, "field_static_motion_scale", 0.05))
        self.field_static_opacity_scale = float(getattr(args, "field_static_opacity_scale", 0.02))
        self.field_static_app_scale = float(getattr(args, "field_static_app_scale", 0.05))
        self.field_static_use_global_gate = bool(getattr(args, "field_static_use_global_gate", 0))
        self.field_static_prior_floor = float(getattr(args, "field_static_prior_floor", 0.25))
        self.field_soft_route_slope = float(getattr(args, "field_soft_route_slope", 8.0))
        self.field_soft_static_threshold = float(getattr(args, "field_soft_static_threshold", 0.45))
        self.field_soft_dynamic_threshold = float(getattr(args, "field_soft_dynamic_threshold", 0.25))
        self.field_staged_training = bool(getattr(args, "field_staged_training", 1))
        self.field_disable_legacy_aux = bool(getattr(args, "field_disable_legacy_aux", 1))
        self.field_disable_ems_main = bool(getattr(args, "field_disable_ems_main", 1))
        self.field_disable_global_omega_split = bool(getattr(args, "field_disable_global_omega_split", 1))
        self.field_warmup_iters = int(getattr(args, "field_warmup_iters", 9000))
        self.field_problem_mining_start = int(getattr(args, "field_problem_mining_start", self.field_warmup_iters))
        self.field_category_activate_iter = int(getattr(args, "field_category_activate_iter", 11000))
        self.field_activate_iter = int(getattr(args, "field_activate_iter", self.field_category_activate_iter))
        self.field_fast_activate_iter = int(getattr(args, "field_fast_activate_iter", 13000))
        self.field_mask_update_interval = int(getattr(args, "field_mask_update_interval", 500))
        self.field_score_ema = float(getattr(args, "field_score_ema", 0.05))
        self.field_visibility_ema = float(getattr(args, "field_visibility_ema", 0.05))
        self.field_problem_error_weight = float(getattr(args, "field_problem_error_weight", 0.50))
        self.field_problem_temporal_weight = float(getattr(args, "field_problem_temporal_weight", 0.50))
        self.field_static_error_boost = float(getattr(args, "field_static_error_boost", 0.50))
        self.field_time_center_ema = float(getattr(args, "field_time_center_ema", 0.05))
        self.field_responsibility_on_threshold = float(getattr(args, "field_responsibility_on_threshold", 0.45))
        self.field_responsibility_off_threshold = float(getattr(args, "field_responsibility_off_threshold", 0.25))
        self.field_visibility_static_threshold = float(getattr(args, "field_visibility_static_threshold", 0.60))
        self.field_slow_motion_on_threshold = float(getattr(args, "field_slow_motion_on_threshold", 0.35))
        self.field_slow_motion_off_threshold = float(getattr(args, "field_slow_motion_off_threshold", 0.20))
        self.field_dynamic_on_threshold = float(getattr(args, "field_dynamic_on_threshold", 0.50))
        self.field_dynamic_off_threshold = float(getattr(args, "field_dynamic_off_threshold", 0.30))
        self.field_motion_pre_threshold = float(getattr(args, "field_motion_pre_threshold", 0.20))
        self.field_motion_accel_pre_threshold = float(getattr(args, "field_motion_accel_pre_threshold", 0.18))
        self.field_static_pre_threshold = float(getattr(args, "field_static_pre_threshold", 0.15))
        self.field_static_on_threshold = float(getattr(args, "field_static_on_threshold", 0.70))
        self.field_static_off_threshold = float(getattr(args, "field_static_off_threshold", 0.50))
        self.field_static_motion_threshold = float(getattr(args, "field_static_motion_threshold", 0.12))
        self.field_static_accel_threshold = float(getattr(args, "field_static_accel_threshold", 0.12))
        self.field_fast_on_threshold = float(getattr(args, "field_fast_on_threshold", 0.60))
        self.field_fast_off_threshold = float(getattr(args, "field_fast_off_threshold", 0.40))
        self.field_score_motion_weight = float(getattr(args, "field_score_motion_weight", 0.35))
        self.field_score_accel_weight = float(getattr(args, "field_score_accel_weight", 0.15))
        self.field_score_error_weight = float(getattr(args, "field_score_error_weight", 0.25))
        self.field_score_screen_weight = float(getattr(args, "field_score_screen_weight", 0.15))
        self.field_score_xyz_weight = float(getattr(args, "field_score_xyz_weight", 0.10))
        self.field_score_static_residual_weight = float(getattr(args, "field_score_static_residual_weight", 0.15))
        self.field_fast_score_motion_weight = float(getattr(args, "field_fast_score_motion_weight", 0.45))
        self.field_fast_score_accel_weight = float(getattr(args, "field_fast_score_accel_weight", 0.35))
        self.field_fast_score_error_weight = float(getattr(args, "field_fast_score_error_weight", 0.20))
        self.field_fast_score_screen_weight = float(getattr(args, "field_fast_score_screen_weight", 0.05))
        self.field_fast_score_xyz_weight = float(getattr(args, "field_fast_score_xyz_weight", 0.05))
        self.field_fast_score_static_residual_weight = float(getattr(args, "field_fast_score_static_residual_weight", 0.25))
        self.field_static_score_motion_weight = float(getattr(args, "field_static_score_motion_weight", 0.45))
        self.field_static_score_accel_weight = float(getattr(args, "field_static_score_accel_weight", 0.35))
        self.field_static_score_residual_weight = float(getattr(args, "field_static_score_residual_weight", 0.20))
        self.field_fast_opacity_scale = float(getattr(args, "field_fast_opacity_scale", 0.75))
        self.field_fast_motion_scale = float(getattr(args, "field_fast_motion_scale", 1.0))
        self.field_temporal_refine = bool(getattr(args, "field_temporal_refine", 1))
        self.field_temporal_refine_start = int(getattr(args, "field_temporal_refine_start", 7000))
        self.field_temporal_refine_interval = int(getattr(args, "field_temporal_refine_interval", 1000))
        self.field_temporal_split_children = int(getattr(args, "field_temporal_split_children", 2))
        self.field_temporal_center_offset = float(getattr(args, "field_temporal_center_offset", 0.08))
        self.field_temporal_scale_shrink = float(getattr(args, "field_temporal_scale_shrink", 0.5))
        self.field_fast_child_motion_scale = float(getattr(args, "field_fast_child_motion_scale", 1.0))
        self.field_temporal_refine_opacity_threshold = float(getattr(args, "field_temporal_refine_opacity_threshold", 0.2))
        self.field_temporal_refine_score_threshold = float(getattr(args, "field_temporal_refine_score_threshold", 0.65))
        self.field_temporal_refine_max_ratio = float(getattr(args, "field_temporal_refine_max_ratio", 0.02))

    def set_field_camera_scale_hints(self, cameras):
        self._field_camera_scale_hints = []
        for camera in cameras:
            camera_center = getattr(camera, "camera_center", None)
            if camera_center is None:
                continue
            self._field_camera_scale_hints.append(
                {
                    "center": camera_center.detach().float().cpu(),
                    "fovx": float(getattr(camera, "FoVx", 0.0)),
                    "fovy": float(getattr(camera, "FoVy", 0.0)),
                    "width": int(getattr(camera, "image_width", 0)),
                    "height": int(getattr(camera, "image_height", 0)),
                }
            )

    def _parse_field_level_resolutions(self, spec):
        if spec is None or spec == "":
            return None
        if isinstance(spec, str):
            items = []
            for level_spec in spec.replace("|", ";").split(";"):
                level_spec = level_spec.strip()
                if not level_spec:
                    continue
                values = [v for v in level_spec.replace("x", ",").split(",") if v.strip()]
                items.append(tuple(int(v) for v in values))
        elif isinstance(spec, (list, tuple)):
            items = spec
        else:
            return None

        resolutions = []
        for item in items:
            if isinstance(item, int):
                resolution = (item, item, item)
            else:
                if len(item) == 1:
                    resolution = (int(item[0]), int(item[0]), int(item[0]))
                elif len(item) == 3:
                    resolution = tuple(int(v) for v in item)
                else:
                    raise ValueError(f"Invalid field_level_resolutions entry: {item}")
            if min(resolution) < 2:
                raise ValueError(f"Euler grid resolution must be >= 2 on every axis, got {resolution}")
            resolutions.append(resolution)
        return resolutions if resolutions else None

    @staticmethod
    def _format_field_level_resolutions(resolutions):
        return ";".join("{}x{}x{}".format(*resolution) for resolution in resolutions)

    @staticmethod
    def _positive_percentile(values, percentile):
        if values is None:
            return None
        values = values.detach().reshape(-1).float()
        values = values[torch.isfinite(values) & (values > 0)]
        if values.numel() == 0:
            return None
        percentile = max(0.0, min(100.0, float(percentile))) / 100.0
        return float(torch.quantile(values, percentile).item())

    def _estimate_field_pixel_scale(self, bbox_center):
        if not self._field_camera_scale_hints:
            return None
        pixel_scales = []
        for hint in self._field_camera_scale_hints:
            width = max(float(hint["width"]), 1.0)
            height = max(float(hint["height"]), 1.0)
            fovx = max(float(hint["fovx"]), 1e-6)
            fovy = max(float(hint["fovy"]), 1e-6)
            center = hint["center"].to(device=bbox_center.device, dtype=bbox_center.dtype)
            depth = torch.linalg.norm(center - bbox_center).clamp_min(1e-6)
            scale_x = 2.0 * depth * math.tan(0.5 * fovx) / width
            scale_y = 2.0 * depth * math.tan(0.5 * fovy) / height
            pixel_scales.append(torch.maximum(scale_x, scale_y))
        return self._positive_percentile(
            torch.stack(pixel_scales),
            self.field_pixel_scale_percentile,
        )

    def _resolve_field_level_resolutions(self, bbox_min, bbox_max, knn_distances=None, gaussian_scales=None):
        manual_resolutions = self._parse_field_level_resolutions(self.field_level_resolutions)
        if manual_resolutions is not None:
            self._field_resolution_stats = {"mode": "manual"}
            return manual_resolutions

        checkpoint_resolutions = self._parse_field_level_resolutions(self.field_resolved_level_resolutions)
        if checkpoint_resolutions is not None and knn_distances is None and gaussian_scales is None:
            self._field_resolution_stats = {"mode": "checkpoint"}
            return checkpoint_resolutions

        if str(self.field_resolution_mode).lower() != "auto_physical":
            self._field_resolution_stats = {"mode": "fixed"}
            return [
                (self.field_base_resolution * (2 ** level),) * 3
                for level in range(self.field_num_levels)
            ]

        bbox_span = torch.clamp((bbox_max - bbox_min).detach().float(), min=1e-6)
        max_span = float(torch.max(bbox_span).item())
        bbox_center = 0.5 * (bbox_min + bbox_max)

        scale_components = {}
        knn_scale = self._positive_percentile(knn_distances, self.field_knn_scale_percentile)
        if knn_scale is not None:
            scale_components["knn"] = knn_scale * self.field_knn_scale_weight
        gaussian_scale = self._positive_percentile(gaussian_scales, self.field_gaussian_scale_percentile)
        if gaussian_scale is not None:
            scale_components["gaussian"] = gaussian_scale * self.field_gaussian_scale_weight
        pixel_scale = self._estimate_field_pixel_scale(bbox_center)
        if pixel_scale is not None:
            scale_components["pixel"] = pixel_scale * self.field_pixel_scale_weight

        finest_cell = 0.0
        if scale_components:
            finest_cell = max(scale_components.values()) * max(float(self.field_min_cell_scale), 0.0)

        base_resolution = max(int(self.field_base_resolution), 2)
        max_resolution = max(int(self.field_max_resolution), base_resolution)
        growth = max(float(self.field_resolution_growth), 1.01)
        max_levels = max(int(self.field_num_levels), 1)

        resolutions = []
        for level in range(max_levels):
            target_resolution = int(round(base_resolution * (growth ** level)))
            target_resolution = max(2, min(target_resolution, max_resolution))
            cell_size = max_span / max(target_resolution - 1, 1)
            if resolutions and finest_cell > 0.0 and cell_size < finest_cell:
                break
            level_resolution = torch.ceil(bbox_span / cell_size).long() + 1
            level_resolution = torch.clamp(level_resolution, min=2, max=max_resolution)
            item = tuple(int(v.item()) for v in level_resolution)
            if resolutions and item == resolutions[-1]:
                break
            resolutions.append(item)
            if target_resolution >= max_resolution:
                break

        if not resolutions:
            resolutions = [(base_resolution, base_resolution, base_resolution)]

        self._field_resolution_stats = {
            "mode": "auto_physical",
            "finest_cell": finest_cell,
            "components": scale_components,
        }
        return resolutions

    def _build_euler_modules(self, bbox_min, bbox_max, knn_distances=None, gaussian_scales=None):
        if not self.use_euler_field:
            return
        level_resolutions = self._resolve_field_level_resolutions(
            bbox_min,
            bbox_max,
            knn_distances=knn_distances,
            gaussian_scales=gaussian_scales,
        )
        self.field_num_levels = len(level_resolutions)
        self.field_resolved_level_resolutions = self._format_field_level_resolutions(level_resolutions)
        stats = getattr(self, "_field_resolution_stats", {})
        components = stats.get("components", {})
        component_text = ", ".join("{}={:.6g}".format(k, v) for k, v in components.items())
        print(
            "[STEGF] Euler grid mode={}, levels={}, resolutions={}, finest_cell={}, components={}".format(
                stats.get("mode", self.field_resolution_mode),
                self.field_num_levels,
                self.field_resolved_level_resolutions,
                stats.get("finest_cell", "n/a"),
                component_text if component_text else "n/a",
            )
        )
        self.euler_field = EulerField(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            base_resolution=self.field_base_resolution,
            num_levels=self.field_num_levels,
            feature_dim=self.field_feature_dim,
            fourier_degree=self.field_fourier_degree,
            level_resolutions=level_resolutions,
            enable_dynamic_grid=not self.field_disable_dynamic_grid,
        ).cuda()
        router_input_dim = self.field_feature_dim + 2 * self.field_level_fourier_degree
        self.field_router = EulerLevelRouter(
            input_dim=router_input_dim,
            hidden_dim=self.field_decoder_hidden,
            num_levels=self.field_num_levels,
        ).cuda()
        gate_input_dim = self.field_feature_dim * 3 + 2 * self.field_level_fourier_degree + 2
        self.field_query_gate = EulerQueryFusionGate(
            input_dim=gate_input_dim,
            hidden_dim=self.field_decoder_hidden,
            bias_init=self.field_query_gate_bias,
            motion_scale_init=self.field_query_motion_scale,
        ).cuda()
        self.field_decoder = EulerResidualDecoder(
            feature_dim=self.field_feature_dim,
            hidden_dim=self.field_decoder_hidden,
        ).cuda()
        self.field_temporal_opacity_head = nn.Linear(self.field_feature_dim, 1, bias=False).cuda()
        nn.init.normal_(self.field_temporal_opacity_head.weight, mean=0.0, std=1e-3)
        self.field_static_view_mapper = nn.Sequential(
            nn.Linear(self.field_feature_dim + 3, self.field_decoder_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(self.field_decoder_hidden, self.field_feature_dim, bias=False),
        ).cuda()
        nn.init.normal_(self.field_static_view_mapper[-1].weight, mean=0.0, std=1e-3)
        self.field_static_app_head = nn.Sequential(
            nn.Linear(self.field_feature_dim, self.field_decoder_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(self.field_decoder_hidden, 6, bias=False),
        ).cuda()
        nn.init.normal_(self.field_static_app_head[-1].weight, mean=0.0, std=1e-3)

    def _init_static_level_logits(self, num_points, values=None):
        if not self.use_euler_field:
            self._static_level_logits = torch.empty(0, device="cuda")
            return
        if values is None:
            values = torch.zeros((num_points, self.field_num_levels), device="cuda")
        else:
            values = values.to(device="cuda", dtype=torch.float32)
        self._static_level_logits = nn.Parameter(values.requires_grad_(True))

    def _init_dynamic_level_logits(self, num_points, values=None):
        if not self.use_euler_field:
            self._dynamic_level_logits = torch.empty(0, device="cuda")
            return
        if values is None:
            values = torch.zeros((num_points, self.field_num_levels), device="cuda")
        else:
            values = values.to(device="cuda", dtype=torch.float32)
        self._dynamic_level_logits = nn.Parameter(values.requires_grad_(True))

    def _init_dynamic_level_time_coeff(self, num_points, values=None):
        if not self.use_euler_field or self.field_level_fourier_degree <= 0:
            self._dynamic_level_time_coeff = torch.empty(0, device="cuda")
            return
        coeff_dim = 2 * self.field_level_fourier_degree
        if values is None:
            values = torch.zeros((num_points, self.field_num_levels, coeff_dim), device="cuda")
        else:
            values = values.to(device="cuda", dtype=torch.float32)
        self._dynamic_level_time_coeff = nn.Parameter(values.requires_grad_(True))

    def _init_static_route_logits(self, num_points, values=None):
        if (not self.use_euler_field) or self.field_static_route_mode != "learned":
            self._static_route_logits = torch.empty(0, device="cuda")
            return
        if values is None:
            values = torch.full(
                (num_points, 1),
                float(self.field_static_route_init),
                device="cuda",
                dtype=torch.float32,
            )
        else:
            values = values.to(device="cuda", dtype=torch.float32)
        self._static_route_logits = nn.Parameter(values.requires_grad_(True))

    def _init_field_residual_gate(self, values=None):
        if not self.use_euler_field:
            self._field_residual_gate = torch.empty(0, device="cuda")
            return
        if values is None:
            values = torch.zeros((2,), device="cuda")
        else:
            values = values.to(device="cuda", dtype=torch.float32)
        self._field_residual_gate = nn.Parameter(values.requires_grad_(True))

    def _time_fourier_basis(self, timestamp, degree, device, dtype):
        if degree <= 0:
            return torch.empty(0, device=device, dtype=dtype)
        if not torch.is_tensor(timestamp):
            timestamp = torch.tensor(timestamp, device=device, dtype=dtype)
        timestamp = timestamp.reshape(1).to(device=device, dtype=dtype)
        harmonics = torch.arange(1, degree + 1, device=device, dtype=dtype)
        angles = 2.0 * math.pi * harmonics * timestamp
        basis = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
        return basis.reshape(-1)

    def _get_router_level_delta(self, level_features, timestamp):
        if self.field_router is None:
            return None
        coarse_levels = min(3, level_features.shape[1])
        coarse_feature = level_features[:, :coarse_levels].mean(dim=1).detach()
        router_inputs = [coarse_feature]
        if self.field_level_fourier_degree > 0:
            basis = self._time_fourier_basis(
                timestamp,
                self.field_level_fourier_degree,
                coarse_feature.device,
                coarse_feature.dtype,
            ).unsqueeze(0).expand(coarse_feature.shape[0], -1)
            router_inputs.append(basis)
        return self.field_router(torch.cat(router_inputs, dim=1))

    def _get_motion_strength(self, motion_offset):
        motion_norm = torch.norm(motion_offset.detach(), dim=1, keepdim=True)
        if self.euler_field is not None:
            bbox_scale = torch.linalg.norm(self.euler_field.bbox_span, dim=1, keepdim=True).to(
                device=motion_norm.device,
                dtype=motion_norm.dtype,
            ).clamp_min(1e-6)
        else:
            bbox_scale = torch.ones((1, 1), device=motion_norm.device, dtype=motion_norm.dtype)
        normalized_motion = motion_norm / bbox_scale
        return torch.log1p(10.0 * normalized_motion)

    def _get_motion_acceleration(self, time_offset):
        acceleration = 2.0 * self._motion[:, 3:6] + 6.0 * self._motion[:, 6:9] * time_offset
        acceleration_norm = torch.norm(acceleration.detach(), dim=1, keepdim=True)
        if self.euler_field is not None:
            bbox_scale = torch.linalg.norm(self.euler_field.bbox_span, dim=1, keepdim=True).to(
                device=acceleration_norm.device,
                dtype=acceleration_norm.dtype,
            ).clamp_min(1e-6)
        else:
            bbox_scale = torch.ones((1, 1), device=acceleration_norm.device, dtype=acceleration_norm.dtype)
        normalized_accel = acceleration_norm / bbox_scale
        return torch.log1p(10.0 * normalized_accel)

    def _get_motion_state_weights(self, motion_offset, time_offset):
        motion_strength = self._get_motion_strength(motion_offset)
        motion_acceleration = self._get_motion_acceleration(time_offset)
        dynamic_weight = torch.sigmoid(self.field_dyn_slope * (motion_strength - self.field_dyn_threshold))
        fast_weight = torch.sigmoid(self.field_fast_slope * (motion_strength - self.field_fast_threshold))
        return motion_strength, motion_acceleration, dynamic_weight, fast_weight

    def _get_temporal_child_support(self, trbf_center, trbf_scale, motion, error_prior=None, copies_per_parent=1):
        centers = trbf_center.repeat(copies_per_parent, 1)
        scales = trbf_scale.repeat(copies_per_parent, 1)
        if (not self.use_euler_field) or centers.numel() == 0 or self.field_disable_legacy_aux:
            return centers, scales

        parent_strength = self._get_motion_strength(motion[:, 0:3]).clamp(max=1.5)
        if error_prior is None:
            error_gate = torch.zeros_like(parent_strength)
        else:
            error_gate = error_prior.detach().reshape(-1, 1).to(device=parent_strength.device, dtype=parent_strength.dtype).clamp(0.0, 1.0)
        motion_gate = torch.clamp((parent_strength - 0.1) / 0.3, min=0.0, max=1.0)
        split_gate = error_gate * motion_gate

        repeated_strength = parent_strength.repeat(copies_per_parent, 1)
        repeated_gate = split_gate.repeat(copies_per_parent, 1)
        offset_scale = 0.06 * repeated_strength * repeated_gate

        if copies_per_parent == 1:
            direction = torch.sign(
                motion[:, 0:1]
                + motion[:, 1:2]
                + motion[:, 2:3]
                + 0.5 * (motion[:, 3:4] + motion[:, 4:5] + motion[:, 5:6])
            )
            direction = torch.where(direction == 0, torch.ones_like(direction), direction)
            offsets = direction * offset_scale
        else:
            direction = torch.sign(
                motion[:, 0:1]
                + motion[:, 1:2]
                + motion[:, 2:3]
                + 0.5 * (motion[:, 3:4] + motion[:, 4:5] + motion[:, 5:6])
            )
            direction = torch.where(direction == 0, torch.ones_like(direction), direction)
            coefficients = torch.linspace(
                -0.25,
                1.0,
                steps=copies_per_parent,
                device=trbf_center.device,
                dtype=trbf_center.dtype,
            )
            coefficients = coefficients.view(copies_per_parent, 1, 1).expand(-1, trbf_center.shape[0], -1).reshape(-1, 1)
            offsets = coefficients * direction.repeat(copies_per_parent, 1) * offset_scale

        centers = (centers + offsets).clamp(0.0, 1.0)
        scales = scales - 0.02 * repeated_strength * repeated_gate
        return centers, scales

    def _get_query_fusion_alpha(self, canonical_feature, motion_feature, timestamp, motion_offset, time_offset):
        if self.field_query_gate is None:
            return torch.ones_like(time_offset)
        motion_strength = self._get_motion_strength(motion_offset)
        gate_inputs = [
            canonical_feature.detach(),
            motion_feature.detach(),
            (motion_feature - canonical_feature).detach(),
            motion_strength,
            torch.abs(time_offset.detach()),
        ]
        if self.field_level_fourier_degree > 0:
            basis = self._time_fourier_basis(
                timestamp,
                self.field_level_fourier_degree,
                canonical_feature.device,
                canonical_feature.dtype,
            ).unsqueeze(0).expand(canonical_feature.shape[0], -1)
            gate_inputs.append(basis)
        gate_logits = self.field_query_gate(torch.cat(gate_inputs, dim=1), motion_strength=motion_strength)
        return torch.sigmoid(gate_logits)

    def _get_soft_route_weights(self, dynamic_weight, gate_alpha, stage):
        num_points = dynamic_weight.shape[0]
        device = dynamic_weight.device
        dtype = dynamic_weight.dtype
        if self.field_v23_compat:
            if stage != "fast_refine" or self.field_disable_dynamic_grid:
                static_route = torch.ones((num_points, 1), device=device, dtype=dtype)
                dynamic_route = torch.zeros_like(static_route)
                return static_route, dynamic_route

            dynamic_score = self._point_state_or_zeros(self._dynamic_score_ema, num_points, device, dtype)
            dynamic_active = self._point_state_or_zeros(self._dynamic_active_mask, num_points, device, dtype)
            slope = max(float(self.field_soft_route_slope), 1e-6)
            score_prior = torch.sigmoid(slope * (dynamic_score - self.field_dynamic_off_threshold))
            dynamic_prior = torch.maximum(dynamic_weight, torch.maximum(score_prior, dynamic_active))
            dynamic_route = torch.clamp(gate_alpha * dynamic_prior, min=0.0, max=1.0)
            static_route = torch.clamp(1.0 - dynamic_route, min=0.0, max=1.0)
            return static_route, dynamic_route

        responsibility = self._point_state_or_zeros(self._responsibility_ema, num_points, device, dtype)
        visibility = self._point_state_or_zeros(self._visibility_persistence_ema, num_points, device, dtype)
        fast_score = self._point_state_or_zeros(self._fast_score_ema, num_points, device, dtype)

        slope = max(float(self.field_soft_route_slope), 1e-6)
        if self.field_static_route_mode == "learned" and self._static_route_logits.numel() == num_points:
            learned_gate = torch.sigmoid(self._static_route_logits.to(device=device, dtype=dtype))
        else:
            static_score = self._point_state_or_zeros(self._static_support_ema, num_points, device, dtype)
            learned_gate = torch.sigmoid(slope * (static_score - self.field_soft_static_threshold))

        static_score = self._point_state_or_zeros(self._static_support_ema, num_points, device, dtype)
        visibility_prior = torch.sigmoid(slope * (visibility - self.field_visibility_static_threshold))
        support_prior = torch.sigmoid(slope * (static_score - self.field_soft_static_threshold))
        responsibility_prior = torch.sigmoid(slope * (self.field_responsibility_off_threshold - responsibility))
        static_prior = visibility_prior * support_prior * responsibility_prior
        prior_floor = float(self.field_static_prior_floor)
        static_prior = prior_floor + (1.0 - prior_floor) * static_prior
        fast_signal = torch.maximum(fast_score, dynamic_weight)
        fast_suppression = 1.0 - torch.sigmoid(slope * (fast_signal - self.field_fast_off_threshold))
        static_route = torch.clamp(learned_gate * static_prior * fast_suppression, min=0.0, max=1.0)

        dynamic_allowed = (stage == "fast_refine") and (not self.field_disable_dynamic_grid)
        if dynamic_allowed:
            responsibility_gate = torch.sigmoid(slope * (responsibility - self.field_soft_dynamic_threshold))
            fast_gate = torch.sigmoid(slope * (fast_score - self.field_fast_off_threshold))
            score_prior = responsibility_gate * fast_gate
            dynamic_prior = torch.maximum(dynamic_weight, score_prior)
            dynamic_route = torch.clamp(gate_alpha * dynamic_prior, min=0.0, max=1.0)
        else:
            dynamic_route = torch.zeros_like(dynamic_weight)

        static_route = torch.clamp(static_route * (1.0 - dynamic_route), min=0.0, max=1.0)
        return static_route, dynamic_route

    def _get_static_level_logits(self):
        return self._static_level_logits

    def _get_dynamic_level_logits(self, timestamp, level_features=None):
        level_logits = self._dynamic_level_logits
        if self._dynamic_level_time_coeff.numel() > 0:
            basis = self._time_fourier_basis(
                timestamp,
                self.field_level_fourier_degree,
                level_logits.device,
                level_logits.dtype,
            )
            level_delta = torch.einsum("nlk,k->nl", self._dynamic_level_time_coeff, basis)
            level_logits = level_logits + level_delta
        if level_features is not None:
            router_delta = self._get_router_level_delta(level_features, timestamp)
            if router_delta is not None:
                level_logits = level_logits + router_delta
        return level_logits

    def _get_dynamic_level_logits_subset(self, point_mask, timestamp, level_features=None):
        level_logits = self._dynamic_level_logits[point_mask]
        if self._dynamic_level_time_coeff.numel() > 0:
            basis = self._time_fourier_basis(
                timestamp,
                self.field_level_fourier_degree,
                level_logits.device,
                level_logits.dtype,
            )
            level_delta = torch.einsum("nlk,k->nl", self._dynamic_level_time_coeff[point_mask], basis)
            level_logits = level_logits + level_delta
        if level_features is not None:
            router_delta = self._get_router_level_delta(level_features, timestamp)
            if router_delta is not None:
                level_logits = level_logits + router_delta
        return level_logits

    def _apply_field_residual(self, motion, opacity_param, residual):
        gates = torch.tanh(self._field_residual_gate)
        motion = motion + gates[0] * residual[:, 0:9]
        opacity_param = opacity_param + gates[1] * residual[:, 9:10]
        return motion, opacity_param

    def _get_static_residual_warmup(self):
        if not torch.is_grad_enabled():
            return 1.0
        current_iter = int(getattr(self, "field_current_iteration", 0))
        start_iter = int(getattr(self, "field_static_start_iter", 0))
        if current_iter < start_iter:
            return 0.0
        warmup_iters = max(int(getattr(self, "field_static_warmup_iters", 0)), 0)
        if warmup_iters == 0:
            return 1.0
        return min(float(current_iter - start_iter + 1) / float(warmup_iters), 1.0)

    def _apply_static_field_residual(self, motion, opacity_param, residual, scale):
        if scale <= 0.0:
            return motion, opacity_param
        if self.field_static_use_global_gate and self._field_residual_gate.numel() > 0:
            gates = torch.tanh(self._field_residual_gate)
            motion_scale = gates[0]
            opacity_scale = gates[1]
        else:
            motion_scale = float(self.field_static_motion_scale)
            opacity_scale = float(self.field_static_opacity_scale)
        motion = motion + scale * motion_scale * residual[:, 0:9]
        opacity_param = opacity_param + scale * opacity_scale * residual[:, 9:10]
        return motion, opacity_param

    def _init_module_grad_cache(self):
        self.rgb_grd = {}
        if self.rgbdecoder is not None:
            for name, param in self.rgbdecoder.named_parameters():
                self.rgb_grd[name] = torch.zeros_like(param, requires_grad=False, device=param.device)

        self.field_grd = {}
        if self.use_euler_field and self.euler_field is not None:
            for name, param in self.euler_field.named_parameters():
                self.field_grd[name] = torch.zeros_like(param, requires_grad=False, device=param.device)

        self.field_router_grd = {}
        if self.use_euler_field and self.field_router is not None:
            for name, param in self.field_router.named_parameters():
                self.field_router_grd[name] = torch.zeros_like(param, requires_grad=False, device=param.device)

        self.field_query_gate_grd = {}
        if self.use_euler_field and self.field_query_gate is not None:
            for name, param in self.field_query_gate.named_parameters():
                self.field_query_gate_grd[name] = torch.zeros_like(param, requires_grad=False, device=param.device)

        self.field_decoder_grd = {}
        if self.use_euler_field and self.field_decoder is not None:
            for name, param in self.field_decoder.named_parameters():
                self.field_decoder_grd[name] = torch.zeros_like(param, requires_grad=False, device=param.device)
        self.field_temporal_opacity_head_grd = {}
        if self.use_euler_field and self.field_temporal_opacity_head is not None:
            for name, param in self.field_temporal_opacity_head.named_parameters():
                self.field_temporal_opacity_head_grd[name] = torch.zeros_like(param, requires_grad=False, device=param.device)
        self.field_static_view_mapper_grd = {}
        if self.use_euler_field and self.field_static_view_mapper is not None:
            for name, param in self.field_static_view_mapper.named_parameters():
                self.field_static_view_mapper_grd[name] = torch.zeros_like(param, requires_grad=False, device=param.device)
        self.field_static_app_head_grd = {}
        if self.use_euler_field and self.field_static_app_head is not None:
            for name, param in self.field_static_app_head.named_parameters():
                self.field_static_app_head_grd[name] = torch.zeros_like(param, requires_grad=False, device=param.device)

    def compose_time_conditioned_attributes(self, timestamp, basicfunction, camera_center=None):
        pointtimes = torch.ones((self.get_xyz.shape[0], 1), dtype=self.get_xyz.dtype, requires_grad=False, device="cuda")
        base_motion = self._motion
        motion = base_motion
        opacity_param = self._opacity
        features_dc = self._features_dc
        features_t = self._features_t
        app_residual = None
        static_warmup = 0.0
        trbfdistanceoffset = timestamp * pointtimes - self.get_trbfcenter
        tforpoly = trbfdistanceoffset.detach()

        if self.use_euler_field and self.euler_field is not None and self.field_decoder is not None:
            canonical_points = self._xyz
            motion_query_offset = self._motion[:, 0:3] * tforpoly + self._motion[:, 3:6] * tforpoly * tforpoly + self._motion[:, 6:9] * tforpoly * tforpoly * tforpoly
            motion_query_points = canonical_points + motion_query_offset
            motion_strength, motion_acceleration, dynamic_weight, fast_weight = self._get_motion_state_weights(motion_query_offset, tforpoly)
            if self.field_query_detach:
                canonical_points = canonical_points.detach()
                motion_query_points = motion_query_points.detach()
            stage = self.field_stage if self.field_staged_training else "fast_refine"
            if self.field_v23_compat:
                static_feature = None
                static_residual = None
                static_residual_motion = torch.zeros((self.get_xyz.shape[0], 1), device=self.get_xyz.device, dtype=self.get_xyz.dtype)
                if stage != "baseline_warmup":
                    static_level_features = self.euler_field.query_static_level_features(canonical_points)
                    static_level_logits = self._get_static_level_logits()
                    static_feature = self.euler_field.blend_level_features(static_level_features, static_level_logits)
                    static_residual = self.field_decoder(static_feature)
                    static_residual_motion = torch.norm(static_residual[:, 0:9].detach(), dim=1, keepdim=True)

                residual = None
                static_route = torch.zeros((self.get_xyz.shape[0], 1), device=self.get_xyz.device, dtype=self.get_xyz.dtype)
                dynamic_route = torch.zeros_like(static_route)
                if stage in ("category_activation", "fast_refine"):
                    if static_feature is not None and static_residual is not None:
                        residual = torch.zeros((self.get_xyz.shape[0], 10), device=self.get_xyz.device, dtype=self.get_xyz.dtype)
                        gate_alpha = torch.zeros_like(dynamic_weight)
                        dynamic_residual = None
                        dynamic_feature = None
                        if stage == "fast_refine" and not self.field_disable_dynamic_grid:
                            dynamic_level_features = self.euler_field.query_dynamic_level_features(
                                motion_query_points,
                                timestamp,
                                mode="nearest",
                            )
                            dynamic_level_logits = self._get_dynamic_level_logits(timestamp, dynamic_level_features)
                            dynamic_feature = self.euler_field.blend_level_features(dynamic_level_features, dynamic_level_logits)

                            if self.field_query_mode == "coarse_motion":
                                gate_alpha = torch.ones_like(dynamic_weight)
                            elif self.field_query_mode == "hybrid":
                                gate_alpha = self._get_query_fusion_alpha(
                                    static_feature,
                                    dynamic_feature,
                                    timestamp,
                                    motion_query_offset,
                                    tforpoly,
                                )
                            else:
                                gate_alpha = torch.zeros_like(dynamic_weight)
                            dynamic_residual = self.field_decoder(dynamic_feature)

                        static_route, dynamic_route = self._get_soft_route_weights(dynamic_weight, gate_alpha, stage)
                        residual = residual + static_route * static_residual
                        if dynamic_residual is not None and dynamic_feature is not None:
                            residual = residual + dynamic_route * dynamic_residual
                            temporal_opacity_delta = self.field_temporal_opacity_head(dynamic_feature)
                            opacity_param = opacity_param + (
                                self.field_fast_opacity_scale
                                * dynamic_route
                                * fast_weight
                                * temporal_opacity_delta
                            )

                self._last_field_aux = {
                    "motion_strength": motion_strength.detach(),
                    "motion_acceleration": motion_acceleration.detach(),
                    "static_residual_motion": static_residual_motion,
                    "static_route": static_route.detach(),
                    "dynamic_route": dynamic_route.detach(),
                }
                if residual is not None:
                    motion, opacity_param = self._apply_field_residual(motion, opacity_param, residual)
            else:
                static_warmup = self._get_static_residual_warmup()
                static_feature = None
                static_residual_motion = torch.zeros((self.get_xyz.shape[0], 1), device=self.get_xyz.device, dtype=self.get_xyz.dtype)
                if static_warmup > 0.0:
                    static_level_features = self.euler_field.query_static_level_features(
                        canonical_points,
                        camera_center=camera_center,
                        view_mapper=self.field_static_view_mapper,
                        view_scale=self.field_static_app_scale,
                    )
                    static_level_logits = self._get_static_level_logits()
                    static_feature = self.euler_field.blend_level_features(static_level_features, static_level_logits)

                dynamic_field_residual = None
                static_route = torch.zeros((self.get_xyz.shape[0], 1), device=self.get_xyz.device, dtype=self.get_xyz.dtype)
                dynamic_route = torch.zeros_like(static_route)
                app_residual = None
                if static_feature is not None:
                    if self.field_static_app_head is not None and self.field_static_app_scale > 0.0:
                        app_residual = self.field_static_app_head(static_feature)
                    gate_alpha = torch.zeros_like(dynamic_weight)
                    dynamic_residual = None
                    dynamic_feature = None
                    if stage == "fast_refine" and not self.field_disable_dynamic_grid:
                        dynamic_level_features = self.euler_field.query_dynamic_level_features(
                            motion_query_points,
                            timestamp,
                            mode="nearest",
                        )
                        dynamic_level_logits = self._get_dynamic_level_logits(timestamp, dynamic_level_features)
                        dynamic_feature = self.euler_field.blend_level_features(dynamic_level_features, dynamic_level_logits)

                        if self.field_query_mode == "coarse_motion":
                            gate_alpha = torch.ones_like(dynamic_weight)
                        elif self.field_query_mode == "hybrid":
                            gate_alpha = self._get_query_fusion_alpha(
                                static_feature,
                                dynamic_feature,
                                timestamp,
                                motion_query_offset,
                                tforpoly,
                            )
                        else:
                            gate_alpha = torch.zeros_like(dynamic_weight)
                        dynamic_residual = self.field_decoder(dynamic_feature)

                    static_route, dynamic_route = self._get_soft_route_weights(dynamic_weight, gate_alpha, stage)
                    if dynamic_residual is not None and dynamic_feature is not None:
                        dynamic_field_residual = dynamic_route * dynamic_residual
                        temporal_opacity_delta = self.field_temporal_opacity_head(dynamic_feature)
                        opacity_param = opacity_param + (
                            self.field_fast_opacity_scale
                            * dynamic_route
                            * fast_weight
                            * temporal_opacity_delta
                        )

                self._last_field_aux = {
                    "motion_strength": motion_strength.detach(),
                    "motion_acceleration": motion_acceleration.detach(),
                    "static_residual_motion": static_residual_motion,
                    "static_route": static_route.detach(),
                    "dynamic_route": dynamic_route.detach(),
                    "static_warmup": static_warmup,
                }
                if dynamic_field_residual is not None:
                    motion, opacity_param = self._apply_field_residual(motion, opacity_param, dynamic_field_residual)

        trbfdistance = trbfdistanceoffset / torch.exp(self._trbf_scale)
        trbfoutput = basicfunction(trbfdistance)
        opacity = self.opacity_activation(opacity_param) * trbfoutput
        means3D = self.get_xyz + motion[:, 0:3] * tforpoly + motion[:, 3:6] * tforpoly * tforpoly + motion[:, 6:9] * tforpoly * tforpoly * tforpoly
        rotations = self.get_rotation(tforpoly)
        colors_precomp = torch.cat((features_dc, tforpoly * features_t), dim=1)
        if self.use_euler_field and (not self.field_v23_compat) and app_residual is not None:
            app_delta = torch.cat((app_residual, torch.zeros_like(features_t)), dim=1)
            colors_precomp = colors_precomp + static_warmup * self.field_static_app_scale * app_delta
        self.trbfoutput = trbfoutput
        return means3D, opacity, rotations, colors_precomp

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):

        if self.preprocesspoints == 3:
            pcd = interpolate_point(pcd, 4) 
        
        elif self.preprocesspoints == 4:
            pcd = interpolate_point(pcd, 2) 
        
        elif self.preprocesspoints == 5:
            pcd = interpolate_point(pcd, 6) 

        elif self.preprocesspoints == 6:
            pcd = interpolate_point(pcd, 8) 
        
        elif self.preprocesspoints == 7:
            pcd = interpolate_point(pcd, 16) 
        elif self.preprocesspoints == 8:
            pcd = interpolate_pointv3(pcd, 4) 
        elif self.preprocesspoints == 14:
            pcd = interpolate_partuse(pcd, 2) 
        
        elif self.preprocesspoints == 15:
            pcd = interpolate_partuse(pcd, 4) 

        elif self.preprocesspoints == 16:
            pcd = interpolate_partuse(pcd, 8) 
        
        elif self.preprocesspoints == 17:
            pcd = interpolate_partuse(pcd, 16) 
        else:
            pass 
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        times = torch.tensor(np.asarray(pcd.times)).float().cuda()


        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 1.0)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        features9channel = torch.cat((fused_color, fused_color), dim=1)

        self._features_dc = nn.Parameter(features9channel.contiguous().requires_grad_(True))
        
        N, _ = fused_color.shape

        fomega = torch.zeros((N, 3), dtype=torch.float, device="cuda")
        self._features_t =  nn.Parameter(fomega.contiguous().requires_grad_(True))
        
        

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        omega = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        self._omega = nn.Parameter(omega.requires_grad_(True))
        
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        motion = torch.zeros((fused_point_cloud.shape[0], 9), device="cuda")# x1, x2, x3,  y1,y2,y3, z1,z2,z3
        self._motion = nn.Parameter(motion.requires_grad_(True))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        


        self._trbf_center = nn.Parameter(times.contiguous().requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.ones((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True)) 

        ## store gradients


        if self.trbfslinit is not None:
            nn.init.constant_(self._trbf_scale, self.trbfslinit) # too large ?
        else:
            nn.init.constant_(self._trbf_scale, 0) # too large ?

        nn.init.constant_(self._features_t, 0)
        nn.init.constant_(self._omega, 0)



        self.maxz, self.minz = torch.amax(self._xyz[:,2]), torch.amin(self._xyz[:,2]) 
        self.maxy, self.miny = torch.amax(self._xyz[:,1]), torch.amin(self._xyz[:,1]) 
        self.maxx, self.minx = torch.amax(self._xyz[:,0]), torch.amin(self._xyz[:,0]) 
        self.maxz = min((self.maxz, 200.0)) # some outliers in the n4d datasets.. 
        if self.use_euler_field:
            bbox_min = torch.amin(self._xyz.detach(), dim=0)
            bbox_max = torch.amax(self._xyz.detach(), dim=0)
            self._build_euler_modules(
                bbox_min,
                bbox_max,
                knn_distances=torch.sqrt(dist2.detach()),
                gaussian_scales=torch.exp(scales.detach()).mean(dim=1),
            )
            self._init_static_level_logits(self.get_xyz.shape[0])
            self._init_dynamic_level_logits(self.get_xyz.shape[0])
            self._init_dynamic_level_time_coeff(self.get_xyz.shape[0])
            self._init_static_route_logits(self.get_xyz.shape[0])
            self._init_field_residual_gate()
        self._init_ems_mask(self.get_xyz.shape[0])
        self._init_dynamic_score_state(self.get_xyz.shape[0])
    def cache_gradient(self):
        self._xyz_grd += self._xyz.grad.clone()
        self._features_dc_grd += self._features_dc.grad.clone()
        self._features_t_grd += self._features_t.grad.clone() # self._features_t_grd
        self._scaling_grd += self._scaling.grad.clone()
        self._rotation_grd += self._rotation.grad.clone()
        self._opacity_grd += self._opacity.grad.clone()
        self._trbf_center_grd += self._trbf_center.grad.clone()
        self._trbf_scale_grd += self._trbf_scale.grad.clone()
        self._motion_grd += self._motion.grad.clone()
        self._omega_grd += self._omega.grad.clone()
        if self.use_euler_field and self._static_level_logits.numel() > 0 and self._static_level_logits.grad is not None:
            self._static_level_logits_grd += self._static_level_logits.grad.clone()
        if self.use_euler_field and self._dynamic_level_logits.numel() > 0 and self._dynamic_level_logits.grad is not None:
            self._dynamic_level_logits_grd += self._dynamic_level_logits.grad.clone()
        if self.use_euler_field and self._dynamic_level_time_coeff.numel() > 0 and self._dynamic_level_time_coeff.grad is not None:
            self._dynamic_level_time_coeff_grd += self._dynamic_level_time_coeff.grad.clone()
        if self.use_euler_field and self._static_route_logits.numel() > 0 and self._static_route_logits.grad is not None:
            self._static_route_logits_grd += self._static_route_logits.grad.clone()
        if self.use_euler_field and self._field_residual_gate.numel() > 0 and self._field_residual_gate.grad is not None:
            self._field_residual_gate_grd += self._field_residual_gate.grad.clone()
        
        if self.rgbdecoder is not None:
            for name, param in self.rgbdecoder.named_parameters():
                if param.grad is not None:
                    self.rgb_grd[name] = self.rgb_grd[name] + param.grad.clone()
        if self.use_euler_field and self.euler_field is not None:
            for name, param in self.euler_field.named_parameters():
                if param.grad is not None:
                    self.field_grd[name] = self.field_grd[name] + param.grad.clone()
        if self.use_euler_field and self.field_router is not None:
            for name, param in self.field_router.named_parameters():
                if param.grad is not None:
                    self.field_router_grd[name] = self.field_router_grd[name] + param.grad.clone()
        if self.use_euler_field and self.field_query_gate is not None:
            for name, param in self.field_query_gate.named_parameters():
                if param.grad is not None:
                    self.field_query_gate_grd[name] = self.field_query_gate_grd[name] + param.grad.clone()
        if self.use_euler_field and self.field_decoder is not None:
            for name, param in self.field_decoder.named_parameters():
                if param.grad is not None:
                    self.field_decoder_grd[name] = self.field_decoder_grd[name] + param.grad.clone()
        if self.use_euler_field and self.field_temporal_opacity_head is not None:
            for name, param in self.field_temporal_opacity_head.named_parameters():
                if param.grad is not None:
                    self.field_temporal_opacity_head_grd[name] = self.field_temporal_opacity_head_grd[name] + param.grad.clone()
        if self.use_euler_field and self.field_static_view_mapper is not None:
            for name, param in self.field_static_view_mapper.named_parameters():
                if param.grad is not None:
                    self.field_static_view_mapper_grd[name] = self.field_static_view_mapper_grd[name] + param.grad.clone()
        if self.use_euler_field and self.field_static_app_head is not None:
            for name, param in self.field_static_app_head.named_parameters():
                if param.grad is not None:
                    self.field_static_app_head_grd[name] = self.field_static_app_head_grd[name] + param.grad.clone()
    def zero_gradient_cache(self):

        self._xyz_grd = torch.zeros_like(self._xyz, requires_grad=False)
        self._features_dc_grd = torch.zeros_like(self._features_dc, requires_grad=False)
        self._features_t_grd = torch.zeros_like(self._features_t, requires_grad=False)


        self._scaling_grd = torch.zeros_like(self._scaling, requires_grad=False)
        self._rotation_grd = torch.zeros_like(self._rotation, requires_grad=False)
        self._opacity_grd = torch.zeros_like(self._opacity, requires_grad=False)
        self._trbf_center_grd = torch.zeros_like(self._trbf_center, requires_grad=False)
        self._trbf_scale_grd = torch.zeros_like(self._trbf_scale, requires_grad=False)
        self._motion_grd = torch.zeros_like(self._motion, requires_grad=False)
        self._omega_grd = torch.zeros_like(self._omega, requires_grad=False)
        if self.use_euler_field and self._static_level_logits.numel() > 0:
            self._static_level_logits_grd = torch.zeros_like(self._static_level_logits, requires_grad=False)
        if self.use_euler_field and self._dynamic_level_logits.numel() > 0:
            self._dynamic_level_logits_grd = torch.zeros_like(self._dynamic_level_logits, requires_grad=False)
        if self.use_euler_field and self._dynamic_level_time_coeff.numel() > 0:
            self._dynamic_level_time_coeff_grd = torch.zeros_like(self._dynamic_level_time_coeff, requires_grad=False)
        if self.use_euler_field and self._static_route_logits.numel() > 0:
            self._static_route_logits_grd = torch.zeros_like(self._static_route_logits, requires_grad=False)
        if self.use_euler_field and self._field_residual_gate.numel() > 0:
            self._field_residual_gate_grd = torch.zeros_like(self._field_residual_gate, requires_grad=False)




        for name in self.rgb_grd.keys():
            self.rgb_grd[name].zero_()
        for name in self.field_grd.keys():
            self.field_grd[name].zero_()
        for name in self.field_router_grd.keys():
            self.field_router_grd[name].zero_()
        for name in self.field_query_gate_grd.keys():
            self.field_query_gate_grd[name].zero_()
        for name in self.field_decoder_grd.keys():
            self.field_decoder_grd[name].zero_()
        for name in self.field_temporal_opacity_head_grd.keys():
            self.field_temporal_opacity_head_grd[name].zero_()
        for name in self.field_static_view_mapper_grd.keys():
            self.field_static_view_mapper_grd[name].zero_()
        for name in self.field_static_app_head_grd.keys():
            self.field_static_app_head_grd[name].zero_()

    def set_batch_gradient(self, cnt):
        ratio = 1/cnt
        self._features_dc.grad = self._features_dc_grd * ratio
        self._features_t.grad = self._features_t_grd * ratio 
        self._xyz.grad = self._xyz_grd * ratio
        self._scaling.grad = self._scaling_grd * ratio
        self._rotation.grad = self._rotation_grd * ratio
        self._opacity.grad = self._opacity_grd * ratio
        self._trbf_center.grad = self._trbf_center_grd * ratio
        self._trbf_scale.grad = self._trbf_scale_grd* ratio
        self._motion.grad = self._motion_grd * ratio
        self._omega.grad = self._omega_grd * ratio
        if self.use_euler_field and self._static_level_logits.numel() > 0:
            self._static_level_logits.grad = self._static_level_logits_grd * ratio
        if self.use_euler_field and self._dynamic_level_logits.numel() > 0:
            self._dynamic_level_logits.grad = self._dynamic_level_logits_grd * ratio
        if self.use_euler_field and self._dynamic_level_time_coeff.numel() > 0:
            self._dynamic_level_time_coeff.grad = self._dynamic_level_time_coeff_grd * ratio
        if self.use_euler_field and self._static_route_logits.numel() > 0:
            self._static_route_logits.grad = self._static_route_logits_grd * ratio
        if self.use_euler_field and self._field_residual_gate.numel() > 0:
            self._field_residual_gate.grad = self._field_residual_gate_grd * ratio

        if self.rgbdecoder is not None:
            for name, param in self.rgbdecoder.named_parameters():
                param.grad = self.rgb_grd[name] * ratio
        if self.use_euler_field and self.euler_field is not None:
            for name, param in self.euler_field.named_parameters():
                param.grad = self.field_grd[name] * ratio
        if self.use_euler_field and self.field_router is not None:
            for name, param in self.field_router.named_parameters():
                param.grad = self.field_router_grd[name] * ratio
        if self.use_euler_field and self.field_query_gate is not None:
            for name, param in self.field_query_gate.named_parameters():
                param.grad = self.field_query_gate_grd[name] * ratio
        if self.use_euler_field and self.field_decoder is not None:
            for name, param in self.field_decoder.named_parameters():
                param.grad = self.field_decoder_grd[name] * ratio
        if self.use_euler_field and self.field_temporal_opacity_head is not None:
            for name, param in self.field_temporal_opacity_head.named_parameters():
                param.grad = self.field_temporal_opacity_head_grd[name] * ratio
        if self.use_euler_field and self.field_static_view_mapper is not None:
            for name, param in self.field_static_view_mapper.named_parameters():
                param.grad = self.field_static_view_mapper_grd[name] * ratio
        if self.use_euler_field and self.field_static_app_head is not None:
            for name, param in self.field_static_app_head.named_parameters():
                param.grad = self.field_static_app_head_grd[name] * ratio


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if self.rgbdecoder is not None:
            self.rgbdecoder.cuda()
        if self.use_euler_field and self.euler_field is not None:
            self.euler_field.cuda()
            self.field_router.cuda()
            self.field_query_gate.cuda()
            self.field_decoder.cuda()
            if self.field_temporal_opacity_head is not None:
                self.field_temporal_opacity_head.cuda()
            if self.field_static_view_mapper is not None:
                self.field_static_view_mapper.cuda()
            if self.field_static_app_head is not None:
                self.field_static_app_head.cuda()
        self._init_module_grad_cache()
         # self._features_t
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_t], 'lr': training_args.featuret_lr, "name": "f_t"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._omega], 'lr': training_args.omega_lr, "name": "omega"},
            {'params': [self._trbf_center], 'lr': training_args.trbfc_lr, "name": "trbf_center"},
            {'params': [self._trbf_scale], 'lr': training_args.trbfs_lr, "name": "trbf_scale"},
            {'params': [self._motion], 'lr':  training_args.position_lr_init * self.spatial_lr_scale * 0.5 * training_args.movelr , "name": "motion"},
        ]
        if self.rgbdecoder is not None:
            l.append({'params': list(self.rgbdecoder.parameters()), 'lr': training_args.rgb_lr, "name": "decoder"})
        if self.use_euler_field and self.euler_field is not None:
            l.append({'params': [self._static_level_logits], 'lr': training_args.grid_logits_lr, "name": "static_grid_logits"})
            l.append({'params': [self._dynamic_level_logits], 'lr': training_args.grid_logits_lr, "name": "dynamic_grid_logits"})
            if self._dynamic_level_time_coeff.numel() > 0:
                l.append({'params': [self._dynamic_level_time_coeff], 'lr': training_args.grid_logits_lr, "name": "dynamic_grid_time_coeff"})
            if self._static_route_logits.numel() > 0 and self.field_static_route_mode == "learned":
                l.append({'params': [self._static_route_logits], 'lr': training_args.field_gate_lr, "name": "static_route_logits"})
            if self._field_residual_gate.numel() > 0:
                l.append({'params': [self._field_residual_gate], 'lr': training_args.field_gate_lr, "name": "field_gate"})
            l.extend([
                {'params': list(self.euler_field.parameters()), 'lr': training_args.field_lr, "name": "field"},
                {'params': list(self.field_router.parameters()), 'lr': training_args.field_decoder_lr, "name": "field_router"},
                {'params': list(self.field_query_gate.parameters()), 'lr': training_args.field_decoder_lr, "name": "field_query_gate"},
                {'params': list(self.field_decoder.parameters()), 'lr': training_args.field_decoder_lr, "name": "field_decoder"},
                {'params': list(self.field_temporal_opacity_head.parameters()), 'lr': training_args.field_decoder_lr, "name": "field_temporal_opacity"},
                {'params': list(self.field_static_view_mapper.parameters()), 'lr': training_args.field_decoder_lr, "name": "field_static_view_mapper"},
                {'params': list(self.field_static_app_head.parameters()), 'lr': training_args.field_decoder_lr, "name": "field_static_app"},
            ])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        print("move decoder to cuda")
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
    
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z','trbf_center', 'trbf_scale' ,'nx', 'ny', 'nz'] # 'trbf_center', 'trbf_scale' 
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        #     l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        #     l.append('f_rest_{}'.format(i))
        for i in range(self._motion.shape[1]):
            l.append('motion_{}'.format(i))

        for i in range(self._features_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1]):
        #     l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._omega.shape[1]):
            l.append('omega_{}'.format(i))
        


        for i in range(self._features_t.shape[1]):
            l.append('f_t_{}'.format(i))
        
        return l

    def _checkpoint_field_config(self):
        return {
            "use_euler_field": self.use_euler_field,
            "field_base_resolution": self.field_base_resolution,
            "field_num_levels": self.field_num_levels,
            "field_resolution_mode": self.field_resolution_mode,
            "field_level_resolutions": self.field_level_resolutions,
            "field_resolved_level_resolutions": self.field_resolved_level_resolutions,
            "field_resolution_growth": self.field_resolution_growth,
            "field_max_resolution": self.field_max_resolution,
            "field_knn_scale_percentile": self.field_knn_scale_percentile,
            "field_gaussian_scale_percentile": self.field_gaussian_scale_percentile,
            "field_pixel_scale_percentile": self.field_pixel_scale_percentile,
            "field_knn_scale_weight": self.field_knn_scale_weight,
            "field_gaussian_scale_weight": self.field_gaussian_scale_weight,
            "field_pixel_scale_weight": self.field_pixel_scale_weight,
            "field_min_cell_scale": self.field_min_cell_scale,
            "field_feature_dim": self.field_feature_dim,
            "field_fourier_degree": self.field_fourier_degree,
            "field_level_fourier_degree": self.field_level_fourier_degree,
            "field_decoder_hidden": self.field_decoder_hidden,
            "field_residual_mode": self.field_residual_mode,
            "field_query_mode": self.field_query_mode,
            "field_query_detach": int(self.field_query_detach),
            "field_query_gate_bias": self.field_query_gate_bias,
            "field_query_motion_scale": self.field_query_motion_scale,
            "field_dyn_threshold": self.field_dyn_threshold,
            "field_fast_threshold": self.field_fast_threshold,
            "field_dyn_slope": self.field_dyn_slope,
            "field_fast_slope": self.field_fast_slope,
            "field_fast_temperature": self.field_fast_temperature,
            "field_disable_dynamic_grid": int(self.field_disable_dynamic_grid),
            "field_v23_compat": int(self.field_v23_compat),
            "field_static_route_mode": self.field_static_route_mode,
            "field_static_route_init": self.field_static_route_init,
            "field_static_start_iter": self.field_static_start_iter,
            "field_static_warmup_iters": self.field_static_warmup_iters,
            "field_static_motion_scale": self.field_static_motion_scale,
            "field_static_opacity_scale": self.field_static_opacity_scale,
            "field_static_app_scale": self.field_static_app_scale,
            "field_static_use_global_gate": int(self.field_static_use_global_gate),
            "field_static_prior_floor": self.field_static_prior_floor,
            "field_soft_route_slope": self.field_soft_route_slope,
            "field_soft_static_threshold": self.field_soft_static_threshold,
            "field_soft_dynamic_threshold": self.field_soft_dynamic_threshold,
            "field_staged_training": int(self.field_staged_training),
            "field_disable_legacy_aux": int(self.field_disable_legacy_aux),
            "field_disable_ems_main": int(self.field_disable_ems_main),
            "field_disable_global_omega_split": int(self.field_disable_global_omega_split),
            "field_warmup_iters": self.field_warmup_iters,
            "field_problem_mining_start": self.field_problem_mining_start,
            "field_category_activate_iter": self.field_category_activate_iter,
            "field_activate_iter": self.field_activate_iter,
            "field_fast_activate_iter": self.field_fast_activate_iter,
            "field_mask_update_interval": self.field_mask_update_interval,
            "field_score_ema": self.field_score_ema,
            "field_visibility_ema": self.field_visibility_ema,
            "field_problem_error_weight": self.field_problem_error_weight,
            "field_problem_temporal_weight": self.field_problem_temporal_weight,
            "field_static_error_boost": self.field_static_error_boost,
            "field_time_center_ema": self.field_time_center_ema,
            "field_responsibility_on_threshold": self.field_responsibility_on_threshold,
            "field_responsibility_off_threshold": self.field_responsibility_off_threshold,
            "field_visibility_static_threshold": self.field_visibility_static_threshold,
            "field_slow_motion_on_threshold": self.field_slow_motion_on_threshold,
            "field_slow_motion_off_threshold": self.field_slow_motion_off_threshold,
            "field_dynamic_on_threshold": self.field_dynamic_on_threshold,
            "field_dynamic_off_threshold": self.field_dynamic_off_threshold,
            "field_motion_pre_threshold": self.field_motion_pre_threshold,
            "field_motion_accel_pre_threshold": self.field_motion_accel_pre_threshold,
            "field_static_pre_threshold": self.field_static_pre_threshold,
            "field_static_on_threshold": self.field_static_on_threshold,
            "field_static_off_threshold": self.field_static_off_threshold,
            "field_static_motion_threshold": self.field_static_motion_threshold,
            "field_static_accel_threshold": self.field_static_accel_threshold,
            "field_fast_on_threshold": self.field_fast_on_threshold,
            "field_fast_off_threshold": self.field_fast_off_threshold,
            "field_score_motion_weight": self.field_score_motion_weight,
            "field_score_accel_weight": self.field_score_accel_weight,
            "field_score_error_weight": self.field_score_error_weight,
            "field_score_screen_weight": self.field_score_screen_weight,
            "field_score_xyz_weight": self.field_score_xyz_weight,
            "field_score_static_residual_weight": self.field_score_static_residual_weight,
            "field_fast_score_motion_weight": self.field_fast_score_motion_weight,
            "field_fast_score_accel_weight": self.field_fast_score_accel_weight,
            "field_fast_score_error_weight": self.field_fast_score_error_weight,
            "field_fast_score_screen_weight": self.field_fast_score_screen_weight,
            "field_fast_score_xyz_weight": self.field_fast_score_xyz_weight,
            "field_fast_score_static_residual_weight": self.field_fast_score_static_residual_weight,
            "field_static_score_motion_weight": self.field_static_score_motion_weight,
            "field_static_score_accel_weight": self.field_static_score_accel_weight,
            "field_static_score_residual_weight": self.field_static_score_residual_weight,
            "field_fast_opacity_scale": self.field_fast_opacity_scale,
            "field_fast_motion_scale": self.field_fast_motion_scale,
            "field_temporal_refine": int(self.field_temporal_refine),
            "field_temporal_refine_start": self.field_temporal_refine_start,
            "field_temporal_refine_interval": self.field_temporal_refine_interval,
            "field_temporal_split_children": self.field_temporal_split_children,
            "field_temporal_center_offset": self.field_temporal_center_offset,
            "field_temporal_scale_shrink": self.field_temporal_scale_shrink,
            "field_fast_child_motion_scale": self.field_fast_child_motion_scale,
            "field_temporal_refine_opacity_threshold": self.field_temporal_refine_opacity_threshold,
            "field_temporal_refine_score_threshold": self.field_temporal_refine_score_threshold,
            "field_temporal_refine_max_ratio": self.field_temporal_refine_max_ratio,
        }

    def _load_aux_payload(self, path):
        ckpt = torch.load(path.replace(".ply", ".pt"), map_location="cpu")
        if isinstance(ckpt, dict) and "rgbdecoder" in ckpt:
            if self.rgbdecoder is not None and ckpt.get("rgbdecoder") is not None:
                self.rgbdecoder.load_state_dict(ckpt["rgbdecoder"])
            return ckpt
        if self.rgbdecoder is not None:
            self.rgbdecoder.load_state_dict(ckpt)
        return {}

    def _load_module_state_compatible(self, module, state_dict):
        module_state = module.state_dict()
        compatible_state = {}
        for key, value in state_dict.items():
            if key in module_state and module_state[key].shape == value.shape:
                compatible_state[key] = value
        module.load_state_dict(compatible_state, strict=False)

    def _apply_loaded_field_state(self, payload, num_points, bbox_min, bbox_max, mask=None, append=False):
        config = payload.get("field_config", {})
        if config:
            self.use_euler_field = bool(config.get("use_euler_field", self.use_euler_field))
            self.field_base_resolution = int(config.get("field_base_resolution", self.field_base_resolution))
            self.field_num_levels = int(config.get("field_num_levels", self.field_num_levels))
            self.field_resolution_mode = str(config.get("field_resolution_mode", self.field_resolution_mode))
            self.field_level_resolutions = config.get("field_level_resolutions", self.field_level_resolutions)
            self.field_resolved_level_resolutions = config.get(
                "field_resolved_level_resolutions",
                self.field_resolved_level_resolutions,
            )
            self.field_resolution_growth = float(config.get("field_resolution_growth", self.field_resolution_growth))
            self.field_max_resolution = int(config.get("field_max_resolution", self.field_max_resolution))
            self.field_knn_scale_percentile = float(config.get("field_knn_scale_percentile", self.field_knn_scale_percentile))
            self.field_gaussian_scale_percentile = float(config.get("field_gaussian_scale_percentile", self.field_gaussian_scale_percentile))
            self.field_pixel_scale_percentile = float(config.get("field_pixel_scale_percentile", self.field_pixel_scale_percentile))
            self.field_knn_scale_weight = float(config.get("field_knn_scale_weight", self.field_knn_scale_weight))
            self.field_gaussian_scale_weight = float(config.get("field_gaussian_scale_weight", self.field_gaussian_scale_weight))
            self.field_pixel_scale_weight = float(config.get("field_pixel_scale_weight", self.field_pixel_scale_weight))
            self.field_min_cell_scale = float(config.get("field_min_cell_scale", self.field_min_cell_scale))
            self.field_feature_dim = int(config.get("field_feature_dim", self.field_feature_dim))
            self.field_fourier_degree = int(config.get("field_fourier_degree", self.field_fourier_degree))
            self.field_level_fourier_degree = int(config.get("field_level_fourier_degree", self.field_level_fourier_degree))
            self.field_decoder_hidden = int(config.get("field_decoder_hidden", self.field_decoder_hidden))
            self.field_residual_mode = str(config.get("field_residual_mode", self.field_residual_mode))
            self.field_query_mode = str(config.get("field_query_mode", self.field_query_mode))
            self.field_query_detach = bool(config.get("field_query_detach", int(self.field_query_detach)))
            self.field_query_gate_bias = float(config.get("field_query_gate_bias", self.field_query_gate_bias))
            self.field_query_motion_scale = float(config.get("field_query_motion_scale", self.field_query_motion_scale))
            self.field_dyn_threshold = float(config.get("field_dyn_threshold", self.field_dyn_threshold))
            self.field_fast_threshold = float(config.get("field_fast_threshold", self.field_fast_threshold))
            self.field_dyn_slope = float(config.get("field_dyn_slope", self.field_dyn_slope))
            self.field_fast_slope = float(config.get("field_fast_slope", self.field_fast_slope))
            self.field_fast_temperature = float(config.get("field_fast_temperature", self.field_fast_temperature))
            self.field_disable_dynamic_grid = bool(config.get("field_disable_dynamic_grid", int(self.field_disable_dynamic_grid)))
            self.field_v23_compat = bool(config.get("field_v23_compat", int(self.field_v23_compat)))
            self.field_static_route_mode = str(config.get("field_static_route_mode", self.field_static_route_mode))
            self.field_static_route_init = float(config.get("field_static_route_init", self.field_static_route_init))
            self.field_static_start_iter = int(config.get("field_static_start_iter", self.field_static_start_iter))
            self.field_static_warmup_iters = int(config.get("field_static_warmup_iters", self.field_static_warmup_iters))
            self.field_static_motion_scale = float(config.get("field_static_motion_scale", self.field_static_motion_scale))
            self.field_static_opacity_scale = float(config.get("field_static_opacity_scale", self.field_static_opacity_scale))
            self.field_static_app_scale = float(config.get("field_static_app_scale", self.field_static_app_scale))
            self.field_static_use_global_gate = bool(config.get("field_static_use_global_gate", int(self.field_static_use_global_gate)))
            self.field_static_prior_floor = float(config.get("field_static_prior_floor", self.field_static_prior_floor))
            self.field_soft_route_slope = float(config.get("field_soft_route_slope", self.field_soft_route_slope))
            self.field_soft_static_threshold = float(config.get("field_soft_static_threshold", self.field_soft_static_threshold))
            self.field_soft_dynamic_threshold = float(config.get("field_soft_dynamic_threshold", self.field_soft_dynamic_threshold))
            self.field_staged_training = bool(config.get("field_staged_training", int(self.field_staged_training)))
            self.field_disable_legacy_aux = bool(config.get("field_disable_legacy_aux", int(self.field_disable_legacy_aux)))
            self.field_disable_ems_main = bool(config.get("field_disable_ems_main", int(self.field_disable_ems_main)))
            self.field_disable_global_omega_split = bool(config.get("field_disable_global_omega_split", int(self.field_disable_global_omega_split)))
            self.field_warmup_iters = int(config.get("field_warmup_iters", self.field_warmup_iters))
            self.field_problem_mining_start = int(config.get("field_problem_mining_start", self.field_problem_mining_start))
            self.field_category_activate_iter = int(config.get("field_category_activate_iter", self.field_category_activate_iter))
            self.field_activate_iter = int(config.get("field_activate_iter", self.field_activate_iter))
            self.field_fast_activate_iter = int(config.get("field_fast_activate_iter", self.field_fast_activate_iter))
            self.field_mask_update_interval = int(config.get("field_mask_update_interval", self.field_mask_update_interval))
            self.field_score_ema = float(config.get("field_score_ema", self.field_score_ema))
            self.field_visibility_ema = float(config.get("field_visibility_ema", self.field_visibility_ema))
            self.field_problem_error_weight = float(config.get("field_problem_error_weight", self.field_problem_error_weight))
            self.field_problem_temporal_weight = float(config.get("field_problem_temporal_weight", self.field_problem_temporal_weight))
            self.field_static_error_boost = float(config.get("field_static_error_boost", self.field_static_error_boost))
            self.field_time_center_ema = float(config.get("field_time_center_ema", self.field_time_center_ema))
            self.field_responsibility_on_threshold = float(config.get("field_responsibility_on_threshold", self.field_responsibility_on_threshold))
            self.field_responsibility_off_threshold = float(config.get("field_responsibility_off_threshold", self.field_responsibility_off_threshold))
            self.field_visibility_static_threshold = float(config.get("field_visibility_static_threshold", self.field_visibility_static_threshold))
            self.field_slow_motion_on_threshold = float(config.get("field_slow_motion_on_threshold", self.field_slow_motion_on_threshold))
            self.field_slow_motion_off_threshold = float(config.get("field_slow_motion_off_threshold", self.field_slow_motion_off_threshold))
            self.field_dynamic_on_threshold = float(config.get("field_dynamic_on_threshold", self.field_dynamic_on_threshold))
            self.field_dynamic_off_threshold = float(config.get("field_dynamic_off_threshold", self.field_dynamic_off_threshold))
            self.field_motion_pre_threshold = float(config.get("field_motion_pre_threshold", self.field_motion_pre_threshold))
            self.field_motion_accel_pre_threshold = float(config.get("field_motion_accel_pre_threshold", self.field_motion_accel_pre_threshold))
            self.field_static_pre_threshold = float(config.get("field_static_pre_threshold", self.field_static_pre_threshold))
            self.field_static_on_threshold = float(config.get("field_static_on_threshold", self.field_static_on_threshold))
            self.field_static_off_threshold = float(config.get("field_static_off_threshold", self.field_static_off_threshold))
            self.field_static_motion_threshold = float(config.get("field_static_motion_threshold", self.field_static_motion_threshold))
            self.field_static_accel_threshold = float(config.get("field_static_accel_threshold", self.field_static_accel_threshold))
            self.field_fast_on_threshold = float(config.get("field_fast_on_threshold", self.field_fast_on_threshold))
            self.field_fast_off_threshold = float(config.get("field_fast_off_threshold", self.field_fast_off_threshold))
            self.field_score_motion_weight = float(config.get("field_score_motion_weight", self.field_score_motion_weight))
            self.field_score_accel_weight = float(config.get("field_score_accel_weight", self.field_score_accel_weight))
            self.field_score_error_weight = float(config.get("field_score_error_weight", self.field_score_error_weight))
            self.field_score_screen_weight = float(config.get("field_score_screen_weight", self.field_score_screen_weight))
            self.field_score_xyz_weight = float(config.get("field_score_xyz_weight", self.field_score_xyz_weight))
            self.field_score_static_residual_weight = float(config.get("field_score_static_residual_weight", self.field_score_static_residual_weight))
            self.field_fast_score_motion_weight = float(config.get("field_fast_score_motion_weight", self.field_fast_score_motion_weight))
            self.field_fast_score_accel_weight = float(config.get("field_fast_score_accel_weight", self.field_fast_score_accel_weight))
            self.field_fast_score_error_weight = float(config.get("field_fast_score_error_weight", self.field_fast_score_error_weight))
            self.field_fast_score_screen_weight = float(config.get("field_fast_score_screen_weight", self.field_fast_score_screen_weight))
            self.field_fast_score_xyz_weight = float(config.get("field_fast_score_xyz_weight", self.field_fast_score_xyz_weight))
            self.field_fast_score_static_residual_weight = float(config.get("field_fast_score_static_residual_weight", self.field_fast_score_static_residual_weight))
            self.field_static_score_motion_weight = float(config.get("field_static_score_motion_weight", self.field_static_score_motion_weight))
            self.field_static_score_accel_weight = float(config.get("field_static_score_accel_weight", self.field_static_score_accel_weight))
            self.field_static_score_residual_weight = float(config.get("field_static_score_residual_weight", self.field_static_score_residual_weight))
            self.field_fast_opacity_scale = float(config.get("field_fast_opacity_scale", self.field_fast_opacity_scale))
            self.field_fast_motion_scale = float(config.get("field_fast_motion_scale", self.field_fast_motion_scale))
            self.field_temporal_refine = bool(config.get("field_temporal_refine", int(self.field_temporal_refine)))
            self.field_temporal_refine_start = int(config.get("field_temporal_refine_start", self.field_temporal_refine_start))
            self.field_temporal_refine_interval = int(config.get("field_temporal_refine_interval", self.field_temporal_refine_interval))
            self.field_temporal_split_children = int(config.get("field_temporal_split_children", self.field_temporal_split_children))
            self.field_temporal_center_offset = float(config.get("field_temporal_center_offset", self.field_temporal_center_offset))
            self.field_temporal_scale_shrink = float(config.get("field_temporal_scale_shrink", self.field_temporal_scale_shrink))
            self.field_fast_child_motion_scale = float(config.get("field_fast_child_motion_scale", self.field_fast_child_motion_scale))
            self.field_temporal_refine_opacity_threshold = float(config.get("field_temporal_refine_opacity_threshold", self.field_temporal_refine_opacity_threshold))
            self.field_temporal_refine_score_threshold = float(config.get("field_temporal_refine_score_threshold", self.field_temporal_refine_score_threshold))
            self.field_temporal_refine_max_ratio = float(config.get("field_temporal_refine_max_ratio", self.field_temporal_refine_max_ratio))

        if not self.use_euler_field:
            self._static_level_logits = torch.empty(0, device="cuda")
            self._dynamic_level_logits = torch.empty(0, device="cuda")
            self._dynamic_level_time_coeff = torch.empty(0, device="cuda")
            self._static_route_logits = torch.empty(0, device="cuda")
            self._field_residual_gate = torch.empty(0, device="cuda")
            self.field_static_view_mapper = None
            self.field_static_app_head = None
            return

        if self.euler_field is None or self.field_router is None or self.field_query_gate is None or self.field_decoder is None or self.field_temporal_opacity_head is None or self.field_static_view_mapper is None or self.field_static_app_head is None:
            self._build_euler_modules(bbox_min, bbox_max)

        if payload.get("euler_field") is not None:
            self.euler_field.load_state_dict(payload["euler_field"])
        if payload.get("field_router") is not None:
            self.field_router.load_state_dict(payload["field_router"])
        if payload.get("field_query_gate") is not None:
            self._load_module_state_compatible(self.field_query_gate, payload["field_query_gate"])
        if payload.get("field_decoder") is not None:
            self._load_module_state_compatible(self.field_decoder, payload["field_decoder"])
        if payload.get("field_temporal_opacity_head") is not None:
            self._load_module_state_compatible(self.field_temporal_opacity_head, payload["field_temporal_opacity_head"])
        if payload.get("field_static_view_mapper") is not None and self.field_static_view_mapper is not None:
            self._load_module_state_compatible(self.field_static_view_mapper, payload["field_static_view_mapper"])
        if payload.get("field_static_app_head") is not None and self.field_static_app_head is not None:
            self._load_module_state_compatible(self.field_static_app_head, payload["field_static_app_head"])

        saved_static_logits = payload.get("static_level_logits")
        if saved_static_logits is None:
            saved_static_logits = payload.get("grid_level_logits")
        saved_dynamic_logits = payload.get("dynamic_level_logits")
        if saved_dynamic_logits is None:
            saved_dynamic_logits = payload.get("grid_level_logits")
        saved_dynamic_time_coeff = payload.get("dynamic_level_time_coeff")
        if saved_dynamic_time_coeff is None:
            saved_dynamic_time_coeff = payload.get("grid_level_time_coeff")
        saved_static_route_logits = payload.get("static_route_logits")
        saved_gate = payload.get("field_residual_gate")
        if saved_static_logits is not None:
            if mask is not None:
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask.astype(np.bool_))
                elif torch.is_tensor(mask):
                    mask = mask.detach().cpu()
                saved_static_logits = saved_static_logits[mask]
                if saved_dynamic_logits is not None:
                    saved_dynamic_logits = saved_dynamic_logits[mask]
                if saved_dynamic_time_coeff is not None:
                    saved_dynamic_time_coeff = saved_dynamic_time_coeff[mask]
                if saved_static_route_logits is not None:
                    saved_static_route_logits = saved_static_route_logits[mask]
            static_logits = saved_static_logits.to(device="cuda", dtype=torch.float32)
        else:
            static_logits = torch.zeros((num_points, self.field_num_levels), device="cuda")

        if saved_dynamic_logits is not None:
            dynamic_logits = saved_dynamic_logits.to(device="cuda", dtype=torch.float32)
        else:
            dynamic_logits = torch.zeros((num_points, self.field_num_levels), device="cuda")

        if saved_dynamic_time_coeff is not None:
            dynamic_time_coeff = saved_dynamic_time_coeff.to(device="cuda", dtype=torch.float32)
        else:
            coeff_dim = 2 * self.field_level_fourier_degree
            dynamic_time_coeff = torch.zeros((num_points, self.field_num_levels, coeff_dim), device="cuda")
        if self.field_static_route_mode == "learned":
            if saved_static_route_logits is not None:
                static_route_logits = saved_static_route_logits.to(device="cuda", dtype=torch.float32)
            else:
                static_route_logits = torch.full(
                    (num_points, 1),
                    float(self.field_static_route_init),
                    device="cuda",
                    dtype=torch.float32,
                )
        else:
            static_route_logits = torch.empty(0, 1, device="cuda", dtype=torch.float32)
        if saved_gate is not None:
            gate = saved_gate.to(device="cuda", dtype=torch.float32)
        else:
            gate = torch.zeros((2,), device="cuda")
        target_gate_dim = 2
        if gate.shape[0] < target_gate_dim:
            gate = torch.cat((gate, torch.zeros((target_gate_dim - gate.shape[0],), device="cuda", dtype=gate.dtype)), dim=0)
        elif gate.shape[0] > target_gate_dim:
            gate = gate[:target_gate_dim]

        if append:
            if self._static_level_logits.numel() > 0:
                base_static_logits = self._static_level_logits.detach()
            else:
                base_static_logits = torch.zeros((0, self.field_num_levels), device="cuda")
            static_logits = torch.cat((base_static_logits, static_logits), dim=0)

            if self._dynamic_level_logits.numel() > 0:
                base_dynamic_logits = self._dynamic_level_logits.detach()
            else:
                base_dynamic_logits = torch.zeros((0, self.field_num_levels), device="cuda")
            dynamic_logits = torch.cat((base_dynamic_logits, dynamic_logits), dim=0)

            if self._dynamic_level_time_coeff.numel() > 0:
                base_time_coeff = self._dynamic_level_time_coeff.detach()
            else:
                coeff_dim = 2 * self.field_level_fourier_degree
                base_time_coeff = torch.zeros((0, self.field_num_levels, coeff_dim), device="cuda")
            dynamic_time_coeff = torch.cat((base_time_coeff, dynamic_time_coeff), dim=0)

            if self._static_route_logits.numel() > 0 and static_route_logits.numel() > 0:
                base_static_route_logits = self._static_route_logits.detach()
            elif static_route_logits.numel() > 0:
                base_static_route_logits = torch.zeros((0, 1), device="cuda")
            else:
                base_static_route_logits = None
            if base_static_route_logits is not None:
                static_route_logits = torch.cat((base_static_route_logits, static_route_logits), dim=0)

        self._static_level_logits = nn.Parameter(static_logits.requires_grad_(True))
        self._dynamic_level_logits = nn.Parameter(dynamic_logits.requires_grad_(True))
        self._dynamic_level_time_coeff = nn.Parameter(dynamic_time_coeff.requires_grad_(True))
        if static_route_logits.numel() > 0:
            self._static_route_logits = nn.Parameter(static_route_logits.requires_grad_(True))
        else:
            self._static_route_logits = torch.empty(0, device="cuda")
        self._field_residual_gate = nn.Parameter(gate.requires_grad_(True))
        saved_ems_mask = payload.get("error_prior")
        if saved_ems_mask is None:
            saved_ems_mask = payload.get("ems_mask")
        if saved_ems_mask is not None:
            if mask is not None:
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask.astype(np.bool_))
                elif torch.is_tensor(mask):
                    mask = mask.detach().cpu()
                saved_ems_mask = saved_ems_mask[mask]
            ems_mask = saved_ems_mask.to(device="cuda", dtype=torch.float32)
        else:
            ems_mask = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        if append:
            if self.maskforems is not None and self.maskforems.numel() > 0:
                ems_mask = torch.cat((self.maskforems.detach(), ems_mask), dim=0)
        self.maskforems = ems_mask
        saved_dynamic_score = payload.get("dynamic_score_ema")
        saved_dynamic_active = payload.get("dynamic_active_mask")
        saved_responsibility = payload.get("responsibility_ema")
        saved_responsibility_time = payload.get("responsibility_time_center_ema")
        saved_slow_score = payload.get("slow_motion_score_ema")
        saved_slow_mask = payload.get("slow_motion_mask")
        saved_fast_score = payload.get("fast_score_ema")
        saved_fast_active = payload.get("fast_active_mask")
        saved_static_support_score = payload.get("static_support_ema")
        saved_static_support_mask = payload.get("static_support_mask")
        saved_visibility = payload.get("visibility_persistence_ema")
        if saved_dynamic_score is not None:
            if mask is not None:
                saved_dynamic_score = saved_dynamic_score[mask]
            dynamic_score = saved_dynamic_score.to(device="cuda", dtype=torch.float32)
        else:
            dynamic_score = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        if saved_dynamic_active is not None:
            if mask is not None:
                saved_dynamic_active = saved_dynamic_active[mask]
            dynamic_active = saved_dynamic_active.to(device="cuda", dtype=torch.float32)
        else:
            dynamic_active = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        if saved_responsibility is not None:
            if mask is not None:
                saved_responsibility = saved_responsibility[mask]
            responsibility_score = saved_responsibility.to(device="cuda", dtype=torch.float32)
        else:
            responsibility_score = dynamic_score.clone()
        if saved_responsibility_time is not None:
            if mask is not None:
                saved_responsibility_time = saved_responsibility_time[mask]
            responsibility_time = saved_responsibility_time.to(device="cuda", dtype=torch.float32)
        else:
            responsibility_time = torch.full((num_points, 1), -1.0, device="cuda", dtype=torch.float32)
        if saved_slow_score is not None:
            if mask is not None:
                saved_slow_score = saved_slow_score[mask]
            slow_score = saved_slow_score.to(device="cuda", dtype=torch.float32)
        else:
            slow_score = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        if saved_slow_mask is not None:
            if mask is not None:
                saved_slow_mask = saved_slow_mask[mask]
            slow_active = saved_slow_mask.to(device="cuda", dtype=torch.float32)
        else:
            slow_active = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        if saved_fast_score is not None:
            if mask is not None:
                saved_fast_score = saved_fast_score[mask]
            fast_score = saved_fast_score.to(device="cuda", dtype=torch.float32)
        else:
            fast_score = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        if saved_fast_active is not None:
            if mask is not None:
                saved_fast_active = saved_fast_active[mask]
            fast_active = saved_fast_active.to(device="cuda", dtype=torch.float32)
        else:
            fast_active = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        if saved_static_support_score is not None:
            if mask is not None:
                saved_static_support_score = saved_static_support_score[mask]
            static_support_score = saved_static_support_score.to(device="cuda", dtype=torch.float32)
        else:
            static_support_score = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        if saved_static_support_mask is not None:
            if mask is not None:
                saved_static_support_mask = saved_static_support_mask[mask]
            static_support_mask = saved_static_support_mask.to(device="cuda", dtype=torch.float32)
        else:
            static_support_mask = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        if saved_visibility is not None:
            if mask is not None:
                saved_visibility = saved_visibility[mask]
            visibility_score = saved_visibility.to(device="cuda", dtype=torch.float32)
        else:
            visibility_score = torch.zeros((num_points, 1), device="cuda", dtype=torch.float32)
        if append:
            if self._dynamic_score_ema is not None and self._dynamic_score_ema.numel() > 0:
                dynamic_score = torch.cat((self._dynamic_score_ema.detach(), dynamic_score), dim=0)
            if self._dynamic_active_mask is not None and self._dynamic_active_mask.numel() > 0:
                dynamic_active = torch.cat((self._dynamic_active_mask.detach(), dynamic_active), dim=0)
            if self._responsibility_ema is not None and self._responsibility_ema.numel() > 0:
                responsibility_score = torch.cat((self._responsibility_ema.detach(), responsibility_score), dim=0)
            if self._responsibility_time_center_ema is not None and self._responsibility_time_center_ema.numel() > 0:
                responsibility_time = torch.cat((self._responsibility_time_center_ema.detach(), responsibility_time), dim=0)
            if self._slow_motion_score_ema is not None and self._slow_motion_score_ema.numel() > 0:
                slow_score = torch.cat((self._slow_motion_score_ema.detach(), slow_score), dim=0)
            if self._slow_motion_mask is not None and self._slow_motion_mask.numel() > 0:
                slow_active = torch.cat((self._slow_motion_mask.detach(), slow_active), dim=0)
            if self._fast_score_ema is not None and self._fast_score_ema.numel() > 0:
                fast_score = torch.cat((self._fast_score_ema.detach(), fast_score), dim=0)
            if self._fast_active_mask is not None and self._fast_active_mask.numel() > 0:
                fast_active = torch.cat((self._fast_active_mask.detach(), fast_active), dim=0)
            if self._static_support_ema is not None and self._static_support_ema.numel() > 0:
                static_support_score = torch.cat((self._static_support_ema.detach(), static_support_score), dim=0)
            if self._static_support_mask is not None and self._static_support_mask.numel() > 0:
                static_support_mask = torch.cat((self._static_support_mask.detach(), static_support_mask), dim=0)
            if self._visibility_persistence_ema is not None and self._visibility_persistence_ema.numel() > 0:
                visibility_score = torch.cat((self._visibility_persistence_ema.detach(), visibility_score), dim=0)
        self._dynamic_score_ema = dynamic_score
        self._dynamic_active_mask = dynamic_active
        self._responsibility_ema = responsibility_score
        self._responsibility_time_center_ema = responsibility_time
        self._slow_motion_score_ema = slow_score
        self._slow_motion_mask = slow_active
        self._fast_score_ema = fast_score
        self._fast_active_mask = fast_active
        self._static_support_ema = static_support_score
        self._static_support_mask = static_support_mask
        self._visibility_persistence_ema = visibility_score

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().cpu().numpy()
        #f_rest = self._features_rest.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()


        trbf_center= self._trbf_center.detach().cpu().numpy()

        trbf_scale = self._trbf_scale.detach().cpu().numpy()
        motion = self._motion.detach().cpu().numpy()

        omega = self._omega.detach().cpu().numpy()

        f_t =  self._features_t.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, trbf_center, trbf_scale, normals, motion, f_dc, opacities, scale, rotation, omega, f_t), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        model_fname = path.replace(".ply", ".pt")
        print(f'Saving model checkpoint to: {model_fname}')
        payload = {
            "rgbdecoder": self.rgbdecoder.state_dict() if self.rgbdecoder is not None else None,
            "field_config": self._checkpoint_field_config(),
            "static_level_logits": self._static_level_logits.detach().cpu() if self.use_euler_field and self._static_level_logits.numel() > 0 else None,
            "dynamic_level_logits": self._dynamic_level_logits.detach().cpu() if self.use_euler_field and self._dynamic_level_logits.numel() > 0 else None,
            "dynamic_level_time_coeff": self._dynamic_level_time_coeff.detach().cpu() if self.use_euler_field and self._dynamic_level_time_coeff.numel() > 0 else None,
            "static_route_logits": self._static_route_logits.detach().cpu() if self.use_euler_field and self._static_route_logits.numel() > 0 else None,
            "grid_level_logits": self._dynamic_level_logits.detach().cpu() if self.use_euler_field and self._dynamic_level_logits.numel() > 0 else None,
            "grid_level_time_coeff": self._dynamic_level_time_coeff.detach().cpu() if self.use_euler_field and self._dynamic_level_time_coeff.numel() > 0 else None,
            "field_residual_gate": self._field_residual_gate.detach().cpu() if self.use_euler_field and self._field_residual_gate.numel() > 0 else None,
            "euler_field": self.euler_field.state_dict() if self.use_euler_field and self.euler_field is not None else None,
            "field_router": self.field_router.state_dict() if self.use_euler_field and self.field_router is not None else None,
            "field_query_gate": self.field_query_gate.state_dict() if self.use_euler_field and self.field_query_gate is not None else None,
            "field_decoder": self.field_decoder.state_dict() if self.use_euler_field and self.field_decoder is not None else None,
            "field_temporal_opacity_head": self.field_temporal_opacity_head.state_dict() if self.use_euler_field and self.field_temporal_opacity_head is not None else None,
            "field_static_view_mapper": self.field_static_view_mapper.state_dict() if self.use_euler_field and self.field_static_view_mapper is not None else None,
            "field_static_app_head": self.field_static_app_head.state_dict() if self.use_euler_field and self.field_static_app_head is not None else None,
            "error_prior": self.maskforems.detach().cpu() if self.maskforems is not None and self.maskforems.numel() > 0 else None,
            "ems_mask": self.maskforems.detach().cpu() if self.maskforems is not None and self.maskforems.numel() > 0 else None,
            "dynamic_score_ema": self._dynamic_score_ema.detach().cpu() if self._dynamic_score_ema is not None and self._dynamic_score_ema.numel() > 0 else None,
            "dynamic_active_mask": self._dynamic_active_mask.detach().cpu() if self._dynamic_active_mask is not None and self._dynamic_active_mask.numel() > 0 else None,
            "responsibility_ema": self._responsibility_ema.detach().cpu() if self._responsibility_ema is not None and self._responsibility_ema.numel() > 0 else None,
            "responsibility_time_center_ema": self._responsibility_time_center_ema.detach().cpu() if self._responsibility_time_center_ema is not None and self._responsibility_time_center_ema.numel() > 0 else None,
            "slow_motion_score_ema": self._slow_motion_score_ema.detach().cpu() if self._slow_motion_score_ema is not None and self._slow_motion_score_ema.numel() > 0 else None,
            "slow_motion_mask": self._slow_motion_mask.detach().cpu() if self._slow_motion_mask is not None and self._slow_motion_mask.numel() > 0 else None,
            "fast_score_ema": self._fast_score_ema.detach().cpu() if self._fast_score_ema is not None and self._fast_score_ema.numel() > 0 else None,
            "fast_active_mask": self._fast_active_mask.detach().cpu() if self._fast_active_mask is not None and self._fast_active_mask.numel() > 0 else None,
            "static_support_ema": self._static_support_ema.detach().cpu() if self._static_support_ema is not None and self._static_support_ema.numel() > 0 else None,
            "static_support_mask": self._static_support_mask.detach().cpu() if self._static_support_mask is not None and self._static_support_mask.numel() > 0 else None,
            "visibility_persistence_ema": self._visibility_persistence_ema.detach().cpu() if self._visibility_persistence_ema is not None and self._visibility_persistence_ema.numel() > 0 else None,
        }
        torch.save(payload, model_fname)


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def zero_omega(self, threhold=0.15):
        scales = self.get_scaling
        omegamask = torch.sum(torch.abs(self._omega), dim=1) > threhold # default 
        scalemask = torch.max(scales, dim=1).values.unsqueeze(1) > 0.2
        scalemaskb = torch.max(scales, dim=1).values.unsqueeze(1) < 0.6
        pointopacity = self.get_opacity
        opacitymask = pointopacity > 0.7

        mask = torch.logical_and(torch.logical_and(omegamask.unsqueeze(1), scalemask), torch.logical_and(scalemaskb, opacitymask))
        omeganew = mask.float() * self._omega
        optimizable_tensors = self.replace_tensor_to_optimizer(omeganew, "omega")
        self._omega = optimizable_tensors["omega"]
        return mask
    def zero_omegabymotion(self, threhold=0.15):
        scales = self.get_scaling
        omegamask = torch.sum(torch.abs(self._motion[:, 0:3]), dim=1) > 0.3 #  #torch.sum(torch.abs(self._omega), dim=1) > threhold # default 
        scalemask = torch.max(scales, dim=1).values.unsqueeze(1) > 0.2
        scalemaskb = torch.max(scales, dim=1).values.unsqueeze(1) < 0.6
        pointopacity = self.get_opacity
        opacitymask = pointopacity > 0.7

        

        mask = torch.logical_and(torch.logical_and(omegamask.unsqueeze(1), scalemask), torch.logical_and(scalemaskb, opacitymask))
        
        
        omeganew = mask.float() * self._omega
        optimizable_tensors = self.replace_tensor_to_optimizer(omeganew, "omega")
        self._omega = optimizable_tensors["omega"]
        return mask


    def zero_omegav2(self, threhold=0.15):
        scales = self.get_scaling
        omegamask = torch.sum(torch.abs(self._omega), dim=1) > threhold # default 
        scalemask = torch.max(scales, dim=1).values.unsqueeze(1) > 0.2
        scalemaskb = torch.max(scales, dim=1).values.unsqueeze(1) < 0.6
        pointopacity = self.get_opacity
        opacitymask = pointopacity > 0.7

        mask = torch.logical_and(torch.logical_and(omegamask.unsqueeze(1), scalemask), torch.logical_and(scalemaskb, opacitymask))
        omeganew = mask.float() * self._omega
        rotationew = self.get_rotation(self.delta_t)


        optimizable_tensors = self.replace_tensor_to_optimizer(omeganew, "omega")
        self._omega = optimizable_tensors["omega"]


        optimizable_tensors = self.replace_tensor_to_optimizer(rotationew, "rotation")
        self._rotation = optimizable_tensors["rotation"]
        return mask

    def load_plyandminmax(self, path,  maxx, maxy, maxz,  minx, miny, minz):
        def logicalorlist(listoftensor):
            mask = None 
            for idx, ele in enumerate(listoftensor):
                if idx == 0 :
                    mask = ele 
                else:
                    mask = np.logical_or(mask, ele)
            return mask 

        plydata = PlyData.read(path)
        payload = self._load_aux_payload(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        trbf_center= np.asarray(plydata.elements[0]["trbf_center"])[..., np.newaxis]
        trbf_scale = np.asarray(plydata.elements[0]["trbf_scale"])[..., np.newaxis]

        motion_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("motion")]
        nummotion = 9
        motion = np.zeros((xyz.shape[0], nummotion))
        for i in range(nummotion):
            motion[:, i] = np.asarray(plydata.elements[0]["motion_"+str(i)])


        dc_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc")]
        num_dc_features = len(dc_f_names)

        features_dc = np.zeros((xyz.shape[0], num_dc_features))
        for i in range(num_dc_features):
            features_dc[:, i] = np.asarray(plydata.elements[0]["f_dc_"+str(i)])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])


        ft_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_t")]
        ftomegas = np.zeros((xyz.shape[0], len(ft_names)))
        for idx, attr_name in enumerate(ft_names):
            ftomegas[:, idx] = np.asarray(plydata.elements[0][attr_name])
      

        mask0 = xyz[:,0] > maxx.item()
        mask1 = xyz[:,1] > maxy.item()
        mask2 = xyz[:,2] > maxz.item()

        mask3 = xyz[:,0] < minx.item()
        mask4 = xyz[:,1] < miny.item()
        mask5 = xyz[:,2] < minz.item()
        mask =  logicalorlist([mask0, mask1, mask2, mask3, mask4, mask5])
        mask = np.logical_not(mask)

        
        
        self._xyz = nn.Parameter(torch.tensor(xyz[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_center = nn.Parameter(torch.tensor(trbf_center[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.tensor(trbf_scale[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._motion = nn.Parameter(torch.tensor(motion[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._omega = nn.Parameter(torch.tensor(omegas[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_t = nn.Parameter(torch.tensor(ftomegas[mask], dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        bbox_min = torch.amin(self._xyz.detach(), dim=0)
        bbox_max = torch.amax(self._xyz.detach(), dim=0)
        self._apply_loaded_field_state(payload, self._xyz.shape[0], bbox_min, bbox_max, mask=mask)

    def load_plyandminmaxY(self, path,  maxx, maxy, maxz,  minx, miny, minz):
        def logicalorlist(listoftensor):
            mask = None 
            for idx, ele in enumerate(listoftensor):
                if idx == 0 :
                    mask = ele 
                else:
                    mask = np.logical_or(mask, ele)
            return mask 

        plydata = PlyData.read(path)
        payload = self._load_aux_payload(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        trbf_center= np.asarray(plydata.elements[0]["trbf_center"])[..., np.newaxis]
        trbf_scale = np.asarray(plydata.elements[0]["trbf_scale"])[..., np.newaxis]

        motion_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("motion")]
        nummotion = 9
        motion = np.zeros((xyz.shape[0], nummotion))
        for i in range(nummotion):
            motion[:, i] = np.asarray(plydata.elements[0]["motion_"+str(i)])


        dc_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc")]
        num_dc_features = len(dc_f_names)

        features_dc = np.zeros((xyz.shape[0], num_dc_features))
        for i in range(num_dc_features):
            features_dc[:, i] = np.asarray(plydata.elements[0]["f_dc_"+str(i)])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        #assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])


        ft_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_t")]
        ftomegas = np.zeros((xyz.shape[0], len(ft_names)))
        for idx, attr_name in enumerate(ft_names):
            ftomegas[:, idx] = np.asarray(plydata.elements[0][attr_name])
      

        mask1 = xyz[:,1] > maxy.item()

        mask4 = xyz[:,1] < miny.item()
        mask =  logicalorlist([mask1 , mask4])
        mask = np.logical_not(mask)

        
        
        self._xyz = nn.Parameter(torch.tensor(xyz[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_center = nn.Parameter(torch.tensor(trbf_center[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.tensor(trbf_scale[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._motion = nn.Parameter(torch.tensor(motion[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._omega = nn.Parameter(torch.tensor(omegas[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_t = nn.Parameter(torch.tensor(ftomegas[mask], dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        bbox_min = torch.amin(self._xyz.detach(), dim=0)
        bbox_max = torch.amax(self._xyz.detach(), dim=0)
        self._apply_loaded_field_state(payload, self._xyz.shape[0], bbox_min, bbox_max, mask=mask)


    def load_plyandminmaxall(self, path,  maxx, maxy, maxz,  minx, miny, minz):
        def logicalorlist(listoftensor):
            mask = None 
            for idx, ele in enumerate(listoftensor):
                if idx == 0 :
                    mask = ele 
                else:
                    mask = np.logical_or(mask, ele)
            return mask 

        plydata = PlyData.read(path)
        payload = self._load_aux_payload(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        trbf_center= np.asarray(plydata.elements[0]["trbf_center"])[..., np.newaxis]
        trbf_scale = np.asarray(plydata.elements[0]["trbf_scale"])[..., np.newaxis]

        motion_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("motion")]
        nummotion = 9
        motion = np.zeros((xyz.shape[0], nummotion))
        for i in range(nummotion):
            motion[:, i] = np.asarray(plydata.elements[0]["motion_"+str(i)])


        dc_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc")]
        num_dc_features = len(dc_f_names)

        features_dc = np.zeros((xyz.shape[0], num_dc_features))
        for i in range(num_dc_features):
            features_dc[:, i] = np.asarray(plydata.elements[0]["f_dc_"+str(i)])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])


        ft_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_t")]
        ftomegas = np.zeros((xyz.shape[0], len(ft_names)))
        for idx, attr_name in enumerate(ft_names):
            ftomegas[:, idx] = np.asarray(plydata.elements[0][attr_name])
      

        mask0 = xyz[:,0] > maxx.item()
        mask1 = xyz[:,1] > maxy.item()
        mask2 = xyz[:,2] > maxz.item()

        mask3 = xyz[:,0] < minx.item()
        mask4 = xyz[:,1] < miny.item()
        mask5 = xyz[:,2] < minz.item()
        mask =  logicalorlist([mask0, mask1, mask2, mask3, mask4, mask5])
        #mask = np.logical_not(mask)# now the reset point is within the boundray

        unstablepoints = np.sum(np.abs(motion[:, 0:3]),axis=1) 
        movingpoints = unstablepoints > 0.03
        trbfmask = trbf_scale < 3 # temporal unstable points

        maskst = np.logical_or(trbfmask.squeeze(1), movingpoints)

        mask = np.logical_or(mask, maskst) # only use large tscale points.
        # replace points with input ?

        mask  = np.logical_not(mask)# remaining good points. todo remove good mask's NN 

        xyz = torch.cat((self._xyz, torch.tensor(xyz[mask], dtype=torch.float, device="cuda")))
        
        self._xyz = nn.Parameter(xyz.requires_grad_(True))

        features_dc= torch.cat((self._features_dc, torch.tensor(features_dc[mask], dtype=torch.float, device="cuda")))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))

        opacities = torch.cat((self._opacity, torch.tensor(opacities[mask], dtype=torch.float, device="cuda")))
        self._opacity = nn.Parameter(opacities).requires_grad_(True)

        scales = torch.cat((self._scaling, torch.tensor(scales[mask], dtype=torch.float, device="cuda")))

        self._scaling = nn.Parameter(scales).requires_grad_(True)
        rots = torch.cat((self._rotation, torch.tensor(rots[mask], dtype=torch.float, device="cuda")))

        self._rotation = nn.Parameter(rots).requires_grad_(True)
        trbf_center =  torch.cat((self._trbf_center, torch.tensor(trbf_center[mask], dtype=torch.float, device="cuda")))
        self._trbf_center = nn.Parameter(trbf_center).requires_grad_(True)
        trbf_scale =  torch.cat((self._trbf_scale, torch.tensor(trbf_scale[mask], dtype=torch.float, device="cuda")))


        self._trbf_scale = nn.Parameter(trbf_scale.requires_grad_(True))

        motion =  torch.cat((self._motion, torch.tensor(motion[mask], dtype=torch.float, device="cuda")))

        self._motion = nn.Parameter(motion.requires_grad_(True))
        omegas = torch.cat((self._omega, torch.tensor(omegas[mask], dtype=torch.float, device="cuda")))
        self._omega = nn.Parameter(omegas.requires_grad_(True))

        ftomegas = torch.cat((self._features_t, torch.tensor(ftomegas[mask], dtype=torch.float, device="cuda")))
        self._features_t = nn.Parameter(ftomegas.requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        bbox_min = torch.amin(self._xyz.detach(), dim=0)
        bbox_max = torch.amax(self._xyz.detach(), dim=0)
        self._apply_loaded_field_state(payload, int(np.sum(mask)), bbox_min, bbox_max, mask=mask, append=True)
    def load_ply(self, path):
        plydata = PlyData.read(path)
        payload = self._load_aux_payload(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        trbf_center= np.asarray(plydata.elements[0]["trbf_center"])[..., np.newaxis]
        trbf_scale = np.asarray(plydata.elements[0]["trbf_scale"])[..., np.newaxis]

        motion_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("motion")]
        nummotion = 9
        motion = np.zeros((xyz.shape[0], nummotion))
        for i in range(nummotion):
            motion[:, i] = np.asarray(plydata.elements[0]["motion_"+str(i)])


        dc_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc")]
        num_dc_features = len(dc_f_names)

        features_dc = np.zeros((xyz.shape[0], num_dc_features))
        for i in range(num_dc_features):
            features_dc[:, i] = np.asarray(plydata.elements[0]["f_dc_"+str(i)])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        #assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])


        ft_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_t")]
        ftomegas = np.zeros((xyz.shape[0], len(ft_names)))
        for idx, attr_name in enumerate(ft_names):
            ftomegas[:, idx] = np.asarray(plydata.elements[0][attr_name])



        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_center = nn.Parameter(torch.tensor(trbf_center, dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.tensor(trbf_scale, dtype=torch.float, device="cuda").requires_grad_(True))
        self._motion = nn.Parameter(torch.tensor(motion, dtype=torch.float, device="cuda").requires_grad_(True))
        self._omega = nn.Parameter(torch.tensor(omegas, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_t = nn.Parameter(torch.tensor(ftomegas, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        
        self.computedopacity =self.opacity_activation(self._opacity)
        self.computedscales = torch.exp(self._scaling) # change not very large
        self.computedtrbfscale = torch.exp(self._trbf_scale) 
        bbox_min = torch.amin(self._xyz.detach(), dim=0)
        bbox_max = torch.amax(self._xyz.detach(), dim=0)
        self._apply_loaded_field_state(payload, self._xyz.shape[0], bbox_min, bbox_max)

        

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) == 1 and group["name"] not in ['decoder', 'field_gate', 'field_temporal_opacity']:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        self._features_t = optimizable_tensors["f_t"]
        if self.use_euler_field and "static_grid_logits" in optimizable_tensors:
            self._static_level_logits = optimizable_tensors["static_grid_logits"]
        if self.use_euler_field and "dynamic_grid_logits" in optimizable_tensors:
            self._dynamic_level_logits = optimizable_tensors["dynamic_grid_logits"]
        if self.use_euler_field and "dynamic_grid_time_coeff" in optimizable_tensors:
            self._dynamic_level_time_coeff = optimizable_tensors["dynamic_grid_time_coeff"]
        if self.use_euler_field and "static_route_logits" in optimizable_tensors:
            self._static_route_logits = optimizable_tensors["static_route_logits"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.omegamask is not None:
            if self.omegamask.shape[0] == valid_points_mask.shape[0]:
                self.omegamask = self.omegamask[valid_points_mask]
            else:
                self.omegamask = None
        if self.maskforems is not None and self.maskforems.numel() > 0:
            self.maskforems = self.maskforems[valid_points_mask]
        if self._dynamic_score_ema is not None and self._dynamic_score_ema.numel() > 0:
            self._dynamic_score_ema = self._dynamic_score_ema[valid_points_mask]
        if self._dynamic_active_mask is not None and self._dynamic_active_mask.numel() > 0:
            self._dynamic_active_mask = self._dynamic_active_mask[valid_points_mask]
        if self._responsibility_ema is not None and self._responsibility_ema.numel() > 0:
            self._responsibility_ema = self._responsibility_ema[valid_points_mask]
        if self._responsibility_time_center_ema is not None and self._responsibility_time_center_ema.numel() > 0:
            self._responsibility_time_center_ema = self._responsibility_time_center_ema[valid_points_mask]
        if self._slow_motion_score_ema is not None and self._slow_motion_score_ema.numel() > 0:
            self._slow_motion_score_ema = self._slow_motion_score_ema[valid_points_mask]
        if self._slow_motion_mask is not None and self._slow_motion_mask.numel() > 0:
            self._slow_motion_mask = self._slow_motion_mask[valid_points_mask]
        if self._fast_score_ema is not None and self._fast_score_ema.numel() > 0:
            self._fast_score_ema = self._fast_score_ema[valid_points_mask]
        if self._fast_active_mask is not None and self._fast_active_mask.numel() > 0:
            self._fast_active_mask = self._fast_active_mask[valid_points_mask]
        if self._static_support_ema is not None and self._static_support_ema.numel() > 0:
            self._static_support_ema = self._static_support_ema[valid_points_mask]
        if self._static_support_mask is not None and self._static_support_mask.numel() > 0:
            self._static_support_mask = self._static_support_mask[valid_points_mask]
        if self._visibility_persistence_ema is not None and self._visibility_persistence_ema.numel() > 0:
            self._visibility_persistence_ema = self._visibility_persistence_ema[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) == 1 and group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation, new_trbf_center, new_trbfscale, new_motion, new_omega, new_featuret, new_static_level_logits=None, new_dynamic_level_logits=None, new_dynamic_level_time_coeff=None, new_ems_mask=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "trbf_center" : new_trbf_center,
        "trbf_scale" : new_trbfscale,
        "motion": new_motion,
        "omega": new_omega,
        "f_t": new_featuret}
        if self.use_euler_field and new_static_level_logits is not None:
            d["static_grid_logits"] = new_static_level_logits
        if self.use_euler_field and new_dynamic_level_logits is not None:
            d["dynamic_grid_logits"] = new_dynamic_level_logits
        if self.use_euler_field and new_dynamic_level_time_coeff is not None:
            d["dynamic_grid_time_coeff"] = new_dynamic_level_time_coeff
        if self.use_euler_field and self.field_static_route_mode == "learned" and self._static_route_logits.numel() > 0:
            d["static_route_logits"] = torch.full(
                (new_xyz.shape[0], 1),
                float(self.field_static_route_init),
                device="cuda",
                dtype=torch.float32,
            )

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_t = optimizable_tensors["f_t"]
        #self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        if self.use_euler_field and "static_grid_logits" in optimizable_tensors:
            self._static_level_logits = optimizable_tensors["static_grid_logits"]
        if self.use_euler_field and "dynamic_grid_logits" in optimizable_tensors:
            self._dynamic_level_logits = optimizable_tensors["dynamic_grid_logits"]
        if self.use_euler_field and "dynamic_grid_time_coeff" in optimizable_tensors:
            self._dynamic_level_time_coeff = optimizable_tensors["dynamic_grid_time_coeff"]
        if self.use_euler_field and "static_route_logits" in optimizable_tensors:
            self._static_route_logits = optimizable_tensors["static_route_logits"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if self.omegamask is not None:
            new_omegamask = torch.zeros((new_xyz.shape[0], 1), device="cuda", dtype=torch.bool)
            self.omegamask = torch.cat((self.omegamask, new_omegamask), dim=0)
        if new_ems_mask is None:
            new_ems_mask = torch.zeros((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        if self.maskforems is None or self.maskforems.numel() == 0:
            self.maskforems = new_ems_mask
        else:
            self.maskforems = torch.cat((self.maskforems, new_ems_mask), dim=0)
        new_dynamic_score = torch.zeros((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        new_dynamic_active = torch.zeros((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        new_responsibility = torch.zeros((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        new_responsibility_time = torch.full((new_xyz.shape[0], 1), -1.0, device="cuda", dtype=torch.float32)
        new_slow_score = torch.zeros((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        new_slow_mask = torch.zeros((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        new_fast_score = torch.zeros((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        new_fast_active = torch.zeros((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        new_static_support_score = torch.zeros((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        new_static_support_mask = torch.zeros((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        new_visibility = torch.zeros((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        if self._dynamic_score_ema is None or self._dynamic_score_ema.numel() == 0:
            self._dynamic_score_ema = new_dynamic_score
        else:
            self._dynamic_score_ema = torch.cat((self._dynamic_score_ema, new_dynamic_score), dim=0)
        if self._dynamic_active_mask is None or self._dynamic_active_mask.numel() == 0:
            self._dynamic_active_mask = new_dynamic_active
        else:
            self._dynamic_active_mask = torch.cat((self._dynamic_active_mask, new_dynamic_active), dim=0)
        if self._responsibility_ema is None or self._responsibility_ema.numel() == 0:
            self._responsibility_ema = new_responsibility
        else:
            self._responsibility_ema = torch.cat((self._responsibility_ema, new_responsibility), dim=0)
        if self._responsibility_time_center_ema is None or self._responsibility_time_center_ema.numel() == 0:
            self._responsibility_time_center_ema = new_responsibility_time
        else:
            self._responsibility_time_center_ema = torch.cat((self._responsibility_time_center_ema, new_responsibility_time), dim=0)
        if self._slow_motion_score_ema is None or self._slow_motion_score_ema.numel() == 0:
            self._slow_motion_score_ema = new_slow_score
        else:
            self._slow_motion_score_ema = torch.cat((self._slow_motion_score_ema, new_slow_score), dim=0)
        if self._slow_motion_mask is None or self._slow_motion_mask.numel() == 0:
            self._slow_motion_mask = new_slow_mask
        else:
            self._slow_motion_mask = torch.cat((self._slow_motion_mask, new_slow_mask), dim=0)
        if self._fast_score_ema is None or self._fast_score_ema.numel() == 0:
            self._fast_score_ema = new_fast_score
        else:
            self._fast_score_ema = torch.cat((self._fast_score_ema, new_fast_score), dim=0)
        if self._fast_active_mask is None or self._fast_active_mask.numel() == 0:
            self._fast_active_mask = new_fast_active
        else:
            self._fast_active_mask = torch.cat((self._fast_active_mask, new_fast_active), dim=0)
        if self._static_support_ema is None or self._static_support_ema.numel() == 0:
            self._static_support_ema = new_static_support_score
        else:
            self._static_support_ema = torch.cat((self._static_support_ema, new_static_support_score), dim=0)
        if self._static_support_mask is None or self._static_support_mask.numel() == 0:
            self._static_support_mask = new_static_support_mask
        else:
            self._static_support_mask = torch.cat((self._static_support_mask, new_static_support_mask), dim=0)
        if self._visibility_persistence_ema is None or self._visibility_persistence_ema.numel() == 0:
            self._visibility_persistence_ema = new_visibility
        else:
            self._visibility_persistence_ema = torch.cat((self._visibility_persistence_ema, new_visibility), dim=0)

    

    def densify_and_splitv2(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1) # n,1,1 to n1
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        parent_trbf_center = self._trbf_center[selected_pts_mask]
        parent_trbf_scale = self._trbf_scale[selected_pts_mask]
        parent_motion = self._motion[selected_pts_mask]
        parent_ems_mask = self.maskforems[selected_pts_mask] if self.maskforems is not None and self.maskforems.numel() > 0 else None
        new_trbf_center, new_trbf_scale = self._get_temporal_child_support(
            parent_trbf_center,
            parent_trbf_scale,
            parent_motion,
            error_prior=parent_ems_mask,
            copies_per_parent=N,
        )
        new_motion = parent_motion.repeat(N,1)
        new_omega = self._omega[selected_pts_mask].repeat(N,1)
        new_feature_t = self._features_t[selected_pts_mask].repeat(N,1)
        new_static_level_logits = None
        new_dynamic_level_logits = None
        new_dynamic_level_time_coeff = None
        if self.use_euler_field:
            new_static_level_logits = self._static_level_logits[selected_pts_mask].repeat(N,1)
            new_dynamic_level_logits = self._dynamic_level_logits[selected_pts_mask].repeat(N,1)
            if self._dynamic_level_time_coeff.numel() > 0:
                new_dynamic_level_time_coeff = self._dynamic_level_time_coeff[selected_pts_mask].repeat(N,1,1)
        new_ems_mask = parent_ems_mask.repeat(N,1) * 0.75 if parent_ems_mask is not None else None

        self.densification_postfix(new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation, new_trbf_center, new_trbf_scale, new_motion, new_omega, new_feature_t, new_static_level_logits, new_dynamic_level_logits, new_dynamic_level_time_coeff, new_ems_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    


    def densify_and_splitim(self, grads, grad_threshold, scene_extent, N=2):  # numpy bmm, change parameter, no random.
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        # new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        numpytmp = rots.cpu().numpy() @ samples.unsqueeze(-1).cpu().numpy() # numpy better than cublas..., cublas use stohastic for bmm 
        new_xyz =torch.from_numpy(numpytmp).cuda().squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.55*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1) # n,1,1 to n1
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        parent_trbf_center = self._trbf_center[selected_pts_mask]
        parent_trbf_scale = self._trbf_scale[selected_pts_mask]
        parent_motion = self._motion[selected_pts_mask]
        parent_ems_mask = self.maskforems[selected_pts_mask] if self.maskforems is not None and self.maskforems.numel() > 0 else None
        new_trbf_center, new_trbf_scale = self._get_temporal_child_support(
            parent_trbf_center,
            parent_trbf_scale,
            parent_motion,
            error_prior=parent_ems_mask,
            copies_per_parent=N,
        )
        new_motion = parent_motion.repeat(N,1)
        new_omega = self._omega[selected_pts_mask].repeat(N,1)
        new_feature_t = self._features_t[selected_pts_mask].repeat(N,1)
        new_static_level_logits = None
        new_dynamic_level_logits = None
        new_dynamic_level_time_coeff = None
        if self.use_euler_field:
            new_static_level_logits = self._static_level_logits[selected_pts_mask].repeat(N,1)
            new_dynamic_level_logits = self._dynamic_level_logits[selected_pts_mask].repeat(N,1)
            if self._dynamic_level_time_coeff.numel() > 0:
                new_dynamic_level_time_coeff = self._dynamic_level_time_coeff[selected_pts_mask].repeat(N,1,1)
        new_ems_mask = parent_ems_mask.repeat(N,1) * 0.75 if parent_ems_mask is not None else None

        self.densification_postfix(new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation, new_trbf_center, new_trbf_scale, new_motion, new_omega, new_feature_t, new_static_level_logits, new_dynamic_level_logits, new_dynamic_level_time_coeff, new_ems_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2): #  numpy bmm for rotation and no random
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        # new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        numpytmp = rots.cpu().numpy() @ samples.unsqueeze(-1).cpu().numpy() # numpy better than cublas..., cublas use stohastic for bmm 
        new_xyz =torch.from_numpy(numpytmp).cuda().squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1) # n,1,1 to n1
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        parent_trbf_center = self._trbf_center[selected_pts_mask]
        parent_trbf_scale = self._trbf_scale[selected_pts_mask]
        parent_motion = self._motion[selected_pts_mask]
        parent_ems_mask = self.maskforems[selected_pts_mask] if self.maskforems is not None and self.maskforems.numel() > 0 else None
        new_trbf_center, new_trbf_scale = self._get_temporal_child_support(
            parent_trbf_center,
            parent_trbf_scale,
            parent_motion,
            error_prior=parent_ems_mask,
            copies_per_parent=N,
        )
        new_motion = parent_motion.repeat(N,1)
        new_omega = self._omega[selected_pts_mask].repeat(N,1)
        new_feature_t = self._features_t[selected_pts_mask].repeat(N,1)
        new_static_level_logits = None
        new_dynamic_level_logits = None
        new_dynamic_level_time_coeff = None
        if self.use_euler_field:
            new_static_level_logits = self._static_level_logits[selected_pts_mask].repeat(N,1)
            new_dynamic_level_logits = self._dynamic_level_logits[selected_pts_mask].repeat(N,1)
            if self._dynamic_level_time_coeff.numel() > 0:
                new_dynamic_level_time_coeff = self._dynamic_level_time_coeff[selected_pts_mask].repeat(N,1,1)
        new_ems_mask = parent_ems_mask.repeat(N,1) * 0.75 if parent_ems_mask is not None else None

        self.densification_postfix(new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation, new_trbf_center, new_trbf_scale, new_motion, new_omega, new_feature_t, new_static_level_logits, new_dynamic_level_logits, new_dynamic_level_time_coeff, new_ems_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_motion = self._motion[selected_pts_mask]
        new_trbf_center = self._trbf_center[selected_pts_mask]
        new_trbfscale = self._trbf_scale[selected_pts_mask]
        new_omega = self._omega[selected_pts_mask]
        new_featuret = self._features_t[selected_pts_mask]
        N, c= new_featuret.shape
        new_static_level_logits = None
        new_dynamic_level_logits = None
        new_dynamic_level_time_coeff = None
        if self.use_euler_field:
            new_static_level_logits = self._static_level_logits[selected_pts_mask]
            new_dynamic_level_logits = self._dynamic_level_logits[selected_pts_mask]
            if self._dynamic_level_time_coeff.numel() > 0:
                new_dynamic_level_time_coeff = self._dynamic_level_time_coeff[selected_pts_mask]
        new_ems_mask = self.maskforems[selected_pts_mask] if self.maskforems is not None and self.maskforems.numel() > 0 else None
        self.densification_postfix(new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation, new_trbf_center, new_trbfscale, new_motion, new_omega, new_featuret, new_static_level_logits, new_dynamic_level_logits, new_dynamic_level_time_coeff, new_ems_mask)


    def densify_and_cloneim(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        # new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_motion = self._motion[selected_pts_mask]
        new_trbf_center = self._trbf_center[selected_pts_mask]
        new_trbfscale = self._trbf_scale[selected_pts_mask]
        new_omega = self._omega[selected_pts_mask]
        new_featuret = self._features_t[selected_pts_mask]
        N, c= new_featuret.shape
        #self.trbfoutput = torch.cat((self.trbfoutput, torch.zeros(N , 1).to(self.trbfoutput)))
        new_static_level_logits = None
        new_dynamic_level_logits = None
        new_dynamic_level_time_coeff = None
        if self.use_euler_field:
            new_static_level_logits = self._static_level_logits[selected_pts_mask]
            new_dynamic_level_logits = self._dynamic_level_logits[selected_pts_mask]
            if self._dynamic_level_time_coeff.numel() > 0:
                new_dynamic_level_time_coeff = self._dynamic_level_time_coeff[selected_pts_mask]
        new_ems_mask = self.maskforems[selected_pts_mask] if self.maskforems is not None and self.maskforems.numel() > 0 else None
        self.densification_postfix(new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation, new_trbf_center, new_trbfscale, new_motion, new_omega, new_featuret, new_static_level_logits, new_dynamic_level_logits, new_dynamic_level_time_coeff, new_ems_mask)




  
    def densify_prunecloneim(self, max_grad, min_opacity, extent, max_screen_size, splitN=1):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        print("befre clone", self._xyz.shape[0])
        self.densify_and_cloneim(grads, max_grad, extent)
        print("after clone", self._xyz.shape[0])

        self.densify_and_splitim(grads, max_grad, extent, 2)
        print("after split", self._xyz.shape[0])

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size  
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        torch.cuda.empty_cache()

    def densify_pruneclone(self, max_grad, min_opacity, extent, max_screen_size, splitN=1):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        print("befre clone", self._xyz.shape[0])
        self.densify_and_clone(grads, max_grad, extent)
        print("after clone", self._xyz.shape[0])

        self.densify_and_splitv2(grads, max_grad, extent, 2)
        print("after split", self._xyz.shape[0])

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size  
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        torch.cuda.empty_cache()
    # this is not using random and use numpy bmm for densify
    def densify_prunecloneimgeneral(self, max_grad, min_opacity, extent, max_screen_size, splitN=1):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        print("befre clone", self._xyz.shape[0])
        self.densify_and_cloneim(grads, max_grad, extent)
        print("after clone", self._xyz.shape[0])

        self.densify_and_split(grads, max_grad, extent, 2)
        print("after split", self._xyz.shape[0])

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size  
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1



    def addgaussians(self, baduvidx, viewpoint_cam, depthmap, gt_image, numperay=3, ratioend=2, trbfcenter=0.5,depthmax=None,shuffle=False):
        def pix2ndc(v, S):
            return (v * 2.0 + 1.0) / S - 1.0
        ratiaolist = torch.linspace(self.raystart, ratioend, numperay) # 0.7 to ratiostart
        rgbs = gt_image[:, baduvidx[:,0], baduvidx[:,1]]
        rgbs = rgbs.permute(1,0)
        featuredc = torch.cat((rgbs, torch.zeros_like(rgbs)), dim=1)# should we add the feature dc with non zero values?

        depths = depthmap[:, baduvidx[:,0], baduvidx[:,1]]
        depths = depths.permute(1,0) # only use depth map > 15 .

        depths = torch.ones_like(depths) * depthmax # use the max local depth for the scene ?

        
        u = baduvidx[:,0] # hight y
        v = baduvidx[:,1] # weidth  x 
        Npoints = u.shape[0]
          
        new_xyz = []
        new_scaling = []
        new_rotation = []
        new_features_dc = []
        new_opacity = []
        new_trbf_center = []
        new_trbf_scale = []
        new_motion = []
        new_omega = []
        new_featuret = [ ]
        new_static_level_logits = []
        new_dynamic_level_logits = []
        new_dynamic_level_time_coeff = []

        camera2wold = viewpoint_cam.world_view_transform.T.inverse()
        projectinverse = viewpoint_cam.projection_matrix.T.inverse()
        maxz, minz = self.maxz, self.minz 
        maxy, miny = self.maxy, self.miny 
        maxx, minx = self.maxx, self.minx  
        

        for zscale in ratiaolist :
            ndcu, ndcv = pix2ndc(u, viewpoint_cam.image_height), pix2ndc(v, viewpoint_cam.image_width)
            # targetPz = depths*zscale # depth in local cameras..
            if shuffle == True:
                randomdepth = torch.rand_like(depths) - 0.5 # -0.5 to 0.5
                targetPz = (depths + depths/10*(randomdepth)) *zscale 
            else:
                targetPz = depths*zscale # depth in local cameras..
            
            ndcu = ndcu.unsqueeze(1)
            ndcv = ndcv.unsqueeze(1)


            ndccamera = torch.cat((ndcv, ndcu,   torch.ones_like(ndcu) * (1.0) , torch.ones_like(ndcu)), 1) # N,4 ...
            
            localpointuv = ndccamera @ projectinverse.T 

            diretioninlocal = localpointuv / localpointuv[:,3:] # ray direction in camera space 


            rate = targetPz / diretioninlocal[:, 2:3] #  
            
            localpoint = diretioninlocal * rate

            localpoint[:, -1] = 1
            
            
            worldpointH = localpoint @ camera2wold.T  #myproduct4x4batch(localpoint, camera2wold) # 
            worldpoint = worldpointH / worldpointH[:, 3:] #  

            xyz = worldpoint[:, :3] 
            distancetocameracenter = viewpoint_cam.camera_center - xyz
            distancetocameracenter = torch.norm(distancetocameracenter, dim=1)

            xmask = torch.logical_and(xyz[:, 0] > minx, xyz[:, 0] < maxx )
            selectedmask = torch.logical_or(xmask, torch.logical_not(xmask))  #torch.logical_and(xmask, ymask)
            new_xyz.append(xyz[selectedmask]) 

            new_features_dc.append(featuredc.cuda(0)[selectedmask])
            
            selectnumpoints = torch.sum(selectedmask).item()
            new_trbf_center.append(torch.rand((selectnumpoints, 1)).cuda())

            assert self.trbfslinit < 1 
            new_trbf_scale.append(self.trbfslinit * torch.ones((selectnumpoints, 1), device="cuda"))
            new_motion.append(torch.zeros((selectnumpoints, 9), device="cuda")) 
            new_omega.append(torch.zeros((selectnumpoints, 4), device="cuda"))
            new_featuret.append(torch.zeros((selectnumpoints, 3), device="cuda"))
            if self.use_euler_field:
                new_static_level_logits.append(torch.zeros((selectnumpoints, self.field_num_levels), device="cuda"))
                new_dynamic_level_logits.append(torch.zeros((selectnumpoints, self.field_num_levels), device="cuda"))
                if self.field_level_fourier_degree > 0:
                    coeff_dim = 2 * self.field_level_fourier_degree
                    new_dynamic_level_time_coeff.append(torch.zeros((selectnumpoints, self.field_num_levels, coeff_dim), device="cuda"))

        new_xyz = torch.cat(new_xyz, dim=0)
        new_rotation = torch.zeros((new_xyz.shape[0],4), device="cuda")
        new_rotation[:, 1]= 0
        
        new_features_dc = torch.cat(new_features_dc, dim=0)
        new_opacity = inverse_sigmoid(0.1 *torch.ones_like(new_xyz[:, 0:1]))
        new_trbf_center = torch.cat(new_trbf_center, dim=0)
        new_trbf_scale = torch.cat(new_trbf_scale, dim=0)
        new_motion = torch.cat(new_motion, dim=0)
        new_omega = torch.cat(new_omega, dim=0)
        new_featuret = torch.cat(new_featuret, dim=0)
        if self.use_euler_field:
            new_static_level_logits = torch.cat(new_static_level_logits, dim=0)
            new_dynamic_level_logits = torch.cat(new_dynamic_level_logits, dim=0)
            if len(new_dynamic_level_time_coeff) > 0:
                new_dynamic_level_time_coeff = torch.cat(new_dynamic_level_time_coeff, dim=0)
            else:
                new_dynamic_level_time_coeff = None
        else:
            new_static_level_logits = None
            new_dynamic_level_logits = None
            new_dynamic_level_time_coeff = None
        new_ems_mask = torch.ones((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)

         

        tmpxyz = torch.cat((new_xyz, self._xyz), dim=0)
        dist2 = torch.clamp_min(distCUDA2(tmpxyz), 0.0000001)
        dist2 = dist2[:new_xyz.shape[0]]
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 1.0)
        new_scaling = scales 


        self.densification_postfix(new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation, new_trbf_center, new_trbf_scale, new_motion, new_omega,new_featuret, new_static_level_logits, new_dynamic_level_logits, new_dynamic_level_time_coeff, new_ems_mask)
        return new_xyz.shape[0]




    def prune_pointswithemsmask(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        self._features_t = optimizable_tensors["f_t"]
        if self.use_euler_field and "static_grid_logits" in optimizable_tensors:
            self._static_level_logits = optimizable_tensors["static_grid_logits"]
        if self.use_euler_field and "dynamic_grid_logits" in optimizable_tensors:
            self._dynamic_level_logits = optimizable_tensors["dynamic_grid_logits"]
        if self.use_euler_field and "dynamic_grid_time_coeff" in optimizable_tensors:
            self._dynamic_level_time_coeff = optimizable_tensors["dynamic_grid_time_coeff"]
        if self.use_euler_field and "static_route_logits" in optimizable_tensors:
            self._static_route_logits = optimizable_tensors["static_route_logits"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.omegamask is not None:
            if self.omegamask.shape[0] == valid_points_mask.shape[0]:
                self.omegamask = self.omegamask[valid_points_mask]
            else:
                self.omegamask = None

        if self.maskforems is not None and self.maskforems.numel() > 0:
            self.maskforems = self.maskforems[valid_points_mask] # we only remain valid mask from ems 
        if self._dynamic_score_ema is not None and self._dynamic_score_ema.numel() > 0:
            self._dynamic_score_ema = self._dynamic_score_ema[valid_points_mask]
        if self._dynamic_active_mask is not None and self._dynamic_active_mask.numel() > 0:
            self._dynamic_active_mask = self._dynamic_active_mask[valid_points_mask]
        if self._responsibility_ema is not None and self._responsibility_ema.numel() > 0:
            self._responsibility_ema = self._responsibility_ema[valid_points_mask]
        if self._responsibility_time_center_ema is not None and self._responsibility_time_center_ema.numel() > 0:
            self._responsibility_time_center_ema = self._responsibility_time_center_ema[valid_points_mask]
        if self._slow_motion_score_ema is not None and self._slow_motion_score_ema.numel() > 0:
            self._slow_motion_score_ema = self._slow_motion_score_ema[valid_points_mask]
        if self._slow_motion_mask is not None and self._slow_motion_mask.numel() > 0:
            self._slow_motion_mask = self._slow_motion_mask[valid_points_mask]
        if self._fast_score_ema is not None and self._fast_score_ema.numel() > 0:
            self._fast_score_ema = self._fast_score_ema[valid_points_mask]
        if self._fast_active_mask is not None and self._fast_active_mask.numel() > 0:
            self._fast_active_mask = self._fast_active_mask[valid_points_mask]
        if self._static_support_ema is not None and self._static_support_ema.numel() > 0:
            self._static_support_ema = self._static_support_ema[valid_points_mask]
        if self._static_support_mask is not None and self._static_support_mask.numel() > 0:
            self._static_support_mask = self._static_support_mask[valid_points_mask]
        if self._visibility_persistence_ema is not None and self._visibility_persistence_ema.numel() > 0:
            self._visibility_persistence_ema = self._visibility_persistence_ema[valid_points_mask]
