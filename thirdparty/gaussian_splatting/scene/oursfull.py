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


class ContentExposureHead(nn.Module):
    def __init__(self, hidden=8, mode="affine", max_log_scale=0.2, max_bias=0.05, max_wb_log_gain=0.08):
        super().__init__()
        hidden = max(int(hidden), 1)
        mode = str(mode).lower()
        if mode not in {"affine", "luma_wb"}:
            mode = "affine"
        self.hidden = hidden
        self.mode = mode
        self.max_log_scale = float(max_log_scale)
        self.max_bias = float(max_bias)
        self.max_wb_log_gain = float(max_wb_log_gain)
        output_dim = 4 if mode == "luma_wb" else 2
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, output_dim),
        )
        # Identity initialization keeps the first iterations equivalent to the baseline renderer.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, stats):
        raw = self.net(stats)
        log_scale = self.max_log_scale * torch.tanh(raw[:, 0:1])
        bias = self.max_bias * torch.tanh(raw[:, 1:2])
        if self.mode != "luma_wb":
            return {
                "mode": self.mode,
                "log_scale": log_scale,
                "bias": bias,
            }
        delta_r = self.max_wb_log_gain * torch.tanh(raw[:, 2:3])
        delta_b = self.max_wb_log_gain * torch.tanh(raw[:, 3:4])
        return {
            "mode": self.mode,
            "log_scale": log_scale,
            "bias": bias,
            "delta_r": delta_r,
            "delta_b": delta_b,
        }


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
        self._static_radiance_level_logits = torch.empty(0)
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
        self.content_exposure_head = None
    
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
        self.field_bbox_expand_scale = 0.0
        self.field_bbox_expand_xyz = ""
        self.field_bbox_extra_min = ""
        self.field_bbox_extra_max = ""
        self.field_bbox_preserve_cell_size = True
        self.field_bbox_frustum_expand = False
        self.field_bbox_frustum_grid = 5
        self.field_bbox_frustum_depth_base = "bbox_corners"
        self.field_bbox_frustum_depth_scales = "1.0,1.5,2.0"
        self.field_bbox_frustum_margin = 0.0
        self.field_bbox_frustum_max_expand_xyz = ""
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
        self.field_static_temporal_residual = False
        self.field_static_temporal_frames = 50
        self.field_static_temporal_scale = 1.0
        self.field_static_radiance_branch = False
        self.field_static_radiance_start = 3000
        self.field_static_radiance_warmup = 1000
        self.field_static_radiance_scale = 0.05
        self.field_static_radiance_depth_multiplier = 5.0
        self.field_static_radiance_samples = 4
        self.field_static_radiance_max_pixels = 0
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
        self.field_bg_prior = False
        self.field_bg_prior_source = "background"
        self.field_bg_prior_color_source = "median"
        self.field_bg_prior_start = 3200
        self.field_bg_prior_until = 9000
        self.field_bg_prior_interval = 500
        self.field_bg_prior_loss_weight = 0.03
        self.field_bg_prior_visible_threshold = 0.08
        self.field_bg_prior_stability_threshold = 0.12
        self.field_bg_prior_error_quantile = 0.97
        self.field_bg_prior_depth_quantile = 0.80
        self.field_bg_prior_max_pixels = 1024
        self.field_bg_prior_num_per_ray = 1
        self.field_bg_prior_depth_scale = 1.02
        self.field_bg_prior_depth_values = ""
        self.field_bg_prior_opacity = 0.05
        self.field_bg_prior_color_init = "gt"
        self.field_bg_prior_scale_init = "knn"
        self.field_bg_prior_fixed_scale = 0.01
        self.field_bg_prior_hybrid_knn_scale_threshold = 5.0
        self.field_bg_prior_trbf_center = 0.5
        self.field_bg_prior_trbf_scale = 0.0
        self.field_bg_prior_protect_iters = 1500
        self.field_bg_prior_mature_prune = True
        self.field_bg_prior_mature_prune_interval = 500
        self.field_bg_prior_mature_min_opacity = 0.01
        self.field_bg_prior_mature_min_visibility = 0.01
        self.field_bg_prior_debug = False
        self.field_bg_prior_debug_max_events = 0
        self.field_bg_prior_debug_mode = "first_per_camera"
        self.field_bg_prior_schedule_mode = "scan"
        self.field_bg_prior_scan_views_per_event = 2
        self.field_bg_prior_scan_time_indices = ""
        self.field_bg_prior_block_size = 32
        self.field_bg_prior_pixels_per_block = 8
        self.field_bg_prior_strict_max_pixels = 32
        self.field_bg_prior_strict_pixels_per_block = 2
        self.field_bg_prior_recall_max_pixels = 32
        self.field_bg_prior_recall_pixels_per_block = 2
        self.field_bg_prior_recall_error_quantile = 0.95
        self.field_bg_prior_recall_min_visible_ratio = 0.15
        self.field_bg_prior_recall_min_stable_ratio = 0.35
        self.field_bg_prior_recall_max_occlusion_ratio = 0.25
        self.field_bg_prior_unreliable_max_pixels = 32
        self.field_bg_prior_unreliable_pixels_per_block = 2
        self.field_bg_prior_unreliable_error_quantile = 0.95
        self.field_bg_prior_min_visible_ratio = 0.35
        self.field_bg_prior_min_stable_ratio = 0.65
        self.field_bg_prior_max_occlusion_ratio = 0.05
        self.field_bg_prior_occlusion_threshold = 0.12
        self.field_bg_prior_occlusion_dilate = 7
        self.field_bg_prior_exposure_robust = True
        self.field_bg_prior_structural_weight = 0.5
        self.field_bg_prior_local_window = 31
        self.field_bg_prior_fixed_depth = True
        self.field_bg_prior_fixed_depth_ratio = 0.95
        self.field_bg_prior_depth_max = 15.0
        self.field_bg_prior_suppress = False
        self.field_bg_prior_suppress_decay = 0.02
        self.field_bg_prior_suppress_max_points = 512
        self.field_bg_prior_suppress_depth_margin = 1.0
        self.field_bg_prior_suppress_opacity_threshold = 0.05
        self.field_bg_prior_suppress_scale_quantile = 0.75
        self.field_bg_prior_clone_split = False
        self.field_bg_prior_clone_stat_start = 9000
        self.field_bg_prior_clone_start = 9500
        self.field_bg_prior_clone_until = 16000
        self.field_bg_prior_clone_interval = 500
        self.field_bg_prior_clone_grad_threshold = 0.0002
        self.field_bg_prior_clone_max_ratio = 0.05
        self.field_bg_prior_clone_max_points = 3000
        self.field_bg_prior_clone_min_age = 500
        self.field_bg_prior_clone_min_opacity = 0.01
        self.field_bg_prior_clone_min_visibility = 0.0
        self.field_bg_prior_clone_split_children = 2
        self.field_bg_prior_keep_split_parent = False
        self.field_bg_dense_add = False
        self.field_bg_dense_add_iter = 3000
        self.field_bg_dense_add_time_indices = "0,12,25,37,49"
        self.field_bg_dense_depth_base = "render"
        self.field_bg_dense_depth_scales = "0.75,1.09,1.58,2.29,3.32,4.82,7"
        self.field_bg_dense_depth_values = ""
        self.field_bg_dense_mask_source = "instant"
        self.field_bg_dense_sample_block_size = 3
        self.field_bg_dense_pixels_per_block = 1
        self.field_bg_dense_max_pixels_per_camera = 512
        self.field_bg_dense_debug = False
        self.field_bg_dense_debug_max_events = 0
        self.field_bg_dense_da3_filter = False
        self.field_bg_dense_da3_path = ""
        self.field_bg_dense_da3_foreground_quantile = 0.45
        self.field_bg_dense_beit_filter = False
        self.field_bg_dense_beit_path = ""
        self.field_bg_dense_beit_band_low = 0.10
        self.field_bg_dense_beit_band_high = 0.30
        self.field_bg_dense_beit_threshold = 0.50
        self.field_bg_dense_cell_dedup = False
        self.field_bg_dense_dedup_level = 3
        self.field_bg_dense_dedup_priority = "center"
        self.field_bg_dense_max_per_cell = 1
        self.field_bg_dense_skip_control_at_add_iter = False
        self.field_bg_dense_clip_to_bbox = False
        self.field_bg_dense_bbox_clip_margin = 0.999
        self.field_highfreq_densify = False
        self.field_highfreq_densify_sigma_divisor = 64.0
        self.field_highfreq_densify_eps = 0.001
        self.field_highfreq_densify_y_min = 0.03
        self.field_highfreq_densify_y_max = 0.97
        self.field_highfreq_densify_min_pixels = 64
        self.field_highfreq_densify_gate_start = 0.3
        self.field_highfreq_densify_gate_width = 0.4
        self.field_appearance_only_train = False
        self.field_appearance_only_start = 20000
        self.field_appearance_only_allow = "f_dc,f_t,decoder"
        self.field_soft_geometry_lr = False
        self.field_soft_geometry_start = 20000
        self.field_soft_geometry_lr_scale = 0.5
        self.field_soft_geometry_full_lr_groups = "f_dc,f_t,decoder,field_static_app,field_static_view_mapper"
        self.field_content_exposure = False
        self.field_content_exposure_lr = 0.001
        self.field_content_exposure_hidden = 8
        self.field_content_exposure_mode = "affine"
        self.field_content_exposure_max_log_scale = 0.2
        self.field_content_exposure_max_bias = 0.05
        self.field_content_exposure_max_wb_log_gain = 0.08
        self.field_content_exposure_reg_weight = 0.0
        self.field_content_exposure_wb_reg_weight = 5.0
        self.field_content_exposure_eps = 0.001
        self.field_content_exposure_detach_stats = True
        self._last_content_exposure_params = None
        self.field_depthpro_supervision = False
        self.field_depthpro_path = ""
        self.field_depthpro_start = 3000
        self.field_depthpro_until = -1
        self.field_depthpro_loss_weight = 0.0
        self.field_depthpro_max_depth = 2.0
        self.field_depthpro_min_pixels = 256
        self.field_depthpro_error_clamp = 1.0
        self.field_depthpro_use_beit_mask = True
        self.field_depthpro_exclude_unreliable = True
        self.field_scale_reg = False
        self.field_scale_reg_start = 9000
        self.field_scale_reg_until = -1
        self.field_scale_reg_weight = 0.0
        self.field_scale_reg_base_limit = 0.3
        self.field_scale_reg_depth_ref = 8.0
        self.field_scale_reg_depth_mode = "euclidean"
        self.field_scale_reg_depth_gamma = 0.75
        self.field_scale_reg_max_boost = 8.0
        self.field_bg_candidate_grad_boost = False
        self.field_bg_candidate_feature_grad_scale = 3.0
        self.field_bg_candidate_opacity_grad_scale = 2.0
        self.field_bg_candidate_scaling_grad_scale = 1.5
        self.field_bg_only_train = False
        self.field_bg_only_start = 3000
        self.field_bg_only_until = 12000
        self.field_bg_only_interval = 1
        self.field_bg_only_loss_weight = 1.0
        self.field_bg_only_min_pixels = 128
        self.field_bg_only_da3_filter = True
        self.field_bg_only_update_modules = False
        self.field_obs_reliability = False
        self.field_obs_reliability_floor = 0.35
        self.field_obs_reliability_mad_threshold = 0.045
        self.field_obs_reliability_diff_threshold = 0.12
        self.field_obs_reliability_motion_threshold = 0.12
        self.field_obs_reliability_mad_weight = 0.40
        self.field_obs_reliability_diff_weight = 0.40
        self.field_obs_reliability_motion_weight = 0.20
        self.field_obs_reliability_unreliable_threshold = 0.55
        self.field_obs_reliability_debug = False
        self.field_obs_reliability_start = 1500
        self.field_obs_reliability_until = -1
        self.field_obs_reliability_ema = 0.05
        self.field_obs_reliability_error_quantile = 0.90
        self.field_obs_reliability_error_threshold = 0.0
        self.field_obs_reliability_min_error = 0.03
        self.field_obs_reliability_dynamic_dilate = 5
        self.field_obs_reliability_structural_weight = 0.5
        self.field_obs_reliability_local_window = 31
        self.field_obs_boost_unreliable_loss = False
        self.field_obs_boost_weight = 2.0
        self.field_obs_reset = False
        self.field_obs_reset_mode = "batch"
        self.field_obs_reset_start = 1500
        self.field_obs_reset_until = 9000
        self.field_obs_reset_interval = 500
        self.field_obs_reset_schedule = ""
        self.field_obs_reset_opacity = 0.01
        self.field_obs_reset_min_opacity = 0.05
        self.field_obs_reset_max_points = 512
        self.field_obs_reset_selection_mode = "center"
        self.field_obs_reset_min_masked_contrib = 0.0
        self.field_obs_reset_min_contrib_ratio = 0.05
        self.field_obs_reset_debug = False
        self.field_obs_reset_debug_max_events = 32
        self.field_obs_reset_log_zero = True
        self.field_obs_reset_scan_time_indices = "0,12,25,37,49"
        self.field_obs_reset_scan_views_per_time = 0
        self.field_obs_reset_scan_min_hits = 2
        self.field_obs_reset_scan_top_ratio = 0.2
        self.field_obs_reset_scan_max_points = 0
        self.field_obs_reset_scan_update_ema = False
        self.field_global_reset = False
        self.field_global_reset_schedule = ""
        self.field_freq_prior = False
        self.field_freq_prior_start = 3500
        self.field_freq_prior_until = 12000
        self.field_freq_prior_weight = 0.01
        self.field_freq_prior_patch_size = 32
        self.field_freq_prior_highpass = 0.25
        self.field_freq_prior_max_patches = 16
        self.field_freq_prior_min_mask_ratio = 0.05
        self.field_freq_prior_reference = "median"
        self.field_freq_prior_on_reset_only = False
        self.field_freq_prior_debug = False
        self.field_freq_prior_debug_max_events = 32
        self.field_freq_prior_debug_mode = "first_per_camera"
        self.field_bg_median_loss = False
        self.field_bg_median_loss_weight = 0.05
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
        self._bg_candidate_mask = torch.empty(0)
        self._bg_birth_iter = torch.empty(0)
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
        self.field_bbox_expand_scale = float(getattr(args, "field_bbox_expand_scale", 0.0))
        self.field_bbox_expand_xyz = getattr(args, "field_bbox_expand_xyz", "")
        self.field_bbox_extra_min = getattr(args, "field_bbox_extra_min", "")
        self.field_bbox_extra_max = getattr(args, "field_bbox_extra_max", "")
        self.field_bbox_preserve_cell_size = bool(getattr(args, "field_bbox_preserve_cell_size", 1))
        self.field_bbox_frustum_expand = bool(getattr(args, "field_bbox_frustum_expand", 0))
        self.field_bbox_frustum_grid = int(getattr(args, "field_bbox_frustum_grid", 5))
        self.field_bbox_frustum_depth_base = str(getattr(args, "field_bbox_frustum_depth_base", "bbox_corners"))
        self.field_bbox_frustum_depth_scales = str(getattr(args, "field_bbox_frustum_depth_scales", "1.0,1.5,2.0"))
        self.field_bbox_frustum_margin = float(getattr(args, "field_bbox_frustum_margin", 0.0))
        self.field_bbox_frustum_max_expand_xyz = getattr(args, "field_bbox_frustum_max_expand_xyz", "")
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
        self.field_static_temporal_residual = bool(getattr(args, "field_static_temporal_residual", 0))
        self.field_static_temporal_frames = int(getattr(args, "field_static_temporal_frames", getattr(args, "duration", 50)))
        self.field_static_temporal_scale = float(getattr(args, "field_static_temporal_scale", 1.0))
        self.field_static_radiance_branch = bool(getattr(args, "field_static_radiance_branch", 0))
        self.field_static_radiance_start = int(getattr(args, "field_static_radiance_start", 3000))
        self.field_static_radiance_warmup = int(getattr(args, "field_static_radiance_warmup", 1000))
        self.field_static_radiance_scale = float(getattr(args, "field_static_radiance_scale", 0.05))
        self.field_static_radiance_depth_multiplier = float(getattr(args, "field_static_radiance_depth_multiplier", 5.0))
        self.field_static_radiance_samples = int(getattr(args, "field_static_radiance_samples", 4))
        self.field_static_radiance_max_pixels = int(getattr(args, "field_static_radiance_max_pixels", 0))
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
        self.field_bg_prior = bool(getattr(args, "field_bg_prior", 0))
        self.field_bg_prior_source = str(getattr(args, "field_bg_prior_source", "background"))
        self.field_bg_prior_color_source = str(getattr(args, "field_bg_prior_color_source", "median"))
        self.field_bg_prior_start = int(getattr(args, "field_bg_prior_start", 3200))
        self.field_bg_prior_until = int(getattr(args, "field_bg_prior_until", 9000))
        self.field_bg_prior_interval = int(getattr(args, "field_bg_prior_interval", 500))
        self.field_bg_prior_loss_weight = float(getattr(args, "field_bg_prior_loss_weight", 0.03))
        self.field_bg_prior_visible_threshold = float(getattr(args, "field_bg_prior_visible_threshold", 0.08))
        self.field_bg_prior_stability_threshold = float(getattr(args, "field_bg_prior_stability_threshold", 0.12))
        self.field_bg_prior_error_quantile = float(getattr(args, "field_bg_prior_error_quantile", 0.97))
        self.field_bg_prior_depth_quantile = float(getattr(args, "field_bg_prior_depth_quantile", 0.80))
        self.field_bg_prior_max_pixels = int(getattr(args, "field_bg_prior_max_pixels", 1024))
        self.field_bg_prior_num_per_ray = int(getattr(args, "field_bg_prior_num_per_ray", 1))
        self.field_bg_prior_depth_scale = float(getattr(args, "field_bg_prior_depth_scale", 1.02))
        self.field_bg_prior_depth_values = str(getattr(args, "field_bg_prior_depth_values", ""))
        self.field_bg_prior_opacity = float(getattr(args, "field_bg_prior_opacity", 0.05))
        self.field_bg_prior_color_init = str(getattr(args, "field_bg_prior_color_init", "gt"))
        self.field_bg_prior_scale_init = str(getattr(args, "field_bg_prior_scale_init", "knn"))
        self.field_bg_prior_fixed_scale = float(getattr(args, "field_bg_prior_fixed_scale", 0.01))
        self.field_bg_prior_hybrid_knn_scale_threshold = float(getattr(args, "field_bg_prior_hybrid_knn_scale_threshold", 5.0))
        self.field_bg_prior_trbf_center = float(getattr(args, "field_bg_prior_trbf_center", 0.5))
        self.field_bg_prior_trbf_scale = float(getattr(args, "field_bg_prior_trbf_scale", 0.0))
        self.field_bg_prior_protect_iters = int(getattr(args, "field_bg_prior_protect_iters", 1500))
        self.field_bg_prior_mature_prune = bool(getattr(args, "field_bg_prior_mature_prune", 1))
        self.field_bg_prior_mature_prune_interval = int(getattr(args, "field_bg_prior_mature_prune_interval", 500))
        self.field_bg_prior_mature_min_opacity = float(getattr(args, "field_bg_prior_mature_min_opacity", 0.01))
        self.field_bg_prior_mature_min_visibility = float(getattr(args, "field_bg_prior_mature_min_visibility", 0.01))
        self.field_bg_prior_debug = bool(getattr(args, "field_bg_prior_debug", 0))
        self.field_bg_prior_debug_max_events = int(getattr(args, "field_bg_prior_debug_max_events", 0))
        self.field_bg_prior_debug_mode = str(getattr(args, "field_bg_prior_debug_mode", "first_per_camera"))
        self.field_bg_prior_schedule_mode = str(getattr(args, "field_bg_prior_schedule_mode", "scan"))
        self.field_bg_prior_scan_views_per_event = int(getattr(args, "field_bg_prior_scan_views_per_event", 2))
        self.field_bg_prior_scan_time_indices = str(getattr(args, "field_bg_prior_scan_time_indices", ""))
        self.field_bg_prior_block_size = int(getattr(args, "field_bg_prior_block_size", 32))
        self.field_bg_prior_pixels_per_block = int(getattr(args, "field_bg_prior_pixels_per_block", 8))
        self.field_bg_prior_strict_max_pixels = int(getattr(args, "field_bg_prior_strict_max_pixels", 32))
        self.field_bg_prior_strict_pixels_per_block = int(getattr(args, "field_bg_prior_strict_pixels_per_block", 2))
        self.field_bg_prior_recall_max_pixels = int(getattr(args, "field_bg_prior_recall_max_pixels", 32))
        self.field_bg_prior_recall_pixels_per_block = int(getattr(args, "field_bg_prior_recall_pixels_per_block", 2))
        self.field_bg_prior_recall_error_quantile = float(getattr(args, "field_bg_prior_recall_error_quantile", 0.95))
        self.field_bg_prior_recall_min_visible_ratio = float(getattr(args, "field_bg_prior_recall_min_visible_ratio", 0.15))
        self.field_bg_prior_recall_min_stable_ratio = float(getattr(args, "field_bg_prior_recall_min_stable_ratio", 0.35))
        self.field_bg_prior_recall_max_occlusion_ratio = float(getattr(args, "field_bg_prior_recall_max_occlusion_ratio", 0.25))
        self.field_bg_prior_unreliable_max_pixels = int(getattr(args, "field_bg_prior_unreliable_max_pixels", 32))
        self.field_bg_prior_unreliable_pixels_per_block = int(getattr(args, "field_bg_prior_unreliable_pixels_per_block", 2))
        self.field_bg_prior_unreliable_error_quantile = float(getattr(args, "field_bg_prior_unreliable_error_quantile", 0.95))
        self.field_bg_prior_min_visible_ratio = float(getattr(args, "field_bg_prior_min_visible_ratio", 0.35))
        self.field_bg_prior_min_stable_ratio = float(getattr(args, "field_bg_prior_min_stable_ratio", 0.65))
        self.field_bg_prior_max_occlusion_ratio = float(getattr(args, "field_bg_prior_max_occlusion_ratio", 0.05))
        self.field_bg_prior_occlusion_threshold = float(getattr(args, "field_bg_prior_occlusion_threshold", 0.12))
        self.field_bg_prior_occlusion_dilate = int(getattr(args, "field_bg_prior_occlusion_dilate", 7))
        self.field_bg_prior_exposure_robust = bool(getattr(args, "field_bg_prior_exposure_robust", 1))
        self.field_bg_prior_structural_weight = float(getattr(args, "field_bg_prior_structural_weight", 0.5))
        self.field_bg_prior_local_window = int(getattr(args, "field_bg_prior_local_window", 31))
        self.field_bg_prior_fixed_depth = bool(getattr(args, "field_bg_prior_fixed_depth", 1))
        self.field_bg_prior_fixed_depth_ratio = float(getattr(args, "field_bg_prior_fixed_depth_ratio", 0.95))
        self.field_bg_prior_depth_max = float(getattr(args, "field_bg_prior_depth_max", 15.0))
        self.field_bg_prior_suppress = bool(getattr(args, "field_bg_prior_suppress", 0))
        self.field_bg_prior_suppress_decay = float(getattr(args, "field_bg_prior_suppress_decay", 0.02))
        self.field_bg_prior_suppress_max_points = int(getattr(args, "field_bg_prior_suppress_max_points", 512))
        self.field_bg_prior_suppress_depth_margin = float(getattr(args, "field_bg_prior_suppress_depth_margin", 1.0))
        self.field_bg_prior_suppress_opacity_threshold = float(getattr(args, "field_bg_prior_suppress_opacity_threshold", 0.05))
        self.field_bg_prior_suppress_scale_quantile = float(getattr(args, "field_bg_prior_suppress_scale_quantile", 0.75))
        self.field_bg_prior_clone_split = bool(getattr(args, "field_bg_prior_clone_split", 0))
        self.field_bg_prior_clone_stat_start = int(getattr(args, "field_bg_prior_clone_stat_start", 9000))
        self.field_bg_prior_clone_start = int(getattr(args, "field_bg_prior_clone_start", 9500))
        self.field_bg_prior_clone_until = int(getattr(args, "field_bg_prior_clone_until", 16000))
        self.field_bg_prior_clone_interval = int(getattr(args, "field_bg_prior_clone_interval", 500))
        self.field_bg_prior_clone_grad_threshold = float(getattr(args, "field_bg_prior_clone_grad_threshold", 0.0002))
        self.field_bg_prior_clone_max_ratio = float(getattr(args, "field_bg_prior_clone_max_ratio", 0.05))
        self.field_bg_prior_clone_max_points = int(getattr(args, "field_bg_prior_clone_max_points", 3000))
        self.field_bg_prior_clone_min_age = int(getattr(args, "field_bg_prior_clone_min_age", 500))
        self.field_bg_prior_clone_min_opacity = float(getattr(args, "field_bg_prior_clone_min_opacity", 0.01))
        self.field_bg_prior_clone_min_visibility = float(getattr(args, "field_bg_prior_clone_min_visibility", 0.0))
        self.field_bg_prior_clone_split_children = int(getattr(args, "field_bg_prior_clone_split_children", 2))
        self.field_bg_prior_keep_split_parent = bool(getattr(args, "field_bg_prior_keep_split_parent", 0))
        self.field_bg_dense_add = bool(getattr(args, "field_bg_dense_add", 0))
        self.field_bg_dense_add_iter = int(getattr(args, "field_bg_dense_add_iter", 3000))
        self.field_bg_dense_add_time_indices = str(getattr(args, "field_bg_dense_add_time_indices", "0,12,25,37,49"))
        self.field_bg_dense_depth_base = str(getattr(args, "field_bg_dense_depth_base", "render"))
        self.field_bg_dense_depth_scales = str(getattr(args, "field_bg_dense_depth_scales", "0.75,1.09,1.58,2.29,3.32,4.82,7"))
        self.field_bg_dense_depth_values = str(getattr(args, "field_bg_dense_depth_values", ""))
        self.field_bg_dense_mask_source = str(getattr(args, "field_bg_dense_mask_source", "instant"))
        self.field_bg_dense_sample_block_size = int(getattr(args, "field_bg_dense_sample_block_size", 3))
        self.field_bg_dense_pixels_per_block = int(getattr(args, "field_bg_dense_pixels_per_block", 1))
        self.field_bg_dense_max_pixels_per_camera = int(getattr(args, "field_bg_dense_max_pixels_per_camera", 512))
        self.field_bg_dense_debug = bool(getattr(args, "field_bg_dense_debug", 0))
        self.field_bg_dense_debug_max_events = int(getattr(args, "field_bg_dense_debug_max_events", 0))
        self.field_bg_dense_da3_filter = bool(getattr(args, "field_bg_dense_da3_filter", 0))
        self.field_bg_dense_da3_path = str(getattr(args, "field_bg_dense_da3_path", ""))
        self.field_bg_dense_da3_foreground_quantile = float(getattr(args, "field_bg_dense_da3_foreground_quantile", 0.45))
        self.field_bg_dense_beit_filter = bool(getattr(args, "field_bg_dense_beit_filter", 0))
        self.field_bg_dense_beit_path = str(getattr(args, "field_bg_dense_beit_path", ""))
        self.field_bg_dense_beit_band_low = float(getattr(args, "field_bg_dense_beit_band_low", 0.10))
        self.field_bg_dense_beit_band_high = float(getattr(args, "field_bg_dense_beit_band_high", 0.30))
        self.field_bg_dense_beit_threshold = float(getattr(args, "field_bg_dense_beit_threshold", 0.50))
        self.field_bg_dense_cell_dedup = bool(getattr(args, "field_bg_dense_cell_dedup", 0))
        self.field_bg_dense_dedup_level = int(getattr(args, "field_bg_dense_dedup_level", 3))
        self.field_bg_dense_dedup_priority = str(getattr(args, "field_bg_dense_dedup_priority", "center"))
        self.field_bg_dense_max_per_cell = int(getattr(args, "field_bg_dense_max_per_cell", 1))
        self.field_bg_dense_skip_control_at_add_iter = bool(getattr(args, "field_bg_dense_skip_control_at_add_iter", 0))
        self.field_bg_dense_clip_to_bbox = bool(getattr(args, "field_bg_dense_clip_to_bbox", 0))
        self.field_bg_dense_bbox_clip_margin = float(getattr(args, "field_bg_dense_bbox_clip_margin", 0.999))
        self.field_highfreq_densify = bool(getattr(args, "field_highfreq_densify", 0))
        self.field_highfreq_densify_sigma_divisor = float(getattr(args, "field_highfreq_densify_sigma_divisor", 64.0))
        self.field_highfreq_densify_eps = float(getattr(args, "field_highfreq_densify_eps", 0.001))
        self.field_highfreq_densify_y_min = float(getattr(args, "field_highfreq_densify_y_min", 0.03))
        self.field_highfreq_densify_y_max = float(getattr(args, "field_highfreq_densify_y_max", 0.97))
        self.field_highfreq_densify_min_pixels = int(getattr(args, "field_highfreq_densify_min_pixels", 64))
        self.field_highfreq_densify_gate_start = float(getattr(args, "field_highfreq_densify_gate_start", 0.3))
        self.field_highfreq_densify_gate_width = float(getattr(args, "field_highfreq_densify_gate_width", 0.4))
        self.field_appearance_only_train = bool(getattr(args, "field_appearance_only_train", 0))
        self.field_appearance_only_start = int(getattr(args, "field_appearance_only_start", 20000))
        self.field_appearance_only_allow = str(getattr(args, "field_appearance_only_allow", "f_dc,f_t,decoder"))
        self.field_soft_geometry_lr = bool(getattr(args, "field_soft_geometry_lr", 0))
        self.field_soft_geometry_start = int(getattr(args, "field_soft_geometry_start", 20000))
        self.field_soft_geometry_lr_scale = float(getattr(args, "field_soft_geometry_lr_scale", 0.5))
        self.field_soft_geometry_full_lr_groups = str(getattr(args, "field_soft_geometry_full_lr_groups", "f_dc,f_t,decoder,field_static_app,field_static_view_mapper"))
        self.field_content_exposure = bool(getattr(args, "field_content_exposure", 0))
        self.field_content_exposure_lr = float(getattr(args, "field_content_exposure_lr", 0.001))
        self.field_content_exposure_hidden = int(getattr(args, "field_content_exposure_hidden", 8))
        self.field_content_exposure_mode = str(getattr(args, "field_content_exposure_mode", "affine")).lower()
        self.field_content_exposure_max_log_scale = float(getattr(args, "field_content_exposure_max_log_scale", 0.2))
        self.field_content_exposure_max_bias = float(getattr(args, "field_content_exposure_max_bias", 0.05))
        self.field_content_exposure_max_wb_log_gain = float(getattr(args, "field_content_exposure_max_wb_log_gain", 0.08))
        self.field_content_exposure_reg_weight = float(getattr(args, "field_content_exposure_reg_weight", 0.0))
        self.field_content_exposure_wb_reg_weight = float(getattr(args, "field_content_exposure_wb_reg_weight", 5.0))
        self.field_content_exposure_eps = float(getattr(args, "field_content_exposure_eps", 0.001))
        self.field_content_exposure_detach_stats = bool(getattr(args, "field_content_exposure_detach_stats", 1))
        self._ensure_content_exposure_head()
        self.field_depthpro_supervision = bool(getattr(args, "field_depthpro_supervision", 0))
        self.field_depthpro_path = str(getattr(args, "field_depthpro_path", ""))
        self.field_depthpro_start = int(getattr(args, "field_depthpro_start", 3000))
        self.field_depthpro_until = int(getattr(args, "field_depthpro_until", -1))
        self.field_depthpro_loss_weight = float(getattr(args, "field_depthpro_loss_weight", 0.0))
        self.field_depthpro_max_depth = float(getattr(args, "field_depthpro_max_depth", 2.0))
        self.field_depthpro_min_pixels = int(getattr(args, "field_depthpro_min_pixels", 256))
        self.field_depthpro_error_clamp = float(getattr(args, "field_depthpro_error_clamp", 1.0))
        self.field_depthpro_use_beit_mask = bool(getattr(args, "field_depthpro_use_beit_mask", 1))
        self.field_depthpro_exclude_unreliable = bool(getattr(args, "field_depthpro_exclude_unreliable", 1))
        self.field_scale_reg = bool(getattr(args, "field_scale_reg", 0))
        self.field_scale_reg_start = int(getattr(args, "field_scale_reg_start", 9000))
        self.field_scale_reg_until = int(getattr(args, "field_scale_reg_until", -1))
        self.field_scale_reg_weight = float(getattr(args, "field_scale_reg_weight", 0.0))
        self.field_scale_reg_base_limit = float(getattr(args, "field_scale_reg_base_limit", 0.3))
        self.field_scale_reg_depth_ref = float(getattr(args, "field_scale_reg_depth_ref", 8.0))
        self.field_scale_reg_depth_mode = str(getattr(args, "field_scale_reg_depth_mode", "euclidean"))
        self.field_scale_reg_depth_gamma = float(getattr(args, "field_scale_reg_depth_gamma", 0.75))
        self.field_scale_reg_max_boost = float(getattr(args, "field_scale_reg_max_boost", 8.0))
        self.field_bg_candidate_grad_boost = bool(getattr(args, "field_bg_candidate_grad_boost", 0))
        self.field_bg_candidate_feature_grad_scale = float(getattr(args, "field_bg_candidate_feature_grad_scale", 3.0))
        self.field_bg_candidate_opacity_grad_scale = float(getattr(args, "field_bg_candidate_opacity_grad_scale", 2.0))
        self.field_bg_candidate_scaling_grad_scale = float(getattr(args, "field_bg_candidate_scaling_grad_scale", 1.5))
        self.field_bg_only_train = bool(getattr(args, "field_bg_only_train", 0))
        self.field_bg_only_start = int(getattr(args, "field_bg_only_start", 3000))
        self.field_bg_only_until = int(getattr(args, "field_bg_only_until", 12000))
        self.field_bg_only_interval = int(getattr(args, "field_bg_only_interval", 1))
        self.field_bg_only_loss_weight = float(getattr(args, "field_bg_only_loss_weight", 1.0))
        self.field_bg_only_min_pixels = int(getattr(args, "field_bg_only_min_pixels", 128))
        self.field_bg_only_da3_filter = bool(getattr(args, "field_bg_only_da3_filter", 1))
        self.field_bg_only_update_modules = bool(getattr(args, "field_bg_only_update_modules", 0))
        self.field_obs_reliability = bool(getattr(args, "field_obs_reliability", 0))
        self.field_obs_reliability_floor = float(getattr(args, "field_obs_reliability_floor", 0.35))
        self.field_obs_reliability_mad_threshold = float(getattr(args, "field_obs_reliability_mad_threshold", 0.045))
        self.field_obs_reliability_diff_threshold = float(getattr(args, "field_obs_reliability_diff_threshold", 0.12))
        self.field_obs_reliability_motion_threshold = float(getattr(args, "field_obs_reliability_motion_threshold", 0.12))
        self.field_obs_reliability_mad_weight = float(getattr(args, "field_obs_reliability_mad_weight", 0.40))
        self.field_obs_reliability_diff_weight = float(getattr(args, "field_obs_reliability_diff_weight", 0.40))
        self.field_obs_reliability_motion_weight = float(getattr(args, "field_obs_reliability_motion_weight", 0.20))
        self.field_obs_reliability_unreliable_threshold = float(getattr(args, "field_obs_reliability_unreliable_threshold", 0.55))
        self.field_obs_reliability_debug = bool(getattr(args, "field_obs_reliability_debug", 0))
        self.field_obs_reliability_start = int(getattr(args, "field_obs_reliability_start", 1500))
        self.field_obs_reliability_until = int(getattr(args, "field_obs_reliability_until", -1))
        self.field_obs_reliability_ema = float(getattr(args, "field_obs_reliability_ema", 0.05))
        self.field_obs_reliability_error_quantile = float(getattr(args, "field_obs_reliability_error_quantile", 0.90))
        self.field_obs_reliability_error_threshold = float(getattr(args, "field_obs_reliability_error_threshold", 0.0))
        self.field_obs_reliability_min_error = float(getattr(args, "field_obs_reliability_min_error", 0.03))
        self.field_obs_reliability_dynamic_dilate = int(getattr(args, "field_obs_reliability_dynamic_dilate", 5))
        self.field_obs_reliability_structural_weight = float(getattr(args, "field_obs_reliability_structural_weight", 0.5))
        self.field_obs_reliability_local_window = int(getattr(args, "field_obs_reliability_local_window", 31))
        self.field_obs_boost_unreliable_loss = bool(getattr(args, "field_obs_boost_unreliable_loss", 0))
        self.field_obs_boost_weight = float(getattr(args, "field_obs_boost_weight", 2.0))
        self.field_obs_reset = bool(getattr(args, "field_obs_reset", 0))
        self.field_obs_reset_mode = str(getattr(args, "field_obs_reset_mode", "batch"))
        self.field_obs_reset_start = int(getattr(args, "field_obs_reset_start", 1500))
        self.field_obs_reset_until = int(getattr(args, "field_obs_reset_until", 9000))
        self.field_obs_reset_interval = int(getattr(args, "field_obs_reset_interval", 500))
        self.field_obs_reset_schedule = str(getattr(args, "field_obs_reset_schedule", ""))
        self.field_obs_reset_opacity = float(getattr(args, "field_obs_reset_opacity", 0.01))
        self.field_obs_reset_min_opacity = float(getattr(args, "field_obs_reset_min_opacity", 0.05))
        self.field_obs_reset_max_points = int(getattr(args, "field_obs_reset_max_points", 512))
        self.field_obs_reset_selection_mode = str(getattr(args, "field_obs_reset_selection_mode", "center"))
        self.field_obs_reset_min_masked_contrib = float(getattr(args, "field_obs_reset_min_masked_contrib", 0.0))
        self.field_obs_reset_min_contrib_ratio = float(getattr(args, "field_obs_reset_min_contrib_ratio", 0.05))
        self.field_obs_reset_debug = bool(getattr(args, "field_obs_reset_debug", 0))
        self.field_obs_reset_debug_max_events = int(getattr(args, "field_obs_reset_debug_max_events", 32))
        self.field_obs_reset_log_zero = bool(getattr(args, "field_obs_reset_log_zero", 1))
        self.field_obs_reset_scan_time_indices = str(getattr(args, "field_obs_reset_scan_time_indices", "0,12,25,37,49"))
        self.field_obs_reset_scan_views_per_time = int(getattr(args, "field_obs_reset_scan_views_per_time", 0))
        self.field_obs_reset_scan_min_hits = int(getattr(args, "field_obs_reset_scan_min_hits", 2))
        self.field_obs_reset_scan_top_ratio = float(getattr(args, "field_obs_reset_scan_top_ratio", 0.2))
        self.field_obs_reset_scan_max_points = int(getattr(args, "field_obs_reset_scan_max_points", 0))
        self.field_obs_reset_scan_update_ema = bool(getattr(args, "field_obs_reset_scan_update_ema", 0))
        self.field_global_reset = bool(getattr(args, "field_global_reset", 0))
        self.field_global_reset_schedule = str(getattr(args, "field_global_reset_schedule", ""))
        self.field_freq_prior = bool(getattr(args, "field_freq_prior", 0))
        self.field_freq_prior_start = int(getattr(args, "field_freq_prior_start", 3500))
        self.field_freq_prior_until = int(getattr(args, "field_freq_prior_until", 12000))
        self.field_freq_prior_weight = float(getattr(args, "field_freq_prior_weight", 0.01))
        self.field_freq_prior_patch_size = int(getattr(args, "field_freq_prior_patch_size", 32))
        self.field_freq_prior_highpass = float(getattr(args, "field_freq_prior_highpass", 0.25))
        self.field_freq_prior_max_patches = int(getattr(args, "field_freq_prior_max_patches", 16))
        self.field_freq_prior_min_mask_ratio = float(getattr(args, "field_freq_prior_min_mask_ratio", 0.05))
        self.field_freq_prior_reference = str(getattr(args, "field_freq_prior_reference", "median"))
        self.field_freq_prior_on_reset_only = bool(getattr(args, "field_freq_prior_on_reset_only", 0))
        self.field_freq_prior_debug = bool(getattr(args, "field_freq_prior_debug", 0))
        self.field_freq_prior_debug_max_events = int(getattr(args, "field_freq_prior_debug_max_events", 32))
        self.field_freq_prior_debug_mode = str(getattr(args, "field_freq_prior_debug_mode", "first_per_camera"))
        self.field_bg_median_loss = bool(getattr(args, "field_bg_median_loss", 0))
        self.field_bg_median_loss_weight = float(getattr(args, "field_bg_median_loss_weight", 0.05))

    def set_field_camera_scale_hints(self, cameras):
        self._field_camera_scale_hints = []
        for camera in cameras:
            camera_center = getattr(camera, "camera_center", None)
            if camera_center is None:
                continue
            camera_to_world = None
            world_view_transform = getattr(camera, "world_view_transform", None)
            if world_view_transform is not None:
                try:
                    camera_to_world = world_view_transform.detach().float().cpu().T.inverse()
                except Exception:
                    camera_to_world = None
            self._field_camera_scale_hints.append(
                {
                    "center": camera_center.detach().float().cpu(),
                    "c2w_rotation": None if camera_to_world is None else camera_to_world[:3, :3].contiguous(),
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
    def _parse_field_bbox_vector(value, device, dtype, name):
        if value is None:
            return torch.zeros(3, device=device, dtype=dtype)
        if torch.is_tensor(value):
            values = value.detach().to(device=device, dtype=dtype).reshape(-1)
        elif isinstance(value, (list, tuple)):
            values = torch.tensor([float(v) for v in value], device=device, dtype=dtype).reshape(-1)
        else:
            text = str(value).strip()
            if not text:
                return torch.zeros(3, device=device, dtype=dtype)
            for sep in (";", "x", "X"):
                text = text.replace(sep, ",")
            parts = [part.strip() for part in text.split(",") if part.strip()]
            values = torch.tensor([float(part) for part in parts], device=device, dtype=dtype)
        if values.numel() == 1:
            values = values.repeat(3)
        if values.numel() != 3:
            raise ValueError(f"{name} expects 1 or 3 values, got {values.numel()}: {value}")
        return values.view(3)

    @staticmethod
    def _parse_field_float_list(value, default, name):
        if value is None:
            return list(default)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return list(default)
            for sep in (";", "x", "X", "|"):
                text = text.replace(sep, ",")
            parts = [part.strip() for part in text.split(",") if part.strip()]
            values = [float(part) for part in parts]
        elif isinstance(value, (list, tuple)):
            values = [float(part) for part in value]
        else:
            values = [float(value)]
        values = [v for v in values if math.isfinite(v) and v > 0.0]
        if not values:
            raise ValueError(f"{name} must contain at least one positive finite value: {value}")
        return values

    def _sample_field_frustum_bbox_points(self, bbox_min, bbox_max):
        if not bool(getattr(self, "field_bbox_frustum_expand", False)):
            return None, {}
        if not self._field_camera_scale_hints:
            return None, {"enabled": True, "points": 0, "reason": "no_camera_hints"}

        device = bbox_min.device
        dtype = bbox_min.dtype
        span = torch.clamp(bbox_max - bbox_min, min=1e-6)
        bbox_center = 0.5 * (bbox_min + bbox_max)
        bbox_diag = torch.linalg.norm(span).clamp_min(1e-6)
        bbox_max_span = torch.max(span).clamp_min(1e-6)
        corners = torch.stack(
            [
                torch.stack((x, y, z))
                for x in (bbox_min[0], bbox_max[0])
                for y in (bbox_min[1], bbox_max[1])
                for z in (bbox_min[2], bbox_max[2])
            ],
            dim=0,
        )

        grid = max(int(getattr(self, "field_bbox_frustum_grid", 5)), 2)
        coords = torch.linspace(-1.0, 1.0, grid, device=device, dtype=dtype)
        depth_scales = self._parse_field_float_list(
            getattr(self, "field_bbox_frustum_depth_scales", "1.0,1.5,2.0"),
            default=[1.0, 1.5, 2.0],
            name="field_bbox_frustum_depth_scales",
        )
        depth_mode = str(getattr(self, "field_bbox_frustum_depth_base", "bbox_corners")).lower()
        points = []
        base_depths = []
        used_cameras = 0

        for hint in self._field_camera_scale_hints:
            rotation = hint.get("c2w_rotation", None)
            if rotation is None:
                continue
            fovx = float(hint.get("fovx", 0.0))
            fovy = float(hint.get("fovy", 0.0))
            if not (math.isfinite(fovx) and math.isfinite(fovy) and fovx > 0.0 and fovy > 0.0):
                continue
            center = hint["center"].to(device=device, dtype=dtype)
            rotation = rotation.to(device=device, dtype=dtype)

            if depth_mode in ("diag", "bbox_diag"):
                base_depth = bbox_diag
            elif depth_mode in ("span", "max_span", "bbox_span"):
                base_depth = bbox_max_span
            elif depth_mode in ("center", "bbox_center"):
                base_depth = torch.linalg.norm(bbox_center - center).clamp_min(1e-6)
            else:
                local_corners = (corners - center.view(1, 3)) @ rotation
                positive_z = local_corners[:, 2][local_corners[:, 2] > 1e-6]
                if positive_z.numel() > 0:
                    base_depth = positive_z.max().clamp_min(1e-6)
                else:
                    base_depth = torch.linalg.norm(corners - center.view(1, 3), dim=1).max().clamp_min(1e-6)

            tanx = math.tan(0.5 * fovx)
            tany = math.tan(0.5 * fovy)
            base_depths.append(float(base_depth.detach().cpu().item()))
            used_cameras += 1

            for scale in depth_scales:
                depth = base_depth * float(scale)
                local_rows = []
                for y in coords:
                    for x in coords:
                        local_rows.append(torch.stack((x * tanx * depth, y * tany * depth, depth)))
                local = torch.stack(local_rows, dim=0)
                world = center.view(1, 3) + local @ rotation.T
                points.append(world)

        if not points:
            return None, {"enabled": True, "points": 0, "reason": "no_valid_camera_frustums"}

        points = torch.cat(points, dim=0)
        points = points[torch.isfinite(points).all(dim=1)]
        if points.numel() == 0:
            return None, {"enabled": True, "points": 0, "reason": "nonfinite_points"}

        stats = {
            "enabled": True,
            "points": int(points.shape[0]),
            "cameras": used_cameras,
            "grid": grid,
            "depth_mode": depth_mode,
            "depth_scales": depth_scales,
            "base_depth_min": min(base_depths) if base_depths else 0.0,
            "base_depth_max": max(base_depths) if base_depths else 0.0,
        }
        return points, stats

    def _expand_field_bbox(self, bbox_min, bbox_max):
        bbox_min = bbox_min.detach().float()
        bbox_max = bbox_max.detach().float()
        span = torch.clamp(bbox_max - bbox_min, min=1e-6)
        device = bbox_min.device
        dtype = bbox_min.dtype
        uniform = max(float(getattr(self, "field_bbox_expand_scale", 0.0)), 0.0)
        expand_ratio = torch.full((3,), uniform, device=device, dtype=dtype)
        expand_ratio = expand_ratio + torch.clamp(
            self._parse_field_bbox_vector(
                getattr(self, "field_bbox_expand_xyz", ""),
                device,
                dtype,
                "field_bbox_expand_xyz",
            ),
            min=0.0,
        )
        extra_min = torch.clamp(
            self._parse_field_bbox_vector(
                getattr(self, "field_bbox_extra_min", ""),
                device,
                dtype,
                "field_bbox_extra_min",
            ),
            min=0.0,
        )
        extra_max = torch.clamp(
            self._parse_field_bbox_vector(
                getattr(self, "field_bbox_extra_max", ""),
                device,
                dtype,
                "field_bbox_extra_max",
            ),
            min=0.0,
        )
        pad = span * expand_ratio
        expanded_min = bbox_min - pad - extra_min
        expanded_max = bbox_max + pad + extra_max
        frustum_points, frustum_stats = self._sample_field_frustum_bbox_points(bbox_min, bbox_max)
        frustum_min = None
        frustum_max = None
        frustum_margin = max(float(getattr(self, "field_bbox_frustum_margin", 0.0)), 0.0)
        if frustum_points is not None:
            frustum_min = frustum_points.min(dim=0).values
            frustum_max = frustum_points.max(dim=0).values
            if frustum_margin > 0.0:
                frustum_min = frustum_min - frustum_margin
                frustum_max = frustum_max + frustum_margin
            expanded_min = torch.minimum(expanded_min, frustum_min)
            expanded_max = torch.maximum(expanded_max, frustum_max)

            max_expand_spec = getattr(self, "field_bbox_frustum_max_expand_xyz", "")
            if max_expand_spec is not None and str(max_expand_spec).strip():
                max_expand = torch.clamp(
                    self._parse_field_bbox_vector(
                        max_expand_spec,
                        device,
                        dtype,
                        "field_bbox_frustum_max_expand_xyz",
                    ),
                    min=0.0,
                )
                expanded_min = torch.maximum(expanded_min, bbox_min - span * max_expand)
                expanded_max = torch.minimum(expanded_max, bbox_max + span * max_expand)
                frustum_stats["max_expand_xyz"] = max_expand.detach().cpu()
        self._field_bbox_stats = {
            "raw_min": bbox_min.detach().cpu(),
            "raw_max": bbox_max.detach().cpu(),
            "expanded_min": expanded_min.detach().cpu(),
            "expanded_max": expanded_max.detach().cpu(),
            "expand_ratio": expand_ratio.detach().cpu(),
            "extra_min": extra_min.detach().cpu(),
            "extra_max": extra_max.detach().cpu(),
            "frustum": frustum_stats,
            "frustum_min": None if frustum_min is None else frustum_min.detach().cpu(),
            "frustum_max": None if frustum_max is None else frustum_max.detach().cpu(),
        }
        if bool(getattr(self, "field_bbox_preserve_cell_size", True)):
            self._field_resolution_reference_span = span.detach().float()
        else:
            self._field_resolution_reference_span = None
        return expanded_min, expanded_max

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
        reference_span = getattr(self, "_field_resolution_reference_span", None)
        if reference_span is not None:
            reference_span = torch.clamp(reference_span.detach().float().to(device=bbox_span.device), min=1e-6)
        else:
            reference_span = bbox_span
        max_span = float(torch.max(reference_span).item())
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
        bbox_min, bbox_max = self._expand_field_bbox(bbox_min, bbox_max)
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
            "[STEGF] Euler grid mode={}, levels={}, resolutions={}, finest_cell={}, components={}, static_temporal_bins={}".format(
                stats.get("mode", self.field_resolution_mode),
                self.field_num_levels,
                self.field_resolved_level_resolutions,
                stats.get("finest_cell", "n/a"),
                component_text if component_text else "n/a",
                EulerField._resolve_static_temporal_bins(self.field_num_levels, self.field_static_temporal_frames)
                if self.field_static_temporal_residual
                else "off",
            )
        )
        bbox_stats = getattr(self, "_field_bbox_stats", None)
        if bbox_stats is not None:
            expand_ratio = bbox_stats["expand_ratio"]
            extra_min = bbox_stats["extra_min"]
            extra_max = bbox_stats["extra_max"]
            frustum_stats = bbox_stats.get("frustum", {})
            frustum_points = int(frustum_stats.get("points", 0)) if isinstance(frustum_stats, dict) else 0
            if bool(torch.any(expand_ratio > 0).item() or torch.any(extra_min > 0).item() or torch.any(extra_max > 0).item() or frustum_points > 0):
                frustum_text = "off"
                if isinstance(frustum_stats, dict) and frustum_stats.get("enabled", False):
                    frustum_text = (
                        "points={}, cameras={}, grid={}, depth_mode={}, depth_scales={}, base_depth=[{:.6g},{:.6g}]".format(
                            int(frustum_stats.get("points", 0)),
                            int(frustum_stats.get("cameras", 0)),
                            int(frustum_stats.get("grid", 0)),
                            frustum_stats.get("depth_mode", "n/a"),
                            [round(float(v), 6) for v in frustum_stats.get("depth_scales", [])],
                            float(frustum_stats.get("base_depth_min", 0.0)),
                            float(frustum_stats.get("base_depth_max", 0.0)),
                        )
                    )
                    if "max_expand_xyz" in frustum_stats:
                        frustum_text += ", max_expand_xyz={}".format(
                            [round(float(v), 6) for v in frustum_stats["max_expand_xyz"].view(-1)]
                        )
                print(
                    "[STEGF] Euler grid bbox expanded: raw_min={}, raw_max={}, expanded_min={}, expanded_max={}, ratio={}, extra_min={}, extra_max={}, frustum={}, preserve_cell_size={}".format(
                        [round(float(v), 6) for v in bbox_stats["raw_min"].view(-1)],
                        [round(float(v), 6) for v in bbox_stats["raw_max"].view(-1)],
                        [round(float(v), 6) for v in bbox_stats["expanded_min"].view(-1)],
                        [round(float(v), 6) for v in bbox_stats["expanded_max"].view(-1)],
                        [round(float(v), 6) for v in expand_ratio.view(-1)],
                        [round(float(v), 6) for v in extra_min.view(-1)],
                        [round(float(v), 6) for v in extra_max.view(-1)],
                        frustum_text,
                        bool(getattr(self, "field_bbox_preserve_cell_size", True)),
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
            enable_static_temporal_residual=self.field_static_temporal_residual,
            static_temporal_frames=self.field_static_temporal_frames,
            static_temporal_scale=self.field_static_temporal_scale,
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
        if self.field_static_radiance_branch:
            self._static_radiance_level_logits = nn.Parameter(
                torch.zeros((self.field_num_levels,), device="cuda", dtype=torch.float32).requires_grad_(True)
            )
        else:
            self._static_radiance_level_logits = torch.empty(0, device="cuda")

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

    def _ensure_content_exposure_head(self):
        if not self.field_content_exposure:
            self.content_exposure_head = None
            return
        needs_rebuild = (
            self.content_exposure_head is None
            or getattr(self.content_exposure_head, "hidden", None) != int(self.field_content_exposure_hidden)
            or getattr(self.content_exposure_head, "mode", None) != str(self.field_content_exposure_mode).lower()
            or getattr(self.content_exposure_head, "max_log_scale", None) != float(self.field_content_exposure_max_log_scale)
            or getattr(self.content_exposure_head, "max_bias", None) != float(self.field_content_exposure_max_bias)
            or getattr(self.content_exposure_head, "max_wb_log_gain", None) != float(self.field_content_exposure_max_wb_log_gain)
        )
        if needs_rebuild:
            self.content_exposure_head = ContentExposureHead(
                hidden=self.field_content_exposure_hidden,
                mode=self.field_content_exposure_mode,
                max_log_scale=self.field_content_exposure_max_log_scale,
                max_bias=self.field_content_exposure_max_bias,
                max_wb_log_gain=self.field_content_exposure_max_wb_log_gain,
            )

    def _content_exposure_stats(self, image):
        source = image.detach() if self.field_content_exposure_detach_stats else image
        source = source.clamp(0.0, 1.0)
        luminance = (
            0.299 * source[0:1, :, :]
            + 0.587 * source[1:2, :, :]
            + 0.114 * source[2:3, :, :]
        )
        flat = luminance.reshape(-1)
        eps = max(float(self.field_content_exposure_eps), 1e-8)
        stats = torch.stack(
            [
                torch.log(flat + eps).mean(),
                flat.mean(),
                torch.quantile(flat, 0.90),
                torch.quantile(flat, 0.95),
                (flat > 0.90).to(flat.dtype).mean(),
                (flat < 0.05).to(flat.dtype).mean(),
            ]
        )
        return stats.view(1, 6)

    def apply_content_exposure(self, image):
        if (not self.field_content_exposure) or self.content_exposure_head is None:
            self._last_content_exposure_params = None
            return image
        stats = self._content_exposure_stats(image).to(device=image.device, dtype=image.dtype)
        params = self.content_exposure_head(stats)
        self._last_content_exposure_params = params
        log_scale = params["log_scale"].to(device=image.device, dtype=image.dtype).view(1, 1, 1)
        bias = params["bias"].to(device=image.device, dtype=image.dtype).view(1, 1, 1)
        if params.get("mode", "affine") != "luma_wb":
            return torch.exp(log_scale) * image + bias
        delta_r = params["delta_r"].to(device=image.device, dtype=image.dtype).view(1, 1, 1)
        delta_b = params["delta_b"].to(device=image.device, dtype=image.dtype).view(1, 1, 1)
        wb = torch.cat(
            [
                torch.exp(delta_r),
                torch.ones_like(delta_r),
                torch.exp(delta_b),
            ],
            dim=0,
        )
        return torch.exp(log_scale) * wb * image + bias

    def get_content_exposure_reg_loss(self):
        weight = float(getattr(self, "field_content_exposure_reg_weight", 0.0))
        if weight <= 0.0:
            return None
        params = self._last_content_exposure_params
        if not params:
            return None
        loss = params["log_scale"].square().mean() + params["bias"].square().mean()
        if params.get("mode", "affine") == "luma_wb":
            wb_weight = float(getattr(self, "field_content_exposure_wb_reg_weight", 5.0))
            loss = loss + wb_weight * (
                params["delta_r"].square().mean() + params["delta_b"].square().mean()
            )
        return weight * loss

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
        self.content_exposure_grd = {}
        if self.content_exposure_head is not None:
            for name, param in self.content_exposure_head.named_parameters():
                self.content_exposure_grd[name] = torch.zeros_like(param, requires_grad=False, device=param.device)

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
                    static_level_features = self.euler_field.query_static_level_features(canonical_points, timestamp=timestamp)
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
                        timestamp=timestamp,
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

    def static_far_segment_radiance_features(self, viewpoint_camera, depth, unreliable_mask, iteration):
        if not bool(getattr(self, "field_static_radiance_branch", False)):
            return None
        if int(iteration) < int(getattr(self, "field_static_radiance_start", 3000)):
            return None
        if self.euler_field is None or self.field_static_app_head is None:
            return None
        if unreliable_mask is None or depth is None:
            return None
        if getattr(viewpoint_camera, "rayd", None) is None:
            return None

        depth = depth.squeeze(0)
        mask = unreliable_mask.to(device=depth.device, dtype=torch.bool)
        if mask.shape != depth.shape or torch.count_nonzero(mask) == 0:
            return None

        pixel_indices = torch.nonzero(mask, as_tuple=False)
        max_pixels = int(getattr(self, "field_static_radiance_max_pixels", 0))
        if max_pixels > 0 and pixel_indices.shape[0] > max_pixels:
            pick = torch.linspace(
                0,
                pixel_indices.shape[0] - 1,
                max_pixels,
                device=pixel_indices.device,
            ).round().long()
            pixel_indices = pixel_indices[pick]
        if pixel_indices.numel() == 0:
            return None

        y = pixel_indices[:, 0]
        x = pixel_indices[:, 1]
        ray_dirs = viewpoint_camera.rayd[0, :, y, x].transpose(0, 1).to(device=depth.device, dtype=depth.dtype)
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=1, eps=1e-6)
        ray_origin = viewpoint_camera.camera_center.to(device=depth.device, dtype=depth.dtype).view(1, 3)
        bbox_min = self.euler_field.bbox_min.to(device=depth.device, dtype=depth.dtype).view(1, 3)
        bbox_max = self.euler_field.bbox_max.to(device=depth.device, dtype=depth.dtype).view(1, 3)

        near = depth[y, x].to(dtype=depth.dtype) * float(getattr(self, "field_static_radiance_depth_multiplier", 5.0))
        finite_depth = torch.isfinite(near) & (near > 1e-4)
        eps = torch.tensor(1e-6, device=depth.device, dtype=depth.dtype)
        parallel = torch.abs(ray_dirs) < eps
        inside_parallel = (ray_origin >= bbox_min) & (ray_origin <= bbox_max)
        denom = torch.where(parallel, torch.ones_like(ray_dirs), ray_dirs)
        t0 = (bbox_min - ray_origin) / denom
        t1 = (bbox_max - ray_origin) / denom
        t_axis_min = torch.minimum(t0, t1)
        t_axis_max = torch.maximum(t0, t1)
        neg_inf = torch.full_like(t_axis_min, -float("inf"))
        pos_inf = torch.full_like(t_axis_max, float("inf"))
        t_axis_min = torch.where(parallel & inside_parallel, neg_inf, t_axis_min)
        t_axis_max = torch.where(parallel & inside_parallel, pos_inf, t_axis_max)
        invalid_parallel = torch.any(parallel & (~inside_parallel), dim=1)
        t_enter = torch.max(t_axis_min, dim=1).values
        t_exit = torch.min(t_axis_max, dim=1).values

        start = torch.maximum(near, t_enter + 1e-4)
        valid = finite_depth & (~invalid_parallel) & torch.isfinite(t_exit) & (t_exit > start + 1e-4)
        if torch.count_nonzero(valid) == 0:
            return None

        y = y[valid]
        x = x[valid]
        ray_dirs = ray_dirs[valid]
        start = start[valid]
        t_exit = t_exit[valid]
        samples = max(int(getattr(self, "field_static_radiance_samples", 4)), 1)
        alpha = (torch.arange(samples, device=depth.device, dtype=depth.dtype) + 0.5) / float(samples)
        sample_depth = start[:, None] * (1.0 - alpha[None, :]) + t_exit[:, None] * alpha[None, :]
        points = ray_origin.view(1, 1, 3) + sample_depth[:, :, None] * ray_dirs[:, None, :]
        flat_points = points.reshape(-1, 3)

        level_features = self.euler_field.query_static_level_features(
            flat_points,
            camera_center=viewpoint_camera.camera_center,
            view_mapper=self.field_static_view_mapper,
            view_scale=self.field_static_app_scale,
            timestamp=getattr(viewpoint_camera, "timestamp", None),
        )
        # Ray samples are independent query points, not existing Gaussians, so
        # they use a global learnable level preference instead of per-Gaussian logits.
        if self._static_radiance_level_logits.numel() == level_features.shape[1]:
            level_logits = self._static_radiance_level_logits.to(
                device=level_features.device,
                dtype=level_features.dtype,
            ).view(1, -1).expand(flat_points.shape[0], -1)
        else:
            level_logits = torch.zeros(
                (flat_points.shape[0], level_features.shape[1]),
                device=level_features.device,
                dtype=level_features.dtype,
            )
        feature = self.euler_field.blend_level_features(level_features, level_logits)
        app_residual = self.field_static_app_head(feature).view(-1, samples, 6).mean(dim=1)
        zeros_t = torch.zeros((app_residual.shape[0], 3), device=app_residual.device, dtype=app_residual.dtype)
        radiance_feature = torch.cat((app_residual, zeros_t), dim=1)

        start_iter = int(getattr(self, "field_static_radiance_start", 3000))
        warmup_iters = max(int(getattr(self, "field_static_radiance_warmup", 1000)), 1)
        warmup = max(0.0, min((float(iteration) - float(start_iter)) / float(warmup_iters), 1.0))
        scale = float(getattr(self, "field_static_radiance_scale", 0.05)) * warmup
        if scale <= 0.0:
            return None

        out = torch.zeros((9, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)), device=depth.device, dtype=app_residual.dtype)
        out[:, y, x] = (scale * radiance_feature).transpose(0, 1)
        return out

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

    def boost_background_candidate_gradients(self, iteration):
        if not bool(getattr(self, "field_bg_candidate_grad_boost", False)):
            return 0
        if self._bg_candidate_mask.numel() == 0 or self._bg_birth_iter.numel() == 0:
            return 0
        num_points = self._xyz.shape[0]
        if self._bg_candidate_mask.shape[0] != num_points or self._bg_birth_iter.shape[0] != num_points:
            return 0

        candidate_mask = (self._bg_candidate_mask.detach().reshape(-1) > 0.5)
        birth_iter = self._bg_birth_iter.detach().reshape(-1)
        protect_iters = max(int(getattr(self, "field_bg_prior_protect_iters", 1500)), 0)
        valid_birth = birth_iter >= 0
        if protect_iters > 0:
            age = float(iteration) - birth_iter
            candidate_mask = candidate_mask & valid_birth & (age >= 0) & (age < float(protect_iters))
        else:
            candidate_mask = candidate_mask & valid_birth

        if torch.count_nonzero(candidate_mask) == 0:
            return 0

        def scale_point_grad(parameter, scale):
            if parameter.grad is None:
                return
            if parameter.grad.shape[0] != num_points:
                return
            scale = float(scale)
            if scale == 1.0:
                return
            factor = torch.ones((num_points,), device=parameter.grad.device, dtype=parameter.grad.dtype)
            factor[candidate_mask.to(device=parameter.grad.device)] = scale
            while factor.dim() < parameter.grad.dim():
                factor = factor.unsqueeze(-1)
            parameter.grad.mul_(factor)

        scale_point_grad(self._features_dc, getattr(self, "field_bg_candidate_feature_grad_scale", 3.0))
        scale_point_grad(self._features_t, getattr(self, "field_bg_candidate_feature_grad_scale", 3.0))
        scale_point_grad(self._opacity, getattr(self, "field_bg_candidate_opacity_grad_scale", 2.0))
        scale_point_grad(self._scaling, getattr(self, "field_bg_candidate_scaling_grad_scale", 1.5))
        return int(torch.count_nonzero(candidate_mask).item())

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
        if self.content_exposure_head is not None:
            for name, param in self.content_exposure_head.named_parameters():
                if param.grad is not None:
                    self.content_exposure_grd[name] = self.content_exposure_grd[name] + param.grad.clone()
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
        for name in self.content_exposure_grd.keys():
            self.content_exposure_grd[name].zero_()

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
        if self.content_exposure_head is not None:
            for name, param in self.content_exposure_head.named_parameters():
                param.grad = self.content_exposure_grd[name] * ratio

    def apply_appearance_only_gradients(self, iteration):
        if not self.field_appearance_only_train or iteration < self.field_appearance_only_start:
            return False
        if self.optimizer is None:
            return False
        allowed = {
            name.strip()
            for name in self.field_appearance_only_allow.split(",")
            if name.strip()
        }
        for group in self.optimizer.param_groups:
            if group.get("name") in allowed:
                continue
            for param in group.get("params", []):
                if param is not None:
                    param.grad = None
        return True

    def apply_soft_geometry_lr(self, iteration):
        if not self.field_soft_geometry_lr or self.optimizer is None:
            return False
        full_lr_groups = {
            name.strip()
            for name in self.field_soft_geometry_full_lr_groups.split(",")
            if name.strip()
        }
        active = iteration >= self.field_soft_geometry_start
        scale = max(float(self.field_soft_geometry_lr_scale), 0.0) if active else 1.0
        for group in self.optimizer.param_groups:
            if group.get("name") in full_lr_groups:
                if "_stegf_base_lr" in group and group.get("name") != "xyz":
                    group["lr"] = group["_stegf_base_lr"]
                continue
            if group.get("name") == "xyz":
                group["lr"] = group["lr"] * scale
            else:
                base_lr = group.get("_stegf_base_lr", group["lr"])
                group["lr"] = base_lr * scale
        return active


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if self.rgbdecoder is not None:
            self.rgbdecoder.cuda()
        self._ensure_content_exposure_head()
        if self.content_exposure_head is not None:
            self.content_exposure_head.cuda()
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
        if self.content_exposure_head is not None:
            l.append({'params': list(self.content_exposure_head.parameters()), 'lr': self.field_content_exposure_lr, "name": "content_exposure"})
        if self.use_euler_field and self.euler_field is not None:
            l.append({'params': [self._static_level_logits], 'lr': training_args.grid_logits_lr, "name": "static_grid_logits"})
            if self._static_radiance_level_logits.numel() > 0:
                l.append({'params': [self._static_radiance_level_logits], 'lr': training_args.grid_logits_lr, "name": "static_radiance_level_logits"})
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
        for group in self.optimizer.param_groups:
            group["_stegf_base_lr"] = group["lr"]
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
            "field_bbox_expand_scale": self.field_bbox_expand_scale,
            "field_bbox_expand_xyz": self.field_bbox_expand_xyz,
            "field_bbox_extra_min": self.field_bbox_extra_min,
            "field_bbox_extra_max": self.field_bbox_extra_max,
            "field_bbox_preserve_cell_size": int(self.field_bbox_preserve_cell_size),
            "field_bbox_frustum_expand": int(self.field_bbox_frustum_expand),
            "field_bbox_frustum_grid": self.field_bbox_frustum_grid,
            "field_bbox_frustum_depth_base": self.field_bbox_frustum_depth_base,
            "field_bbox_frustum_depth_scales": self.field_bbox_frustum_depth_scales,
            "field_bbox_frustum_margin": self.field_bbox_frustum_margin,
            "field_bbox_frustum_max_expand_xyz": self.field_bbox_frustum_max_expand_xyz,
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
            "field_static_temporal_residual": int(self.field_static_temporal_residual),
            "field_static_temporal_frames": self.field_static_temporal_frames,
            "field_static_temporal_scale": self.field_static_temporal_scale,
            "field_static_radiance_branch": int(self.field_static_radiance_branch),
            "field_static_radiance_start": self.field_static_radiance_start,
            "field_static_radiance_warmup": self.field_static_radiance_warmup,
            "field_static_radiance_scale": self.field_static_radiance_scale,
            "field_static_radiance_depth_multiplier": self.field_static_radiance_depth_multiplier,
            "field_static_radiance_samples": self.field_static_radiance_samples,
            "field_static_radiance_max_pixels": self.field_static_radiance_max_pixels,
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
            "field_bg_prior": int(self.field_bg_prior),
            "field_bg_prior_source": self.field_bg_prior_source,
            "field_bg_prior_color_source": self.field_bg_prior_color_source,
            "field_bg_prior_start": self.field_bg_prior_start,
            "field_bg_prior_until": self.field_bg_prior_until,
            "field_bg_prior_interval": self.field_bg_prior_interval,
            "field_bg_prior_loss_weight": self.field_bg_prior_loss_weight,
            "field_bg_prior_visible_threshold": self.field_bg_prior_visible_threshold,
            "field_bg_prior_stability_threshold": self.field_bg_prior_stability_threshold,
            "field_bg_prior_error_quantile": self.field_bg_prior_error_quantile,
            "field_bg_prior_depth_quantile": self.field_bg_prior_depth_quantile,
            "field_bg_prior_max_pixels": self.field_bg_prior_max_pixels,
            "field_bg_prior_num_per_ray": self.field_bg_prior_num_per_ray,
            "field_bg_prior_depth_scale": self.field_bg_prior_depth_scale,
            "field_bg_prior_depth_values": self.field_bg_prior_depth_values,
            "field_bg_prior_opacity": self.field_bg_prior_opacity,
            "field_bg_prior_color_init": self.field_bg_prior_color_init,
            "field_bg_prior_scale_init": self.field_bg_prior_scale_init,
            "field_bg_prior_fixed_scale": self.field_bg_prior_fixed_scale,
            "field_bg_prior_hybrid_knn_scale_threshold": self.field_bg_prior_hybrid_knn_scale_threshold,
            "field_bg_prior_trbf_center": self.field_bg_prior_trbf_center,
            "field_bg_prior_trbf_scale": self.field_bg_prior_trbf_scale,
            "field_bg_prior_protect_iters": self.field_bg_prior_protect_iters,
            "field_bg_prior_mature_prune": int(self.field_bg_prior_mature_prune),
            "field_bg_prior_mature_prune_interval": self.field_bg_prior_mature_prune_interval,
            "field_bg_prior_mature_min_opacity": self.field_bg_prior_mature_min_opacity,
            "field_bg_prior_mature_min_visibility": self.field_bg_prior_mature_min_visibility,
            "field_bg_prior_debug": int(self.field_bg_prior_debug),
            "field_bg_prior_debug_max_events": self.field_bg_prior_debug_max_events,
            "field_bg_prior_debug_mode": self.field_bg_prior_debug_mode,
            "field_bg_prior_schedule_mode": self.field_bg_prior_schedule_mode,
            "field_bg_prior_scan_views_per_event": self.field_bg_prior_scan_views_per_event,
            "field_bg_prior_scan_time_indices": self.field_bg_prior_scan_time_indices,
            "field_bg_prior_block_size": self.field_bg_prior_block_size,
            "field_bg_prior_pixels_per_block": self.field_bg_prior_pixels_per_block,
            "field_bg_prior_strict_max_pixels": self.field_bg_prior_strict_max_pixels,
            "field_bg_prior_strict_pixels_per_block": self.field_bg_prior_strict_pixels_per_block,
            "field_bg_prior_recall_max_pixels": self.field_bg_prior_recall_max_pixels,
            "field_bg_prior_recall_pixels_per_block": self.field_bg_prior_recall_pixels_per_block,
            "field_bg_prior_recall_error_quantile": self.field_bg_prior_recall_error_quantile,
            "field_bg_prior_recall_min_visible_ratio": self.field_bg_prior_recall_min_visible_ratio,
            "field_bg_prior_recall_min_stable_ratio": self.field_bg_prior_recall_min_stable_ratio,
            "field_bg_prior_recall_max_occlusion_ratio": self.field_bg_prior_recall_max_occlusion_ratio,
            "field_bg_prior_unreliable_max_pixels": self.field_bg_prior_unreliable_max_pixels,
            "field_bg_prior_unreliable_pixels_per_block": self.field_bg_prior_unreliable_pixels_per_block,
            "field_bg_prior_unreliable_error_quantile": self.field_bg_prior_unreliable_error_quantile,
            "field_bg_prior_min_visible_ratio": self.field_bg_prior_min_visible_ratio,
            "field_bg_prior_min_stable_ratio": self.field_bg_prior_min_stable_ratio,
            "field_bg_prior_max_occlusion_ratio": self.field_bg_prior_max_occlusion_ratio,
            "field_bg_prior_occlusion_threshold": self.field_bg_prior_occlusion_threshold,
            "field_bg_prior_occlusion_dilate": self.field_bg_prior_occlusion_dilate,
            "field_bg_prior_exposure_robust": int(self.field_bg_prior_exposure_robust),
            "field_bg_prior_structural_weight": self.field_bg_prior_structural_weight,
            "field_bg_prior_local_window": self.field_bg_prior_local_window,
            "field_bg_prior_fixed_depth": int(self.field_bg_prior_fixed_depth),
            "field_bg_prior_fixed_depth_ratio": self.field_bg_prior_fixed_depth_ratio,
            "field_bg_prior_depth_max": self.field_bg_prior_depth_max,
            "field_bg_prior_suppress": int(self.field_bg_prior_suppress),
            "field_bg_prior_suppress_decay": self.field_bg_prior_suppress_decay,
            "field_bg_prior_suppress_max_points": self.field_bg_prior_suppress_max_points,
            "field_bg_prior_suppress_depth_margin": self.field_bg_prior_suppress_depth_margin,
            "field_bg_prior_suppress_opacity_threshold": self.field_bg_prior_suppress_opacity_threshold,
            "field_bg_prior_suppress_scale_quantile": self.field_bg_prior_suppress_scale_quantile,
            "field_bg_prior_clone_split": int(self.field_bg_prior_clone_split),
            "field_bg_prior_clone_stat_start": self.field_bg_prior_clone_stat_start,
            "field_bg_prior_clone_start": self.field_bg_prior_clone_start,
            "field_bg_prior_clone_until": self.field_bg_prior_clone_until,
            "field_bg_prior_clone_interval": self.field_bg_prior_clone_interval,
            "field_bg_prior_clone_grad_threshold": self.field_bg_prior_clone_grad_threshold,
            "field_bg_prior_clone_max_ratio": self.field_bg_prior_clone_max_ratio,
            "field_bg_prior_clone_max_points": self.field_bg_prior_clone_max_points,
            "field_bg_prior_clone_min_age": self.field_bg_prior_clone_min_age,
            "field_bg_prior_clone_min_opacity": self.field_bg_prior_clone_min_opacity,
            "field_bg_prior_clone_min_visibility": self.field_bg_prior_clone_min_visibility,
            "field_bg_prior_clone_split_children": self.field_bg_prior_clone_split_children,
            "field_bg_prior_keep_split_parent": int(self.field_bg_prior_keep_split_parent),
            "field_bg_dense_add": int(self.field_bg_dense_add),
            "field_bg_dense_add_iter": self.field_bg_dense_add_iter,
            "field_bg_dense_add_time_indices": self.field_bg_dense_add_time_indices,
            "field_bg_dense_depth_base": self.field_bg_dense_depth_base,
            "field_bg_dense_depth_scales": self.field_bg_dense_depth_scales,
            "field_bg_dense_depth_values": self.field_bg_dense_depth_values,
            "field_bg_dense_mask_source": self.field_bg_dense_mask_source,
            "field_bg_dense_sample_block_size": self.field_bg_dense_sample_block_size,
            "field_bg_dense_pixels_per_block": self.field_bg_dense_pixels_per_block,
            "field_bg_dense_max_pixels_per_camera": self.field_bg_dense_max_pixels_per_camera,
            "field_bg_dense_debug": int(self.field_bg_dense_debug),
            "field_bg_dense_debug_max_events": self.field_bg_dense_debug_max_events,
            "field_bg_dense_da3_filter": int(self.field_bg_dense_da3_filter),
            "field_bg_dense_da3_path": self.field_bg_dense_da3_path,
            "field_bg_dense_da3_foreground_quantile": self.field_bg_dense_da3_foreground_quantile,
            "field_bg_dense_beit_filter": int(self.field_bg_dense_beit_filter),
            "field_bg_dense_beit_path": self.field_bg_dense_beit_path,
            "field_bg_dense_beit_band_low": self.field_bg_dense_beit_band_low,
            "field_bg_dense_beit_band_high": self.field_bg_dense_beit_band_high,
            "field_bg_dense_beit_threshold": self.field_bg_dense_beit_threshold,
            "field_bg_dense_cell_dedup": int(self.field_bg_dense_cell_dedup),
            "field_bg_dense_dedup_level": self.field_bg_dense_dedup_level,
            "field_bg_dense_dedup_priority": self.field_bg_dense_dedup_priority,
            "field_bg_dense_max_per_cell": self.field_bg_dense_max_per_cell,
            "field_bg_dense_skip_control_at_add_iter": int(self.field_bg_dense_skip_control_at_add_iter),
            "field_bg_dense_clip_to_bbox": int(self.field_bg_dense_clip_to_bbox),
            "field_bg_dense_bbox_clip_margin": self.field_bg_dense_bbox_clip_margin,
            "field_highfreq_densify": int(self.field_highfreq_densify),
            "field_highfreq_densify_sigma_divisor": self.field_highfreq_densify_sigma_divisor,
            "field_highfreq_densify_eps": self.field_highfreq_densify_eps,
            "field_highfreq_densify_y_min": self.field_highfreq_densify_y_min,
            "field_highfreq_densify_y_max": self.field_highfreq_densify_y_max,
            "field_highfreq_densify_min_pixels": self.field_highfreq_densify_min_pixels,
            "field_highfreq_densify_gate_start": self.field_highfreq_densify_gate_start,
            "field_highfreq_densify_gate_width": self.field_highfreq_densify_gate_width,
            "field_appearance_only_train": int(self.field_appearance_only_train),
            "field_appearance_only_start": self.field_appearance_only_start,
            "field_appearance_only_allow": self.field_appearance_only_allow,
            "field_soft_geometry_lr": int(self.field_soft_geometry_lr),
            "field_soft_geometry_start": self.field_soft_geometry_start,
            "field_soft_geometry_lr_scale": self.field_soft_geometry_lr_scale,
            "field_soft_geometry_full_lr_groups": self.field_soft_geometry_full_lr_groups,
            "field_content_exposure": int(self.field_content_exposure),
            "field_content_exposure_lr": self.field_content_exposure_lr,
            "field_content_exposure_hidden": self.field_content_exposure_hidden,
            "field_content_exposure_mode": self.field_content_exposure_mode,
            "field_content_exposure_max_log_scale": self.field_content_exposure_max_log_scale,
            "field_content_exposure_max_bias": self.field_content_exposure_max_bias,
            "field_content_exposure_max_wb_log_gain": self.field_content_exposure_max_wb_log_gain,
            "field_content_exposure_reg_weight": self.field_content_exposure_reg_weight,
            "field_content_exposure_wb_reg_weight": self.field_content_exposure_wb_reg_weight,
            "field_content_exposure_eps": self.field_content_exposure_eps,
            "field_content_exposure_detach_stats": int(self.field_content_exposure_detach_stats),
            "field_depthpro_supervision": int(self.field_depthpro_supervision),
            "field_depthpro_path": self.field_depthpro_path,
            "field_depthpro_start": self.field_depthpro_start,
            "field_depthpro_until": self.field_depthpro_until,
            "field_depthpro_loss_weight": self.field_depthpro_loss_weight,
            "field_depthpro_max_depth": self.field_depthpro_max_depth,
            "field_depthpro_min_pixels": self.field_depthpro_min_pixels,
            "field_depthpro_error_clamp": self.field_depthpro_error_clamp,
            "field_depthpro_use_beit_mask": int(self.field_depthpro_use_beit_mask),
            "field_depthpro_exclude_unreliable": int(self.field_depthpro_exclude_unreliable),
            "field_scale_reg": int(self.field_scale_reg),
            "field_scale_reg_start": self.field_scale_reg_start,
            "field_scale_reg_until": self.field_scale_reg_until,
            "field_scale_reg_weight": self.field_scale_reg_weight,
            "field_scale_reg_base_limit": self.field_scale_reg_base_limit,
            "field_scale_reg_depth_ref": self.field_scale_reg_depth_ref,
            "field_scale_reg_depth_mode": self.field_scale_reg_depth_mode,
            "field_scale_reg_depth_gamma": self.field_scale_reg_depth_gamma,
            "field_scale_reg_max_boost": self.field_scale_reg_max_boost,
            "field_bg_candidate_grad_boost": int(self.field_bg_candidate_grad_boost),
            "field_bg_candidate_feature_grad_scale": self.field_bg_candidate_feature_grad_scale,
            "field_bg_candidate_opacity_grad_scale": self.field_bg_candidate_opacity_grad_scale,
            "field_bg_candidate_scaling_grad_scale": self.field_bg_candidate_scaling_grad_scale,
            "field_bg_only_train": int(self.field_bg_only_train),
            "field_bg_only_start": self.field_bg_only_start,
            "field_bg_only_until": self.field_bg_only_until,
            "field_bg_only_interval": self.field_bg_only_interval,
            "field_bg_only_loss_weight": self.field_bg_only_loss_weight,
            "field_bg_only_min_pixels": self.field_bg_only_min_pixels,
            "field_bg_only_da3_filter": int(self.field_bg_only_da3_filter),
            "field_bg_only_update_modules": int(self.field_bg_only_update_modules),
            "field_obs_reliability": int(self.field_obs_reliability),
            "field_obs_reliability_floor": self.field_obs_reliability_floor,
            "field_obs_reliability_mad_threshold": self.field_obs_reliability_mad_threshold,
            "field_obs_reliability_diff_threshold": self.field_obs_reliability_diff_threshold,
            "field_obs_reliability_motion_threshold": self.field_obs_reliability_motion_threshold,
            "field_obs_reliability_mad_weight": self.field_obs_reliability_mad_weight,
            "field_obs_reliability_diff_weight": self.field_obs_reliability_diff_weight,
            "field_obs_reliability_motion_weight": self.field_obs_reliability_motion_weight,
            "field_obs_reliability_unreliable_threshold": self.field_obs_reliability_unreliable_threshold,
            "field_obs_reliability_debug": int(self.field_obs_reliability_debug),
            "field_obs_reliability_start": self.field_obs_reliability_start,
            "field_obs_reliability_until": self.field_obs_reliability_until,
            "field_obs_reliability_ema": self.field_obs_reliability_ema,
            "field_obs_reliability_error_quantile": self.field_obs_reliability_error_quantile,
            "field_obs_reliability_error_threshold": self.field_obs_reliability_error_threshold,
            "field_obs_reliability_min_error": self.field_obs_reliability_min_error,
            "field_obs_reliability_dynamic_dilate": self.field_obs_reliability_dynamic_dilate,
            "field_obs_reliability_structural_weight": self.field_obs_reliability_structural_weight,
            "field_obs_reliability_local_window": self.field_obs_reliability_local_window,
            "field_obs_boost_unreliable_loss": int(self.field_obs_boost_unreliable_loss),
            "field_obs_boost_weight": self.field_obs_boost_weight,
            "field_obs_reset": int(self.field_obs_reset),
            "field_obs_reset_mode": self.field_obs_reset_mode,
            "field_obs_reset_start": self.field_obs_reset_start,
            "field_obs_reset_until": self.field_obs_reset_until,
            "field_obs_reset_interval": self.field_obs_reset_interval,
            "field_obs_reset_schedule": self.field_obs_reset_schedule,
            "field_obs_reset_opacity": self.field_obs_reset_opacity,
            "field_obs_reset_min_opacity": self.field_obs_reset_min_opacity,
            "field_obs_reset_max_points": self.field_obs_reset_max_points,
            "field_obs_reset_selection_mode": self.field_obs_reset_selection_mode,
            "field_obs_reset_min_masked_contrib": self.field_obs_reset_min_masked_contrib,
            "field_obs_reset_min_contrib_ratio": self.field_obs_reset_min_contrib_ratio,
            "field_obs_reset_debug": int(self.field_obs_reset_debug),
            "field_obs_reset_debug_max_events": self.field_obs_reset_debug_max_events,
            "field_obs_reset_log_zero": int(self.field_obs_reset_log_zero),
            "field_obs_reset_scan_time_indices": self.field_obs_reset_scan_time_indices,
            "field_obs_reset_scan_views_per_time": self.field_obs_reset_scan_views_per_time,
            "field_obs_reset_scan_min_hits": self.field_obs_reset_scan_min_hits,
            "field_obs_reset_scan_top_ratio": self.field_obs_reset_scan_top_ratio,
            "field_obs_reset_scan_max_points": self.field_obs_reset_scan_max_points,
            "field_obs_reset_scan_update_ema": int(self.field_obs_reset_scan_update_ema),
            "field_global_reset": int(self.field_global_reset),
            "field_global_reset_schedule": self.field_global_reset_schedule,
            "field_freq_prior": int(self.field_freq_prior),
            "field_freq_prior_start": self.field_freq_prior_start,
            "field_freq_prior_until": self.field_freq_prior_until,
            "field_freq_prior_weight": self.field_freq_prior_weight,
            "field_freq_prior_patch_size": self.field_freq_prior_patch_size,
            "field_freq_prior_highpass": self.field_freq_prior_highpass,
            "field_freq_prior_max_patches": self.field_freq_prior_max_patches,
            "field_freq_prior_min_mask_ratio": self.field_freq_prior_min_mask_ratio,
            "field_freq_prior_reference": self.field_freq_prior_reference,
            "field_freq_prior_on_reset_only": int(self.field_freq_prior_on_reset_only),
            "field_freq_prior_debug": int(self.field_freq_prior_debug),
            "field_freq_prior_debug_max_events": self.field_freq_prior_debug_max_events,
            "field_freq_prior_debug_mode": self.field_freq_prior_debug_mode,
            "field_bg_median_loss": int(self.field_bg_median_loss),
            "field_bg_median_loss_weight": self.field_bg_median_loss_weight,
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
            self.field_bbox_expand_scale = float(config.get("field_bbox_expand_scale", self.field_bbox_expand_scale))
            self.field_bbox_expand_xyz = config.get("field_bbox_expand_xyz", self.field_bbox_expand_xyz)
            self.field_bbox_extra_min = config.get("field_bbox_extra_min", self.field_bbox_extra_min)
            self.field_bbox_extra_max = config.get("field_bbox_extra_max", self.field_bbox_extra_max)
            self.field_bbox_preserve_cell_size = bool(config.get("field_bbox_preserve_cell_size", int(self.field_bbox_preserve_cell_size)))
            self.field_bbox_frustum_expand = bool(config.get("field_bbox_frustum_expand", int(self.field_bbox_frustum_expand)))
            self.field_bbox_frustum_grid = int(config.get("field_bbox_frustum_grid", self.field_bbox_frustum_grid))
            self.field_bbox_frustum_depth_base = str(config.get("field_bbox_frustum_depth_base", self.field_bbox_frustum_depth_base))
            self.field_bbox_frustum_depth_scales = config.get("field_bbox_frustum_depth_scales", self.field_bbox_frustum_depth_scales)
            self.field_bbox_frustum_margin = float(config.get("field_bbox_frustum_margin", self.field_bbox_frustum_margin))
            self.field_bbox_frustum_max_expand_xyz = config.get("field_bbox_frustum_max_expand_xyz", self.field_bbox_frustum_max_expand_xyz)
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
            self.field_static_temporal_residual = bool(config.get("field_static_temporal_residual", int(self.field_static_temporal_residual)))
            self.field_static_temporal_frames = int(config.get("field_static_temporal_frames", self.field_static_temporal_frames))
            self.field_static_temporal_scale = float(config.get("field_static_temporal_scale", self.field_static_temporal_scale))
            self.field_static_radiance_branch = bool(config.get("field_static_radiance_branch", int(self.field_static_radiance_branch)))
            self.field_static_radiance_start = int(config.get("field_static_radiance_start", self.field_static_radiance_start))
            self.field_static_radiance_warmup = int(config.get("field_static_radiance_warmup", self.field_static_radiance_warmup))
            self.field_static_radiance_scale = float(config.get("field_static_radiance_scale", self.field_static_radiance_scale))
            self.field_static_radiance_depth_multiplier = float(config.get("field_static_radiance_depth_multiplier", self.field_static_radiance_depth_multiplier))
            self.field_static_radiance_samples = int(config.get("field_static_radiance_samples", self.field_static_radiance_samples))
            self.field_static_radiance_max_pixels = int(config.get("field_static_radiance_max_pixels", self.field_static_radiance_max_pixels))
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
            self.field_bg_prior = bool(config.get("field_bg_prior", int(self.field_bg_prior)))
            self.field_bg_prior_start = int(config.get("field_bg_prior_start", self.field_bg_prior_start))
            self.field_bg_prior_source = str(config.get("field_bg_prior_source", self.field_bg_prior_source))
            self.field_bg_prior_color_source = str(config.get("field_bg_prior_color_source", self.field_bg_prior_color_source))
            self.field_bg_prior_until = int(config.get("field_bg_prior_until", self.field_bg_prior_until))
            self.field_bg_prior_interval = int(config.get("field_bg_prior_interval", self.field_bg_prior_interval))
            self.field_bg_prior_loss_weight = float(config.get("field_bg_prior_loss_weight", self.field_bg_prior_loss_weight))
            self.field_bg_prior_visible_threshold = float(config.get("field_bg_prior_visible_threshold", self.field_bg_prior_visible_threshold))
            self.field_bg_prior_stability_threshold = float(config.get("field_bg_prior_stability_threshold", self.field_bg_prior_stability_threshold))
            self.field_bg_prior_error_quantile = float(config.get("field_bg_prior_error_quantile", self.field_bg_prior_error_quantile))
            self.field_bg_prior_depth_quantile = float(config.get("field_bg_prior_depth_quantile", self.field_bg_prior_depth_quantile))
            self.field_bg_prior_max_pixels = int(config.get("field_bg_prior_max_pixels", self.field_bg_prior_max_pixels))
            self.field_bg_prior_num_per_ray = int(config.get("field_bg_prior_num_per_ray", self.field_bg_prior_num_per_ray))
            self.field_bg_prior_depth_scale = float(config.get("field_bg_prior_depth_scale", self.field_bg_prior_depth_scale))
            self.field_bg_prior_depth_values = str(config.get("field_bg_prior_depth_values", self.field_bg_prior_depth_values))
            self.field_bg_prior_opacity = float(config.get("field_bg_prior_opacity", self.field_bg_prior_opacity))
            self.field_bg_prior_color_init = str(config.get("field_bg_prior_color_init", self.field_bg_prior_color_init))
            self.field_bg_prior_scale_init = str(config.get("field_bg_prior_scale_init", self.field_bg_prior_scale_init))
            self.field_bg_prior_fixed_scale = float(config.get("field_bg_prior_fixed_scale", self.field_bg_prior_fixed_scale))
            self.field_bg_prior_hybrid_knn_scale_threshold = float(config.get("field_bg_prior_hybrid_knn_scale_threshold", self.field_bg_prior_hybrid_knn_scale_threshold))
            self.field_bg_prior_trbf_center = float(config.get("field_bg_prior_trbf_center", self.field_bg_prior_trbf_center))
            self.field_bg_prior_trbf_scale = float(config.get("field_bg_prior_trbf_scale", self.field_bg_prior_trbf_scale))
            self.field_bg_prior_protect_iters = int(config.get("field_bg_prior_protect_iters", self.field_bg_prior_protect_iters))
            self.field_bg_prior_mature_prune = bool(config.get("field_bg_prior_mature_prune", int(self.field_bg_prior_mature_prune)))
            self.field_bg_prior_mature_prune_interval = int(config.get("field_bg_prior_mature_prune_interval", self.field_bg_prior_mature_prune_interval))
            self.field_bg_prior_mature_min_opacity = float(config.get("field_bg_prior_mature_min_opacity", self.field_bg_prior_mature_min_opacity))
            self.field_bg_prior_mature_min_visibility = float(config.get("field_bg_prior_mature_min_visibility", self.field_bg_prior_mature_min_visibility))
            self.field_bg_prior_debug = bool(config.get("field_bg_prior_debug", int(self.field_bg_prior_debug)))
            self.field_bg_prior_debug_max_events = int(config.get("field_bg_prior_debug_max_events", self.field_bg_prior_debug_max_events))
            self.field_bg_prior_debug_mode = str(config.get("field_bg_prior_debug_mode", self.field_bg_prior_debug_mode))
            self.field_bg_prior_schedule_mode = str(config.get("field_bg_prior_schedule_mode", self.field_bg_prior_schedule_mode))
            self.field_bg_prior_scan_views_per_event = int(config.get("field_bg_prior_scan_views_per_event", self.field_bg_prior_scan_views_per_event))
            self.field_bg_prior_scan_time_indices = str(config.get("field_bg_prior_scan_time_indices", self.field_bg_prior_scan_time_indices))
            self.field_bg_prior_block_size = int(config.get("field_bg_prior_block_size", self.field_bg_prior_block_size))
            self.field_bg_prior_pixels_per_block = int(config.get("field_bg_prior_pixels_per_block", self.field_bg_prior_pixels_per_block))
            self.field_bg_prior_strict_max_pixels = int(config.get("field_bg_prior_strict_max_pixels", self.field_bg_prior_strict_max_pixels))
            self.field_bg_prior_strict_pixels_per_block = int(config.get("field_bg_prior_strict_pixels_per_block", self.field_bg_prior_strict_pixels_per_block))
            self.field_bg_prior_recall_max_pixels = int(config.get("field_bg_prior_recall_max_pixels", self.field_bg_prior_recall_max_pixels))
            self.field_bg_prior_recall_pixels_per_block = int(config.get("field_bg_prior_recall_pixels_per_block", self.field_bg_prior_recall_pixels_per_block))
            self.field_bg_prior_recall_error_quantile = float(config.get("field_bg_prior_recall_error_quantile", self.field_bg_prior_recall_error_quantile))
            self.field_bg_prior_recall_min_visible_ratio = float(config.get("field_bg_prior_recall_min_visible_ratio", self.field_bg_prior_recall_min_visible_ratio))
            self.field_bg_prior_recall_min_stable_ratio = float(config.get("field_bg_prior_recall_min_stable_ratio", self.field_bg_prior_recall_min_stable_ratio))
            self.field_bg_prior_recall_max_occlusion_ratio = float(config.get("field_bg_prior_recall_max_occlusion_ratio", self.field_bg_prior_recall_max_occlusion_ratio))
            self.field_bg_prior_unreliable_max_pixels = int(config.get("field_bg_prior_unreliable_max_pixels", self.field_bg_prior_unreliable_max_pixels))
            self.field_bg_prior_unreliable_pixels_per_block = int(config.get("field_bg_prior_unreliable_pixels_per_block", self.field_bg_prior_unreliable_pixels_per_block))
            self.field_bg_prior_unreliable_error_quantile = float(config.get("field_bg_prior_unreliable_error_quantile", self.field_bg_prior_unreliable_error_quantile))
            self.field_bg_prior_min_visible_ratio = float(config.get("field_bg_prior_min_visible_ratio", self.field_bg_prior_min_visible_ratio))
            self.field_bg_prior_min_stable_ratio = float(config.get("field_bg_prior_min_stable_ratio", self.field_bg_prior_min_stable_ratio))
            self.field_bg_prior_max_occlusion_ratio = float(config.get("field_bg_prior_max_occlusion_ratio", self.field_bg_prior_max_occlusion_ratio))
            self.field_bg_prior_occlusion_threshold = float(config.get("field_bg_prior_occlusion_threshold", self.field_bg_prior_occlusion_threshold))
            self.field_bg_prior_occlusion_dilate = int(config.get("field_bg_prior_occlusion_dilate", self.field_bg_prior_occlusion_dilate))
            self.field_bg_prior_exposure_robust = bool(config.get("field_bg_prior_exposure_robust", int(self.field_bg_prior_exposure_robust)))
            self.field_bg_prior_structural_weight = float(config.get("field_bg_prior_structural_weight", self.field_bg_prior_structural_weight))
            self.field_bg_prior_local_window = int(config.get("field_bg_prior_local_window", self.field_bg_prior_local_window))
            self.field_bg_prior_fixed_depth = bool(config.get("field_bg_prior_fixed_depth", int(self.field_bg_prior_fixed_depth)))
            self.field_bg_prior_fixed_depth_ratio = float(config.get("field_bg_prior_fixed_depth_ratio", self.field_bg_prior_fixed_depth_ratio))
            self.field_bg_prior_depth_max = float(config.get("field_bg_prior_depth_max", self.field_bg_prior_depth_max))
            self.field_bg_prior_suppress = bool(config.get("field_bg_prior_suppress", int(self.field_bg_prior_suppress)))
            self.field_bg_prior_suppress_decay = float(config.get("field_bg_prior_suppress_decay", self.field_bg_prior_suppress_decay))
            self.field_bg_prior_suppress_max_points = int(config.get("field_bg_prior_suppress_max_points", self.field_bg_prior_suppress_max_points))
            self.field_bg_prior_suppress_depth_margin = float(config.get("field_bg_prior_suppress_depth_margin", self.field_bg_prior_suppress_depth_margin))
            self.field_bg_prior_suppress_opacity_threshold = float(config.get("field_bg_prior_suppress_opacity_threshold", self.field_bg_prior_suppress_opacity_threshold))
            self.field_bg_prior_suppress_scale_quantile = float(config.get("field_bg_prior_suppress_scale_quantile", self.field_bg_prior_suppress_scale_quantile))
            self.field_bg_prior_clone_split = bool(config.get("field_bg_prior_clone_split", int(self.field_bg_prior_clone_split)))
            self.field_bg_prior_clone_stat_start = int(config.get("field_bg_prior_clone_stat_start", self.field_bg_prior_clone_stat_start))
            self.field_bg_prior_clone_start = int(config.get("field_bg_prior_clone_start", self.field_bg_prior_clone_start))
            self.field_bg_prior_clone_until = int(config.get("field_bg_prior_clone_until", self.field_bg_prior_clone_until))
            self.field_bg_prior_clone_interval = int(config.get("field_bg_prior_clone_interval", self.field_bg_prior_clone_interval))
            self.field_bg_prior_clone_grad_threshold = float(config.get("field_bg_prior_clone_grad_threshold", self.field_bg_prior_clone_grad_threshold))
            self.field_bg_prior_clone_max_ratio = float(config.get("field_bg_prior_clone_max_ratio", self.field_bg_prior_clone_max_ratio))
            self.field_bg_prior_clone_max_points = int(config.get("field_bg_prior_clone_max_points", self.field_bg_prior_clone_max_points))
            self.field_bg_prior_clone_min_age = int(config.get("field_bg_prior_clone_min_age", self.field_bg_prior_clone_min_age))
            self.field_bg_prior_clone_min_opacity = float(config.get("field_bg_prior_clone_min_opacity", self.field_bg_prior_clone_min_opacity))
            self.field_bg_prior_clone_min_visibility = float(config.get("field_bg_prior_clone_min_visibility", self.field_bg_prior_clone_min_visibility))
            self.field_bg_prior_clone_split_children = int(config.get("field_bg_prior_clone_split_children", self.field_bg_prior_clone_split_children))
            self.field_bg_prior_keep_split_parent = bool(config.get("field_bg_prior_keep_split_parent", int(self.field_bg_prior_keep_split_parent)))
            self.field_bg_dense_add = bool(config.get("field_bg_dense_add", int(self.field_bg_dense_add)))
            self.field_bg_dense_add_iter = int(config.get("field_bg_dense_add_iter", self.field_bg_dense_add_iter))
            self.field_bg_dense_add_time_indices = str(config.get("field_bg_dense_add_time_indices", self.field_bg_dense_add_time_indices))
            self.field_bg_dense_depth_base = str(config.get("field_bg_dense_depth_base", self.field_bg_dense_depth_base))
            self.field_bg_dense_depth_scales = str(config.get("field_bg_dense_depth_scales", self.field_bg_dense_depth_scales))
            self.field_bg_dense_depth_values = str(config.get("field_bg_dense_depth_values", self.field_bg_dense_depth_values))
            self.field_bg_dense_mask_source = str(config.get("field_bg_dense_mask_source", self.field_bg_dense_mask_source))
            self.field_bg_dense_sample_block_size = int(config.get("field_bg_dense_sample_block_size", self.field_bg_dense_sample_block_size))
            self.field_bg_dense_pixels_per_block = int(config.get("field_bg_dense_pixels_per_block", self.field_bg_dense_pixels_per_block))
            self.field_bg_dense_max_pixels_per_camera = int(config.get("field_bg_dense_max_pixels_per_camera", self.field_bg_dense_max_pixels_per_camera))
            self.field_bg_dense_debug = bool(config.get("field_bg_dense_debug", int(self.field_bg_dense_debug)))
            self.field_bg_dense_debug_max_events = int(config.get("field_bg_dense_debug_max_events", self.field_bg_dense_debug_max_events))
            self.field_bg_dense_da3_filter = bool(config.get("field_bg_dense_da3_filter", int(self.field_bg_dense_da3_filter)))
            self.field_bg_dense_da3_path = str(config.get("field_bg_dense_da3_path", self.field_bg_dense_da3_path))
            self.field_bg_dense_da3_foreground_quantile = float(config.get("field_bg_dense_da3_foreground_quantile", self.field_bg_dense_da3_foreground_quantile))
            self.field_bg_dense_beit_filter = bool(config.get("field_bg_dense_beit_filter", int(self.field_bg_dense_beit_filter)))
            self.field_bg_dense_beit_path = str(config.get("field_bg_dense_beit_path", self.field_bg_dense_beit_path))
            self.field_bg_dense_beit_band_low = float(config.get("field_bg_dense_beit_band_low", self.field_bg_dense_beit_band_low))
            self.field_bg_dense_beit_band_high = float(config.get("field_bg_dense_beit_band_high", self.field_bg_dense_beit_band_high))
            self.field_bg_dense_beit_threshold = float(config.get("field_bg_dense_beit_threshold", self.field_bg_dense_beit_threshold))
            self.field_bg_dense_cell_dedup = bool(config.get("field_bg_dense_cell_dedup", int(self.field_bg_dense_cell_dedup)))
            self.field_bg_dense_dedup_level = int(config.get("field_bg_dense_dedup_level", self.field_bg_dense_dedup_level))
            self.field_bg_dense_dedup_priority = str(config.get("field_bg_dense_dedup_priority", self.field_bg_dense_dedup_priority))
            self.field_bg_dense_max_per_cell = int(config.get("field_bg_dense_max_per_cell", self.field_bg_dense_max_per_cell))
            self.field_bg_dense_skip_control_at_add_iter = bool(config.get("field_bg_dense_skip_control_at_add_iter", int(self.field_bg_dense_skip_control_at_add_iter)))
            self.field_bg_dense_clip_to_bbox = bool(config.get("field_bg_dense_clip_to_bbox", int(self.field_bg_dense_clip_to_bbox)))
            self.field_bg_dense_bbox_clip_margin = float(config.get("field_bg_dense_bbox_clip_margin", self.field_bg_dense_bbox_clip_margin))
            self.field_highfreq_densify = bool(config.get("field_highfreq_densify", int(self.field_highfreq_densify)))
            self.field_highfreq_densify_sigma_divisor = float(config.get("field_highfreq_densify_sigma_divisor", self.field_highfreq_densify_sigma_divisor))
            self.field_highfreq_densify_eps = float(config.get("field_highfreq_densify_eps", self.field_highfreq_densify_eps))
            self.field_highfreq_densify_y_min = float(config.get("field_highfreq_densify_y_min", self.field_highfreq_densify_y_min))
            self.field_highfreq_densify_y_max = float(config.get("field_highfreq_densify_y_max", self.field_highfreq_densify_y_max))
            self.field_highfreq_densify_min_pixels = int(config.get("field_highfreq_densify_min_pixels", self.field_highfreq_densify_min_pixels))
            self.field_highfreq_densify_gate_start = float(config.get("field_highfreq_densify_gate_start", self.field_highfreq_densify_gate_start))
            self.field_highfreq_densify_gate_width = float(config.get("field_highfreq_densify_gate_width", self.field_highfreq_densify_gate_width))
            self.field_appearance_only_train = bool(config.get("field_appearance_only_train", int(self.field_appearance_only_train)))
            self.field_appearance_only_start = int(config.get("field_appearance_only_start", self.field_appearance_only_start))
            self.field_appearance_only_allow = str(config.get("field_appearance_only_allow", self.field_appearance_only_allow))
            self.field_soft_geometry_lr = bool(config.get("field_soft_geometry_lr", int(self.field_soft_geometry_lr)))
            self.field_soft_geometry_start = int(config.get("field_soft_geometry_start", self.field_soft_geometry_start))
            self.field_soft_geometry_lr_scale = float(config.get("field_soft_geometry_lr_scale", self.field_soft_geometry_lr_scale))
            self.field_soft_geometry_full_lr_groups = str(config.get("field_soft_geometry_full_lr_groups", self.field_soft_geometry_full_lr_groups))
            self.field_content_exposure = bool(config.get("field_content_exposure", int(self.field_content_exposure)))
            self.field_content_exposure_lr = float(config.get("field_content_exposure_lr", self.field_content_exposure_lr))
            self.field_content_exposure_hidden = int(config.get("field_content_exposure_hidden", self.field_content_exposure_hidden))
            self.field_content_exposure_mode = str(config.get("field_content_exposure_mode", self.field_content_exposure_mode)).lower()
            self.field_content_exposure_max_log_scale = float(config.get("field_content_exposure_max_log_scale", self.field_content_exposure_max_log_scale))
            self.field_content_exposure_max_bias = float(config.get("field_content_exposure_max_bias", self.field_content_exposure_max_bias))
            self.field_content_exposure_max_wb_log_gain = float(config.get("field_content_exposure_max_wb_log_gain", self.field_content_exposure_max_wb_log_gain))
            self.field_content_exposure_reg_weight = float(config.get("field_content_exposure_reg_weight", self.field_content_exposure_reg_weight))
            self.field_content_exposure_wb_reg_weight = float(config.get("field_content_exposure_wb_reg_weight", self.field_content_exposure_wb_reg_weight))
            self.field_content_exposure_eps = float(config.get("field_content_exposure_eps", self.field_content_exposure_eps))
            self.field_content_exposure_detach_stats = bool(config.get("field_content_exposure_detach_stats", int(self.field_content_exposure_detach_stats)))
            self.field_depthpro_supervision = bool(config.get("field_depthpro_supervision", int(self.field_depthpro_supervision)))
            self.field_depthpro_path = str(config.get("field_depthpro_path", self.field_depthpro_path))
            self.field_depthpro_start = int(config.get("field_depthpro_start", self.field_depthpro_start))
            self.field_depthpro_until = int(config.get("field_depthpro_until", self.field_depthpro_until))
            self.field_depthpro_loss_weight = float(config.get("field_depthpro_loss_weight", self.field_depthpro_loss_weight))
            self.field_depthpro_max_depth = float(config.get("field_depthpro_max_depth", self.field_depthpro_max_depth))
            self.field_depthpro_min_pixels = int(config.get("field_depthpro_min_pixels", self.field_depthpro_min_pixels))
            self.field_depthpro_error_clamp = float(config.get("field_depthpro_error_clamp", self.field_depthpro_error_clamp))
            self.field_depthpro_use_beit_mask = bool(config.get("field_depthpro_use_beit_mask", int(self.field_depthpro_use_beit_mask)))
            self.field_depthpro_exclude_unreliable = bool(config.get("field_depthpro_exclude_unreliable", int(self.field_depthpro_exclude_unreliable)))
            self.field_scale_reg = bool(config.get("field_scale_reg", int(self.field_scale_reg)))
            self.field_scale_reg_start = int(config.get("field_scale_reg_start", self.field_scale_reg_start))
            self.field_scale_reg_until = int(config.get("field_scale_reg_until", self.field_scale_reg_until))
            self.field_scale_reg_weight = float(config.get("field_scale_reg_weight", self.field_scale_reg_weight))
            self.field_scale_reg_base_limit = float(config.get("field_scale_reg_base_limit", self.field_scale_reg_base_limit))
            self.field_scale_reg_depth_ref = float(config.get("field_scale_reg_depth_ref", self.field_scale_reg_depth_ref))
            self.field_scale_reg_depth_mode = str(config.get("field_scale_reg_depth_mode", self.field_scale_reg_depth_mode))
            self.field_scale_reg_depth_gamma = float(config.get("field_scale_reg_depth_gamma", self.field_scale_reg_depth_gamma))
            self.field_scale_reg_max_boost = float(config.get("field_scale_reg_max_boost", self.field_scale_reg_max_boost))
            self.field_bg_candidate_grad_boost = bool(config.get("field_bg_candidate_grad_boost", int(self.field_bg_candidate_grad_boost)))
            self.field_bg_candidate_feature_grad_scale = float(config.get("field_bg_candidate_feature_grad_scale", self.field_bg_candidate_feature_grad_scale))
            self.field_bg_candidate_opacity_grad_scale = float(config.get("field_bg_candidate_opacity_grad_scale", self.field_bg_candidate_opacity_grad_scale))
            self.field_bg_candidate_scaling_grad_scale = float(config.get("field_bg_candidate_scaling_grad_scale", self.field_bg_candidate_scaling_grad_scale))
            self.field_bg_only_train = bool(config.get("field_bg_only_train", int(self.field_bg_only_train)))
            self.field_bg_only_start = int(config.get("field_bg_only_start", self.field_bg_only_start))
            self.field_bg_only_until = int(config.get("field_bg_only_until", self.field_bg_only_until))
            self.field_bg_only_interval = int(config.get("field_bg_only_interval", self.field_bg_only_interval))
            self.field_bg_only_loss_weight = float(config.get("field_bg_only_loss_weight", self.field_bg_only_loss_weight))
            self.field_bg_only_min_pixels = int(config.get("field_bg_only_min_pixels", self.field_bg_only_min_pixels))
            self.field_bg_only_da3_filter = bool(config.get("field_bg_only_da3_filter", int(self.field_bg_only_da3_filter)))
            self.field_bg_only_update_modules = bool(config.get("field_bg_only_update_modules", int(self.field_bg_only_update_modules)))
            self.field_obs_reliability = bool(config.get("field_obs_reliability", int(self.field_obs_reliability)))
            self.field_obs_reliability_floor = float(config.get("field_obs_reliability_floor", self.field_obs_reliability_floor))
            self.field_obs_reliability_mad_threshold = float(config.get("field_obs_reliability_mad_threshold", self.field_obs_reliability_mad_threshold))
            self.field_obs_reliability_diff_threshold = float(config.get("field_obs_reliability_diff_threshold", self.field_obs_reliability_diff_threshold))
            self.field_obs_reliability_motion_threshold = float(config.get("field_obs_reliability_motion_threshold", self.field_obs_reliability_motion_threshold))
            self.field_obs_reliability_mad_weight = float(config.get("field_obs_reliability_mad_weight", self.field_obs_reliability_mad_weight))
            self.field_obs_reliability_diff_weight = float(config.get("field_obs_reliability_diff_weight", self.field_obs_reliability_diff_weight))
            self.field_obs_reliability_motion_weight = float(config.get("field_obs_reliability_motion_weight", self.field_obs_reliability_motion_weight))
            self.field_obs_reliability_unreliable_threshold = float(config.get("field_obs_reliability_unreliable_threshold", self.field_obs_reliability_unreliable_threshold))
            self.field_obs_reliability_debug = bool(config.get("field_obs_reliability_debug", int(self.field_obs_reliability_debug)))
            self.field_obs_reliability_start = int(config.get("field_obs_reliability_start", self.field_obs_reliability_start))
            self.field_obs_reliability_until = int(config.get("field_obs_reliability_until", self.field_obs_reliability_until))
            self.field_obs_reliability_ema = float(config.get("field_obs_reliability_ema", self.field_obs_reliability_ema))
            self.field_obs_reliability_error_quantile = float(config.get("field_obs_reliability_error_quantile", self.field_obs_reliability_error_quantile))
            self.field_obs_reliability_error_threshold = float(config.get("field_obs_reliability_error_threshold", self.field_obs_reliability_error_threshold))
            self.field_obs_reliability_min_error = float(config.get("field_obs_reliability_min_error", self.field_obs_reliability_min_error))
            self.field_obs_reliability_dynamic_dilate = int(config.get("field_obs_reliability_dynamic_dilate", self.field_obs_reliability_dynamic_dilate))
            self.field_obs_reliability_structural_weight = float(config.get("field_obs_reliability_structural_weight", self.field_obs_reliability_structural_weight))
            self.field_obs_reliability_local_window = int(config.get("field_obs_reliability_local_window", self.field_obs_reliability_local_window))
            self.field_obs_boost_unreliable_loss = bool(config.get("field_obs_boost_unreliable_loss", int(self.field_obs_boost_unreliable_loss)))
            self.field_obs_boost_weight = float(config.get("field_obs_boost_weight", self.field_obs_boost_weight))
            self.field_obs_reset = bool(config.get("field_obs_reset", int(self.field_obs_reset)))
            self.field_obs_reset_mode = str(config.get("field_obs_reset_mode", self.field_obs_reset_mode))
            self.field_obs_reset_start = int(config.get("field_obs_reset_start", self.field_obs_reset_start))
            self.field_obs_reset_until = int(config.get("field_obs_reset_until", self.field_obs_reset_until))
            self.field_obs_reset_interval = int(config.get("field_obs_reset_interval", self.field_obs_reset_interval))
            self.field_obs_reset_schedule = str(config.get("field_obs_reset_schedule", self.field_obs_reset_schedule))
            self.field_obs_reset_opacity = float(config.get("field_obs_reset_opacity", self.field_obs_reset_opacity))
            self.field_obs_reset_min_opacity = float(config.get("field_obs_reset_min_opacity", self.field_obs_reset_min_opacity))
            self.field_obs_reset_max_points = int(config.get("field_obs_reset_max_points", self.field_obs_reset_max_points))
            self.field_obs_reset_selection_mode = str(config.get("field_obs_reset_selection_mode", self.field_obs_reset_selection_mode))
            self.field_obs_reset_min_masked_contrib = float(config.get("field_obs_reset_min_masked_contrib", self.field_obs_reset_min_masked_contrib))
            self.field_obs_reset_min_contrib_ratio = float(config.get("field_obs_reset_min_contrib_ratio", self.field_obs_reset_min_contrib_ratio))
            self.field_obs_reset_debug = bool(config.get("field_obs_reset_debug", int(self.field_obs_reset_debug)))
            self.field_obs_reset_debug_max_events = int(config.get("field_obs_reset_debug_max_events", self.field_obs_reset_debug_max_events))
            self.field_obs_reset_log_zero = bool(config.get("field_obs_reset_log_zero", int(self.field_obs_reset_log_zero)))
            self.field_obs_reset_scan_time_indices = str(config.get("field_obs_reset_scan_time_indices", self.field_obs_reset_scan_time_indices))
            self.field_obs_reset_scan_views_per_time = int(config.get("field_obs_reset_scan_views_per_time", self.field_obs_reset_scan_views_per_time))
            self.field_obs_reset_scan_min_hits = int(config.get("field_obs_reset_scan_min_hits", self.field_obs_reset_scan_min_hits))
            self.field_obs_reset_scan_top_ratio = float(config.get("field_obs_reset_scan_top_ratio", self.field_obs_reset_scan_top_ratio))
            self.field_obs_reset_scan_max_points = int(config.get("field_obs_reset_scan_max_points", self.field_obs_reset_scan_max_points))
            self.field_obs_reset_scan_update_ema = bool(config.get("field_obs_reset_scan_update_ema", int(self.field_obs_reset_scan_update_ema)))
            self.field_global_reset = bool(config.get("field_global_reset", int(self.field_global_reset)))
            self.field_global_reset_schedule = str(config.get("field_global_reset_schedule", self.field_global_reset_schedule))
            self.field_freq_prior = bool(config.get("field_freq_prior", int(self.field_freq_prior)))
            self.field_freq_prior_start = int(config.get("field_freq_prior_start", self.field_freq_prior_start))
            self.field_freq_prior_until = int(config.get("field_freq_prior_until", self.field_freq_prior_until))
            self.field_freq_prior_weight = float(config.get("field_freq_prior_weight", self.field_freq_prior_weight))
            self.field_freq_prior_patch_size = int(config.get("field_freq_prior_patch_size", self.field_freq_prior_patch_size))
            self.field_freq_prior_highpass = float(config.get("field_freq_prior_highpass", self.field_freq_prior_highpass))
            self.field_freq_prior_max_patches = int(config.get("field_freq_prior_max_patches", self.field_freq_prior_max_patches))
            self.field_freq_prior_min_mask_ratio = float(config.get("field_freq_prior_min_mask_ratio", self.field_freq_prior_min_mask_ratio))
            self.field_freq_prior_reference = str(config.get("field_freq_prior_reference", self.field_freq_prior_reference))
            self.field_freq_prior_on_reset_only = bool(config.get("field_freq_prior_on_reset_only", int(self.field_freq_prior_on_reset_only)))
            self.field_freq_prior_debug = bool(config.get("field_freq_prior_debug", int(self.field_freq_prior_debug)))
            self.field_freq_prior_debug_max_events = int(config.get("field_freq_prior_debug_max_events", self.field_freq_prior_debug_max_events))
            self.field_freq_prior_debug_mode = str(config.get("field_freq_prior_debug_mode", self.field_freq_prior_debug_mode))
            self.field_bg_median_loss = bool(config.get("field_bg_median_loss", int(self.field_bg_median_loss)))
            self.field_bg_median_loss_weight = float(config.get("field_bg_median_loss_weight", self.field_bg_median_loss_weight))

        self._ensure_content_exposure_head()
        if payload.get("content_exposure_head") is not None and self.content_exposure_head is not None:
            self._load_module_state_compatible(self.content_exposure_head, payload["content_exposure_head"])

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
        saved_static_radiance_logits = payload.get("static_radiance_level_logits")
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
        if saved_static_radiance_logits is not None:
            static_radiance_logits = saved_static_radiance_logits.to(device="cuda", dtype=torch.float32).view(-1)
            if static_radiance_logits.shape[0] < self.field_num_levels:
                static_radiance_logits = torch.cat(
                    (
                        static_radiance_logits,
                        torch.zeros((self.field_num_levels - static_radiance_logits.shape[0],), device="cuda", dtype=static_radiance_logits.dtype),
                    ),
                    dim=0,
                )
            elif static_radiance_logits.shape[0] > self.field_num_levels:
                static_radiance_logits = static_radiance_logits[: self.field_num_levels]
        elif self.field_static_radiance_branch:
            static_radiance_logits = torch.zeros((self.field_num_levels,), device="cuda")
        else:
            static_radiance_logits = torch.empty(0, device="cuda")

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
        if static_radiance_logits.numel() > 0:
            self._static_radiance_level_logits = nn.Parameter(static_radiance_logits.requires_grad_(True))
        else:
            self._static_radiance_level_logits = torch.empty(0, device="cuda")
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
            "content_exposure_head": self.content_exposure_head.state_dict() if self.content_exposure_head is not None else None,
            "field_config": self._checkpoint_field_config(),
            "static_level_logits": self._static_level_logits.detach().cpu() if self.use_euler_field and self._static_level_logits.numel() > 0 else None,
            "static_radiance_level_logits": self._static_radiance_level_logits.detach().cpu() if self.use_euler_field and self._static_radiance_level_logits.numel() > 0 else None,
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

    def reset_observation_region_opacity(self, region_mask, viewpoint_cam, means3D, visibility_filter, iteration, radii=None, return_stats=False, contrib_total=None, contrib_masked=None):
        def make_result(reason, stats=None):
            if not return_stats:
                return 0
            result = {
                "reason": reason,
                "selected_points": 0,
                "candidate_points": 0,
                "in_region_points": 0,
                "valid_projected_points": 0,
                "visible_points": 0,
            }
            if stats is not None:
                result.update(stats)
            result["reason"] = reason
            return result

        def summarize_tensor(stats, prefix, values):
            if values is None:
                return
            values = values.detach().float()
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                return
            stats[f"{prefix}_mean"] = float(values.mean().item())
            stats[f"{prefix}_min"] = float(values.min().item())
            stats[f"{prefix}_max"] = float(values.max().item())
            stats[f"{prefix}_p50"] = float(torch.quantile(values, 0.50).item())
            stats[f"{prefix}_p90"] = float(torch.quantile(values, 0.90).item())

        if not bool(getattr(self, "field_obs_reset", False)):
            return make_result("disabled")
        schedule_raw = str(getattr(self, "field_obs_reset_schedule", "")).strip()
        if schedule_raw:
            scheduled_iters = set()
            for item in schedule_raw.split(","):
                item = item.strip()
                if item:
                    scheduled_iters.add(int(item))
            if int(iteration) not in scheduled_iters:
                return make_result("outside_schedule")
        else:
            start = int(getattr(self, "field_obs_reset_start", 1500))
            until = int(getattr(self, "field_obs_reset_until", 9000))
            interval = max(int(getattr(self, "field_obs_reset_interval", 500)), 1)
            if int(iteration) < start or int(iteration) > until or (int(iteration) - start) % interval != 0:
                return make_result("outside_schedule")
        if region_mask is None or torch.count_nonzero(region_mask) == 0:
            return make_result("empty_region")
        if means3D is None or means3D.shape[0] != self._xyz.shape[0]:
            return make_result("invalid_means")
        if visibility_filter is None or visibility_filter.shape[0] != self._xyz.shape[0]:
            return make_result("invalid_visibility")

        with torch.no_grad():
            device = self._xyz.device
            means3D = means3D.to(device=device)
            region_mask = region_mask.to(device=device, dtype=torch.bool)
            visibility_filter = visibility_filter.to(device=device, dtype=torch.bool)
            height, width = region_mask.shape
            stats = {
                "reason": "ok",
                "iteration": int(iteration),
                "camera": str(getattr(viewpoint_cam, "image_name", "")),
                "timestamp": float(getattr(viewpoint_cam, "timestamp", 0.0)),
                "image_height": int(height),
                "image_width": int(width),
                "region_pixels": int(torch.count_nonzero(region_mask).item()),
                "region_ratio": float(torch.count_nonzero(region_mask).float().item() / max(float(height * width), 1.0)),
                "total_points": int(self._xyz.shape[0]),
                "visible_points": int(torch.count_nonzero(visibility_filter).item()),
            }
            projected = geom_transform_points(means3D, viewpoint_cam.full_proj_transform)
            ndc = projected[:, :2]
            valid = (
                visibility_filter
                & torch.isfinite(ndc[:, 0])
                & torch.isfinite(ndc[:, 1])
                & (ndc[:, 0] >= -1.0)
                & (ndc[:, 0] <= 1.0)
                & (ndc[:, 1] >= -1.0)
                & (ndc[:, 1] <= 1.0)
            )
            if torch.count_nonzero(valid) == 0:
                return make_result("no_valid_projected", stats)
            stats["valid_projected_points"] = int(torch.count_nonzero(valid).item())

            x = (((ndc[:, 0] + 1.0) * float(width)) - 1.0) * 0.5
            y = (((ndc[:, 1] + 1.0) * float(height)) - 1.0) * 0.5
            x_idx = torch.round(x).long().clamp(0, width - 1)
            y_idx = torch.round(y).long().clamp(0, height - 1)
            in_region = torch.zeros_like(valid)
            in_region[valid] = region_mask[y_idx[valid], x_idx[valid]]
            stats["in_region_points"] = int(torch.count_nonzero(in_region).item())

            opacity = self.get_opacity.squeeze(1)
            min_opacity = max(float(getattr(self, "field_obs_reset_min_opacity", 0.05)), 0.0)
            selection_mode = str(getattr(self, "field_obs_reset_selection_mode", "center")).lower()
            score = opacity.float()
            if selection_mode == "contribution":
                if contrib_total is None or contrib_masked is None:
                    return make_result("missing_contribution", stats)
                if contrib_total.shape[0] != self._xyz.shape[0] or contrib_masked.shape[0] != self._xyz.shape[0]:
                    return make_result("invalid_contribution", stats)
                contrib_total = contrib_total.to(device=device, dtype=torch.float32)
                contrib_masked = contrib_masked.to(device=device, dtype=torch.float32)
                contrib_ratio = contrib_masked / (contrib_total + 1e-6)
                min_masked = max(float(getattr(self, "field_obs_reset_min_masked_contrib", 0.0)), 0.0)
                min_ratio = max(float(getattr(self, "field_obs_reset_min_contrib_ratio", 0.05)), 0.0)
                candidate = valid & (opacity > min_opacity) & (contrib_masked > min_masked) & (contrib_ratio > min_ratio)
                score = contrib_masked * contrib_ratio * opacity.float()
                stats["selection_mode"] = "contribution"
                stats["min_masked_contrib"] = float(min_masked)
                stats["min_contrib_ratio"] = float(min_ratio)
                summarize_tensor(stats, "candidate_contrib_total", contrib_total[candidate])
                summarize_tensor(stats, "candidate_contrib_masked", contrib_masked[candidate])
                summarize_tensor(stats, "candidate_contrib_ratio", contrib_ratio[candidate])
                summarize_tensor(stats, "candidate_score", score[candidate])
            else:
                candidate = in_region & (opacity > min_opacity)
                stats["selection_mode"] = "center"
            stats["candidate_points"] = int(torch.count_nonzero(candidate).item())
            summarize_tensor(stats, "candidate_opacity", opacity[candidate])
            selected_indices = torch.nonzero(candidate, as_tuple=False).squeeze(1)

            if return_stats:
                masks = {
                    "in_region_points": torch.zeros((height, width), device=device, dtype=torch.bool),
                    "candidate_points": torch.zeros((height, width), device=device, dtype=torch.bool),
                    "reset_points": torch.zeros((height, width), device=device, dtype=torch.bool),
                }
                masks["in_region_points"][y_idx[in_region], x_idx[in_region]] = True
                masks["candidate_points"][y_idx[candidate], x_idx[candidate]] = True
                stats["_debug_masks"] = masks

            if selected_indices.numel() == 0:
                return make_result("no_candidate", stats)

            max_points = max(int(getattr(self, "field_obs_reset_max_points", 512)), 1)
            if selected_indices.numel() > max_points:
                _, topk = torch.topk(score[selected_indices].float(), k=max_points, largest=True)
                selected_indices = selected_indices[topk]

            reset_opacity = max(min(float(getattr(self, "field_obs_reset_opacity", 0.01)), 1.0 - 1e-6), 1e-6)
            old_opacity = self.get_opacity[selected_indices].clamp(1e-6, 1.0 - 1e-6)
            new_opacity = torch.minimum(old_opacity, torch.full_like(old_opacity, reset_opacity))
            self._opacity[selected_indices] = inverse_sigmoid(new_opacity)
            stats["selected_points"] = int(selected_indices.numel())
            stats["limited_by_max_points"] = int(stats["candidate_points"] > selected_indices.numel())
            stats["reset_opacity"] = float(reset_opacity)
            summarize_tensor(stats, "selected_opacity_before", old_opacity.squeeze(1))
            summarize_tensor(stats, "selected_opacity_after", new_opacity.squeeze(1))
            if selection_mode == "contribution":
                summarize_tensor(stats, "selected_contrib_total", contrib_total[selected_indices])
                summarize_tensor(stats, "selected_contrib_masked", contrib_masked[selected_indices])
                summarize_tensor(stats, "selected_contrib_ratio", contrib_ratio[selected_indices])
                summarize_tensor(stats, "selected_score", score[selected_indices])
            camera_center = getattr(viewpoint_cam, "camera_center", None)
            if camera_center is not None:
                camera_center = camera_center.to(device=device, dtype=means3D.dtype).view(1, 3)
                distance = torch.linalg.norm(means3D[selected_indices] - camera_center, dim=1)
                summarize_tensor(stats, "selected_camera_distance", distance)
            scale_max = torch.max(self.get_scaling[selected_indices].detach().float(), dim=1).values
            summarize_tensor(stats, "selected_scale_max", scale_max)
            if radii is not None and radii.shape[0] == self._xyz.shape[0]:
                summarize_tensor(stats, "selected_screen_radius", radii.to(device=device).detach().float()[selected_indices])
            summarize_tensor(stats, "selected_screen_x", x_idx[selected_indices].float())
            summarize_tensor(stats, "selected_screen_y", y_idx[selected_indices].float())
            summarize_tensor(stats, "selected_world_x", means3D[selected_indices, 0])
            summarize_tensor(stats, "selected_world_y", means3D[selected_indices, 1])
            summarize_tensor(stats, "selected_world_z", means3D[selected_indices, 2])
            if return_stats:
                stats["_debug_masks"]["reset_points"][y_idx[selected_indices], x_idx[selected_indices]] = True

            for group in self.optimizer.param_groups:
                if group.get("name", None) != "opacity" or len(group["params"]) != 1:
                    continue
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is None:
                    continue
                if "exp_avg" in stored_state:
                    stored_state["exp_avg"][selected_indices] = 0
                if "exp_avg_sq" in stored_state:
                    stored_state["exp_avg_sq"][selected_indices] = 0
            if return_stats:
                return stats
            return int(selected_indices.numel())

    def reset_observation_score_opacity(self, reset_score, reset_hits, iteration, return_stats=False):
        def make_result(reason, stats=None):
            if not return_stats:
                return 0
            result = {
                "reason": reason,
                "selected_points": 0,
                "candidate_points": 0,
                "total_points": int(self._xyz.shape[0]),
            }
            if stats is not None:
                result.update(stats)
            result["reason"] = reason
            return result

        def summarize_tensor(stats, prefix, values):
            if values is None:
                return
            values = values.detach().float()
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                return
            stats[f"{prefix}_mean"] = float(values.mean().item())
            stats[f"{prefix}_min"] = float(values.min().item())
            stats[f"{prefix}_max"] = float(values.max().item())
            stats[f"{prefix}_p50"] = float(torch.quantile(values, 0.50).item())
            stats[f"{prefix}_p90"] = float(torch.quantile(values, 0.90).item())

        if not bool(getattr(self, "field_obs_reset", False)):
            return make_result("disabled")
        if reset_score is None or reset_hits is None:
            return make_result("missing_score")
        if reset_score.shape[0] != self._xyz.shape[0] or reset_hits.shape[0] != self._xyz.shape[0]:
            return make_result("invalid_score_shape")

        with torch.no_grad():
            device = self._xyz.device
            reset_score = reset_score.to(device=device, dtype=torch.float32)
            reset_hits = reset_hits.to(device=device)
            opacity = self.get_opacity.squeeze(1)
            min_hits = max(int(getattr(self, "field_obs_reset_scan_min_hits", 2)), 1)
            min_opacity = max(float(getattr(self, "field_obs_reset_min_opacity", 0.05)), 0.0)
            candidate = (
                torch.isfinite(reset_score)
                & (reset_score > 0.0)
                & (reset_hits >= min_hits)
                & (opacity > min_opacity)
            )
            candidate_indices = torch.nonzero(candidate, as_tuple=False).squeeze(1)
            stats = {
                "reason": "ok",
                "iteration": int(iteration),
                "selection_mode": "multiview_contribution",
                "total_points": int(self._xyz.shape[0]),
                "candidate_points": int(candidate_indices.numel()),
                "min_hits": int(min_hits),
                "top_ratio": float(getattr(self, "field_obs_reset_scan_top_ratio", 0.2)),
            }
            summarize_tensor(stats, "candidate_score", reset_score[candidate])
            summarize_tensor(stats, "candidate_hits", reset_hits[candidate].float())
            summarize_tensor(stats, "candidate_opacity", opacity[candidate])
            if candidate_indices.numel() == 0:
                return make_result("no_candidate", stats)

            top_ratio = max(0.0, min(float(getattr(self, "field_obs_reset_scan_top_ratio", 0.2)), 1.0))
            select_count = max(int(torch.ceil(torch.tensor(float(candidate_indices.numel()) * top_ratio)).item()), 1)
            max_points = int(getattr(self, "field_obs_reset_scan_max_points", 0))
            if max_points > 0:
                select_count = min(select_count, max_points)
            select_count = min(select_count, int(candidate_indices.numel()))
            _, topk = torch.topk(reset_score[candidate_indices].float(), k=select_count, largest=True)
            selected_indices = candidate_indices[topk]

            reset_opacity = max(min(float(getattr(self, "field_obs_reset_opacity", 0.01)), 1.0 - 1e-6), 1e-6)
            old_opacity = self.get_opacity[selected_indices].clamp(1e-6, 1.0 - 1e-6)
            new_opacity = torch.minimum(old_opacity, torch.full_like(old_opacity, reset_opacity))
            self._opacity[selected_indices] = inverse_sigmoid(new_opacity)
            stats["selected_points"] = int(selected_indices.numel())
            stats["reset_opacity"] = float(reset_opacity)
            stats["limited_by_max_points"] = int(max_points > 0 and int(candidate_indices.numel()) * top_ratio > max_points)
            summarize_tensor(stats, "selected_score", reset_score[selected_indices])
            summarize_tensor(stats, "selected_hits", reset_hits[selected_indices].float())
            summarize_tensor(stats, "selected_opacity_before", old_opacity.squeeze(1))
            summarize_tensor(stats, "selected_opacity_after", new_opacity.squeeze(1))
            summarize_tensor(stats, "selected_scale_max", torch.max(self.get_scaling[selected_indices].detach().float(), dim=1).values)
            summarize_tensor(stats, "selected_world_x", self._xyz[selected_indices, 0])
            summarize_tensor(stats, "selected_world_y", self._xyz[selected_indices, 1])
            summarize_tensor(stats, "selected_world_z", self._xyz[selected_indices, 2])

            for group in self.optimizer.param_groups:
                if group.get("name", None) != "opacity" or len(group["params"]) != 1:
                    continue
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is None:
                    continue
                if "exp_avg" in stored_state:
                    stored_state["exp_avg"][selected_indices] = 0
                if "exp_avg_sq" in stored_state:
                    stored_state["exp_avg_sq"][selected_indices] = 0
            if return_stats:
                return stats
            return int(selected_indices.numel())
    
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
            if len(group["params"]) == 1 and group["name"] not in ['decoder', 'field_gate', 'field_temporal_opacity', 'static_radiance_level_logits']:
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
        if self._bg_candidate_mask is not None and self._bg_candidate_mask.numel() > 0:
            if self._bg_candidate_mask.shape[0] == valid_points_mask.shape[0]:
                self._bg_candidate_mask = self._bg_candidate_mask[valid_points_mask]
            else:
                self._bg_candidate_mask = torch.zeros((int(torch.count_nonzero(valid_points_mask).item()), 1), device="cuda", dtype=torch.float32)
        if self._bg_birth_iter is not None and self._bg_birth_iter.numel() > 0:
            if self._bg_birth_iter.shape[0] == valid_points_mask.shape[0]:
                self._bg_birth_iter = self._bg_birth_iter[valid_points_mask]
            else:
                self._bg_birth_iter = torch.full((int(torch.count_nonzero(valid_points_mask).item()), 1), -1.0, device="cuda", dtype=torch.float32)

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

    def densification_postfix(self, new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation, new_trbf_center, new_trbfscale, new_motion, new_omega, new_featuret, new_static_level_logits=None, new_dynamic_level_logits=None, new_dynamic_level_time_coeff=None, new_ems_mask=None, new_bg_candidate_mask=None, new_bg_birth_iter=None):
        old_count = self._xyz.shape[0]
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
        if new_bg_candidate_mask is None:
            new_bg_candidate_mask = torch.zeros((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        if new_bg_birth_iter is None:
            new_bg_birth_iter = torch.full((new_xyz.shape[0], 1), -1.0, device="cuda", dtype=torch.float32)
        if self._bg_candidate_mask is not None and self._bg_candidate_mask.numel() > 0 and self._bg_candidate_mask.shape[0] == old_count:
            self._bg_candidate_mask = torch.cat((self._bg_candidate_mask, new_bg_candidate_mask), dim=0)
        else:
            old_bg_candidate = torch.zeros((old_count, 1), device="cuda", dtype=torch.float32)
            self._bg_candidate_mask = torch.cat((old_bg_candidate, new_bg_candidate_mask), dim=0)
        if self._bg_birth_iter is not None and self._bg_birth_iter.numel() > 0 and self._bg_birth_iter.shape[0] == old_count:
            self._bg_birth_iter = torch.cat((self._bg_birth_iter, new_bg_birth_iter), dim=0)
        else:
            old_bg_birth = torch.full((old_count, 1), -1.0, device="cuda", dtype=torch.float32)
            self._bg_birth_iter = torch.cat((old_bg_birth, new_bg_birth_iter), dim=0)

    

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

    def add_densification_stats_with_gate(self, viewspace_point_tensor, update_filter, point_gate):
        grad = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        gate = point_gate[update_filter].to(device=grad.device, dtype=grad.dtype).clamp(0.0, 1.0)
        self.xyz_gradient_accum[update_filter] += grad * gate
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
        new_rotation[:, 0]= 1
        
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

    def suppress_background_explainers(self, region_mask, viewpoint_cam, bg_depth, iteration):
        if not self.field_bg_prior or not bool(getattr(self, "field_bg_prior_suppress", False)):
            return 0, torch.zeros_like(region_mask, dtype=torch.bool)
        if region_mask is None or torch.count_nonzero(region_mask) == 0:
            return 0, torch.zeros_like(region_mask, dtype=torch.bool)
        if self.get_xyz.numel() == 0:
            return 0, torch.zeros_like(region_mask, dtype=torch.bool)

        with torch.no_grad():
            xyz = self.get_xyz
            device = xyz.device
            region_mask = region_mask.to(device=device, dtype=torch.bool)
            height, width = region_mask.shape

            projected = geom_transform_points(xyz, viewpoint_cam.full_proj_transform)
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
                return 0, torch.zeros_like(region_mask, dtype=torch.bool)

            x = (((ndc[:, 0] + 1.0) * float(width)) - 1.0) * 0.5
            y = (((ndc[:, 1] + 1.0) * float(height)) - 1.0) * 0.5
            x_idx = torch.round(x).long().clamp(0, width - 1)
            y_idx = torch.round(y).long().clamp(0, height - 1)
            in_region = torch.zeros_like(valid)
            in_region[valid] = region_mask[y_idx[valid], x_idx[valid]]

            camera_center = viewpoint_cam.camera_center.to(device=device, dtype=xyz.dtype)
            depth_like = torch.linalg.norm(xyz - camera_center.view(1, 3), dim=1)
            bg_depth_value = torch.as_tensor(bg_depth, device=device, dtype=xyz.dtype).reshape(()).clamp_min(1e-4)
            depth_margin = max(float(getattr(self, "field_bg_prior_suppress_depth_margin", 1.0)), 0.0)
            in_front = depth_like < (bg_depth_value - depth_margin)

            opacity = self.get_opacity.squeeze(1)
            opacity_gate = opacity > float(getattr(self, "field_bg_prior_suppress_opacity_threshold", 0.05))
            candidate = in_region & in_front & opacity_gate

            if self._bg_candidate_mask is not None and self._bg_candidate_mask.numel() == candidate.shape[0]:
                candidate = candidate & (self._bg_candidate_mask.squeeze(1) <= 0.5)

            if torch.count_nonzero(candidate) == 0:
                return 0, torch.zeros_like(region_mask, dtype=torch.bool)

            scale = self.get_scaling.max(dim=1).values
            scale_quantile = float(getattr(self, "field_bg_prior_suppress_scale_quantile", 0.75))
            if 0.0 <= scale_quantile <= 1.0 and torch.count_nonzero(candidate) > 8:
                scale_threshold = torch.quantile(scale[candidate].float(), scale_quantile)
                candidate = candidate & (scale >= scale_threshold)

            selected_indices = torch.nonzero(candidate, as_tuple=False).squeeze(1)
            if selected_indices.numel() == 0:
                return 0, torch.zeros_like(region_mask, dtype=torch.bool)

            max_points = max(int(getattr(self, "field_bg_prior_suppress_max_points", 512)), 1)
            if selected_indices.numel() > max_points:
                scores = opacity[selected_indices] * torch.clamp(scale[selected_indices], min=1e-6)
                _, topk = torch.topk(scores.float(), k=max_points, largest=True)
                selected_indices = selected_indices[topk]

            decay = max(0.0, min(float(getattr(self, "field_bg_prior_suppress_decay", 0.02)), 0.95))
            old_opacity = self.get_opacity[selected_indices].clamp(1e-6, 1.0 - 1e-6)
            new_opacity = torch.clamp(old_opacity * (1.0 - decay), 1e-6, 1.0 - 1e-6)
            self._opacity[selected_indices] = inverse_sigmoid(new_opacity)

            suppressed_pixels = torch.zeros_like(region_mask, dtype=torch.bool)
            suppressed_pixels[y_idx[selected_indices], x_idx[selected_indices]] = True
            return int(selected_indices.numel()), suppressed_pixels

    def add_static_background_gaussians(self, pixel_indices, viewpoint_cam, depthmap, bg_image, iteration,
                                        numperay=1, depth_scale=1.02, depth_values=None, depth_scales=None):
        if pixel_indices is None or pixel_indices.numel() == 0:
            return 0

        def pix2ndc(v, S):
            return (v * 2.0 + 1.0) / S - 1.0

        def clip_to_bbox(xyz):
            if not bool(getattr(self, "field_bg_dense_clip_to_bbox", 0)):
                return xyz
            if not self.use_euler_field or self.euler_field is None:
                return xyz
            if not hasattr(self.euler_field, "bbox_min") or not hasattr(self.euler_field, "bbox_max"):
                return xyz
            if xyz.numel() == 0:
                return xyz

            bbox_min = self.euler_field.bbox_min.to(device=xyz.device, dtype=xyz.dtype).view(1, 3)
            bbox_max = self.euler_field.bbox_max.to(device=xyz.device, dtype=xyz.dtype).view(1, 3)
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
            margin = max(0.0, min(float(getattr(self, "field_bg_dense_bbox_clip_margin", 0.999)), 1.0))
            t_clip = torch.clamp(t_exit * margin, min=0.0, max=1.0).view(-1, 1)
            clipped = origin + direction * t_clip
            should_clip = (~inside) & intersects & (t_exit < 1.0)
            return torch.where(should_clip.view(-1, 1), clipped, xyz)

        if depthmap.dim() == 2:
            depthmap = depthmap.unsqueeze(0)
        if bg_image.dim() != 3:
            raise ValueError("bg_image must have shape [3, H, W]")

        numperay = max(int(numperay), 1)
        absolute_depths = None
        if depth_values is not None:
            if not torch.is_tensor(depth_values):
                depth_values = torch.tensor(depth_values, device="cuda", dtype=depthmap.dtype)
            absolute_depths = depth_values.to(device="cuda", dtype=depthmap.dtype).flatten()
            absolute_depths = absolute_depths[torch.isfinite(absolute_depths) & (absolute_depths > 0.0)]
            if absolute_depths.numel() == 0:
                absolute_depths = None
        if absolute_depths is None:
            if depth_scales is not None:
                if not torch.is_tensor(depth_scales):
                    depth_scales = torch.tensor(depth_scales, device="cuda", dtype=depthmap.dtype)
                depth_scales = depth_scales.to(device="cuda", dtype=depthmap.dtype).flatten()
                depth_scales = depth_scales[torch.isfinite(depth_scales) & (depth_scales > 0.0)]
                if depth_scales.numel() == 0:
                    depth_scales = None
            if depth_scales is not None:
                pass
            elif numperay == 1:
                depth_scales = torch.tensor([float(depth_scale)], device="cuda", dtype=depthmap.dtype)
            else:
                depth_scales = torch.linspace(1.0, float(depth_scale), numperay, device="cuda", dtype=depthmap.dtype)
        else:
            depth_scales = absolute_depths

        u = pixel_indices[:, 0].long()
        v = pixel_indices[:, 1].long()
        base_depths = depthmap[:, u, v].permute(1, 0).clamp_min(1e-4)
        rgbs = bg_image[:, u, v].permute(1, 0).clamp(0.0, 1.0)
        color_init = str(getattr(self, "field_bg_prior_color_init", "gt")).lower()
        if color_init == "zero":
            featuredc = torch.zeros((rgbs.shape[0], 6), device="cuda", dtype=rgbs.dtype)
        else:
            featuredc = torch.cat((rgbs, torch.zeros_like(rgbs)), dim=1)

        camera2wold = viewpoint_cam.world_view_transform.T.inverse()
        projectinverse = viewpoint_cam.projection_matrix.T.inverse()
        ndcu = pix2ndc(u, viewpoint_cam.image_height).unsqueeze(1)
        ndcv = pix2ndc(v, viewpoint_cam.image_width).unsqueeze(1)
        ndccamera = torch.cat((ndcv, ndcu, torch.ones_like(ndcu), torch.ones_like(ndcu)), dim=1)
        localpointuv = ndccamera @ projectinverse.T
        direction_local = localpointuv / localpointuv[:, 3:]

        new_xyz = []
        new_features_dc = []
        new_trbf_center = []
        new_trbf_scale = []
        new_motion = []
        new_omega = []
        new_featuret = []
        new_depth_scale_tags = []
        new_static_level_logits = []
        new_dynamic_level_logits = []
        new_dynamic_level_time_coeff = []

        for depth_item in depth_scales:
            if absolute_depths is None:
                targetPz = base_depths * depth_item
            else:
                targetPz = torch.full_like(base_depths, float(depth_item.detach().cpu()))
            denom = direction_local[:, 2:3]
            denom = torch.where(torch.abs(denom) < 1e-6, torch.full_like(denom, 1e-6), denom)
            rate = targetPz / denom
            localpoint = direction_local * rate
            localpoint[:, -1] = 1.0
            worldpointH = localpoint @ camera2wold.T
            worldpoint = worldpointH / worldpointH[:, 3:]
            xyz = clip_to_bbox(worldpoint[:, :3])

            selectnumpoints = xyz.shape[0]
            new_xyz.append(xyz)
            new_features_dc.append(featuredc)
            if absolute_depths is None:
                depth_scale_value = float(depth_item.detach().cpu())
            else:
                depth_scale_value = float("inf")
            new_depth_scale_tags.append(torch.full((selectnumpoints, 1), depth_scale_value, device="cuda", dtype=xyz.dtype))
            new_trbf_center.append(torch.full((selectnumpoints, 1), float(self.field_bg_prior_trbf_center), device="cuda"))
            new_trbf_scale.append(torch.full((selectnumpoints, 1), float(self.field_bg_prior_trbf_scale), device="cuda"))
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
        new_features_dc = torch.cat(new_features_dc, dim=0)
        new_trbf_center = torch.cat(new_trbf_center, dim=0)
        new_trbf_scale = torch.cat(new_trbf_scale, dim=0)
        new_motion = torch.cat(new_motion, dim=0)
        new_omega = torch.cat(new_omega, dim=0)
        new_featuret = torch.cat(new_featuret, dim=0)
        new_depth_scale_tags = torch.cat(new_depth_scale_tags, dim=0).reshape(-1)

        new_rotation = torch.zeros((new_xyz.shape[0], 4), device="cuda")
        new_rotation[:, 0] = 1.0
        new_opacity = inverse_sigmoid(float(self.field_bg_prior_opacity) * torch.ones((new_xyz.shape[0], 1), device="cuda"))

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

        new_scaling = self._init_background_gaussian_scaling(new_xyz, new_depth_scale_tags)

        new_ems_mask = torch.ones((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        new_bg_candidate_mask = torch.ones((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        new_bg_birth_iter = torch.full((new_xyz.shape[0], 1), float(iteration), device="cuda", dtype=torch.float32)

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
            new_featuret,
            new_static_level_logits,
            new_dynamic_level_logits,
            new_dynamic_level_time_coeff,
            new_ems_mask,
            new_bg_candidate_mask,
            new_bg_birth_iter,
        )
        return new_xyz.shape[0]

    def _init_background_gaussian_scaling(self, new_xyz, depth_scale_tags=None):
        scale_init = str(getattr(self, "field_bg_prior_scale_init", "knn")).lower()
        fixed_scale = max(float(getattr(self, "field_bg_prior_fixed_scale", 0.01)), 1e-6)
        fixed_scaling = torch.full((new_xyz.shape[0], 3), math.log(fixed_scale), device="cuda", dtype=new_xyz.dtype)

        def knn_scaling():
            tmpxyz = torch.cat((new_xyz, self._xyz), dim=0)
            dist2 = torch.clamp_min(distCUDA2(tmpxyz), 1e-7)
            dist2 = dist2[:new_xyz.shape[0]]
            return torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        if scale_init == "fixed":
            new_scaling = fixed_scaling
        elif scale_init in ("hybrid_far_knn", "hybrid_knn", "fixed_near_knn_far"):
            threshold = float(getattr(self, "field_bg_prior_hybrid_knn_scale_threshold", 5.0))
            if depth_scale_tags is None:
                depth_tags = torch.full((new_xyz.shape[0],), float("inf"), device="cuda", dtype=new_xyz.dtype)
            else:
                depth_tags = depth_scale_tags.to(device="cuda", dtype=new_xyz.dtype).reshape(-1)
            far_mask = (depth_tags > threshold).view(-1, 1)
            new_scaling = torch.where(far_mask, knn_scaling(), fixed_scaling)
        else:
            new_scaling = knn_scaling()
        return torch.clamp(new_scaling, -10, 1.0)

    def add_static_background_gaussians_xyz(self, new_xyz, rgbs, iteration, depth_scale_tags=None):
        if new_xyz is None or new_xyz.numel() == 0:
            return 0
        new_xyz = new_xyz.to(device="cuda", dtype=self._xyz.dtype).contiguous()
        rgbs = rgbs.to(device="cuda", dtype=new_xyz.dtype).reshape(-1, 3).clamp(0.0, 1.0)
        if rgbs.shape[0] != new_xyz.shape[0]:
            raise ValueError("rgbs must have shape [N, 3] and match new_xyz")
        if depth_scale_tags is None:
            new_depth_scale_tags = torch.full((new_xyz.shape[0],), float("inf"), device="cuda", dtype=new_xyz.dtype)
        else:
            new_depth_scale_tags = depth_scale_tags.to(device="cuda", dtype=new_xyz.dtype).reshape(-1)
            if new_depth_scale_tags.shape[0] != new_xyz.shape[0]:
                raise ValueError("depth_scale_tags must have shape [N] and match new_xyz")
        color_init = str(getattr(self, "field_bg_prior_color_init", "gt")).lower()
        if color_init == "zero":
            new_features_dc = torch.zeros((rgbs.shape[0], 6), device="cuda", dtype=new_xyz.dtype)
        else:
            new_features_dc = torch.cat((rgbs, torch.zeros_like(rgbs)), dim=1)

        new_rotation = torch.zeros((new_xyz.shape[0], 4), device="cuda", dtype=new_xyz.dtype)
        new_rotation[:, 0] = 1.0
        new_opacity = inverse_sigmoid(float(self.field_bg_prior_opacity) * torch.ones((new_xyz.shape[0], 1), device="cuda", dtype=new_xyz.dtype))
        new_trbf_center = torch.full((new_xyz.shape[0], 1), float(self.field_bg_prior_trbf_center), device="cuda", dtype=new_xyz.dtype)
        new_trbf_scale = torch.full((new_xyz.shape[0], 1), float(self.field_bg_prior_trbf_scale), device="cuda", dtype=new_xyz.dtype)
        new_motion = torch.zeros((new_xyz.shape[0], 9), device="cuda", dtype=new_xyz.dtype)
        new_omega = torch.zeros((new_xyz.shape[0], 4), device="cuda", dtype=new_xyz.dtype)
        new_featuret = torch.zeros((new_xyz.shape[0], 3), device="cuda", dtype=new_xyz.dtype)

        if self.use_euler_field:
            new_static_level_logits = torch.zeros((new_xyz.shape[0], self.field_num_levels), device="cuda", dtype=new_xyz.dtype)
            new_dynamic_level_logits = torch.zeros((new_xyz.shape[0], self.field_num_levels), device="cuda", dtype=new_xyz.dtype)
            if self.field_level_fourier_degree > 0:
                coeff_dim = 2 * self.field_level_fourier_degree
                new_dynamic_level_time_coeff = torch.zeros((new_xyz.shape[0], self.field_num_levels, coeff_dim), device="cuda", dtype=new_xyz.dtype)
            else:
                new_dynamic_level_time_coeff = None
        else:
            new_static_level_logits = None
            new_dynamic_level_logits = None
            new_dynamic_level_time_coeff = None

        new_scaling = self._init_background_gaussian_scaling(
            new_xyz,
            new_depth_scale_tags,
        )

        new_ems_mask = torch.ones((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        new_bg_candidate_mask = torch.ones((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        new_bg_birth_iter = torch.full((new_xyz.shape[0], 1), float(iteration), device="cuda", dtype=torch.float32)

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
            new_featuret,
            new_static_level_logits,
            new_dynamic_level_logits,
            new_dynamic_level_time_coeff,
            new_ems_mask,
            new_bg_candidate_mask,
            new_bg_birth_iter,
        )
        return new_xyz.shape[0]

    def get_background_candidate_mask(self, iteration=None, min_age=0):
        n_points = self.get_xyz.shape[0]
        if self._bg_candidate_mask is None or self._bg_candidate_mask.numel() == 0:
            return torch.zeros((n_points,), device=self._xyz.device, dtype=torch.bool)
        if self._bg_candidate_mask.shape[0] != n_points:
            return torch.zeros((n_points,), device=self._xyz.device, dtype=torch.bool)
        mask = self._bg_candidate_mask.squeeze(1) > 0.5
        if iteration is not None and min_age > 0:
            if self._bg_birth_iter is None or self._bg_birth_iter.numel() == 0 or self._bg_birth_iter.shape[0] != n_points:
                return torch.zeros_like(mask)
            birth = self._bg_birth_iter.squeeze(1)
            mask = mask & (birth >= 0.0) & ((float(iteration) - birth) >= float(min_age))
        return mask

    def keep_background_candidate_gradients_only(self, update_modules=False):
        candidate = self.get_background_candidate_mask()
        if torch.count_nonzero(candidate) == 0:
            return 0

        def mask_point_grad(param):
            if param is None:
                return
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            if param.grad.shape[0] == candidate.shape[0]:
                view_shape = [candidate.shape[0]] + [1] * (param.grad.dim() - 1)
                param.grad.mul_(candidate.view(*view_shape).to(device=param.grad.device, dtype=param.grad.dtype))

        for param in (
            self._xyz,
            self._features_dc,
            self._features_t,
            self._scaling,
            self._rotation,
            self._opacity,
            self._trbf_center,
            self._trbf_scale,
            self._motion,
            self._omega,
            self._static_level_logits,
            self._dynamic_level_logits,
            self._dynamic_level_time_coeff,
            self._static_route_logits,
            self._field_residual_gate,
        ):
            mask_point_grad(param)

        if not update_modules:
            if self.use_euler_field and self.euler_field is not None:
                for name, param in self.euler_field.named_parameters():
                    if name.startswith("static_grids") or name.startswith("static_temporal_grids"):
                        continue
                    if param.grad is not None:
                        param.grad.zero_()
            modules = [
                self.rgbdecoder,
                self.field_router if self.use_euler_field else None,
                self.field_query_gate if self.use_euler_field else None,
                self.field_decoder if self.use_euler_field else None,
                self.field_temporal_opacity_head if self.use_euler_field else None,
                self.field_static_view_mapper if self.use_euler_field else None,
                self.field_static_app_head if self.use_euler_field else None,
            ]
            for module in modules:
                if module is None:
                    continue
                for param in module.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
        return int(torch.count_nonzero(candidate).item())

    def background_candidate_stats_enabled(self, iteration):
        if not bool(getattr(self, "field_bg_prior_clone_split", False)):
            return False
        if self._bg_candidate_mask is None or self._bg_candidate_mask.numel() == 0:
            return False
        start = int(getattr(self, "field_bg_prior_clone_stat_start", 9000))
        until = int(getattr(self, "field_bg_prior_clone_until", 16000))
        return int(iteration) >= start and int(iteration) <= until

    def densify_background_candidates(self, iteration, grad_threshold, scene_extent):
        stats = {
            "eligible": 0,
            "selected": 0,
            "cloned": 0,
            "split_parents": 0,
            "new_points": 0,
        }
        if not bool(getattr(self, "field_bg_prior_clone_split", False)):
            return stats
        if self._bg_candidate_mask is None or self._bg_candidate_mask.numel() == 0:
            return stats
        iteration = int(iteration)
        start = int(getattr(self, "field_bg_prior_clone_start", 9500))
        until = int(getattr(self, "field_bg_prior_clone_until", 16000))
        interval = max(int(getattr(self, "field_bg_prior_clone_interval", 500)), 1)
        if iteration < start or iteration > until or iteration % interval != 0:
            return stats
        if self.xyz_gradient_accum is None or self.xyz_gradient_accum.numel() == 0:
            return stats
        if self.denom is None or self.denom.numel() == 0:
            return stats
        if self.xyz_gradient_accum.shape[0] != self.get_xyz.shape[0]:
            return stats

        min_age = int(getattr(self, "field_bg_prior_clone_min_age", 500))
        candidate = self.get_background_candidate_mask(iteration=iteration, min_age=min_age)
        if torch.count_nonzero(candidate) == 0:
            return stats

        denom = torch.clamp(self.denom, min=1.0)
        grads = self.xyz_gradient_accum / denom
        grads[grads.isnan()] = 0.0
        grad_norm = torch.norm(grads, dim=-1)

        threshold = float(getattr(self, "field_bg_prior_clone_grad_threshold", grad_threshold))
        if threshold <= 0.0:
            threshold = float(grad_threshold)
        eligible = candidate & (grad_norm >= threshold)

        min_opacity = float(getattr(self, "field_bg_prior_clone_min_opacity", 0.01))
        if min_opacity > 0.0:
            eligible = eligible & (self.get_opacity.squeeze(1) >= min_opacity)

        min_visibility = float(getattr(self, "field_bg_prior_clone_min_visibility", 0.0))
        if min_visibility > 0.0 and self._visibility_persistence_ema is not None and self._visibility_persistence_ema.numel() == self.get_xyz.shape[0]:
            eligible = eligible & (self._visibility_persistence_ema.squeeze(1) >= min_visibility)

        eligible_indices = torch.nonzero(eligible, as_tuple=False).squeeze(1)
        stats["eligible"] = int(eligible_indices.numel())
        if eligible_indices.numel() == 0:
            return stats

        bg_count = int(torch.count_nonzero(candidate).item())
        max_ratio = max(float(getattr(self, "field_bg_prior_clone_max_ratio", 0.05)), 0.0)
        ratio_limit = eligible_indices.numel()
        if max_ratio > 0.0:
            ratio_limit = max(1, int(math.ceil(float(bg_count) * max_ratio)))
        max_points = int(getattr(self, "field_bg_prior_clone_max_points", 3000))
        if max_points > 0:
            ratio_limit = min(ratio_limit, max_points)
        select_count = min(int(eligible_indices.numel()), int(ratio_limit))
        if select_count <= 0:
            return stats

        if eligible_indices.numel() > select_count:
            _, topk = torch.topk(grad_norm[eligible_indices].float(), k=select_count, largest=True)
            selected_indices = eligible_indices[topk]
        else:
            selected_indices = eligible_indices
        selected = torch.zeros_like(eligible)
        selected[selected_indices] = True
        stats["selected"] = int(selected_indices.numel())

        scale_gate = self.percent_dense * scene_extent
        max_scaling = torch.max(self.get_scaling, dim=1).values
        clone_mask = selected & (max_scaling <= scale_gate)
        split_mask = selected & (max_scaling > scale_gate)
        split_children = max(int(getattr(self, "field_bg_prior_clone_split_children", 2)), 2)

        new_xyz_parts = []
        new_features_parts = []
        new_opacity_parts = []
        new_scaling_parts = []
        new_rotation_parts = []
        new_trbf_center_parts = []
        new_trbf_scale_parts = []
        new_motion_parts = []
        new_omega_parts = []
        new_featuret_parts = []
        new_static_logits_parts = []
        new_dynamic_logits_parts = []
        new_dynamic_time_parts = []
        new_ems_parts = []

        clone_count = int(torch.count_nonzero(clone_mask).item())
        if clone_count > 0:
            new_xyz_parts.append(self._xyz[clone_mask])
            new_features_parts.append(self._features_dc[clone_mask])
            new_opacity_parts.append(self._opacity[clone_mask])
            new_scaling_parts.append(self._scaling[clone_mask])
            new_rotation_parts.append(self._rotation[clone_mask])
            new_trbf_center_parts.append(self._trbf_center[clone_mask])
            new_trbf_scale_parts.append(self._trbf_scale[clone_mask])
            new_motion_parts.append(self._motion[clone_mask])
            new_omega_parts.append(self._omega[clone_mask])
            new_featuret_parts.append(self._features_t[clone_mask])
            if self.use_euler_field:
                new_static_logits_parts.append(self._static_level_logits[clone_mask])
                new_dynamic_logits_parts.append(self._dynamic_level_logits[clone_mask])
                if self._dynamic_level_time_coeff.numel() > 0:
                    new_dynamic_time_parts.append(self._dynamic_level_time_coeff[clone_mask])
            if self.maskforems is not None and self.maskforems.numel() > 0 and self.maskforems.shape[0] == self.get_xyz.shape[0]:
                new_ems_parts.append(self.maskforems[clone_mask])
            else:
                new_ems_parts.append(torch.ones((clone_count, 1), device="cuda", dtype=torch.float32))

        split_count = int(torch.count_nonzero(split_mask).item())
        if split_count > 0:
            stds = self.get_scaling[split_mask].repeat(split_children, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[split_mask]).repeat(split_children, 1, 1)
            new_xyz_parts.append(torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[split_mask].repeat(split_children, 1))
            new_features_parts.append(self._features_dc[split_mask].repeat(split_children, 1))
            new_opacity_parts.append(self._opacity[split_mask].repeat(split_children, 1))
            new_scaling_parts.append(self.scaling_inverse_activation(self.get_scaling[split_mask].repeat(split_children, 1) / (0.8 * split_children)))
            new_rotation_parts.append(self._rotation[split_mask].repeat(split_children, 1))

            parent_trbf_center = self._trbf_center[split_mask]
            parent_trbf_scale = self._trbf_scale[split_mask]
            parent_motion = self._motion[split_mask]
            parent_ems_mask = self.maskforems[split_mask] if self.maskforems is not None and self.maskforems.numel() > 0 and self.maskforems.shape[0] == self.get_xyz.shape[0] else None
            new_trbf_center, new_trbf_scale = self._get_temporal_child_support(
                parent_trbf_center,
                parent_trbf_scale,
                parent_motion,
                error_prior=parent_ems_mask,
                copies_per_parent=split_children,
            )
            new_trbf_center_parts.append(new_trbf_center)
            new_trbf_scale_parts.append(new_trbf_scale)
            new_motion_parts.append(parent_motion.repeat(split_children, 1))
            new_omega_parts.append(self._omega[split_mask].repeat(split_children, 1))
            new_featuret_parts.append(self._features_t[split_mask].repeat(split_children, 1))
            if self.use_euler_field:
                new_static_logits_parts.append(self._static_level_logits[split_mask].repeat(split_children, 1))
                new_dynamic_logits_parts.append(self._dynamic_level_logits[split_mask].repeat(split_children, 1))
                if self._dynamic_level_time_coeff.numel() > 0:
                    new_dynamic_time_parts.append(self._dynamic_level_time_coeff[split_mask].repeat(split_children, 1, 1))
            if parent_ems_mask is not None:
                new_ems_parts.append(parent_ems_mask.repeat(split_children, 1) * 0.75)
            else:
                new_ems_parts.append(torch.ones((split_count * split_children, 1), device="cuda", dtype=torch.float32))

        if len(new_xyz_parts) == 0:
            return stats

        new_xyz = torch.cat(new_xyz_parts, dim=0)
        new_features_dc = torch.cat(new_features_parts, dim=0)
        new_opacity = torch.cat(new_opacity_parts, dim=0)
        new_scaling = torch.cat(new_scaling_parts, dim=0)
        new_rotation = torch.cat(new_rotation_parts, dim=0)
        new_trbf_center = torch.cat(new_trbf_center_parts, dim=0)
        new_trbf_scale = torch.cat(new_trbf_scale_parts, dim=0)
        new_motion = torch.cat(new_motion_parts, dim=0)
        new_omega = torch.cat(new_omega_parts, dim=0)
        new_featuret = torch.cat(new_featuret_parts, dim=0)
        new_ems_mask = torch.cat(new_ems_parts, dim=0)

        if self.use_euler_field:
            new_static_level_logits = torch.cat(new_static_logits_parts, dim=0) if len(new_static_logits_parts) > 0 else None
            new_dynamic_level_logits = torch.cat(new_dynamic_logits_parts, dim=0) if len(new_dynamic_logits_parts) > 0 else None
            new_dynamic_level_time_coeff = torch.cat(new_dynamic_time_parts, dim=0) if len(new_dynamic_time_parts) > 0 else None
        else:
            new_static_level_logits = None
            new_dynamic_level_logits = None
            new_dynamic_level_time_coeff = None

        new_bg_candidate_mask = torch.ones((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        new_bg_birth_iter = torch.full((new_xyz.shape[0], 1), float(iteration), device="cuda", dtype=torch.float32)
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
            new_featuret,
            new_static_level_logits,
            new_dynamic_level_logits,
            new_dynamic_level_time_coeff,
            new_ems_mask,
            new_bg_candidate_mask,
            new_bg_birth_iter,
        )

        if split_count > 0 and not bool(getattr(self, "field_bg_prior_keep_split_parent", False)):
            prune_filter = torch.cat((split_mask, torch.zeros(new_xyz.shape[0], device="cuda", dtype=torch.bool)))
            self.prune_points(prune_filter)

        stats["cloned"] = clone_count
        stats["split_parents"] = split_count
        stats["new_points"] = int(new_xyz.shape[0])
        return stats

    def protect_background_candidates_from_prune(self, prune_mask, iteration):
        if self._bg_candidate_mask is None or self._bg_candidate_mask.numel() == 0:
            return prune_mask
        if self._bg_candidate_mask.shape[0] != prune_mask.shape[0]:
            return prune_mask
        candidate = self._bg_candidate_mask.squeeze(1) > 0.5
        if self._bg_birth_iter is not None and self._bg_birth_iter.numel() > 0:
            birth = self._bg_birth_iter.squeeze(1)
        else:
            birth = torch.full_like(prune_mask.float(), -1.0)
        young = candidate & (birth >= 0.0) & ((float(iteration) - birth) < float(self.field_bg_prior_protect_iters))
        return prune_mask & (~young)

    def prune_mature_background_candidates(self, iteration):
        if not self.field_bg_prior:
            return 0
        if not bool(getattr(self, "field_bg_prior_mature_prune", True)):
            return 0
        interval = max(int(self.field_bg_prior_mature_prune_interval), 1)
        if iteration < self.field_bg_prior_start or iteration % interval != 0:
            return 0
        if self._bg_candidate_mask is None or self._bg_candidate_mask.numel() == 0:
            return 0
        if self._bg_candidate_mask.shape[0] != self.get_xyz.shape[0]:
            return 0

        candidate = self._bg_candidate_mask.squeeze(1) > 0.5
        birth = self._bg_birth_iter.squeeze(1)
        age = float(iteration) - birth
        mature = candidate & (birth >= 0.0) & (age >= float(self.field_bg_prior_protect_iters))
        if torch.count_nonzero(mature) == 0:
            return 0

        opacity = self.get_opacity.squeeze(1)
        low_opacity = opacity < float(self.field_bg_prior_mature_min_opacity)
        if self._visibility_persistence_ema is not None and self._visibility_persistence_ema.numel() == self.get_xyz.shape[0]:
            visibility = self._visibility_persistence_ema.squeeze(1)
            low_visibility = visibility < float(self.field_bg_prior_mature_min_visibility)
        else:
            low_visibility = torch.zeros_like(low_opacity)
        stale = age >= float(self.field_bg_prior_protect_iters * 2)
        prune_mask = mature & (low_opacity | (stale & low_visibility & (opacity < max(float(self.field_bg_prior_opacity), 1e-4))))
        prune_count = int(torch.count_nonzero(prune_mask).item())
        if prune_count > 0:
            self.prune_points(prune_mask)
            torch.cuda.empty_cache()
        return prune_count




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
