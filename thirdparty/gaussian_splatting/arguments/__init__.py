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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

    def export_changed_args_to_json(self, args): 
        defaults = {}
        for arg in vars(args).items():
            try:
                if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                    defaultvalue = getattr(self, arg[0])
                    # defaults[ arg[0] ] = defaultvalue
                    if defaultvalue != arg[1]:
                        defaults[arg[0]] = arg[1]
            except:
                pass 
               
        return defaults


class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.veryrify_llff = 0
        self.eval = False
        self.model = "gmodel" # 
        self.loader = "colmap" #
        self.test_photometric_fit = 0
        self.test_photometric_fit_mode = "rgb_affine"
        self.test_photometric_fit_reg = 1e-6
        self.test_photometric_fit_clamp = 1
        self.test_photometric_fit_save_images = 1
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
        self.field_bbox_preserve_cell_size = 1
        self.field_bbox_frustum_expand = 0
        self.field_bbox_frustum_grid = 5
        self.field_bbox_frustum_depth_base = "bbox_corners"
        self.field_bbox_frustum_depth_scales = "1.0,1.5,2.0"
        self.field_bbox_frustum_margin = 0.0
        self.field_bbox_frustum_max_expand_xyz = ""
        self.field_feature_dim = 8
        self.field_fourier_degree = 10
        self.field_decoder_hidden = 32
        self.field_level_fourier_degree = 2
        self.field_residual_mode = "geometry"
        self.field_query_mode = "hybrid"
        self.field_query_detach = 1
        self.field_query_gate_bias = -2.0
        self.field_query_motion_scale = 2.0
        self.field_dyn_threshold = 0.08
        self.field_fast_threshold = 0.18
        self.field_dyn_slope = 10.0
        self.field_fast_slope = 15.0
        self.field_fast_temperature = 0.25
        self.field_disable_dynamic_grid = 1
        self.field_v23_compat = 0
        self.field_static_route_mode = "learned"
        self.field_static_route_init = 0.0
        self.field_static_start_iter = 0
        self.field_static_warmup_iters = 3000
        self.field_static_motion_scale = 0.05
        self.field_static_opacity_scale = 0.02
        self.field_static_app_scale = 0.05
        self.field_static_temporal_residual = 0
        self.field_static_temporal_frames = 50
        self.field_static_temporal_scale = 1.0
        self.field_static_radiance_branch = 0
        self.field_static_radiance_start = 3000
        self.field_static_radiance_warmup = 1000
        self.field_static_radiance_scale = 0.05
        self.field_static_radiance_depth_multiplier = 5.0
        self.field_static_radiance_samples = 4
        self.field_static_radiance_max_pixels = 0
        self.field_static_use_global_gate = 0
        self.field_static_prior_floor = 0.25
        self.field_soft_route_slope = 8.0
        self.field_soft_static_threshold = 0.45
        self.field_soft_dynamic_threshold = 0.25
        self.field_staged_training = 1
        self.field_disable_legacy_aux = 1
        self.field_disable_ems_main = 1
        self.field_disable_global_omega_split = 1
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
        self.field_temporal_refine = 1
        self.field_temporal_refine_start = 7000
        self.field_temporal_refine_interval = 1000
        self.field_temporal_split_children = 2
        self.field_temporal_center_offset = 0.08
        self.field_temporal_scale_shrink = 0.5
        self.field_fast_child_motion_scale = 1.0
        self.field_temporal_refine_opacity_threshold = 0.2
        self.field_temporal_refine_score_threshold = 0.65
        self.field_temporal_refine_max_ratio = 0.02
        self.field_bg_prior = 0
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
        self.field_bg_prior_mature_prune = 1
        self.field_bg_prior_mature_prune_interval = 500
        self.field_bg_prior_mature_min_opacity = 0.01
        self.field_bg_prior_mature_min_visibility = 0.01
        self.field_bg_prior_debug = 0
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
        self.field_bg_prior_exposure_robust = 1
        self.field_bg_prior_structural_weight = 0.5
        self.field_bg_prior_local_window = 31
        self.field_bg_prior_fixed_depth = 1
        self.field_bg_prior_fixed_depth_ratio = 0.95
        self.field_bg_prior_depth_max = 15.0
        self.field_bg_prior_suppress = 0
        self.field_bg_prior_suppress_decay = 0.02
        self.field_bg_prior_suppress_max_points = 512
        self.field_bg_prior_suppress_depth_margin = 1.0
        self.field_bg_prior_suppress_opacity_threshold = 0.05
        self.field_bg_prior_suppress_scale_quantile = 0.75
        self.field_bg_prior_clone_split = 0
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
        self.field_bg_prior_keep_split_parent = 0
        self.field_bg_dense_add = 0
        self.field_bg_dense_add_iter = 3000
        self.field_bg_dense_add_time_indices = "0,12,25,37,49"
        self.field_bg_dense_depth_base = "render"
        self.field_bg_dense_depth_scales = "0.75,1.09,1.58,2.29,3.32,4.82,7"
        self.field_bg_dense_depth_values = ""
        self.field_bg_dense_mask_source = "instant"
        self.field_bg_dense_sample_block_size = 3
        self.field_bg_dense_pixels_per_block = 1
        self.field_bg_dense_max_pixels_per_camera = 512
        self.field_bg_dense_debug = 0
        self.field_bg_dense_debug_max_events = 0
        self.field_bg_dense_da3_filter = 0
        self.field_bg_dense_da3_path = ""
        self.field_bg_dense_da3_foreground_quantile = 0.45
        self.field_bg_dense_beit_filter = 0
        self.field_bg_dense_beit_path = ""
        self.field_bg_dense_beit_band_low = 0.10
        self.field_bg_dense_beit_band_high = 0.30
        self.field_bg_dense_beit_threshold = 0.50
        self.field_bg_dense_cell_dedup = 0
        self.field_bg_dense_dedup_level = 3
        self.field_bg_dense_dedup_priority = "center"
        self.field_bg_dense_max_per_cell = 1
        self.field_bg_dense_skip_control_at_add_iter = 0
        self.field_bg_dense_clip_to_bbox = 0
        self.field_bg_dense_bbox_clip_margin = 0.999
        self.field_highfreq_densify = 0
        self.field_highfreq_densify_sigma_divisor = 64.0
        self.field_highfreq_densify_eps = 0.001
        self.field_highfreq_densify_y_min = 0.03
        self.field_highfreq_densify_y_max = 0.97
        self.field_highfreq_densify_min_pixels = 64
        self.field_highfreq_densify_gate_start = 0.3
        self.field_highfreq_densify_gate_width = 0.4
        self.field_appearance_only_train = 0
        self.field_appearance_only_start = 20000
        self.field_appearance_only_allow = "f_dc,f_t,decoder"
        self.field_soft_geometry_lr = 0
        self.field_soft_geometry_start = 20000
        self.field_soft_geometry_lr_scale = 0.5
        self.field_soft_geometry_full_lr_groups = "f_dc,f_t,decoder,field_static_app,field_static_view_mapper"
        self.field_content_exposure = 0
        self.field_content_exposure_lr = 0.001
        self.field_content_exposure_hidden = 8
        self.field_content_exposure_mode = "affine"
        self.field_content_exposure_max_log_scale = 0.2
        self.field_content_exposure_max_bias = 0.05
        self.field_content_exposure_max_wb_log_gain = 0.08
        self.field_content_exposure_reg_weight = 0.0
        self.field_content_exposure_wb_reg_weight = 5.0
        self.field_content_exposure_eps = 0.001
        self.field_content_exposure_detach_stats = 1
        self.field_depthpro_supervision = 0
        self.field_depthpro_path = ""
        self.field_depthpro_start = 3000
        self.field_depthpro_until = -1
        self.field_depthpro_loss_weight = 0.0
        self.field_depthpro_max_depth = 2.0
        self.field_depthpro_min_pixels = 256
        self.field_depthpro_error_clamp = 1.0
        self.field_depthpro_use_beit_mask = 1
        self.field_depthpro_exclude_unreliable = 1
        self.field_scale_reg = 0
        self.field_scale_reg_start = 9000
        self.field_scale_reg_until = -1
        self.field_scale_reg_weight = 0.0
        self.field_scale_reg_base_limit = 0.3
        self.field_scale_reg_depth_ref = 8.0
        self.field_scale_reg_depth_mode = "euclidean"
        self.field_scale_reg_depth_gamma = 0.75
        self.field_scale_reg_max_boost = 8.0
        self.field_bg_candidate_grad_boost = 0
        self.field_bg_candidate_feature_grad_scale = 3.0
        self.field_bg_candidate_opacity_grad_scale = 2.0
        self.field_bg_candidate_scaling_grad_scale = 1.5
        self.field_bg_only_train = 0
        self.field_bg_only_start = 3000
        self.field_bg_only_until = 12000
        self.field_bg_only_interval = 1
        self.field_bg_only_loss_weight = 1.0
        self.field_bg_only_min_pixels = 128
        self.field_bg_only_da3_filter = 1
        self.field_bg_only_update_modules = 0
        self.field_obs_reliability = 0
        self.field_obs_reliability_floor = 0.35
        self.field_obs_reliability_mad_threshold = 0.045
        self.field_obs_reliability_diff_threshold = 0.12
        self.field_obs_reliability_motion_threshold = 0.12
        self.field_obs_reliability_mad_weight = 0.40
        self.field_obs_reliability_diff_weight = 0.40
        self.field_obs_reliability_motion_weight = 0.20
        self.field_obs_reliability_unreliable_threshold = 0.55
        self.field_obs_reliability_debug = 0
        self.field_obs_reliability_start = 1500
        self.field_obs_reliability_until = -1
        self.field_obs_reliability_ema = 0.05
        self.field_obs_reliability_error_quantile = 0.90
        self.field_obs_reliability_error_threshold = 0.0
        self.field_obs_reliability_min_error = 0.03
        self.field_obs_reliability_dynamic_dilate = 5
        self.field_obs_reliability_structural_weight = 0.5
        self.field_obs_reliability_local_window = 31
        self.field_obs_boost_unreliable_loss = 0
        self.field_obs_boost_weight = 2.0
        self.field_obs_reset = 0
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
        self.field_obs_reset_debug = 0
        self.field_obs_reset_debug_max_events = 32
        self.field_obs_reset_log_zero = 1
        self.field_obs_reset_scan_time_indices = "0,12,25,37,49"
        self.field_obs_reset_scan_views_per_time = 0
        self.field_obs_reset_scan_min_hits = 2
        self.field_obs_reset_scan_top_ratio = 0.2
        self.field_obs_reset_scan_max_points = 0
        self.field_obs_reset_scan_update_ema = 0
        self.field_global_reset = 0
        self.field_global_reset_schedule = ""
        self.field_freq_prior = 0
        self.field_freq_prior_start = 3500
        self.field_freq_prior_until = 12000
        self.field_freq_prior_weight = 0.01
        self.field_freq_prior_patch_size = 32
        self.field_freq_prior_highpass = 0.25
        self.field_freq_prior_max_patches = 16
        self.field_freq_prior_min_mask_ratio = 0.05
        self.field_freq_prior_reference = "median"
        self.field_freq_prior_on_reset_only = 0
        self.field_freq_prior_debug = 0
        self.field_freq_prior_debug_max_events = 32
        self.field_freq_prior_debug_mode = "first_per_camera"
        self.field_bg_median_loss = 0
        self.field_bg_median_loss_weight = 0.05
        self.field_init_depth_debug = 0
        self.field_init_depth_only = 0
        self.field_init_depth_time_indices = "0,12,25,37,49"
        self.field_init_depth_views_per_time = 0
        self.field_init_depth_max_depth = 15.0
        self.field_init_depth_thresholds = "15,25,50"
        self.field_init_point_projection_debug = 0
        self.field_init_point_projection_dot_radius = 1
        


        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.featuret_lr = 0.001
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005

        self.trbfc_lr = 0.0001 # 
        self.trbfs_lr = 0.03
        self.trbfslinit = 0.0 # 
        self.batch = 2
        self.movelr = 3.5

        self.omega_lr = 0.0001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3_000
        self.opacity_reset_at = 10000
        self.densify_from_iter = 500
        self.densify_until_iter = 9000
        self.densify_grad_threshold = 0.0002
        self.rgb_lr = 0.0001
        self.desicnt = 6
        self.reg = 0 
        self.regl = 0.0001 
        self.shrinkscale = 2.0 
        self.randomfeature = 0 
        self.emstype = 0
        self.radials = 10.0
        self.farray = 2 # 
        self.emsstart = 1600 #small for debug
        self.losstart = 200
        self.saveemppoints = 0 #
        self.prunebysize = 0 
        self.emsthr = 0.6  
        self.opthr = 0.005
        self.selectiveview = 0  
        self.preprocesspoints = 0  
        self.fzrotit = 8001
        self.addsphpointsscale = 0.8  
        self.gnumlimit = 330000 
        self.rayends = 7.5
        self.raystart = 0.7
        self.shuffleems = 1
        self.prevpath = "1"
        self.loadall = 0
        self.removescale = 5
        self.gtmask = 0 # 0 means not train with mask for undistorted gt image; 1 means 
        self.gtisint8 = 0 # 0 means gt is used as float . 
        self.field_lr = 0.001
        self.field_decoder_lr = 0.0005
        self.grid_logits_lr = 0.0025
        self.field_gate_lr = 0.001
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
