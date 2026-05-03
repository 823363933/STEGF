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
