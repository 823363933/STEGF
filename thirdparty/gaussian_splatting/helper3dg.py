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
import os
import torch
from random import randint
import random 
import sys 
import uuid
import time 
import json

import numpy as np 
import cv2
from tqdm import tqdm
import shutil

sys.path.append("./thirdparty/gaussian_splatting")

from thirdparty.gaussian_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args


def _truthy(value):
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "none", "no")
    return bool(value)


def _normalize_existence_mode(args):
    mode = str(
        getattr(args, "field_existence_single_expert", "none")
    ).strip().lower()
    aliases = {
        "": "none",
        "off": "none",
        "legacy": "none",
        "p": "persistent",
        "i": "interval",
        "t": "transient",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"none", "persistent", "interval", "transient"}:
        raise ValueError(
            "field_existence_single_expert must be one of "
            "none/persistent/interval/transient, got {!r}".format(mode)
        )
    args.field_existence_single_expert = mode
    if mode != "none":
        carrier_motion = str(
            getattr(args, "field_motion_model", "polynomial")
        ).strip().lower()
        if (
            carrier_motion == "carrier_hybrid"
            and _truthy(getattr(args, "field_carrier_initialization", 0))
            and _truthy(getattr(args, "field_existence_moe", 0))
        ):
            raise ValueError(
                "carrier_hybrid motion requires field_existence_moe=0; "
                "it cannot be combined with existence MoE routing"
            )
        args.field_existence_moe = 0
    return args


def _normalize_motion_model(args):
    mode = str(getattr(args, "field_motion_model", "polynomial")).strip().lower()
    aliases = {
        "": "polynomial",
        "none": "polynomial",
        "off": "polynomial",
        "poly": "polynomial",
        "legacy": "polynomial",
        "h1": "polynomial",
        "shared": "h2",
        "shared_field": "h2",
        "h2_only": "h2",
        "couptest_poly": "couptest_polynomial",
        "polynomial_couptest": "couptest_polynomial",
        "grid_couptest": "couptest_grid",
        "couptest_shared_grid": "couptest_grid",
    }
    mode = aliases.get(mode, mode)
    if mode not in {
        "polynomial",
        "h2",
        "carrier_hybrid",
        "couptest_polynomial",
        "couptest_grid",
    }:
        raise ValueError(
            "field_motion_model must be polynomial, h2, carrier_hybrid, "
            "couptest_polynomial, or couptest_grid, got {!r}".format(
                mode
            )
        )
    args.field_motion_model = mode
    carrier_initialization = _truthy(
        getattr(args, "field_carrier_initialization", 0)
    )
    if carrier_initialization and mode != "carrier_hybrid":
        raise ValueError(
            "field_carrier_initialization=1 requires "
            "field_motion_model=carrier_hybrid"
        )
    if mode in {"couptest_polynomial", "couptest_grid"}:
        couptest_mode = str(
            getattr(args, "field_couptest_mode", "none")
        ).strip().lower()
        if couptest_mode not in {
            "uncoupled",
            "coupled",
            "coupled_detached",
        }:
            raise ValueError(
                "{} motion requires field_couptest_mode="
                "uncoupled, coupled, or coupled_detached, got {!r}".format(
                    mode,
                    couptest_mode,
                )
            )
        if getattr(args, "field_existence_single_expert", "none") != "transient":
            raise ValueError(
                "{} motion requires "
                "field_existence_single_expert=transient".format(mode)
            )
        if carrier_initialization:
            raise ValueError(
                "{} does not use Carrier initialization".format(mode)
            )
        if mode == "couptest_grid" and not _truthy(
            getattr(args, "use_euler_field", 0)
        ):
            raise ValueError("couptest_grid requires use_euler_field=1")
        existence_floor = float(
            getattr(args, "field_couptest_initial_existence_floor", 0.95)
        )
        if not 0.0 < existence_floor < 1.0:
            raise ValueError(
                "field_couptest_initial_existence_floor must be within (0, 1)"
            )
        args.field_couptest_mode = couptest_mode
        args.field_existence_moe = 0
    elif mode == "h2":
        if not _truthy(getattr(args, "use_euler_field", 0)):
            raise ValueError("H2 motion requires use_euler_field=1")
        if getattr(args, "field_existence_single_expert", "none") != "transient":
            raise ValueError(
                "S1.10.24-A H2-only motion requires "
                "field_existence_single_expert=transient"
            )
    elif mode == "carrier_hybrid":
        if not carrier_initialization:
            raise ValueError(
                "carrier_hybrid motion requires field_carrier_initialization=1"
            )
        if getattr(args, "field_existence_single_expert", "none") != "persistent":
            raise ValueError(
                "carrier_hybrid motion requires "
                "field_existence_single_expert=persistent"
            )
        if _truthy(getattr(args, "field_existence_moe", 0)):
            raise ValueError("carrier_hybrid motion requires field_existence_moe=0")
        if int(getattr(args, "preprocesspoints", -1)) != 0:
            raise ValueError("carrier_hybrid motion requires preprocesspoints=0")
        supported_schemas = {
            "stegf_colmap_carrier_initialization_map_v2",
            "stegf_colmap_high_confidence_carrier_initialization_map_v1",
        }
        schema = str(
            getattr(args, "field_carrier_initialization_schema", "")
        ).strip()
        if schema not in supported_schemas:
            raise ValueError(
                "carrier_hybrid motion requires a supported Carrier "
                f"initialization schema, got {schema!r}"
            )
        if schema == "stegf_colmap_high_confidence_carrier_initialization_map_v1":
            if _truthy(getattr(args, "field_bg_dense_add", 0)):
                raise ValueError(
                    "S2.0.1-noadd requires field_bg_dense_add=0"
                )
            if _truthy(getattr(args, "field_bg_prior", 0)):
                raise ValueError("S2.0.1-noadd requires field_bg_prior=0")
            if not _truthy(getattr(args, "field_disable_ems_main", 0)):
                raise ValueError(
                    "S2.0.1-noadd requires field_disable_ems_main=1"
                )
    return args


def _format_summary_value(value):
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def _format_summary_items(values, keys):
    parts = []
    for key in keys:
        if key in values:
            parts.append(f"{key}={_format_summary_value(values[key])}")
    return ", ".join(parts)


def _format_labeled_items(values, items):
    parts = []
    for label, key in items:
        if key in values:
            parts.append(f"{label}={_format_summary_value(values[key])}")
    return ", ".join(parts)


def _print_compact_args_summary(args, title):
    values = vars(args)
    source_path = os.path.normpath(str(values.get("source_path", "")))
    scene_name = os.path.basename(os.path.dirname(source_path)) or "unknown"
    version = str(values.get("stegf_version", "unversioned"))
    run_items = [
        f"version={version}",
        f"scene={scene_name}",
        f"model={values.get('model', 'unknown')}",
    ]
    for label, key in (
        ("iterations", "iterations"),
        ("duration", "duration"),
        ("resolution", "resolution"),
        ("batch", "batch"),
    ):
        if key in values:
            run_items.append(f"{label}={values[key]}")
    print(f"[STEGF] {title}: " + ", ".join(run_items))

    carrier_enabled = _truthy(values.get("field_carrier_initialization", 0))
    dense_enabled = _truthy(values.get("field_dense_initialization", 0))
    carrier_schema = str(values.get("field_carrier_initialization_schema", ""))
    if carrier_schema == "stegf_colmap_high_confidence_carrier_initialization_map_v1":
        init_mode = "carrier_noadd_v1"
    else:
        init_mode = "carrier_v2" if carrier_enabled else "colmap_points"
    if dense_enabled:
        init_mode = "colmap_points+frame0_dense"
    temporal_items = [
        f"init={init_mode}",
        f"existence={values.get('field_existence_single_expert', 'none')}",
        f"motion={values.get('field_motion_model', 'polynomial')}",
    ]
    if (
        str(values.get("field_motion_model", "")).lower()
        in {"couptest_polynomial", "couptest_grid"}
    ):
        temporal_items.append(
            f"coupling={values.get('field_couptest_mode', 'none')}"
        )
    if carrier_enabled:
        temporal_items.append("extrapolation=endpoint_clamped")
    print("[STEGF] Spatiotemporal model: " + ", ".join(temporal_items))
    print(f"[STEGF] Output: {values.get('model_path', '')}")


def _print_args_summary(args, title):
    values = vars(args)
    path_keys = ["model_path", "source_path", "configpath"]
    core_keys = ["model", "loader", "valloader", "iterations", "test_iteration", "save_iterations", "duration", "batch", "densify", "rdpip", "rgbfunction", "timing_repeats"]

    print(f"[STEGF] {title} args: {_format_summary_items(values, path_keys)}")
    core_summary = _format_summary_items(values, core_keys)
    if core_summary:
        print(f"[STEGF] {title} core: {core_summary}")

    bbox_summary = _format_labeled_items(
        values,
        [
            ("expand_xyz", "field_bbox_expand_xyz"),
            ("expand_scale", "field_bbox_expand_scale"),
            ("frustum", "field_bbox_frustum_expand"),
            ("preserve_cell", "field_bbox_preserve_cell_size"),
        ],
    )
    if bbox_summary:
        print(f"[STEGF] Field bbox: {bbox_summary}")

    active_modules = []
    disabled_modules = []
    if _truthy(values.get("field_carrier_initialization", 0)):
        active_modules.append(
            "carrier_init("
            + _format_labeled_items(
                values,
                [
                    ("mode", "field_motion_model"),
                    ("path", "field_carrier_initialization_path"),
                    ("schema", "field_carrier_initialization_schema"),
                    ("existence", "field_existence_single_expert"),
                ],
            )
            + ", extrapolation=endpoint_clamped)"
        )
    else:
        disabled_modules.append("carrier_init")
    if str(values.get("field_motion_model", "polynomial")).lower() == "h2":
        active_modules.append(
            "motion_h2("
            + _format_labeled_items(
                values,
                [
                    ("grids", "field_h2_level_resolutions"),
                    ("feature", "field_h2_feature_dim"),
                    ("hidden", "field_h2_hidden_dim"),
                    ("fourier", "field_h2_fourier_degree"),
                    ("max_speed", "field_h2_max_normalized_speed"),
                    ("steps", "field_h2_integration_steps"),
                    ("method", "field_h2_integration_method"),
                    ("reg", "field_h2_velocity_reg_weight"),
                    ("lr", "field_h2_lr"),
                ],
            )
            + ")"
        )
    else:
        disabled_modules.append("motion_h2")
    if _truthy(values.get("field_bg_dense_add", 0)):
        scale_init = str(values.get("field_bg_prior_scale_init", "")).lower()
        bg_dense_items = [
            ("iter", "field_bg_dense_add_iter"),
            ("mask", "field_bg_dense_mask_source"),
            ("scales", "field_bg_dense_depth_scales"),
            ("beit", "field_bg_dense_beit_filter"),
            ("band_low", "field_bg_dense_beit_band_low"),
            ("band_high", "field_bg_dense_beit_band_high"),
            ("dedup_level", "field_bg_dense_dedup_level"),
            ("dedup_priority", "field_bg_dense_dedup_priority"),
            ("max_per_cell", "field_bg_dense_max_per_cell"),
            ("clip_bbox", "field_bg_dense_clip_to_bbox"),
            ("scale_init", "field_bg_prior_scale_init"),
        ]
        if _truthy(values.get("field_bg_dense_source_time_select", 0)):
            bg_dense_items.extend([
                ("source_times", "field_bg_dense_source_time_select"),
                ("src_support", "field_bg_dense_source_min_support"),
                ("src_bg", "field_bg_dense_source_beit_background_threshold"),
            ])
        if scale_init in ("hybrid_far_knn", "hybrid_knn", "fixed_near_knn_far"):
            bg_dense_items.extend([
                ("hybrid_threshold", "field_bg_prior_hybrid_knn_scale_threshold"),
            ])
        active_modules.append(
            "bg_dense("
            + _format_labeled_items(
                values,
                bg_dense_items,
            )
            + ")"
        )
    else:
        disabled_modules.append("bg_dense")

    if _truthy(values.get("field_obs_reliability", 0)):
        active_modules.append(
            "obs_reliability("
            + _format_labeled_items(
                values,
                [
                    ("start", "field_obs_reliability_start"),
                    ("until", "field_obs_reliability_until"),
                    ("unreliable_thr", "field_obs_reliability_unreliable_threshold"),
                    ("window", "field_obs_reliability_local_window"),
                ],
            )
            + ")"
        )
    else:
        disabled_modules.append("obs_reliability")

    if _truthy(values.get("field_scale_reg", 0)):
        active_modules.append(
            "scale_reg("
            + _format_labeled_items(
                values,
                [
                    ("start", "field_scale_reg_start"),
                    ("until", "field_scale_reg_until"),
                    ("limit", "field_scale_reg_base_limit"),
                    ("weight", "field_scale_reg_weight"),
                    ("depth_mode", "field_scale_reg_depth_mode"),
                    ("gamma", "field_scale_reg_depth_gamma"),
                ],
            )
            + ")"
        )
    else:
        disabled_modules.append("scale_reg")

    if _truthy(values.get("field_highfreq_densify", 0)):
        active_modules.append(
            "highfreq_densify("
            + _format_labeled_items(
                values,
                [
                    ("sigma_div", "field_highfreq_densify_sigma_divisor"),
                    ("eps", "field_highfreq_densify_eps"),
                    ("y_min", "field_highfreq_densify_y_min"),
                    ("y_max", "field_highfreq_densify_y_max"),
                    ("min_pixels", "field_highfreq_densify_min_pixels"),
                    ("gate_start", "field_highfreq_densify_gate_start"),
                    ("gate_width", "field_highfreq_densify_gate_width"),
                ],
            )
            + ")"
        )
    elif "field_highfreq_densify" in values:
        disabled_modules.append("highfreq_densify")

    single_expert = str(
        values.get("field_existence_single_expert", "none")
    ).strip().lower()
    if single_expert in {"persistent", "interval", "transient"}:
        items = [("mode", "field_existence_single_expert")]
        if single_expert == "interval":
            items.extend(
                [
                    ("interval_half", "field_existence_interval_init_half_width"),
                    ("interval_max", "field_existence_interval_max_half_width"),
                    ("interval_tau", "field_existence_interval_transition"),
                    ("center_lr", "field_interval_center_lr"),
                    ("width_lr", "field_interval_width_lr"),
                ]
            )
        elif single_expert == "transient":
            items.extend(
                [
                    ("center_lr", "trbfc_lr"),
                    ("scale_lr", "trbfs_lr"),
                ]
            )
        active_modules.append(
            "existence_single("
            + _format_labeled_items(values, items)
            + (
                ", motion_anchor=decoupled"
                if single_expert == "transient"
                else ""
            )
            + ")"
        )
    elif _truthy(values.get("field_existence_moe", 0)):
        active_modules.append(
            "existence_moe("
            + _format_labeled_items(
                values,
                [
                    ("start", "field_existence_start"),
                    ("temp", "field_existence_temperature_start"),
                    ("temp_end", "field_existence_temperature_end"),
                    ("temp_until", "field_existence_temperature_until"),
                    ("router_init", "field_existence_router_init"),
                    ("interval_half", "field_existence_interval_init_half_width"),
                    ("interval_max", "field_existence_interval_max_half_width"),
                    ("interval_tau", "field_existence_interval_transition"),
                    ("transient_budget", "field_existence_transient_budget"),
                    ("budget_w", "field_existence_budget_weight"),
                    ("transient_width", "field_existence_transient_width_limit"),
                    ("width_route_w", "field_existence_width_route_weight"),
                    ("entropy_w", "field_existence_entropy_weight"),
                    ("coverage_w", "field_existence_coverage_weight"),
                    ("router_lr", "field_existence_lr"),
                ],
            )
            + ")"
        )
    elif "field_existence_moe" in values:
        disabled_modules.append("existence_moe")

    if _truthy(values.get("field_mvstruct", 0)):
        active_modules.append(
            "mvstruct("
            + _format_labeled_items(
                values,
                [
                    ("start", "field_mvstruct_start"),
                    ("until", "field_mvstruct_until"),
                    ("interval", "field_mvstruct_interval"),
                    ("views", "field_mvstruct_views"),
                    ("min_event_views", "field_mvstruct_min_event_views"),
                    ("dssim_w", "field_mvstruct_dssim_weight"),
                    ("densify", "field_mvstruct_densify"),
                    ("densify_start", "field_mvstruct_densify_start"),
                    ("densify_interval", "field_mvstruct_densify_interval"),
                    ("grad_thr", "field_mvstruct_grad_threshold"),
                    ("min_obs", "field_mvstruct_min_observations"),
                    ("vis_ratio", "field_mvstruct_min_visibility_ratio"),
                    ("event_max", "field_mvstruct_event_max_ratio"),
                    ("total_max", "field_mvstruct_total_max_ratio"),
                    ("cooldown", "field_mvstruct_cooldown"),
                    ("oversize", "field_mvstruct_oversize_split"),
                    ("oversize_radius", "field_mvstruct_oversize_radius"),
                    ("oversize_budget", "field_mvstruct_oversize_budget_ratio"),
                    ("hard_time", "field_mvstruct_hard_time"),
                    ("hard_ema", "field_mvstruct_hard_time_ema_decay"),
                    ("hard_sampling", "field_mvstruct_hard_time_sampling"),
                    ("hard_diverse", "field_mvstruct_hard_time_diverse_views"),
                    ("conflict", "field_mvstruct_conflict_split"),
                    ("conf_src", "field_mvstruct_conflict_source"),
                    ("conf_thr", "field_mvstruct_conflict_threshold"),
                    ("conf_events", "field_mvstruct_conflict_min_events"),
                    ("conf_budget", "field_mvstruct_conflict_budget_ratio"),
                    ("conf_radius", "field_mvstruct_conflict_min_radius"),
                    ("conf_children", "field_mvstruct_conflict_children"),
                    ("conf_specialize", "field_mvstruct_conflict_specialize"),
                    ("conf_dirsplit", "field_mvstruct_conflict_directional_split"),
                    ("dir_events", "field_mvstruct_directional_min_events"),
                    ("dir_axis", "field_mvstruct_directional_min_axis_ratio"),
                    ("dir_offset", "field_mvstruct_directional_offset_ratio"),
                    ("spec_events", "field_mvstruct_specialize_min_events"),
                    ("spec_axis", "field_mvstruct_specialize_axis_ratio"),
                    ("spec_radius", "field_mvstruct_specialize_min_radius"),
                    ("spec_delta", "field_mvstruct_specialize_feature_delta"),
                    ("spec_offset", "field_mvstruct_specialize_offset_ratio"),
                    ("spec_scale", "field_mvstruct_specialize_scale_ratio"),
                ],
            )
            + ")"
        )
    elif "field_mvstruct" in values:
        disabled_modules.append("mvstruct")

    if _truthy(values.get("field_appearance_only_train", 0)):
        active_modules.append(
            "appearance_only("
            + _format_labeled_items(
                values,
                [
                    ("start", "field_appearance_only_start"),
                    ("allow", "field_appearance_only_allow"),
                ],
            )
            + ")"
        )
    elif "field_appearance_only_train" in values:
        disabled_modules.append("appearance_only")

    if _truthy(values.get("field_soft_geometry_lr", 0)):
        active_modules.append(
            "soft_geometry_lr("
            + _format_labeled_items(
                values,
                [
                    ("start", "field_soft_geometry_start"),
                    ("scale", "field_soft_geometry_lr_scale"),
                    ("full_lr", "field_soft_geometry_full_lr_groups"),
                ],
            )
            + ")"
        )
    elif "field_soft_geometry_lr" in values:
        disabled_modules.append("soft_geometry_lr")

    if _truthy(values.get("field_layer_responsibility", 0)):
        active_modules.append(
            "layer_resp("
            + _format_labeled_items(
                values,
                [
                    ("start", "field_layer_responsibility_start"),
                    ("until", "field_layer_responsibility_until"),
                    ("interval", "field_layer_responsibility_interval"),
                    ("near_z", "field_layer_near_depth"),
                    ("far_z", "field_layer_far_depth"),
                    ("far_w", "field_layer_far_loss_weight"),
                    ("front_w", "field_layer_front_opacity_weight"),
                    ("front_tau", "field_layer_front_opacity_budget"),
                    ("erode", "field_layer_mask_erode"),
                    ("min_px", "field_layer_min_pixels"),
                    ("debug", "field_layer_debug"),
                ],
            )
            + ")"
        )
    elif "field_layer_responsibility" in values:
        disabled_modules.append("layer_resp")

    if _truthy(values.get("field_content_exposure", 0)):
        active_modules.append(
            "content_exposure("
            + _format_labeled_items(
                values,
                [
                    ("hidden", "field_content_exposure_hidden"),
                    ("mode", "field_content_exposure_mode"),
                    ("max_log_scale", "field_content_exposure_max_log_scale"),
                    ("max_bias", "field_content_exposure_max_bias"),
                    ("max_wb", "field_content_exposure_max_wb_log_gain"),
                    ("reg", "field_content_exposure_reg_weight"),
                    ("wb_reg", "field_content_exposure_wb_reg_weight"),
                    ("detach_stats", "field_content_exposure_detach_stats"),
                    ("lr", "field_content_exposure_lr"),
                ],
            )
            + ")"
        )
    elif "field_content_exposure" in values:
        disabled_modules.append("content_exposure")

    if title.lower().startswith("testing") and _truthy(values.get("test_photometric_fit", 0)):
        active_modules.append(
            "photometric_fit("
            + _format_labeled_items(
                values,
                [
                    ("mode", "test_photometric_fit_mode"),
                    ("reg", "test_photometric_fit_reg"),
                    ("clamp", "test_photometric_fit_clamp"),
                    ("save_images", "test_photometric_fit_save_images"),
                ],
            )
            + ")"
        )

    optional_switches = [
        ("field_depthpro_supervision", "depthpro"),
        ("field_static_radiance_branch", "static_radiance"),
        ("field_bg_only_train", "bg_only"),
        ("field_bg_prior", "bg_prior"),
        ("field_obs_reset", "obs_reset"),
        ("field_freq_prior", "freq_prior"),
        ("field_bg_median_loss", "bg_median"),
        ("field_obs_boost_unreliable_loss", "obs_boost"),
        ("field_init_depth_only", "init_depth"),
    ]
    for switch_key, label in optional_switches:
        if switch_key not in values:
            continue
        if _truthy(values.get(switch_key, 0)):
            active_modules.append(label)
        else:
            disabled_modules.append(label)

    if active_modules:
        print("[STEGF] Active field modules: " + "; ".join(active_modules))
    if disabled_modules:
        print("[STEGF] Disabled field modules: " + ", ".join(disabled_modules))


def getparser():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser) #we put more parameters in optimization params, just for convenience.
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6029)
    parser.add_argument('--debug_from', type=int, default=-2)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10000, 12000, 25_000, 30_000])
    parser.add_argument("--test_iterations", default=-1, type=int)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--densify", type=int, default=1, help="densify =1, we control points on N3d dataset")
    parser.add_argument("--duration", type=int, default=5, help="5 debug , 50 used")
    parser.add_argument("--basicfunction", type=str, default = "gaussian")
    parser.add_argument("--rgbfunction", type=str, default = "rgbv1")
    parser.add_argument("--rdpip", type=str, default = "v2")
    parser.add_argument("--configpath", type=str, default = "None")

    
    
    # 1. Get default values
    defaults = vars(parser.parse_args([]))

    # 2. Get actual user-provided args
    args = parser.parse_args()

    # Always save the final iteration, but keep the list stable and unique.
    args.save_iterations.append(args.iterations)
    args.save_iterations = list(dict.fromkeys(args.save_iterations))

    # 3. Load config if provided
    if os.path.exists(args.configpath) and args.configpath != "None":
        if args.quiet:
            print("[STEGF] Config:", args.configpath)
        else:
            print("Overriding from config:", args.configpath)
        with open(args.configpath) as f:
            config = json.load(f)

        for k, v in config.items():
            if hasattr(args, k):
                current_val = getattr(args, k)
                default_val = defaults.get(k)

                if current_val == default_val:
                    setattr(args, k, v)
                    
                else:
                    print(f"Kept CLI override for '{k}': {current_val}")
            else:
                print(f"Unknown config key '{k}', skipping.")

        if not args.quiet:
            print("Finished loading config.")

    args = _normalize_existence_mode(args)
    args = _normalize_motion_model(args)
    if args.quiet:
        _print_compact_args_summary(args, "Training")
    else:
        _print_args_summary(args, "Training")
        print("Optimizing", args.model_path)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)


    return args, lp.extract(args), op.extract(args), pp.extract(args)

def getrenderparts(render_pkg):
    return render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]




def gettestparse():
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--test_iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--duration", default=50, type=int)
    parser.add_argument("--rgbfunction", type=str, default = "rgbv1")
    parser.add_argument("--rdpip", type=str, default = "v3")
    parser.add_argument("--valloader", type=str, default = "colmap")
    parser.add_argument("--configpath", type=str, default = "1")
    parser.add_argument("--timing_repeats", default=0, type=int)

    parser.add_argument("--quiet", action="store_true")
    
    # record default value 
    defaults = vars(parser.parse_args([]))

    # update newest terminal command
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    multiview = True if args.valloader.endswith("mv") else False

    # if config file exisits
    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        config = json.load(open(args.configpath))
        for k, v in config.items():
            # not passed in by user
            if hasattr(args, k) and getattr(args, k) == defaults.get(k):
                setattr(args, k, v)
            elif hasattr(args, k):
                print(f"Keeping command line value for '{k}'")

        print("finish load config from " + args.configpath)
        args = _normalize_existence_mode(args)
        args = _normalize_motion_model(args)
        if args.quiet:
            _print_compact_args_summary(args, "Testing")
        else:
            _print_args_summary(args, "Testing")
    else:
        args = _normalize_existence_mode(args)
        args = _normalize_motion_model(args)

    return args, model.extract(args), pipeline.extract(args), multiview
    
def getcolmapsinglen3d(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --database_path " + dbfile+ " --image_path " + inputimagefolder

    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)


    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)

   # threshold is from   https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/scripts/local_colmap_and_resize.sh#L62
    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)


    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel  + " --output_path " + folder  \
    + " --output_type COLMAP" 
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)





def getcolmapsingleimundistort(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor SiftExtraction.max_image_size 6000 --database_path " + dbfile+ " --image_path " + inputimagefolder 

    
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    

    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)


 

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    #Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
   



def getcolmapsingleimdistort(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor SiftExtraction.max_image_size 6000 --database_path " + dbfile+ " --image_path " + inputimagefolder 
    
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    

    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
        

def getcolmapsingletechni(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --database_path " + dbfile+ " --image_path " + inputimagefolder 

    
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    

    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  #
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)


    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
    
    return 
    
