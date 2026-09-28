#!/usr/bin/env python3
"""Export a static-grid-baked STEGF checkpoint for compact inference.

The final STEGF model queries the static appearance grid at canonical Gaussian
positions.  This exporter averages that appearance contribution over the
training cameras, folds the resulting six channels into ``_features_dc``, and
writes a separate checkpoint with ``use_euler_field=0``.  The source training
checkpoint is never modified.
"""

import hashlib
import json
import os
import shutil
import sys
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GAUSSIAN_SPLATTING_ROOT = REPO_ROOT / "thirdparty" / "gaussian_splatting"
if str(GAUSSIAN_SPLATTING_ROOT) not in sys.path:
    sys.path.append(str(GAUSSIAN_SPLATTING_ROOT))

from helper_train import getmodel, getrenderpip, trbfunction
from thirdparty.gaussian_splatting.arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
)
from thirdparty.gaussian_splatting.helper3dg import (
    _normalize_existence_mode,
    _normalize_motion_model,
    get_combined_args,
)
from thirdparty.gaussian_splatting.scene import Scene
from thirdparty.gaussian_splatting.utils.general_utils import safe_state


def _parse_args():
    parser = ArgumentParser(
        description="Export a static-grid-baked compact STEGF checkpoint"
    )
    model_group = ModelParams(parser, sentinel=True)
    OptimizationParams(parser)
    pipeline_group = PipelineParams(parser)

    parser.add_argument("--test_iteration", default=-1, type=int)
    parser.add_argument("--duration", default=None, type=int)
    parser.add_argument("--rgbfunction", default=None, type=str)
    parser.add_argument("--rdpip", default=None, type=str)
    parser.add_argument("--valloader", default=None, type=str)
    parser.add_argument("--configpath", default=None, type=str)
    parser.add_argument("--quiet", action="store_true", default=None)
    parser.add_argument(
        "--output_model_path",
        default="",
        type=str,
        help="Compact model root (default: <model_path>/compact).",
    )
    parser.add_argument(
        "--camera_round_decimals",
        default=6,
        type=int,
        help="Precision used to identify unique and held-out camera centers.",
    )
    parser.add_argument(
        "--verify_indices",
        default="0,25,49",
        type=str,
        help="Validation-frame indices used for an in-memory bake check.",
    )
    parser.add_argument(
        "--verify_max_abs_warning",
        default=1e-3,
        type=float,
        help="Warn when a checked baked render differs by more than this.",
    )
    parser.add_argument(
        "--skip_verify",
        action="store_true",
        help="Skip image-space verification before writing the checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of the same compact checkpoint iteration.",
    )

    args = get_combined_args(parser)
    config_path = getattr(args, "configpath", None)
    if config_path and config_path != "None" and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    fallbacks = {
        "duration": 50,
        "rgbfunction": "rgbv1",
        "rdpip": "train_ours_full",
        "valloader": "colmapvalid",
        "configpath": "None",
        "quiet": False,
    }
    for key, value in fallbacks.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    args = _normalize_existence_mode(args)
    args = _normalize_motion_model(args)
    return args, model_group.extract(args), pipeline_group.extract(args)


def _module_eval(module):
    if module is not None:
        module.cuda()
        module.eval()


def _prepare_model(gaussians):
    for name in (
        "rgbdecoder",
        "content_exposure_head",
        "euler_field",
        "field_static_view_mapper",
        "field_static_app_head",
    ):
        _module_eval(getattr(gaussians, name, None))


def _camera_key_from_center(center, decimals):
    center = center.detach().float().cpu().reshape(-1)
    return tuple(round(float(value), int(decimals)) for value in center.tolist())


def _camera_key(camera, decimals):
    return _camera_key_from_center(camera.camera_center, decimals)


def _checkpoint_camera_records(model_path, device):
    camera_path = Path(model_path) / "cameras.json"
    if not camera_path.is_file():
        raise RuntimeError(
            "Compact export requires cameras.json in the source model root: "
            f"{camera_path}"
        )
    with camera_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    cameras = []
    for entry in entries:
        position = entry.get("position")
        if not isinstance(position, list) or len(position) != 3:
            continue
        cameras.append(
            SimpleNamespace(
                camera_center=torch.tensor(
                    position,
                    device=device,
                    dtype=torch.float32,
                ),
                image_name=str(entry.get("img_name", "unknown")),
            )
        )
    if not cameras:
        raise RuntimeError(f"No camera positions were found in {camera_path}")
    return cameras, camera_path


def _unique_cameras(cameras, decimals):
    unique = {}
    for camera in cameras:
        unique.setdefault(_camera_key(camera, decimals), camera)
    return list(unique.values())


def _training_cameras(all_cameras, test_views, decimals):
    unique_all = _unique_cameras(all_cameras, decimals)
    test_keys = {_camera_key(camera, decimals) for camera in test_views}
    training = [
        camera
        for camera in unique_all
        if _camera_key(camera, decimals) not in test_keys
    ]
    matched_test = {
        _camera_key(camera, decimals)
        for camera in unique_all
        if _camera_key(camera, decimals) in test_keys
    }
    if not training:
        raise RuntimeError(
            "No training camera centers remain after excluding validation "
            "camera centers"
        )
    if not matched_test:
        raise RuntimeError(
            "The validation camera center was not found in cameras.json; "
            "refusing to average cameras because the train/test split cannot "
            "be verified"
        )
    return training, unique_all, test_keys, matched_test


def _static_app_delta(gaussians, camera_center):
    level_features = gaussians.euler_field.query_static_level_features(
        gaussians._xyz.detach(),
        camera_center=camera_center,
        view_mapper=gaussians.field_static_view_mapper,
        view_scale=float(gaussians.field_static_app_scale),
        timestamp=None,
    )
    level_logits = gaussians._get_static_level_logits().detach()
    feature = gaussians.euler_field.blend_level_features(
        level_features,
        level_logits,
    )
    residual = gaussians.field_static_app_head(feature)
    return float(gaussians.field_static_app_scale) * residual


def _validate_baking_conditions(gaussians):
    failures = []
    if not bool(getattr(gaussians, "use_euler_field", False)):
        failures.append("use_euler_field=0")
    if getattr(gaussians, "euler_field", None) is None:
        failures.append("euler_field is missing")
    if getattr(gaussians, "field_static_view_mapper", None) is None:
        failures.append("field_static_view_mapper is missing")
    if getattr(gaussians, "field_static_app_head", None) is None:
        failures.append("field_static_app_head is missing")
    if float(getattr(gaussians, "field_static_app_scale", 0.0)) <= 0.0:
        failures.append("field_static_app_scale<=0")
    if bool(getattr(gaussians, "field_static_temporal_residual", False)):
        failures.append("field_static_temporal_residual=1")
    if bool(getattr(gaussians, "field_static_radiance_branch", False)):
        failures.append("field_static_radiance_branch=1")
    if not bool(getattr(gaussians, "field_disable_dynamic_grid", False)):
        failures.append("field_disable_dynamic_grid=0")
    if str(getattr(gaussians, "field_motion_model", "polynomial")) in {
        "h2",
        "couptest_grid",
    }:
        failures.append(
            f"field_motion_model={gaussians.field_motion_model} requires a grid"
        )
    if gaussians._features_dc.ndim != 2 or gaussians._features_dc.shape[1] < 6:
        failures.append("features_dc has fewer than 6 channels")
    if failures:
        raise RuntimeError(
            "Checkpoint is not compatible with isolated static-appearance "
            "baking: " + ", ".join(failures)
        )


def _mean_static_delta(gaussians, cameras):
    mean = torch.zeros(
        (gaussians.get_xyz.shape[0], 6),
        device=gaussians.get_xyz.device,
        dtype=gaussians.get_xyz.dtype,
    )
    second_moment = torch.zeros_like(mean)
    camera_reports = []
    for index, camera in enumerate(cameras, start=1):
        residual = _static_app_delta(gaussians, camera.camera_center)
        mean.add_(residual)
        second_moment.add_(residual.square())
        camera_reports.append(
            {
                "image_name": camera.image_name,
                "camera_center": [
                    float(value)
                    for value in camera.camera_center.detach().cpu().reshape(-1)
                ],
                "residual_rms": float(
                    torch.sqrt(residual.float().square().mean()).item()
                ),
            }
        )
        print(
            "[STEGF][Compact] Accumulating training-camera grid output: "
            f"{index}/{len(cameras)}",
            flush=True,
        )
    count = float(len(cameras))
    mean.div_(count)
    variance = (second_moment / count - mean.square()).clamp_min(0.0)
    std = torch.sqrt(variance)
    mean_rms = float(torch.sqrt(mean.float().square().mean()).item())
    std_rms = float(torch.sqrt(std.float().square().mean()).item())
    return mean, {
        "mean_residual_rms": mean_rms,
        "cross_view_std_rms": std_rms,
        "cross_view_std_to_mean_residual_rms": std_rms / max(mean_rms, 1e-12),
        "per_camera": camera_reports,
    }


def _parse_indices(raw, count):
    indices = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            index = int(token)
            if 0 <= index < count:
                indices.append(index)
    return sorted(set(indices))


@contextmanager
def _baked_variant(gaussians, base_features, mean_delta):
    original_use_field = bool(gaussians.use_euler_field)
    try:
        with torch.no_grad():
            gaussians._features_dc.copy_(base_features)
            gaussians._features_dc[:, :6].add_(mean_delta)
            gaussians.use_euler_field = False
        yield
    finally:
        with torch.no_grad():
            gaussians._features_dc.copy_(base_features)
        gaussians.use_euler_field = original_use_field


def _render_image(camera, gaussians, pipeline, background, render, settings, rasterizer):
    package = render(
        camera,
        gaussians,
        pipeline,
        background,
        scaling_modifier=1.0,
        basicfunction=trbfunction,
        GRsetting=settings,
        GRzer=rasterizer,
    )
    image = package["render"]
    if hasattr(gaussians, "apply_content_exposure"):
        image = gaussians.apply_content_exposure(image)
    return image.clamp(0.0, 1.0)


def _verify_bake(gaussians, test_views, pipeline, mean_delta, indices):
    if not indices:
        return {"enabled": False, "frames": []}
    render, settings, rasterizer = getrenderpip("test_ours_full")
    background = torch.zeros((9,), dtype=torch.float32, device="cuda")
    if gaussians.ts is None:
        view = test_views[0]
        gaussians.ts = torch.ones(
            (1, 1, view.image_height, view.image_width),
            device="cuda",
        )
    base_features = gaussians._features_dc.detach().clone()
    frames = []
    with torch.no_grad():
        for index in indices:
            camera = test_views[index]
            gaussians.use_euler_field = True
            gaussians._features_dc.copy_(base_features)
            full = _render_image(
                camera,
                gaussians,
                pipeline,
                background,
                render,
                settings,
                rasterizer,
            )
            with _baked_variant(gaussians, base_features, mean_delta):
                baked = _render_image(
                    camera,
                    gaussians,
                    pipeline,
                    background,
                    render,
                    settings,
                    rasterizer,
                )
            difference = (baked - full).abs()
            mse = difference.square().mean().clamp_min(1e-12)
            result = {
                "time_index": int(index),
                "timestamp": float(camera.timestamp),
                "image_name": str(camera.image_name),
                "l1_to_full": float(difference.mean().item()),
                "rmse_to_full": float(torch.sqrt(mse).item()),
                "psnr_to_full": float((-10.0 * torch.log10(mse)).item()),
                "max_abs_to_full": float(difference.max().item()),
            }
            frames.append(result)
            print(
                "[STEGF][Compact] Verification: "
                f"frame={index}, l1={result['l1_to_full']:.3e}, "
                f"max={result['max_abs_to_full']:.3e}, "
                f"psnr_to_full={result['psnr_to_full']:.3f} dB",
                flush=True,
            )
    gaussians.use_euler_field = True
    with torch.no_grad():
        gaussians._features_dc.copy_(base_features)
    return {
        "enabled": True,
        "frames": frames,
        "mean_l1_to_full": float(np.mean([item["l1_to_full"] for item in frames])),
        "max_abs_to_full": float(max(item["max_abs_to_full"] for item in frames)),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_runtime_metadata(source_root, output_root, config_path, args):
    for name in ("cameras.json", "stegf_config.json"):
        source = source_root / name
        if source.is_file():
            shutil.copy2(source, output_root / name)

    compact_config = None
    compact_config_path = None
    if config_path and config_path != "None" and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            compact_config = json.load(handle)
        compact_config["use_euler_field"] = 0
        compact_config["field_static_app_scale"] = 0.0
        compact_config_path = output_root / "compact_config.json"
        with compact_config_path.open("w", encoding="utf-8") as handle:
            json.dump(compact_config, handle, indent=2, sort_keys=True)
            handle.write("\n")

    export_only_keys = {
        "output_model_path",
        "camera_round_decimals",
        "verify_indices",
        "verify_max_abs_warning",
        "skip_verify",
        "overwrite",
    }
    compact_args = {
        key: value
        for key, value in vars(args).items()
        if key not in export_only_keys
    }
    compact_args["model_path"] = str(output_root)
    compact_args["use_euler_field"] = False
    compact_args["field_static_app_scale"] = 0.0
    if compact_config_path is not None:
        compact_args["configpath"] = str(compact_config_path)
    with (output_root / "cfg_args").open("w", encoding="utf-8") as handle:
        handle.write(str(Namespace(**compact_args)))
    return compact_config_path


def _write_compact_checkpoint(
    gaussians,
    base_features,
    mean_delta,
    output_ply,
    metadata,
    overwrite,
):
    output_pt = output_ply.with_suffix(".pt")
    manifest_path = output_ply.parent / "compact_export.json"
    existing = [path for path in (output_ply, output_pt, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Compact checkpoint already exists; pass --overwrite to replace "
            "this exact iteration:\n" + "\n".join(str(path) for path in existing)
        )

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    temporary_ply = output_ply.with_name("point_cloud.compact_tmp.ply")
    temporary_pt = temporary_ply.with_suffix(".pt")
    for path in (temporary_ply, temporary_pt):
        if path.exists():
            path.unlink()

    original_use_field = bool(gaussians.use_euler_field)
    original_static_app_scale = float(gaussians.field_static_app_scale)
    implicit_motion_anchor = (
        str(gaussians.field_motion_model).strip().lower()
        in {"couptest_polynomial", "couptest_grid"}
    )
    if implicit_motion_anchor and not bool(
        torch.allclose(
            gaussians._motion_time_anchor.detach(),
            gaussians.get_trbfcenter.detach(),
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise RuntimeError(
            "Compact final-model export requires fixed lifecycle centers; "
            "motion_time_anchor differs from trbf_center"
        )
    try:
        with torch.no_grad():
            gaussians._features_dc.copy_(base_features)
            gaussians._features_dc[:, :6].add_(mean_delta)
            gaussians.use_euler_field = False
            gaussians.field_static_app_scale = 0.0
        gaussians.save_ply(str(temporary_ply))

        payload = torch.load(temporary_pt, map_location="cpu")
        training_only_keys = (
            "static_level_logits",
            "static_radiance_level_logits",
            "dynamic_level_logits",
            "dynamic_level_time_coeff",
            "static_route_logits",
            "grid_level_logits",
            "grid_level_time_coeff",
            "field_residual_gate",
            "euler_field",
            "h2_velocity_field",
            "grid_motion_field",
            "field_router",
            "field_query_gate",
            "field_decoder",
            "field_temporal_opacity_head",
            "field_static_view_mapper",
            "field_static_app_head",
            "error_prior",
            "ems_mask",
            "dynamic_score_ema",
            "dynamic_active_mask",
            "responsibility_ema",
            "responsibility_time_center_ema",
            "slow_motion_score_ema",
            "slow_motion_mask",
            "fast_score_ema",
            "fast_active_mask",
            "static_support_ema",
            "static_support_mask",
            "visibility_persistence_ema",
        )
        removed_keys = [key for key in training_only_keys if key in payload]
        for key in removed_keys:
            payload.pop(key, None)
        metadata["removed_auxiliary_keys"] = removed_keys
        removed_redundant_keys = []
        if implicit_motion_anchor and payload.get("motion_time_anchor") is not None:
            payload.pop("motion_time_anchor", None)
            removed_redundant_keys.append("motion_time_anchor")
        metadata["removed_redundant_keys"] = removed_redundant_keys
        metadata["motion_time_anchor_storage"] = (
            "implicit_trbf_center"
            if implicit_motion_anchor
            else "checkpoint_tensor"
        )
        payload["compact_export"] = metadata
        torch.save(payload, temporary_pt)
        os.replace(temporary_ply, output_ply)
        os.replace(temporary_pt, output_pt)
    finally:
        with torch.no_grad():
            gaussians._features_dc.copy_(base_features)
        gaussians.use_euler_field = original_use_field
        gaussians.field_static_app_scale = original_static_app_scale
        for path in (temporary_ply, temporary_pt):
            if path.exists():
                path.unlink()
    return output_pt, manifest_path


def export_compact(args, dataset, pipeline):
    if not torch.cuda.is_available():
        raise RuntimeError("Compact export requires CUDA")

    source_root = Path(dataset.model_path).resolve()
    output_root = (
        Path(args.output_model_path).resolve()
        if str(args.output_model_path).strip()
        else source_root / "compact"
    )
    if output_root == source_root:
        raise RuntimeError("Compact output must differ from the source model path")

    GaussianModel = getmodel(dataset.model)
    gaussians = GaussianModel(dataset.sh_degree, args.rgbfunction)
    if hasattr(gaussians, "configure_euler_field"):
        gaussians.configure_euler_field(dataset)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.test_iteration,
        shuffle=False,
        multiview=False,
        duration=args.duration,
        loader=args.valloader,
    )
    iteration = int(scene.loaded_iter)
    _prepare_model(gaussians)
    _validate_baking_conditions(gaussians)

    test_views = sorted(
        scene.getTestCameras(),
        key=lambda camera: float(camera.timestamp),
    )
    if not test_views:
        raise RuntimeError("The selected loader produced no validation cameras")
    all_records, camera_path = _checkpoint_camera_records(
        source_root,
        gaussians.get_xyz.device,
    )
    training_cameras, unique_all, test_keys, matched_test = _training_cameras(
        all_records,
        test_views,
        args.camera_round_decimals,
    )

    print(
        "[STEGF][Compact] Exporting static-grid-baked checkpoint: "
        f"iteration={iteration}, gaussians={gaussians.get_xyz.shape[0]}, "
        f"training_cameras={len(training_cameras)}, "
        f"excluded_validation_cameras={len(matched_test)}",
        flush=True,
    )
    with torch.no_grad():
        mean_delta, residual_report = _mean_static_delta(
            gaussians,
            training_cameras,
        )

    verify_indices = [] if args.skip_verify else _parse_indices(
        args.verify_indices,
        len(test_views),
    )
    verification = _verify_bake(
        gaussians,
        test_views,
        pipeline,
        mean_delta,
        verify_indices,
    )
    warning_threshold = float(args.verify_max_abs_warning)
    verification["max_abs_warning_threshold"] = warning_threshold
    verification["warning"] = bool(
        verification.get("max_abs_to_full", 0.0) > warning_threshold
    )
    if verification["warning"]:
        print(
            "[STEGF][Compact][Warning] Checked maximum image difference "
            f"{verification['max_abs_to_full']:.3e} exceeds "
            f"{warning_threshold:.3e}; the checkpoint will still be exported.",
            flush=True,
        )

    source_ply = (
        source_root
        / "point_cloud"
        / f"iteration_{iteration}"
        / "point_cloud.ply"
    )
    source_pt = source_ply.with_suffix(".pt")
    missing_source_files = [
        str(path) for path in (source_ply, source_pt) if not path.is_file()
    ]
    if missing_source_files:
        raise FileNotFoundError(
            "Source checkpoint is incomplete:\n" + "\n".join(missing_source_files)
        )
    output_ply = (
        output_root
        / "point_cloud"
        / f"iteration_{iteration}"
        / "point_cloud.ply"
    )
    base_features = gaussians._features_dc.detach().clone()
    metadata = {
        "schema": "stegf_compact_static_grid_bake_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_model_path": str(source_root),
        "source_iteration": iteration,
        "source_point_cloud": str(source_ply),
        "source_camera_metadata": str(camera_path),
        "output_model_path": str(output_root),
        "gaussians": int(gaussians.get_xyz.shape[0]),
        "bake_target": "features_dc[:,0:6]",
        "bake_operation": "training_camera_mean_static_grid_residual",
        "field_query_position": "canonical_xyz",
        "training_camera_centers": len(training_cameras),
        "all_unique_camera_centers": len(unique_all),
        "excluded_validation_camera_centers": len(matched_test),
        "validation_camera_keys": [list(key) for key in sorted(test_keys)],
        "camera_round_decimals": int(args.camera_round_decimals),
        "residual": residual_report,
        "verification": verification,
        "compact_field_config": {
            "use_euler_field": 0,
            "field_static_app_scale": 0.0,
            "static_grid_baked": True,
        },
    }
    output_pt, manifest_path = _write_compact_checkpoint(
        gaussians,
        base_features,
        mean_delta,
        output_ply,
        metadata,
        bool(args.overwrite),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    compact_config_path = _copy_runtime_metadata(
        source_root,
        output_root,
        args.configpath,
        args,
    )

    artifact_report = {
        "source_ply_bytes": source_ply.stat().st_size,
        "source_pt_bytes": source_pt.stat().st_size,
        "compact_ply_bytes": output_ply.stat().st_size,
        "compact_pt_bytes": output_pt.stat().st_size,
        "source_total_bytes": source_ply.stat().st_size + source_pt.stat().st_size,
        "compact_total_bytes": output_ply.stat().st_size + output_pt.stat().st_size,
        "compact_ply_sha256": _sha256(output_ply),
        "compact_pt_sha256": _sha256(output_pt),
    }
    artifact_report["saved_bytes"] = (
        artifact_report["source_total_bytes"]
        - artifact_report["compact_total_bytes"]
    )
    artifact_report["size_reduction_ratio"] = (
        artifact_report["saved_bytes"]
        / max(artifact_report["source_total_bytes"], 1)
    )
    metadata["artifacts"] = artifact_report
    metadata["compact_point_cloud"] = str(output_ply)
    metadata["compact_aux_checkpoint"] = str(output_pt)
    metadata["compact_config"] = (
        str(compact_config_path) if compact_config_path is not None else None
    )
    test_config = compact_config_path or args.configpath
    recommended_test_command = [
        sys.executable,
        str(REPO_ROOT / "script" / "test_all_iterations.py"),
        "--iterations",
        str(iteration),
        "--quiet",
        "--eval",
        "--skip_train",
        "--valloader",
        str(args.valloader),
        "--configpath",
        str(test_config),
        "--model_path",
        str(output_root),
        "--source_path",
        str(Path(dataset.source_path).resolve()),
    ]
    metadata["recommended_test_command"] = recommended_test_command
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        "[STEGF][Compact] Export complete: "
        f"{output_ply}\n"
        f"[STEGF][Compact] Auxiliary checkpoint: {output_pt}\n"
        f"[STEGF][Compact] Manifest: {manifest_path}\n"
        f"[STEGF][Compact] Removed checkpoint bytes: "
        f"{artifact_report['saved_bytes']} "
        f"({100.0 * artifact_report['size_reduction_ratio']:.2f}%)",
        flush=True,
    )
    if compact_config_path is not None:
        print(
            "[STEGF][Compact] Test with config: "
            f"{compact_config_path}",
            flush=True,
        )
    print(
        "[STEGF][Compact] Verification command:\n  "
        + " ".join(recommended_test_command),
        flush=True,
    )


def main():
    args, dataset, pipeline = _parse_args()
    safe_state(bool(args.quiet))
    export_compact(args, dataset, pipeline)


if __name__ == "__main__":
    main()
