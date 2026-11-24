"""Coordinate helpers for background insertion, independent of CUDA rasterizers.

Immersive pixels belong to the final fisheye supervision image, while camera
projection matrices describe the full-resolution wide-pinhole render. MiDaS
arrays produced by precompute_midas_beit_depth_like.py also use that pinhole
domain. Never treat a fisheye pixel as a pinhole pixel or a relative MiDaS
prediction as a metric depth.
"""

import math

import torch
import torch.nn.functional as F


INITIAL_CAMERA_DEPTH_BASE = "initial_camera_max_non15"


@torch.no_grad()
def collect_initial_camera_depths(cameras, render_camera, progress=None):
    """Cache one STGS-style camera-z maximum per training camera at time zero.

    Use the raw pinhole depth before the Immersive warp: bilinear interpolation
    can mix the default depth 15 into otherwise valid pixels. Exactly 15 is a
    rasterizer sentinel, not a far-plane cap; valid depths above 15 are retained.
    Only CPU scalars/metadata survive each render, not depth maps or graphs.
    """
    cameras = sorted(list(cameras), key=lambda camera: str(camera.image_name))
    if not cameras:
        raise RuntimeError("Initial background depth requires time-zero training cameras")
    records = {}
    for camera in cameras:
        name = str(camera.image_name)
        if name in records:
            raise RuntimeError("Duplicate initial background camera: {}".format(name))
        if abs(float(camera.timestamp)) > 1e-8:
            raise RuntimeError("Initial background depth must be rendered at time zero: {}".format(name))
        package = render_camera(camera)
        use_raw_depth = torch.is_tensor(package.get("rawdepth"))
        if getattr(camera, "fisheyemapper", None) is not None and not use_raw_depth:
            raise RuntimeError("Initial fisheye background depth requires rawdepth: {}".format(name))
        depth = package["rawdepth"] if use_raw_depth else package["depth"]
        depth = depth.detach()
        if tuple(depth.shape) not in (
            (int(camera.image_height), int(camera.image_width)),
            (1, int(camera.image_height), int(camera.image_width)),
        ):
            raise RuntimeError("Initial background depth is not in the pinhole domain: {}".format(name))
        valid = torch.isfinite(depth) & (depth > 0.0) & (depth != 15.0)
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            raise RuntimeError(
                "No finite positive non-15 initial depth for camera {}; "
                "refusing to substitute a guessed background depth".format(name)
            )
        records[name] = {
            "base_camera_z": float(depth[valid].max().item()),
            "timestamp": float(camera.timestamp),
            "source_key": "rawdepth" if use_raw_depth else "depth",
            "depth_domain": "pinhole_before_fisheye_warp",
            "depth_shape": list(depth.shape),
            "total_pixels": int(depth.numel()),
            "valid_non15_pixels": valid_count,
            "default_15_pixels": int((depth == 15.0).sum().item()),
            "invalid_pixels": int((~torch.isfinite(depth) | (depth <= 0.0)).sum().item()),
        }
        del package, depth, valid
        if progress is not None:
            progress(len(records), len(cameras), records[name])
    return {
        "depth_base": INITIAL_CAMERA_DEPTH_BASE,
        "computed_at_iteration": 0,
        "time_index": 0,
        "camera_split": "train_only",
        "default_depth_excluded": 15.0,
        "validity": "finite_and_positive_and_exactly_not_15",
        "camera_count": len(records),
        "per_camera": records,
    }


def initial_camera_depth_base(reference_depth, camera_name, initial_depths):
    """Broadcast the cached scalar; never recompute it from a later render."""
    record = initial_depths.get("per_camera", {}).get(str(camera_name))
    if record is None:
        raise RuntimeError("Missing initial background depth for camera {}".format(camera_name))
    value = float(record["base_camera_z"])
    if not math.isfinite(value) or value <= 0.0 or value == 15.0:
        raise RuntimeError("Invalid initial background depth for camera {}".format(camera_name))
    return torch.full_like(reference_depth, value)


def _camera_mapper(camera, device, dtype):
    mapper = getattr(camera, "fisheyemapper", None)
    if mapper is None:
        return None
    mapper = mapper.to(device=device, dtype=dtype)
    if mapper.ndim == 3:
        mapper = mapper.unsqueeze(0)
    if mapper.ndim != 4 or mapper.shape[0] != 1 or mapper.shape[-1] != 2:
        raise ValueError("fisheyemapper must have shape [1, H, W, 2]")
    return mapper


def align_background_depth_like(depth_like, camera, target_shape):
    """Align a pinhole prior to supervision, returning a strict validity mask.

The fisheye branch follows the renderer's bilinear scalar warp and
align_corners=True resize. The ordinary perspective path retains its original
align_corners=False resize, including its numerical behavior.
"""
    if depth_like.ndim != 2:
        raise ValueError("Background depth-like input must be a 2-D array")
    height, width = (int(value) for value in target_shape)
    mapper = _camera_mapper(camera, depth_like.device, depth_like.dtype)
    if mapper is None:
        if depth_like.shape != (height, width):
            depth_like = F.interpolate(
                depth_like[None, None], size=(height, width),
                mode="bilinear", align_corners=False,
            )[0, 0]
        return depth_like, torch.isfinite(depth_like), {
            "input_domain": "pinhole", "output_domain": "pinhole",
            "alignment": "legacy_resize_align_corners_false",
        }

    expected = (int(camera.image_height), int(camera.image_width))
    if tuple(depth_like.shape) != expected:
        raise ValueError(
            "Immersive MiDaS input must use the full pinhole image size {}; "
            "got {}. Precompute it from colmap_<t>/images.".format(
                expected, tuple(depth_like.shape),
            )
        )
    finite_source = torch.isfinite(depth_like)
    finite_mapper = torch.isfinite(mapper).all(dim=-1)
    safe_mapper = torch.where(finite_mapper[..., None], mapper, torch.full_like(mapper, 2.0))
    warped = F.grid_sample(
        torch.where(finite_source, depth_like, torch.zeros_like(depth_like))[None, None],
        safe_mapper, mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    support = F.grid_sample(
        finite_source.to(depth_like.dtype)[None, None], safe_mapper,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    # Zero padding must not masquerade as a small (therefore far) MiDaS value.
    map_inside = finite_mapper & (mapper.abs() <= 1.0).all(dim=-1)
    support = support * map_inside[:, None].to(support.dtype)
    if warped.shape[-2:] != (height, width):
        warped = F.interpolate(warped, size=(height, width), mode="bilinear", align_corners=True)
        support = F.interpolate(support, size=(height, width), mode="bilinear", align_corners=True)
    valid = (support[0, 0] >= 1.0 - 1e-6) & torch.isfinite(warped[0, 0])
    aligned = torch.where(valid, warped[0, 0], torch.full_like(warped[0, 0], float("nan")))
    return aligned, valid, {
        "input_domain": "pinhole", "output_domain": "fisheye_supervision",
        "alignment": "renderer_bilinear_warp_then_resize_align_corners_true",
        "source_shape": list(expected), "target_shape": [height, width],
        "invalid_pixels": int((~valid).sum().item()),
    }


def background_pinhole_pixels(pixel_indices, camera, supervision_shape, dtype=torch.float32):
    """Return full pinhole xy coordinates for supervision row/column indices.

Mapper values are grid_sample coordinates, not rasterizer NDC coordinates.
The conversion through source pixel centers is required before unprojection.
"""
    height, width = (int(value) for value in supervision_shape)
    pixels = pixel_indices.to(dtype=dtype)
    mapper = _camera_mapper(camera, pixel_indices.device, dtype)
    valid = torch.isfinite(pixels).all(dim=1)
    valid = valid & (pixels[:, 0] >= 0) & (pixels[:, 0] < height)
    valid = valid & (pixels[:, 1] >= 0) & (pixels[:, 1] < width)
    if mapper is None:
        # Preserve the original perspective path, whose supervision/render
        # dimensions coincide. Do not apply the Immersive resize to N3D/Tech.
        return pixels[:, [1, 0]], valid
    if pixels.shape[0] == 0:
        return pixels[:, [1, 0]], valid

    query = torch.stack((
        2.0 * pixels[:, 1] / max(width - 1, 1) - 1.0,
        2.0 * pixels[:, 0] / max(height - 1, 1) - 1.0,
    ), dim=-1).view(1, -1, 1, 2)
    mapped = F.grid_sample(
        mapper.permute(0, 3, 1, 2), query,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )[0, :, :, 0].T
    valid = valid & torch.isfinite(mapped).all(dim=1) & (mapped.abs() <= 1.0).all(dim=1)
    source_size = pixels.new_tensor([int(camera.image_width) - 1, int(camera.image_height) - 1])
    return (mapped + 1.0) * 0.5 * source_size, valid


def backproject_background_pixels(pixel_indices, camera, depth_base, depth_value=None, depth_scale=None):
    """Unproject using camera-z depth already aligned to supervision pixels.

Invalid rows remain NaN, preserving index correspondence for later rejection
and auditing. No clipping or guessed replacement ray is applied here.
"""
    if depth_base.ndim == 3:
        depth_base = depth_base.squeeze(0)
    if depth_base.ndim != 2:
        raise ValueError("Background base depth must have shape [H, W] or [1, H, W]")
    pixel_indices = pixel_indices.to(device=depth_base.device)
    if pixel_indices.shape[0] == 0:
        return depth_base.new_empty((0, 3))
    xy, valid = background_pinhole_pixels(pixel_indices, camera, depth_base.shape, depth_base.dtype)
    rows = pixel_indices[:, 0].long().clamp(0, depth_base.shape[0] - 1)
    cols = pixel_indices[:, 1].long().clamp(0, depth_base.shape[1] - 1)
    if depth_value is not None and float(depth_value) > 0.0:
        target_depth = depth_base.new_full((pixel_indices.shape[0], 1), float(depth_value))
    else:
        scale = float(depth_scale) if depth_scale is not None and float(depth_scale) > 0.0 else 1.0
        target_depth = depth_base[rows, cols].view(-1, 1).clamp_min(1e-4) * scale
    valid = valid & torch.isfinite(target_depth[:, 0]) & (target_depth[:, 0] > 0)

    camera_to_world = camera.world_view_transform.T.inverse().to(depth_base)
    projection_inverse = camera.projection_matrix.T.inverse().to(depth_base)
    ndcx = ((xy[:, 0] * 2.0 + 1.0) / camera.image_width - 1.0).unsqueeze(1)
    ndcy = ((xy[:, 1] * 2.0 + 1.0) / camera.image_height - 1.0).unsqueeze(1)
    ndc = torch.cat((ndcx, ndcy, torch.ones_like(ndcx), torch.ones_like(ndcx)), dim=1)
    local_h = ndc @ projection_inverse.T
    direction = local_h / local_h[:, 3:]
    denom = direction[:, 2:3]
    denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
    local = direction * (target_depth / denom)
    local[:, -1] = 1.0
    world_h = local @ camera_to_world.T
    xyz = (world_h / world_h[:, 3:])[:, :3]
    valid = valid & torch.isfinite(xyz).all(dim=1)
    return torch.where(valid[:, None], xyz, torch.full_like(xyz, float("nan")))


def finite_distribution(values):
    values = values.detach().reshape(-1)
    values = values[torch.isfinite(values)].float()
    if values.numel() == 0:
        return {"count": 0}
    quantiles = torch.quantile(values, values.new_tensor([0.0, 0.1, 0.5, 0.9, 1.0]))
    return dict(zip(("min", "p10", "p50", "p90", "max"), quantiles.cpu().tolist()), count=int(values.numel()))


def summarize_background_geometry(raw_xyz, clipped_xyz, camera, field=None):
    """Describe geometry before cell selection; counts do not assert surfaces."""
    finite = torch.isfinite(raw_xyz).all(dim=1)
    finite_after = torch.isfinite(clipped_xyz).all(dim=1)
    changed = finite & finite_after & (raw_xyz != clipped_xyz).any(dim=1)
    result = {
        "input_candidates": int(raw_xyz.shape[0]),
        "invalid_geometry_candidates": int((~finite_after).sum().item()),
        "clipped_candidates": int(changed.sum().item()),
        "grid_available": bool(field is not None and hasattr(field, "bbox_min")),
    }
    world_view = camera.world_view_transform.to(raw_xyz)
    for name, xyz in (("requested_camera_z", raw_xyz), ("after_clip_camera_z", clipped_xyz)):
        local = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1) @ world_view
        result[name] = finite_distribution(local[:, 2])
    if result["grid_available"]:
        lo = field.bbox_min.to(raw_xyz).view(1, 3)
        hi = field.bbox_max.to(raw_xyz).view(1, 3)
        before = finite & ((raw_xyz >= lo) & (raw_xyz <= hi)).all(dim=1)
        after = finite_after & ((clipped_xyz >= lo) & (clipped_xyz <= hi)).all(dim=1)
        result.update({
            "inside_bbox_before_clip": int(before.sum().item()),
            "outside_bbox_before_clip": int((finite & ~before).sum().item()),
            "inside_bbox_after_clip": int(after.sum().item()),
            "outside_bbox_after_clip": int((finite_after & ~after).sum().item()),
        })
    return result
