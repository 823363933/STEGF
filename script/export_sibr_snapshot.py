import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np


PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

C0 = 0.28209479177387814


def read_binary_ply_vertices(path):
    properties = []
    vertex_count = None
    fmt = None
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header in {path}")
            text = line.decode("ascii").strip()
            if text.startswith("format "):
                fmt = text.split()[1]
            elif text.startswith("element vertex"):
                vertex_count = int(text.split()[2])
            elif text.startswith("property ") and vertex_count is not None:
                parts = text.split()
                if len(parts) == 3:
                    properties.append((parts[2], PLY_TYPES[parts[1]]))
            elif text == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"Only binary_little_endian PLY is supported, got {fmt}")
        if vertex_count is None:
            raise ValueError(f"No vertex element in {path}")
        dtype = np.dtype([(name, "<" + typ) for name, typ in properties])
        data = np.fromfile(f, dtype=dtype, count=vertex_count)
    return data


def stack_props(data, prefix, count):
    return np.stack([data[f"{prefix}_{i}"] for i in range(count)], axis=1).astype(np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def inverse_sigmoid(x):
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x)).astype(np.float32)


def normalize_quaternion(q):
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    normalized = q / np.maximum(norm, 1e-8)
    invalid = (~np.isfinite(normalized).all(axis=1)) | (norm.reshape(-1) < 1e-8)
    if np.any(invalid):
        normalized[invalid] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return normalized.astype(np.float32)


def rgb_to_sh(rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    return ((rgb - 0.5) / C0).astype(np.float32)


def make_scene_sh_dc(f_dc, mode):
    mode = str(mode).lower()
    if mode == "original":
        return rgb_to_sh(f_dc[:, 0:3])
    if mode == "white":
        rgb = np.ones((f_dc.shape[0], 3), dtype=np.float32)
        return rgb_to_sh(rgb)
    if mode == "opacity":
        # Filled later by caller if a more useful scalar is available.
        rgb = np.full((f_dc.shape[0], 3), 0.65, dtype=np.float32)
        return rgb_to_sh(rgb)
    if mode == "gray":
        rgb = np.full((f_dc.shape[0], 3), 0.72, dtype=np.float32)
        return rgb_to_sh(rgb)
    raise ValueError(f"Unsupported scene color mode: {mode}")


def write_sibr_ply(path, xyz, sh_dc, opacity_raw, scaling_raw, rotation, sh_degree=3):
    count = xyz.shape[0]
    rest_count = (sh_degree + 1) ** 2 * 3 - 3
    dtype_fields = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("f_dc_0", "<f4"),
        ("f_dc_1", "<f4"),
        ("f_dc_2", "<f4"),
    ]
    dtype_fields.extend((f"f_rest_{i}", "<f4") for i in range(rest_count))
    dtype_fields.extend(
        [
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )
    out = np.zeros(count, dtype=np.dtype(dtype_fields))
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["f_dc_0"], out["f_dc_1"], out["f_dc_2"] = sh_dc[:, 0], sh_dc[:, 1], sh_dc[:, 2]
    out["opacity"] = opacity_raw.reshape(-1)
    for i in range(3):
        out[f"scale_{i}"] = scaling_raw[:, i]
    for i in range(4):
        out[f"rot_{i}"] = rotation[:, i]

    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {count}\n".encode("ascii"))
        for name, _ in dtype_fields:
            f.write(f"property float {name}\n".encode("ascii"))
        f.write(b"end_header\n")
        out.tofile(f)


def parse_resolution_string(spec):
    if not spec:
        return []
    resolutions = []
    for item in str(spec).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.lower().split("x")
        if len(parts) != 3:
            raise ValueError(f"Invalid grid resolution entry: {item}")
        resolutions.append(tuple(int(v) for v in parts))
    return resolutions


def load_static_grid_spec(model_path, point_dir, grid_level):
    import torch

    pt_path = Path(point_dir) / "point_cloud.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing field checkpoint for static grid overlay: {pt_path}")
    payload = torch.load(str(pt_path), map_location="cpu")
    euler_state = payload.get("euler_field") or {}
    field_config = payload.get("field_config") or {}
    bbox_min = euler_state.get("bbox_min")
    bbox_max = euler_state.get("bbox_max")
    if bbox_min is None or bbox_max is None:
        raise ValueError("point_cloud.pt does not contain euler_field bbox_min/bbox_max")
    bbox_min = bbox_min.detach().cpu().numpy().astype(np.float32).reshape(-1)[:3]
    bbox_max = bbox_max.detach().cpu().numpy().astype(np.float32).reshape(-1)[:3]

    resolutions = parse_resolution_string(field_config.get("field_resolved_level_resolutions", ""))
    if not resolutions:
        idx = 0
        while f"static_grids.{idx}" in euler_state:
            grid = euler_state[f"static_grids.{idx}"]
            # Stored as C x X x Y x Z.
            resolutions.append(tuple(int(v) for v in grid.shape[-3:]))
            idx += 1
    if not resolutions:
        raise ValueError("Could not resolve static grid resolutions from checkpoint")
    if grid_level < 0:
        grid_level = len(resolutions) + grid_level
    if grid_level < 0 or grid_level >= len(resolutions):
        raise ValueError(f"Invalid grid level {grid_level}; available levels: 0..{len(resolutions) - 1}")
    return bbox_min, bbox_max, resolutions[grid_level], resolutions


def sample_axis_line(start, end, step):
    length = float(np.linalg.norm(end - start))
    n = max(int(np.ceil(length / max(step, 1e-6))) + 1, 2)
    alpha = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    return start[None, :] * (1.0 - alpha) + end[None, :] * alpha


def make_static_grid_gaussians(bbox_min, bbox_max, resolution, step_scale, line_scale, opacity, color_rgb, max_points):
    nx, ny, nz = resolution
    xs = np.linspace(bbox_min[0], bbox_max[0], nx + 1, dtype=np.float32)
    ys = np.linspace(bbox_min[1], bbox_max[1], ny + 1, dtype=np.float32)
    zs = np.linspace(bbox_min[2], bbox_max[2], nz + 1, dtype=np.float32)
    cell = (bbox_max - bbox_min) / np.asarray([nx, ny, nz], dtype=np.float32)
    min_cell = float(np.min(np.abs(cell)))
    step = max(min_cell * step_scale, 1e-4)

    chunks = []
    for y in ys:
        for z in zs:
            chunks.append(sample_axis_line(np.array([xs[0], y, z], dtype=np.float32), np.array([xs[-1], y, z], dtype=np.float32), step))
    for x in xs:
        for z in zs:
            chunks.append(sample_axis_line(np.array([x, ys[0], z], dtype=np.float32), np.array([x, ys[-1], z], dtype=np.float32), step))
    for x in xs:
        for y in ys:
            chunks.append(sample_axis_line(np.array([x, y, zs[0]], dtype=np.float32), np.array([x, y, zs[-1]], dtype=np.float32), step))
    points = np.concatenate(chunks, axis=0)
    if max_points > 0 and points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        points = points[idx]

    sh_dc = np.repeat(rgb_to_sh(np.asarray(color_rgb, dtype=np.float32)[None, :]), points.shape[0], axis=0)
    opacity_raw = np.full((points.shape[0], 1), inverse_sigmoid(np.asarray([[opacity]], dtype=np.float32))[0, 0], dtype=np.float32)
    scale = np.full((points.shape[0], 3), np.log(max(min_cell * line_scale, 1e-5)), dtype=np.float32)
    rotation = np.zeros((points.shape[0], 4), dtype=np.float32)
    rotation[:, 0] = 1.0
    return points.astype(np.float32), sh_dc, opacity_raw, scale, rotation


def adjust_scene_scaling_raw(scaling_raw, multiplier=1.0, min_scale=0.0, max_scale=0.0):
    adjusted = scaling_raw.astype(np.float32, copy=True)
    multiplier = float(multiplier)
    if multiplier > 0.0 and multiplier != 1.0:
        adjusted = adjusted + np.float32(np.log(multiplier))
    if float(min_scale) > 0.0:
        adjusted = np.maximum(adjusted, np.float32(np.log(float(min_scale))))
    if float(max_scale) > 0.0:
        adjusted = np.minimum(adjusted, np.float32(np.log(float(max_scale))))
    return adjusted.astype(np.float32)


def patch_cfg_args(src_path, dst_path, source_path=None, sh_degree=3):
    text = Path(src_path).read_text()
    text = re.sub(r"sh_degree=\d+", f"sh_degree={sh_degree}", text)
    if source_path:
        source_path = str(Path(source_path).expanduser().resolve())
        text = re.sub(r"source_path='[^']*'", f"source_path='{source_path}'", text)
    Path(dst_path).write_text(text)


def write_launch_script(output_path, args, focal_point):
    viewer_bin = "${SIBR_VIEWER_BIN:-SIBR_gaussianViewer_app}"
    source_path = str(Path(args.source_path).expanduser().resolve()) if args.source_path else ""
    source_line = f'SOURCE_PATH="${{SIBR_SOURCE_PATH:-{source_path}}}"' if source_path else 'SOURCE_PATH="${SIBR_SOURCE_PATH:-}"'
    source_args = '"${SOURCE_ARGS[@]}"'
    script = f"""#!/usr/bin/env bash
set -euo pipefail

VIEWER_BIN="{viewer_bin}"
MODEL_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
{source_line}
SOURCE_ARGS=()
if [[ -n "$SOURCE_PATH" ]]; then
  SOURCE_ARGS=(-s "$SOURCE_PATH")
fi

exec "$VIEWER_BIN" \\
  -m "$MODEL_DIR" \\
  {source_args} \\
  --iteration {int(args.iteration)} \\
  --focal-pt {float(focal_point[0]):.6f} {float(focal_point[1]):.6f} {float(focal_point[2]):.6f} \\
  "$@"
"""
    script_path = Path(output_path) / "launch_sibr_diagnostic.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)


def resolve_timestamp(args):
    if args.timestamp is not None:
        return float(args.timestamp)
    duration = max(int(args.duration), 1)
    if duration == 1:
        return 0.0
    frame_index = max(0, min(int(args.frame_index), duration - 1))
    return float(frame_index) / float(duration - 1)


def export_snapshot(args):
    model_path = Path(args.model_path).expanduser().resolve()
    input_ply = model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not input_ply.exists():
        raise FileNotFoundError(input_ply)

    data = read_binary_ply_vertices(input_ply)
    xyz = stack_props(data, "", 3) if False else np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    motion = stack_props(data, "motion", 9)
    f_dc = stack_props(data, "f_dc", 6)
    opacity_raw_base = np.asarray(data["opacity"], dtype=np.float32).reshape(-1, 1)
    scaling_raw = stack_props(data, "scale", 3)
    rotation_raw = stack_props(data, "rot", 4)
    omega = stack_props(data, "omega", 4)
    trbf_center = np.asarray(data["trbf_center"], dtype=np.float32).reshape(-1, 1)
    trbf_scale = np.asarray(data["trbf_scale"], dtype=np.float32).reshape(-1, 1)

    t = np.float32(resolve_timestamp(args))
    dt = t - trbf_center
    xyz_t = xyz + motion[:, 0:3] * dt + motion[:, 3:6] * dt * dt + motion[:, 6:9] * dt * dt * dt
    rotation_t = normalize_quaternion(rotation_raw + dt * omega)
    trbf = np.exp(-np.square(dt / np.exp(trbf_scale)))
    opacity_t = sigmoid(opacity_raw_base) * trbf
    opacity_t = np.clip(opacity_t * float(args.scene_opacity_scale), 1e-6, float(args.scene_opacity_cap))
    if float(args.scene_ball_opacity) > 0.0:
        opacity_t = np.full_like(opacity_t, np.clip(float(args.scene_ball_opacity), 1e-6, 1.0 - 1e-6))
    opacity_raw_t = inverse_sigmoid(opacity_t)
    scene_scale_mode = "ellipsoid"
    if float(args.scene_ball_radius) > 0.0:
        ball_radius = max(float(args.scene_ball_radius), 1e-6)
        scaling_raw = np.full_like(scaling_raw, np.log(ball_radius), dtype=np.float32)
        scene_scale_mode = "ball"
    else:
        scaling_raw = adjust_scene_scaling_raw(
            scaling_raw,
            multiplier=args.scene_scale_multiplier,
            min_scale=args.scene_scale_min,
            max_scale=args.scene_scale_max,
        )

    # SIBR sees only SH DC plus zero SH residuals here:
    # - f_dc[0:3] is used as approximate RGB by default;
    # - f_rest_* is written as zero, so view-dependent color is disabled;
    # - f_t_* is not exported, so time-dependent color is disabled.
    sh_dc = make_scene_sh_dc(f_dc, args.scene_color_mode)
    if args.scene_color_mode == "opacity":
        scalar = np.clip(opacity_t.reshape(-1, 1) / max(float(args.scene_opacity_cap), 1e-6), 0.0, 1.0)
        rgb = np.repeat(0.15 + 0.75 * scalar, 3, axis=1).astype(np.float32)
        sh_dc = rgb_to_sh(rgb)
    overlay_count = 0
    grid_meta = None
    if args.include_static_grid:
        pt_dir = model_path / "point_cloud" / f"iteration_{args.iteration}"
        bbox_min, bbox_max, resolution, all_resolutions = load_static_grid_spec(model_path, pt_dir, args.grid_level)
        grid_xyz, grid_sh, grid_opacity, grid_scale, grid_rotation = make_static_grid_gaussians(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            resolution=resolution,
            step_scale=args.grid_step_scale,
            line_scale=args.grid_line_scale,
            opacity=args.grid_opacity,
            color_rgb=args.grid_color,
            max_points=args.grid_max_points,
        )
        xyz_t = np.concatenate([xyz_t, grid_xyz], axis=0)
        sh_dc = np.concatenate([sh_dc, grid_sh], axis=0)
        opacity_raw_t = np.concatenate([opacity_raw_t, grid_opacity], axis=0)
        scaling_raw = np.concatenate([scaling_raw, grid_scale], axis=0)
        rotation_t = np.concatenate([rotation_t, grid_rotation], axis=0)
        overlay_count = int(grid_xyz.shape[0])
        grid_meta = {
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "bbox_center": (0.5 * (bbox_min + bbox_max)).tolist(),
            "grid_resolution": list(resolution),
            "all_grid_resolutions": [list(v) for v in all_resolutions],
            "grid_overlay_gaussians": overlay_count,
        }

    if args.output_path is None:
        frame_tag = f"f{int(args.frame_index):03d}" if args.timestamp is None else f"t{float(t):.4f}".replace(".", "p")
        grid_tag = f"_gridL{args.grid_level}" if args.include_static_grid else ""
        output_path = model_path.parent / f"{model_path.name}_sibr_diag_{frame_tag}{grid_tag}"
    else:
        output_path = Path(args.output_path).expanduser().resolve()
    point_dir = output_path / "point_cloud" / f"iteration_{args.iteration}"
    point_dir.mkdir(parents=True, exist_ok=True)
    write_sibr_ply(point_dir / "point_cloud.ply", xyz_t, sh_dc, opacity_raw_t, scaling_raw, rotation_t, sh_degree=3)

    cfg_src = model_path / "cfg_args"
    if cfg_src.exists():
        patch_cfg_args(cfg_src, output_path / "cfg_args", source_path=args.source_path, sh_degree=3)

    for name in ("cameras.json", "input.ply"):
        src = model_path / name
        if src.exists():
            shutil.copy2(src, output_path / name)

    meta = {
        "purpose": "SIBR geometry/grid diagnostic export. Uses f_dc[0:3] as approximate RGB by default.",
        "source_model_path": str(model_path),
        "output_path": str(output_path),
        "iteration": int(args.iteration),
        "frame_index": int(args.frame_index),
        "duration": int(args.duration),
        "timestamp": float(t),
        "scene_color_mode": str(args.scene_color_mode),
        "view_dependent_color": "disabled: exported f_rest_* are zero",
        "time_dependent_color": "disabled: source f_t_* is ignored",
        "position": "motion-polynomial evaluated at timestamp",
        "opacity": "base opacity sigmoid multiplied by temporal RBF kernel at timestamp",
        "scene_opacity_scale": float(args.scene_opacity_scale),
        "scene_opacity_cap": float(args.scene_opacity_cap),
        "scene_ball_radius": float(args.scene_ball_radius),
        "scene_ball_opacity": float(args.scene_ball_opacity),
        "scene_scale_multiplier": float(args.scene_scale_multiplier),
        "scene_scale_min": float(args.scene_scale_min),
        "scene_scale_max": float(args.scene_scale_max),
        "scene_scale_mode": scene_scale_mode,
        "scene_gaussians": int(xyz.shape[0]),
        "total_exported_gaussians": int(xyz_t.shape[0]),
        "static_grid": grid_meta,
    }
    with open(output_path / "sibr_diagnostic_meta.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    if grid_meta is not None:
        write_launch_script(output_path, args, np.asarray(grid_meta["bbox_center"], dtype=np.float32))

    print(f"Exported SIBR-compatible approximate snapshot to: {output_path}")
    print(f"Frame index: {args.frame_index}")
    print(f"Timestamp: {float(t):.6f}")
    print(f"Scene Gaussians: {xyz.shape[0]}")
    if overlay_count:
        print(f"Static grid overlay Gaussians: {overlay_count}")
        print(f"Total exported Gaussians: {xyz_t.shape[0]}")
    print("Use SIBR with:")
    if grid_meta is not None:
        cx, cy, cz = grid_meta["bbox_center"]
        print(f"  SIBR_gaussianViewer_app -m {output_path} --iteration {args.iteration} --focal-pt {cx:.6f} {cy:.6f} {cz:.6f}")
        print(f"or run:")
        print(f"  {output_path / 'launch_sibr_diagnostic.sh'}")
    else:
        print(f"  SIBR_gaussianViewer_app -m {output_path} --iteration {args.iteration}")
    if args.source_path:
        print(f"or explicitly:")
        if grid_meta is not None:
            cx, cy, cz = grid_meta["bbox_center"]
            print(f"  SIBR_gaussianViewer_app -m {output_path} -s {Path(args.source_path).expanduser().resolve()} --iteration {args.iteration} --focal-pt {cx:.6f} {cy:.6f} {cz:.6f}")
        else:
            print(f"  SIBR_gaussianViewer_app -m {output_path} -s {Path(args.source_path).expanduser().resolve()} --iteration {args.iteration}")


def build_arg_parser(description="Export a first-frame STEGF/STGS geometry snapshot for SIBR diagnostics."):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model_path", required=True, help="STEGF model directory.")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--frame_index", type=int, default=0, help="Frame index to freeze. Default 0 means first frame.")
    parser.add_argument("--duration", type=int, default=50, help="Number of normalized frames used to convert frame_index to timestamp.")
    parser.add_argument("--timestamp", type=float, default=None, help="Optional normalized timestamp override. If omitted, frame_index/(duration-1) is used.")
    parser.add_argument("--output_path", default=None, help="Output model directory for SIBR viewer. Defaults next to model_path.")
    parser.add_argument("--source_path", default=None, help="Optional local dataset path to write into cfg_args.")
    parser.add_argument("--scene_color_mode", choices=["original", "gray", "white", "opacity"], default="original", help="Color mode. Default uses f_dc[0:3] as approximate RGB; SH residuals and f_t are zeroed.")
    parser.add_argument("--scene_opacity_scale", type=float, default=1.0, help="Optional extra opacity scale after temporal RBF. Default preserves time-modulated opacity.")
    parser.add_argument("--scene_opacity_cap", type=float, default=1.0, help="Opacity cap after scene_opacity_scale.")
    parser.add_argument("--scene_ball_radius", type=float, default=0.0, help="If >0, export every scene Gaussian as an isotropic ball with this radius.")
    parser.add_argument("--scene_ball_opacity", type=float, default=0.0, help="If >0, override exported scene Gaussian opacity for ball diagnostics.")
    parser.add_argument("--scene_scale_multiplier", type=float, default=1.0, help="Multiplier applied to original anisotropic Gaussian scales when not using --scene_ball_radius. Use 3.0 to visualize an approximate 3-sigma effective support ellipsoid.")
    parser.add_argument("--scene_scale_min", type=float, default=0.0, help="Optional minimum exported anisotropic scale in world units. 0 disables clamp.")
    parser.add_argument("--scene_scale_max", type=float, default=0.0, help="Optional maximum exported anisotropic scale in world units. 0 disables clamp.")
    parser.add_argument("--include_static_grid", action="store_true", default=True, help="Overlay static grid as colored Gaussian line samples.")
    parser.add_argument("--no_static_grid", dest="include_static_grid", action="store_false", help="Disable static grid overlay.")
    parser.add_argument("--grid_level", type=int, default=2, help="Static grid level to overlay. Use 1 for human-counted second level; default code-level L2.")
    parser.add_argument("--grid_opacity", type=float, default=0.75)
    parser.add_argument("--grid_line_scale", type=float, default=0.06, help="Line Gaussian scale as a fraction of the smallest grid cell size.")
    parser.add_argument("--grid_step_scale", type=float, default=0.35, help="Sampling step as a fraction of the smallest grid cell size.")
    parser.add_argument("--grid_max_points", type=int, default=250000, help="Downsample grid overlay if it exceeds this many Gaussians. 0 disables cap.")
    parser.add_argument("--grid_color", type=float, nargs=3, default=(0.0, 1.0, 0.15), help="RGB color for static grid overlay.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    export_snapshot(args)


if __name__ == "__main__":
    main()
