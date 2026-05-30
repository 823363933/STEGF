import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GAUSSIAN_SPLATTING_ROOT = REPO_ROOT / "thirdparty" / "gaussian_splatting"
if str(GAUSSIAN_SPLATTING_ROOT) not in sys.path:
    sys.path.insert(0, str(GAUSSIAN_SPLATTING_ROOT))

from export_sibr_snapshot import (
    inverse_sigmoid,
    load_static_grid_spec,
    make_static_grid_gaussians,
    rgb_to_sh,
    write_launch_script,
    write_sibr_ply,
)


def load_colmap_point_readers():
    loader_path = GAUSSIAN_SPLATTING_ROOT / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("stegf_colmap_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_points3D_binary, module.read_points3D_text


read_points3D_binary, read_points3D_text = load_colmap_point_readers()


def read_ascii_or_binary_init_points(source_path, duration, use_total_ply):
    source_path = Path(source_path).expanduser().resolve()
    if use_total_ply:
        total_ply = source_path / "sparse" / "0" / f"points3D_total{int(duration)}.ply"
        if total_ply.exists():
            data = read_ply_xyz_rgb(total_ply)
            return data["xyz"], data["rgb"], str(total_ply)

    bin_path = source_path / "sparse" / "0" / "points3D.bin"
    txt_path = source_path / "sparse" / "0" / "points3D.txt"
    if bin_path.exists():
        xyz, rgb, _ = read_points3D_binary(str(bin_path))
        return xyz.astype(np.float32), rgb.astype(np.float32), str(bin_path)
    if txt_path.exists():
        xyz, rgb, _ = read_points3D_text(str(txt_path))
        return xyz.astype(np.float32), rgb.astype(np.float32), str(txt_path)
    raise FileNotFoundError(f"No COLMAP points3D found in {source_path / 'sparse' / '0'}")


def read_ply_xyz_rgb(path):
    with open(path, "rb") as f:
        header = []
        property_names = []
        property_types = []
        vertex_count = None
        fmt = None
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            text = line.decode("ascii").strip()
            header.append(text)
            if text.startswith("format "):
                fmt = text.split()[1]
            elif text.startswith("element vertex"):
                vertex_count = int(text.split()[2])
                in_vertex = True
            elif text.startswith("element "):
                in_vertex = False
            elif in_vertex and text.startswith("property "):
                parts = text.split()
                if len(parts) >= 3:
                    property_types.append(parts[1])
                    property_names.append(parts[2])
            elif text == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"Missing vertex count in {path}")
        for name in ("x", "y", "z"):
            if name not in property_names:
                raise ValueError(f"Missing {name} property in {path}")
        xyz_indices = [property_names.index("x"), property_names.index("y"), property_names.index("z")]
        rgb_indices = None
        if all(name in property_names for name in ("red", "green", "blue")):
            rgb_indices = [property_names.index("red"), property_names.index("green"), property_names.index("blue")]
        if fmt == "ascii":
            rows = []
            for _ in range(vertex_count):
                rows.append(f.readline().decode("ascii").strip().split())
            arr = np.asarray(rows, dtype=np.float32)
            xyz = arr[:, xyz_indices]
            if rgb_indices is not None:
                rgb = arr[:, rgb_indices]
            else:
                rgb = np.full((xyz.shape[0], 3), 180.0, dtype=np.float32)
            return {"xyz": xyz.astype(np.float32), "rgb": rgb.astype(np.float32)}
        if fmt == "binary_little_endian":
            type_map = {
                "char": "i1",
                "uchar": "u1",
                "int8": "i1",
                "uint8": "u1",
                "short": "<i2",
                "ushort": "<u2",
                "int16": "<i2",
                "uint16": "<u2",
                "int": "<i4",
                "uint": "<u4",
                "int32": "<i4",
                "uint32": "<u4",
                "float": "<f4",
                "float32": "<f4",
                "double": "<f8",
                "float64": "<f8",
            }
            dtype = np.dtype([(name, type_map[prop_type]) for prop_type, name in zip(property_types, property_names)])
            data = np.fromfile(f, dtype=dtype, count=vertex_count)
            xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
            if rgb_indices is not None:
                rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.float32)
            else:
                rgb = np.full((xyz.shape[0], 3), 180.0, dtype=np.float32)
            return {"xyz": xyz, "rgb": rgb}
    raise ValueError(f"Unsupported PLY format: {fmt}")


def robust_radius(xyz, scale):
    if xyz.shape[0] < 2:
        return float(scale)
    center = np.median(xyz, axis=0, keepdims=True)
    distances = np.linalg.norm(xyz - center, axis=1)
    extent = float(np.quantile(distances, 0.95))
    return max(extent * float(scale), 1e-4)


def _expand_bits_10bit(v):
    v = v.astype(np.uint32) & np.uint32(1023)
    v = (v | (v << np.uint32(16))) & np.uint32(0x030000FF)
    v = (v | (v << np.uint32(8))) & np.uint32(0x0300F00F)
    v = (v | (v << np.uint32(4))) & np.uint32(0x030C30C3)
    v = (v | (v << np.uint32(2))) & np.uint32(0x09249249)
    return v


def morton_codes_10bit(xyz):
    xyz = xyz.astype(np.float32, copy=False)
    bbox_min = np.min(xyz, axis=0)
    bbox_max = np.max(xyz, axis=0)
    span = np.maximum(bbox_max - bbox_min, 1e-6)
    q = np.clip(((xyz - bbox_min[None, :]) / span[None, :]) * 1023.0, 0.0, 1023.0).astype(np.uint32)
    return _expand_bits_10bit(q[:, 0]) | (_expand_bits_10bit(q[:, 1]) << np.uint32(1)) | (_expand_bits_10bit(q[:, 2]) << np.uint32(2))


def _update_three_best(best, rows, values):
    if rows.size == 0:
        return
    current = best[rows]
    combined = np.concatenate([current, values.astype(np.float32, copy=False)[:, None]], axis=1)
    combined.sort(axis=1)
    best[rows] = combined[:, :3]


def estimate_init_gaussian_scaling_raw(xyz, knn_window=64, multiplier=1.0, min_scale=0.0, max_scale=0.0):
    """Approximate the create_from_pcd distCUDA2 initialization on CPU.

    The training code initializes scales as:
      log(sqrt(mean squared distance to 3 nearest neighbors))).repeat(3)
    and clamps raw log-scales to [-10, 1].

    This fallback uses Morton-order local search. It is diagnostic-oriented:
    exact enough for initialization shape/boundary inspection without requiring
    the CUDA simple_knn extension in the local visualization environment.
    """
    xyz = xyz.astype(np.float32, copy=False)
    count = xyz.shape[0]
    if count == 0:
        return np.zeros((0, 3), dtype=np.float32), {"method": "empty", "knn_window": int(knn_window)}
    if count == 1:
        raw = np.zeros((1, 3), dtype=np.float32)
        return raw, {"method": "single_point", "knn_window": int(knn_window)}

    codes = morton_codes_10bit(xyz)
    order = np.argsort(codes, kind="mergesort")
    sorted_xyz = xyz[order]
    best_sorted = np.full((count, 3), np.inf, dtype=np.float32)
    max_offset = max(1, min(int(knn_window), count - 1))
    row_ids = np.arange(count, dtype=np.int64)

    for offset in range(1, max_offset + 1):
        diff = sorted_xyz[offset:] - sorted_xyz[:-offset]
        d2 = np.einsum("ij,ij->i", diff, diff, dtype=np.float32)
        _update_three_best(best_sorted, row_ids[:-offset], d2)
        _update_three_best(best_sorted, row_ids[offset:], d2)

    fallback = np.nanmedian(best_sorted[np.isfinite(best_sorted)])
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1e-7
    best_sorted[~np.isfinite(best_sorted)] = fallback

    best = np.empty_like(best_sorted)
    best[order] = best_sorted
    dist2 = np.maximum(np.mean(best, axis=1), 1e-7)
    scale = np.sqrt(dist2)
    multiplier = float(multiplier)
    if multiplier > 0.0 and multiplier != 1.0:
        scale = scale * multiplier
    if float(min_scale) > 0.0:
        scale = np.maximum(scale, float(min_scale))
    if float(max_scale) > 0.0:
        scale = np.minimum(scale, float(max_scale))
    raw = np.log(np.maximum(scale, 1e-7)).astype(np.float32)
    raw = np.clip(raw, -10.0, 1.0)
    meta = {
        "method": "morton_window_knn3_mean_dist2",
        "knn_window": int(max_offset),
        "scale_min": float(np.exp(np.min(raw))),
        "scale_median": float(np.exp(np.median(raw))),
        "scale_max": float(np.exp(np.max(raw))),
    }
    return np.repeat(raw[:, None], 3, axis=1).astype(np.float32), meta


def export_init_points(args):
    source_path = Path(args.source_path).expanduser().resolve()
    grid_model_path = Path(args.grid_model_path).expanduser().resolve()
    grid_point_dir = grid_model_path / "point_cloud" / f"iteration_{int(args.grid_iteration)}"
    output_path = Path(args.output_path).expanduser().resolve()
    point_dir = output_path / "point_cloud" / f"iteration_{int(args.iteration)}"
    point_dir.mkdir(parents=True, exist_ok=True)

    xyz, rgb, point_source = read_ascii_or_binary_init_points(source_path, args.duration, args.use_total_ply)
    if args.max_points > 0 and xyz.shape[0] > args.max_points:
        indices = np.linspace(0, xyz.shape[0] - 1, int(args.max_points)).astype(np.int64)
        xyz = xyz[indices]
        rgb = rgb[indices]

    rgb01 = np.clip(rgb / 255.0, 0.0, 1.0)
    if args.point_color_mode == "white":
        rgb01 = np.ones_like(rgb01)
    elif args.point_color_mode == "cyan":
        rgb01 = np.repeat(np.asarray([[0.0, 0.8, 1.0]], dtype=np.float32), xyz.shape[0], axis=0)
    sh_dc = rgb_to_sh(rgb01)
    opacity_raw = np.full((xyz.shape[0], 1), inverse_sigmoid(np.asarray([[args.point_opacity]], dtype=np.float32))[0, 0], dtype=np.float32)
    scale_meta = None
    radius = robust_radius(xyz, args.point_radius_scale) if args.point_radius <= 0 else float(args.point_radius)
    if args.point_shape == "init_gaussian":
        scaling_raw, scale_meta = estimate_init_gaussian_scaling_raw(
            xyz,
            knn_window=args.init_knn_window,
            multiplier=args.init_scale_multiplier,
            min_scale=args.init_scale_min,
            max_scale=args.init_scale_max,
        )
        radius = float(scale_meta.get("scale_median", radius))
    else:
        scaling_raw = np.full((xyz.shape[0], 3), np.log(max(radius, 1e-5)), dtype=np.float32)
    rotation = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    rotation[:, 0] = 1.0

    bbox_min, bbox_max, resolution, all_resolutions = load_static_grid_spec(grid_model_path, grid_point_dir, args.grid_level)
    draw_resolution = (1, 1, 1) if args.bbox_only else resolution
    grid_xyz, grid_sh, grid_opacity, grid_scale, grid_rotation = make_static_grid_gaussians(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        resolution=draw_resolution,
        step_scale=args.grid_step_scale,
        line_scale=args.grid_line_scale,
        opacity=args.grid_opacity,
        color_rgb=args.grid_color,
        max_points=args.grid_max_points,
    )

    out_xyz = np.concatenate([xyz.astype(np.float32), grid_xyz], axis=0)
    out_sh = np.concatenate([sh_dc.astype(np.float32), grid_sh], axis=0)
    out_opacity = np.concatenate([opacity_raw, grid_opacity], axis=0)
    out_scaling = np.concatenate([scaling_raw, grid_scale], axis=0)
    out_rotation = np.concatenate([rotation, grid_rotation], axis=0)

    write_sibr_ply(point_dir / "point_cloud.ply", out_xyz, out_sh, out_opacity, out_scaling, out_rotation, sh_degree=3)

    for name in ("cameras.json", "input.ply", "cfg_args"):
        src = grid_model_path / name
        if src.exists():
            shutil.copy2(src, output_path / name)

    bbox_center = 0.5 * (bbox_min + bbox_max)
    write_launch_script(output_path, args, bbox_center)

    meta = {
        "purpose": "SIBR diagnostic export for COLMAP/SfM initialization points plus static grid.",
        "source_path": str(source_path),
        "point_source": point_source,
        "grid_model_path": str(grid_model_path),
        "grid_iteration": int(args.grid_iteration),
        "iteration": int(args.iteration),
        "init_points": int(xyz.shape[0]),
        "point_shape": str(args.point_shape),
        "point_radius": float(radius),
        "scale_meta": scale_meta,
        "grid_points": int(grid_xyz.shape[0]),
        "total_exported_gaussians": int(out_xyz.shape[0]),
        "static_grid": {
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "bbox_center": bbox_center.tolist(),
            "bbox_only": bool(args.bbox_only),
            "grid_resolution": list(resolution),
            "draw_resolution": list(draw_resolution),
            "all_grid_resolutions": [list(v) for v in all_resolutions],
        },
    }
    with open(output_path / "sibr_init_points_meta.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(f"Exported init-point SIBR diagnostic to: {output_path}")
    print(f"Init points: {xyz.shape[0]}")
    if scale_meta is not None:
        print(
            "Initial Gaussian scale: min={:.6f}, median={:.6f}, max={:.6f}, method={}".format(
                scale_meta["scale_min"],
                scale_meta["scale_median"],
                scale_meta["scale_max"],
                scale_meta["method"],
            )
        )
    else:
        print(f"Point radius: {radius:.6f}")
    print(f"Static grid overlay Gaussians: {grid_xyz.shape[0]}")
    print(f"Run: {output_path / 'launch_sibr_diagnostic.sh'}")


def main():
    parser = argparse.ArgumentParser(description="Export COLMAP init points plus static grid for SIBR spatial diagnostics.")
    parser.add_argument("--source_path", required=True, help="COLMAP scene path, e.g. coffee_martini/colmap_0.")
    parser.add_argument("--grid_model_path", required=True, help="Trained STEGF model path used only to read static grid bbox/resolution and cameras.")
    parser.add_argument("--grid_iteration", type=int, default=30000)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration folder name expected by SIBR.")
    parser.add_argument("--duration", type=int, default=50)
    parser.add_argument("--use_total_ply", action="store_true", help="Prefer points3D_total{duration}.ply if present.")
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--point_radius", type=float, default=0.0, help="Explicit isotropic Gaussian radius. <=0 uses robust extent * point_radius_scale.")
    parser.add_argument("--point_radius_scale", type=float, default=0.0025)
    parser.add_argument("--point_opacity", type=float, default=0.8)
    parser.add_argument("--point_color_mode", choices=["rgb", "white", "cyan"], default="rgb")
    parser.add_argument("--point_shape", choices=["ball", "init_gaussian"], default="ball", help="ball uses a fixed diagnostic radius; init_gaussian approximates create_from_pcd initial scales.")
    parser.add_argument("--init_knn_window", type=int, default=64, help="Morton-order search window for CPU init scale approximation.")
    parser.add_argument("--init_scale_multiplier", type=float, default=1.0, help="Optional display multiplier for init Gaussian scales.")
    parser.add_argument("--init_scale_min", type=float, default=0.0, help="Optional minimum init Gaussian scale in world units. 0 disables clamp.")
    parser.add_argument("--init_scale_max", type=float, default=0.0, help="Optional maximum init Gaussian scale in world units. 0 disables clamp.")
    parser.add_argument("--grid_level", type=int, default=2)
    parser.add_argument("--bbox_only", action="store_true", help="Draw only the static-grid bounding box instead of all grid cell lines.")
    parser.add_argument("--grid_opacity", type=float, default=0.75)
    parser.add_argument("--grid_line_scale", type=float, default=0.06)
    parser.add_argument("--grid_step_scale", type=float, default=0.35)
    parser.add_argument("--grid_max_points", type=int, default=250000)
    parser.add_argument("--grid_color", type=float, nargs=3, default=(0.0, 1.0, 0.15))
    args = parser.parse_args()
    export_init_points(args)


if __name__ == "__main__":
    main()
