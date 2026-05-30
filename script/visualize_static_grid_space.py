import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


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
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    return xyz, data


def load_field_state(point_cloud_dir):
    pt_path = Path(point_cloud_dir) / "point_cloud.pt"
    if not pt_path.exists():
        return None
    return torch.load(str(pt_path), map_location="cpu")


def parse_resolution_string(spec):
    if not spec:
        return []
    resolutions = []
    for item in str(spec).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.lower().split("x")
        resolutions.append(tuple(int(v) for v in parts))
    return resolutions


def bbox_corners(bmin, bmax):
    x0, y0, z0 = bmin
    x1, y1, z1 = bmax
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )


BBOX_EDGES = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ],
    dtype=np.int32,
)


def unique_cameras(cameras):
    result = {}
    for cam in cameras:
        name = cam.get("img_name")
        if name not in result:
            result[name] = cam
    return [result[k] for k in sorted(result)]


def camera_frustum_points(cam, near, far):
    width = float(cam["width"])
    height = float(cam["height"])
    fx = float(cam["fx"])
    fy = float(cam["fy"])
    cx = width * 0.5
    cy = height * 0.5
    c = np.asarray(cam["position"], dtype=np.float32)
    rot = np.asarray(cam["rotation"], dtype=np.float32)
    corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    world = [c]
    for depth in (near, far):
        for u, v in corners:
            local = np.array([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], dtype=np.float32)
            world.append(c + rot @ local)
    return np.stack(world, axis=0)


def write_ply_points(path, points, colors=None):
    points = np.asarray(points, dtype=np.float32)
    if colors is None:
        colors = np.full((points.shape[0], 3), 180, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_ply_lines(path, points, edges, colors=None):
    points = np.asarray(points, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int32)
    if colors is None:
        colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {edges.shape[0]}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for a, b in edges:
            f.write(f"{int(a)} {int(b)}\n")


def build_frustum_lines(cameras, near, far):
    all_points = []
    all_edges = []
    for cam in cameras:
        base = len(all_points)
        pts = camera_frustum_points(cam, near, far)
        all_points.extend(pts)
        # 0=center, 1-4 near corners, 5-8 far corners
        local_edges = []
        local_edges.extend([[0, i] for i in range(5, 9)])
        local_edges.extend([[1, 2], [2, 3], [3, 4], [4, 1]])
        local_edges.extend([[5, 6], [6, 7], [7, 8], [8, 5]])
        local_edges.extend([[1, 5], [2, 6], [3, 7], [4, 8]])
        all_edges.extend([[base + a, base + b] for a, b in local_edges])
    return np.asarray(all_points, dtype=np.float32), np.asarray(all_edges, dtype=np.int32)


def sample_points(points, max_points, seed):
    if points.shape[0] <= max_points:
        return points, np.arange(points.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx], idx


def set_equal_axes(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55)
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_bbox_3d(ax, bmin, bmax, color, label):
    corners = bbox_corners(bmin, bmax)
    for i, (a, b) in enumerate(BBOX_EDGES):
        xs, ys, zs = corners[[a, b]].T
        ax.plot(xs, ys, zs, color=color, linewidth=1.5, label=label if i == 0 else None)


def draw_projection(ax, points, dims, grid_min, grid_max, camera_points, title):
    xdim, ydim = dims
    ax.scatter(points[:, xdim], points[:, ydim], s=0.1, c="#8a8a8a", alpha=0.25, rasterized=True)
    if camera_points.size > 0:
        ax.scatter(camera_points[:, xdim], camera_points[:, ydim], s=16, c="#1f77b4", label="cameras")
    rect_x = [grid_min[xdim], grid_max[xdim], grid_max[xdim], grid_min[xdim], grid_min[xdim]]
    rect_y = [grid_min[ydim], grid_min[ydim], grid_max[ydim], grid_max[ydim], grid_min[ydim]]
    ax.plot(rect_x, rect_y, c="red", linewidth=1.2, label="static grid bbox")
    ax.set_xlabel("xyz"[xdim])
    ax.set_ylabel("xyz"[ydim])
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")


def main():
    parser = argparse.ArgumentParser(description="Visualize STEGF Gaussian points and static Euler grid bbox.")
    parser.add_argument("--model_path", default="output/S1.1.x/S1.1/coffee_martini")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--max_points", type=int, default=80000)
    parser.add_argument("--frustum_near", type=float, default=0.7)
    parser.add_argument("--frustum_far", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    point_dir = model_path / "point_cloud" / f"iteration_{args.iteration}"
    ply_path = point_dir / "point_cloud.ply"
    camera_path = model_path / "cameras.json"
    output_dir = Path(args.output_dir) if args.output_dir else model_path / "static_grid_space_debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    xyz, ply_data = read_binary_ply_vertices(ply_path)
    field_state = load_field_state(point_dir)
    if field_state is None or "euler_field" not in field_state:
        grid_min = xyz.min(axis=0)
        grid_max = xyz.max(axis=0)
        field_config = {}
        resolutions = []
    else:
        euler_state = field_state["euler_field"]
        grid_min = euler_state["bbox_min"].reshape(-1, 3)[0].numpy().astype(np.float32)
        grid_max = euler_state["bbox_max"].reshape(-1, 3)[0].numpy().astype(np.float32)
        field_config = field_state.get("field_config", {})
        resolutions = parse_resolution_string(field_config.get("field_resolved_level_resolutions", ""))

    cameras = unique_cameras(json.load(open(camera_path)))
    camera_centers = np.asarray([cam["position"] for cam in cameras], dtype=np.float32)
    frustum_points, frustum_edges = build_frustum_lines(cameras, args.frustum_near, args.frustum_far)

    inside_grid = np.all((xyz >= grid_min) & (xyz <= grid_max), axis=1)
    sampled_xyz, sampled_idx = sample_points(xyz, args.max_points, args.seed)
    sampled_inside = inside_grid[sampled_idx]

    point_colors = np.zeros((sampled_xyz.shape[0], 3), dtype=np.uint8)
    point_colors[sampled_inside] = np.array([160, 160, 160], dtype=np.uint8)
    point_colors[~sampled_inside] = np.array([255, 40, 40], dtype=np.uint8)
    write_ply_points(output_dir / "gaussian_points_sample.ply", sampled_xyz, point_colors)
    write_ply_points(output_dir / "camera_centers.ply", camera_centers, np.full((len(camera_centers), 3), [30, 120, 255], dtype=np.uint8))

    grid_corners = bbox_corners(grid_min, grid_max)
    write_ply_lines(
        output_dir / "static_grid_bbox.ply",
        grid_corners,
        BBOX_EDGES,
        np.full((8, 3), [255, 0, 0], dtype=np.uint8),
    )
    write_ply_lines(
        output_dir / "camera_frustums.ply",
        frustum_points,
        frustum_edges,
        np.full((frustum_points.shape[0], 3), [30, 120, 255], dtype=np.uint8),
    )

    plot_points = sampled_xyz
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(plot_points[:, 0], plot_points[:, 1], plot_points[:, 2], s=0.1, c="#888888", alpha=0.15, rasterized=True)
    ax.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], s=18, c="#1f77b4", label="cameras")
    draw_bbox_3d(ax, grid_min, grid_max, "red", "static grid bbox")
    set_equal_axes(ax, np.concatenate([plot_points, grid_corners, camera_centers], axis=0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("S1.1 Gaussian Points and Static Grid BBox")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "space_3d.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    draw_projection(axes[0], plot_points, (0, 1), grid_min, grid_max, camera_centers, "XY projection")
    draw_projection(axes[1], plot_points, (0, 2), grid_min, grid_max, camera_centers, "XZ projection")
    draw_projection(axes[2], plot_points, (1, 2), grid_min, grid_max, camera_centers, "YZ projection")
    fig.tight_layout()
    fig.savefig(output_dir / "space_projections.png", dpi=220)
    plt.close(fig)

    stats = {
        "model_path": str(model_path),
        "iteration": int(args.iteration),
        "gaussian_count": int(xyz.shape[0]),
        "gaussian_bbox": {
            "min": xyz.min(axis=0).tolist(),
            "max": xyz.max(axis=0).tolist(),
            "span": (xyz.max(axis=0) - xyz.min(axis=0)).tolist(),
        },
        "static_grid_bbox": {
            "min": grid_min.tolist(),
            "max": grid_max.tolist(),
            "span": (grid_max - grid_min).tolist(),
        },
        "static_grid_resolutions": resolutions,
        "points_inside_static_grid_bbox": int(inside_grid.sum()),
        "points_outside_static_grid_bbox": int((~inside_grid).sum()),
        "points_inside_static_grid_bbox_ratio": float(inside_grid.mean()),
        "camera_count": int(len(cameras)),
        "camera_bbox": {
            "min": camera_centers.min(axis=0).tolist(),
            "max": camera_centers.max(axis=0).tolist(),
            "span": (camera_centers.max(axis=0) - camera_centers.min(axis=0)).tolist(),
        },
        "frustum_far": float(args.frustum_far),
        "frustum_bbox": {
            "min": frustum_points.min(axis=0).tolist(),
            "max": frustum_points.max(axis=0).tolist(),
            "span": (frustum_points.max(axis=0) - frustum_points.min(axis=0)).tolist(),
        },
        "field_config_subset": {
            key: field_config.get(key)
            for key in [
                "field_num_levels",
                "field_resolution_mode",
                "field_resolved_level_resolutions",
                "field_max_resolution",
                "field_min_cell_scale",
            ]
        },
    }
    with open(output_dir / "space_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[STEGF] Wrote static grid space debug to {output_dir}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
