import json
import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
THIRDPARTY_ROOT = REPO_ROOT / "thirdparty" / "gaussian_splatting"
if str(THIRDPARTY_ROOT) not in sys.path:
    sys.path.append(str(THIRDPARTY_ROOT))

from helper_train import getmodel, getrenderpip, trbfunction
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from thirdparty.gaussian_splatting.scene import Scene
from thirdparty.gaussian_splatting.utils.general_utils import safe_state
from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix


BBOX_EDGES = (
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


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
            raise ValueError(f"Invalid static grid resolution entry: {item}")
        resolutions.append(tuple(int(v) for v in parts))
    return resolutions


def load_static_grid_spec(model_path, iteration, grid_level):
    point_dir = Path(model_path) / "point_cloud" / f"iteration_{iteration}"
    pt_path = point_dir / "point_cloud.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing static grid checkpoint: {pt_path}")

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
            resolutions.append(tuple(int(v) for v in grid.shape[-3:]))
            idx += 1
    if not resolutions:
        raise ValueError("Could not resolve static grid resolutions from checkpoint")

    if grid_level < 0:
        grid_level = len(resolutions) + grid_level
    if grid_level < 0 or grid_level >= len(resolutions):
        raise ValueError(f"Invalid grid level {grid_level}; available levels: 0..{len(resolutions) - 1}")

    return {
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "grid_level": int(grid_level),
        "grid_resolution": resolutions[grid_level],
        "all_grid_resolutions": resolutions,
        "point_cloud_dir": str(point_dir),
    }


def bbox_corners(bbox_min, bbox_max):
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    return np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x0, y1, z0],
            [x1, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x0, y1, z1],
            [x1, y1, z1],
        ],
        dtype=np.float32,
    )


def bbox_segments(bbox_min, bbox_max):
    corners = bbox_corners(bbox_min, bbox_max)
    return [(corners[i0], corners[i1], [int(i0), int(i1)]) for i0, i1 in BBOX_EDGES]


def static_grid_segments(bbox_min, bbox_max, resolution):
    bbox_min = np.asarray(bbox_min, dtype=np.float32)
    bbox_max = np.asarray(bbox_max, dtype=np.float32)
    nx, ny, nz = [int(v) for v in resolution]
    xs = np.linspace(bbox_min[0], bbox_max[0], nx + 1, dtype=np.float32)
    ys = np.linspace(bbox_min[1], bbox_max[1], ny + 1, dtype=np.float32)
    zs = np.linspace(bbox_min[2], bbox_max[2], nz + 1, dtype=np.float32)

    segments = []
    for y in ys:
        for z in zs:
            segments.append(
                (
                    np.asarray([bbox_min[0], y, z], dtype=np.float32),
                    np.asarray([bbox_max[0], y, z], dtype=np.float32),
                    None,
                )
            )
    for x in xs:
        for z in zs:
            segments.append(
                (
                    np.asarray([x, bbox_min[1], z], dtype=np.float32),
                    np.asarray([x, bbox_max[1], z], dtype=np.float32),
                    None,
                )
            )
    for x in xs:
        for y in ys:
            segments.append(
                (
                    np.asarray([x, y, bbox_min[2]], dtype=np.float32),
                    np.asarray([x, y, bbox_max[2]], dtype=np.float32),
                    None,
                )
            )
    return segments


def transform_points_to_camera(points, camera):
    points_t = torch.as_tensor(points, dtype=torch.float32, device="cuda")
    ones = torch.ones((points_t.shape[0], 1), dtype=points_t.dtype, device=points_t.device)
    camera_points = torch.cat([points_t, ones], dim=1) @ camera.world_view_transform
    return camera_points[:, :3].detach().cpu().numpy()


def transform_points_to_camera_np(points, camera):
    points = np.asarray(points, dtype=np.float32)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    matrix = camera.world_view_transform.detach().cpu().numpy().astype(np.float32)
    return (np.concatenate([points, ones], axis=1) @ matrix)[:, :3]


def project_camera_points(points, camera):
    points_t = torch.as_tensor(points, dtype=torch.float32, device="cuda")
    ones = torch.ones((points_t.shape[0], 1), dtype=points_t.dtype, device=points_t.device)
    clip = torch.cat([points_t, ones], dim=1) @ camera.projection_matrix
    w = clip[:, 3:4]
    ndc = clip[:, :3] / (w + 1e-7)
    finite = torch.isfinite(ndc).all(dim=1) & torch.isfinite(w[:, 0]) & (w[:, 0].abs() > 1e-7)
    x = (((ndc[:, 0] + 1.0) * float(camera.image_width)) - 1.0) * 0.5
    y = (((ndc[:, 1] + 1.0) * float(camera.image_height)) - 1.0) * 0.5
    return torch.stack([x, y], dim=1).detach().cpu().numpy(), finite.detach().cpu().numpy()


def project_camera_points_np(points, camera):
    points = np.asarray(points, dtype=np.float32)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    matrix = camera.projection_matrix.detach().cpu().numpy().astype(np.float32)
    clip = np.concatenate([points, ones], axis=1) @ matrix
    w = clip[:, 3:4]
    ndc = clip[:, :3] / (w + 1e-7)
    finite = np.isfinite(ndc).all(axis=1) & np.isfinite(w[:, 0]) & (np.abs(w[:, 0]) > 1e-7)
    x = (((ndc[:, 0] + 1.0) * float(camera.image_width)) - 1.0) * 0.5
    y = (((ndc[:, 1] + 1.0) * float(camera.image_height)) - 1.0) * 0.5
    return np.stack([x, y], axis=1), finite


def clip_segment_near(p0, p1, near):
    z0 = float(p0[2])
    z1 = float(p1[2])
    if z0 <= near and z1 <= near:
        return None
    if z0 <= near or z1 <= near:
        denom = z1 - z0
        if abs(denom) < 1e-8:
            return None
        t = (near - z0) / denom
        t = float(np.clip(t, 0.0, 1.0))
        clipped = p0 + t * (p1 - p0)
        if z0 <= near:
            p0 = clipped
        else:
            p1 = clipped
    return p0, p1


def tensor_to_uint8_image(image):
    image = torch.clamp(image.detach().float(), 0.0, 1.0)
    image = (image.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return image


def draw_world_segments(draw, camera, segments, color, thickness, halo, record_edges=False):
    width = int(camera.image_width)
    height = int(camera.image_height)
    drawn = 0
    projected_edges = []
    if len(segments) == 0:
        return drawn, projected_edges

    endpoints = np.asarray([[p0, p1] for p0, p1, _ in segments], dtype=np.float32)
    endpoints_cam = transform_points_to_camera_np(endpoints.reshape(-1, 3), camera).reshape(-1, 2, 3)

    for idx, (_, _, edge_id) in enumerate(segments):
        clipped = clip_segment_near(endpoints_cam[idx, 0].copy(), endpoints_cam[idx, 1].copy(), float(camera.znear))
        if clipped is None:
            continue
        projected, valid = project_camera_points_np(np.stack(clipped, axis=0), camera)
        if not bool(valid[0] and valid[1]):
            continue
        p0 = projected[0]
        p1 = projected[1]
        if (
            max(p0[0], p1[0]) < 0
            or min(p0[0], p1[0]) > width - 1
            or max(p0[1], p1[1]) < 0
            or min(p0[1], p1[1]) > height - 1
        ):
            continue
        if halo > 0:
            draw.line(
                [(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))],
                fill=(255, 255, 255),
                width=int(thickness + 2 * halo),
            )
        draw.line(
            [(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))],
            fill=tuple(int(v) for v in color),
            width=int(thickness),
        )
        drawn += 1
        if record_edges:
            projected_edges.append(
                {
                    "edge": edge_id,
                    "p0": [float(p0[0]), float(p0[1])],
                    "p1": [float(p1[0]), float(p1[1])],
                }
            )
    return drawn, projected_edges


def draw_bbox_overlay(rendering, camera, bbox_min, bbox_max, color, thickness, halo=1, grid_resolution=None, args=None):
    image = Image.fromarray(tensor_to_uint8_image(rendering))
    draw = ImageDraw.Draw(image)
    drawn_grid_lines = 0
    if args is not None and int(args.draw_grid_lines) != 0 and grid_resolution is not None:
        grid_segments = static_grid_segments(bbox_min, bbox_max, grid_resolution)
        drawn_grid_lines, _ = draw_world_segments(
            draw,
            camera,
            grid_segments,
            args.grid_cell_line_color,
            args.grid_cell_line_thickness,
            int(args.grid_cell_line_halo),
            record_edges=False,
        )

    drawn_edges, projected_edges = draw_world_segments(
        draw,
        camera,
        bbox_segments(bbox_min, bbox_max),
        color,
        thickness,
        halo,
        record_edges=True,
    )
    return image, {
        "drawn_edges": drawn_edges,
        "drawn_grid_lines": drawn_grid_lines,
        "projected_edges": projected_edges,
    }


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


def make_rays(world_view_transform, projection_matrix, camera_center, width, height):
    device = world_view_transform.device
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    ndc_x = (2.0 * xs + 1.0) / float(width) - 1.0
    ndc_y = (2.0 * ys + 1.0) / float(height) - 1.0
    ndc_camera = torch.stack(
        [ndc_x, ndc_y, torch.ones_like(ndc_x), torch.ones_like(ndc_x)],
        dim=-1,
    )

    project_inverse = projection_matrix.T.inverse()
    camera_to_world = world_view_transform.T.inverse()
    projected = ndc_camera @ project_inverse.T
    direction_local = projected / (projected[..., 3:] + 1e-7)
    rays_d = direction_local[..., :3] @ camera_to_world[:3, :3].T
    rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)
    rays_o = camera_center.view(1, 1, 3).expand_as(rays_d)
    return torch.cat(
        [
            rays_o.permute(2, 0, 1).unsqueeze(0),
            rays_d.permute(2, 0, 1).unsqueeze(0),
        ],
        dim=1,
    )


def make_overview_camera(anchor_camera, bbox_min, bbox_max, xyz_min, xyz_max, args):
    bbox_min = np.asarray(bbox_min, dtype=np.float32)
    bbox_max = np.asarray(bbox_max, dtype=np.float32)
    xyz_min = np.asarray(xyz_min, dtype=np.float32)
    xyz_max = np.asarray(xyz_max, dtype=np.float32)
    if int(args.overview_fit_gaussian_bounds) != 0:
        view_min = np.minimum(bbox_min, xyz_min)
        view_max = np.maximum(bbox_max, xyz_max)
    else:
        view_min = bbox_min
        view_max = bbox_max
    center = 0.5 * (view_min + view_max)
    radius = max(float(np.linalg.norm(view_max - view_min)) * 0.5, 1e-3)

    anchor_center = anchor_camera.camera_center.detach().cpu().numpy().astype(np.float32)
    camera_to_world = anchor_camera.world_view_transform.T.inverse().detach().cpu().numpy()
    up_hint = normalize(camera_to_world[:3, 1])
    front = normalize(anchor_center - center)
    if np.linalg.norm(front) < 1e-6:
        front = normalize(-camera_to_world[:3, 2])

    pitch = math.radians(float(args.overview_pitch_deg))
    width = int(args.overview_width)
    height = int(args.overview_height)
    fovy = math.radians(float(args.overview_fov_deg))
    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * (float(width) / float(height)))
    min_fov = max(min(fovx, fovy), math.radians(1.0))
    if int(args.overview_auto_fit) != 0:
        distance = radius / max(math.sin(min_fov * 0.5), 1e-3)
        distance *= float(args.overview_fit_padding)
    else:
        distance = radius * 2.0 * float(args.overview_distance_scale)

    overview_direction = normalize(front * math.cos(pitch) + up_hint * math.sin(pitch))
    eye = center + overview_direction * distance

    forward = normalize(center - eye)
    right = normalize(np.cross(up_hint, forward))
    if np.linalg.norm(right) < 1e-6:
        right = normalize(np.cross(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), forward))
    if np.linalg.norm(right) < 1e-6:
        right = normalize(np.cross(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), forward))
    up = normalize(np.cross(forward, right))

    camera_to_world_rot = np.stack([right, up, forward], axis=1)
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = camera_to_world_rot.T
    world_to_camera[:3, 3] = -camera_to_world_rot.T @ eye

    znear = 0.01
    zfar = max(100.0, distance + radius * 4.0)

    world_view_transform = torch.tensor(world_to_camera, dtype=torch.float32, device="cuda").transpose(0, 1)
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    rays = make_rays(world_view_transform, projection_matrix, camera_center, width, height)

    return SimpleNamespace(
        uid=-1,
        colmap_id=-1,
        image_name="overview_cam0_front_pitch45",
        image_width=width,
        image_height=height,
        FoVx=fovx,
        FoVy=fovy,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        projection_matrix=projection_matrix,
        full_proj_transform=full_proj_transform,
        camera_center=camera_center,
        timestamp=float(args.overview_timestamp),
        original_image=None,
        rays=rays,
        rayo=rays[:, :3],
        rayd=rays[:, 3:],
        fisheyemapper=None,
    )


def set_eval_mode(gaussians):
    if gaussians.rgbdecoder is not None:
        gaussians.rgbdecoder.cuda()
        gaussians.rgbdecoder.eval()
    if getattr(gaussians, "use_euler_field", False):
        if gaussians.euler_field is not None:
            gaussians.euler_field.cuda()
            gaussians.euler_field.eval()
        if getattr(gaussians, "field_router", None) is not None:
            gaussians.field_router.cuda()
            gaussians.field_router.eval()
        if getattr(gaussians, "field_query_gate", None) is not None:
            gaussians.field_query_gate.cuda()
            gaussians.field_query_gate.eval()
        if gaussians.field_decoder is not None:
            gaussians.field_decoder.cuda()
            gaussians.field_decoder.eval()


def save_image_and_overlay(rendering, camera, bbox_min, bbox_max, grid_resolution, render_path, overlay_path, args):
    torchvision.utils.save_image(torch.clamp(rendering, 0.0, 1.0), str(render_path))
    overlay, stats = draw_bbox_overlay(
        rendering,
        camera,
        bbox_min,
        bbox_max,
        args.grid_line_color,
        args.grid_line_thickness,
        halo=int(args.grid_line_halo),
        grid_resolution=grid_resolution,
        args=args,
    )
    overlay.save(str(overlay_path))
    return stats


def save_bbox_wireframe(camera, bbox_min, bbox_max, grid_resolution, path, args):
    background = torch.ones(3, int(camera.image_height), int(camera.image_width), dtype=torch.float32)
    overlay, stats = draw_bbox_overlay(
        background,
        camera,
        bbox_min,
        bbox_max,
        args.grid_line_color,
        args.grid_line_thickness,
        halo=0,
        grid_resolution=grid_resolution,
        args=args,
    )
    overlay.save(str(path))
    return stats


def render_with_grid(dataset, iteration, pipeline, multiview, args):
    with torch.no_grad():
        print("use model {}".format(dataset.model))
        gaussian_model = getmodel(dataset.model)
        gaussians = gaussian_model(dataset.sh_degree, args.rgbfunction)
        if hasattr(gaussians, "configure_euler_field"):
            gaussians.configure_euler_field(dataset)

        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            multiview=multiview,
            duration=args.duration,
            loader=args.valloader,
        )
        views = scene.getTestCameras()
        if len(views) == 0:
            raise RuntimeError("No test cameras found; use --valloader colmapvalid for the held-out view.")

        if gaussians.ts is None:
            height, width = views[0].image_height, views[0].image_width
            gaussians.ts = torch.ones(1, 1, height, width, device="cuda")

        set_eval_mode(gaussians)
        render_name = "test_ours_full" if args.rdpip == "train_ours_full" else args.rdpip
        render, gr_setting, gr_zero = getrenderpip(render_name)
        background = torch.zeros(9, dtype=torch.float32, device="cuda")

        grid_spec = load_static_grid_spec(dataset.model_path, scene.loaded_iter, args.grid_level)
        bbox_min = grid_spec["bbox_min"]
        bbox_max = grid_spec["bbox_max"]
        xyz = gaussians.get_xyz.detach()
        xyz_min = xyz.amin(dim=0).cpu().numpy()
        xyz_max = xyz.amax(dim=0).cpu().numpy()

        output_root = Path(dataset.model_path) / args.output_name / f"ours_{scene.loaded_iter}"
        render_dir = output_root / "renders"
        overlay_dir = output_root / "bbox_overlay"
        gt_dir = output_root / "gt"
        overview_dir = output_root / "overview"
        render_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        overview_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "iteration": int(scene.loaded_iter),
            "grid_level": grid_spec["grid_level"],
            "grid_resolution": list(grid_spec["grid_resolution"]),
            "draw_grid_lines": int(args.draw_grid_lines),
            "all_grid_resolutions": [list(v) for v in grid_spec["all_grid_resolutions"]],
            "static_grid_bbox_min": bbox_min.tolist(),
            "static_grid_bbox_max": bbox_max.tolist(),
            "gaussian_xyz_min": xyz_min.tolist(),
            "gaussian_xyz_max": xyz_max.tolist(),
            "output_root": str(output_root),
        }
        with open(output_root / "grid_bbox.json", "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        overlay_stats = []
        if not args.skip_test_views:
            for idx, view in enumerate(tqdm(views, desc="Rendering test views with BBOX")):
                render_pkg = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    scaling_modifier=1.0,
                    basicfunction=trbfunction,
                    GRsetting=gr_setting,
                    GRzer=gr_zero,
                )
                rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
                stem = f"{idx:05d}"
                stats = save_image_and_overlay(
                    rendering,
                    view,
                    bbox_min,
                    bbox_max,
                    grid_spec["grid_resolution"],
                    render_dir / f"{stem}.png",
                    overlay_dir / f"{stem}.png",
                    args,
                )
                stats.update(
                    {
                        "index": int(idx),
                        "image_name": str(getattr(view, "image_name", "")),
                        "timestamp": float(getattr(view, "timestamp", 0.0)),
                    }
                )
                overlay_stats.append(stats)
                if getattr(view, "original_image", None) is not None:
                    gt = view.original_image[0:3, :, :].cuda().float()
                    torchvision.utils.save_image(torch.clamp(gt, 0.0, 1.0), str(gt_dir / f"{stem}.png"))
                torch.cuda.empty_cache()
            with open(output_root / "bbox_overlay_stats.json", "w") as f:
                json.dump(overlay_stats, f, indent=2, sort_keys=True)

        if not args.skip_overview:
            overview_camera = make_overview_camera(views[0], bbox_min, bbox_max, xyz_min, xyz_max, args)
            render_pkg = render(
                overview_camera,
                gaussians,
                pipeline,
                background,
                scaling_modifier=float(args.overview_scale_modifier),
                basicfunction=trbfunction,
                GRsetting=gr_setting,
                GRzer=gr_zero,
            )
            overview_render = torch.clamp(render_pkg["render"], 0.0, 1.0)
            overview_stats = save_image_and_overlay(
                overview_render,
                overview_camera,
                bbox_min,
                bbox_max,
                grid_spec["grid_resolution"],
                overview_dir / "render.png",
                overview_dir / "bbox_overlay.png",
                args,
            )
            overview_meta = {
                "image_name": overview_camera.image_name,
                "width": int(overview_camera.image_width),
                "height": int(overview_camera.image_height),
                "fovx_deg": math.degrees(float(overview_camera.FoVx)),
                "fovy_deg": math.degrees(float(overview_camera.FoVy)),
                "timestamp": float(overview_camera.timestamp),
                "camera_center": overview_camera.camera_center.detach().cpu().numpy().tolist(),
                "pitch_deg": float(args.overview_pitch_deg),
                "auto_fit": int(args.overview_auto_fit),
                "fit_padding": float(args.overview_fit_padding),
                "fit_gaussian_bounds": int(args.overview_fit_gaussian_bounds),
                "distance_scale": float(args.overview_distance_scale),
                "scale_modifier": float(args.overview_scale_modifier),
                "bbox_overlay": overview_stats,
            }
            with open(overview_dir / "camera.json", "w") as f:
                json.dump(overview_meta, f, indent=2, sort_keys=True)
            wire_stats = save_bbox_wireframe(
                overview_camera,
                bbox_min,
                bbox_max,
                grid_spec["grid_resolution"],
                overview_dir / "bbox_wireframe_white.png",
                args,
            )
            with open(overview_dir / "bbox_wireframe_stats.json", "w") as f:
                json.dump(wire_stats, f, indent=2, sort_keys=True)

        print(f"[STEGF] Saved grid BBOX render outputs to {output_root}")


def get_grid_parse():
    parser = ArgumentParser(description="Render test views and a far overview camera with static-grid BBOX overlay.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--test_iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--duration", default=50, type=int)
    parser.add_argument("--rgbfunction", type=str, default="rgbv1")
    parser.add_argument("--rdpip", type=str, default="v3")
    parser.add_argument("--valloader", type=str, default="colmap")
    parser.add_argument("--configpath", type=str, default="1")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--output_name", type=str, default="test_grid")
    parser.add_argument("--grid_level", type=int, default=2)
    parser.add_argument("--draw_grid_lines", type=int, default=1)
    parser.add_argument("--grid_line_color", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--grid_line_thickness", type=int, default=2)
    parser.add_argument("--grid_line_halo", type=int, default=1)
    parser.add_argument("--grid_cell_line_color", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--grid_cell_line_thickness", type=int, default=1)
    parser.add_argument("--grid_cell_line_halo", type=int, default=1)
    parser.add_argument("--skip_test_views", action="store_true")
    parser.add_argument("--skip_overview", action="store_true")
    parser.add_argument("--overview_width", type=int, default=1280)
    parser.add_argument("--overview_height", type=int, default=720)
    parser.add_argument("--overview_fov_deg", type=float, default=55.0)
    parser.add_argument("--overview_pitch_deg", type=float, default=45.0)
    parser.add_argument("--overview_distance_scale", type=float, default=2.8)
    parser.add_argument("--overview_auto_fit", type=int, default=1)
    parser.add_argument("--overview_fit_padding", type=float, default=1.25)
    parser.add_argument("--overview_fit_gaussian_bounds", type=int, default=0)
    parser.add_argument("--overview_scale_modifier", type=float, default=3.0)
    parser.add_argument("--overview_timestamp", type=float, default=0.5)

    defaults = vars(parser.parse_args([]))
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    multiview = True if args.valloader.endswith("mv") else False

    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        with open(args.configpath) as f:
            config = json.load(f)
        for k, v in config.items():
            if hasattr(args, k) and getattr(args, k) == defaults.get(k):
                setattr(args, k, v)
            else:
                print(f"Keeping command line value for '{k}'")
        print("finish load config from " + args.configpath)
        print("args: " + str(args))

    if args.skip_test:
        args.skip_test_views = True

    return args, model.extract(args), pipeline.extract(args), multiview


if __name__ == "__main__":
    parsed_args, model_extract, pp_extract, parsed_multiview = get_grid_parse()
    render_with_grid(
        model_extract,
        parsed_args.test_iteration,
        pp_extract,
        parsed_multiview,
        parsed_args,
    )
