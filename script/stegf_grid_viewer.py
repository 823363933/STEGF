import io
import json
import math
import os
import sys
import traceback
from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_ROOT, REPO_ROOT / "thirdparty" / "gaussian_splatting"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from helper_train import getmodel, getrenderpip, trbfunction
from testgrid import draw_bbox_overlay, load_static_grid_spec, make_rays, normalize, tensor_to_uint8_image
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from thirdparty.gaussian_splatting.scene import Scene
from thirdparty.gaussian_splatting.utils.general_utils import safe_state
from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix


def parse_float(query, name, default):
    values = query.get(name)
    if not values:
        return float(default)
    try:
        return float(values[0])
    except ValueError:
        return float(default)


def parse_int(query, name, default):
    values = query.get(name)
    if not values:
        return int(default)
    try:
        return int(values[0])
    except ValueError:
        return int(default)


def rotate_axis(v, axis, angle_rad):
    v = np.asarray(v, dtype=np.float32)
    axis = normalize(axis)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


def make_lookat_camera(eye, target, up_hint, width, height, fovy_deg, timestamp):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up_hint = normalize(up_hint)
    forward = normalize(target - eye)
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

    fovy = math.radians(float(fovy_deg))
    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * (float(width) / float(height)))
    znear = 0.01
    zfar = 1000.0
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
        image_name="stegf_interactive_view",
        image_width=int(width),
        image_height=int(height),
        FoVx=fovx,
        FoVy=fovy,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        projection_matrix=projection_matrix,
        full_proj_transform=full_proj_transform,
        camera_center=camera_center,
        timestamp=float(timestamp),
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


class ViewerState:
    def __init__(self, dataset, pipeline, args, multiview):
        self.args = args
        gaussian_model = getmodel(dataset.model)
        self.gaussians = gaussian_model(dataset.sh_degree, args.rgbfunction)
        if hasattr(self.gaussians, "configure_euler_field"):
            self.gaussians.configure_euler_field(dataset)

        self.scene = Scene(
            dataset,
            self.gaussians,
            load_iteration=args.test_iteration,
            shuffle=False,
            multiview=multiview,
            duration=args.duration,
            loader=args.valloader,
        )
        self.views = self.scene.getTestCameras()
        if len(self.views) == 0:
            raise RuntimeError("No test cameras found. Use --valloader colmapvalid or another loader with test cameras.")

        if self.gaussians.ts is None:
            self.gaussians.ts = torch.ones(
                1,
                1,
                int(args.viewer_height),
                int(args.viewer_width),
                device="cuda",
            )

        set_eval_mode(self.gaussians)
        render_name = "test_ours_full" if args.rdpip == "train_ours_full" else args.rdpip
        self.render_fn, self.gr_setting, self.gr_zero = getrenderpip(render_name)
        self.pipeline = pipeline
        self.background = torch.zeros(9, dtype=torch.float32, device="cuda")

        self.grid_spec = load_static_grid_spec(dataset.model_path, self.scene.loaded_iter, int(args.grid_level))
        self.bbox_min = self.grid_spec["bbox_min"]
        self.bbox_max = self.grid_spec["bbox_max"]
        self.center = 0.5 * (self.bbox_min + self.bbox_max)
        self.radius = max(float(np.linalg.norm(self.bbox_max - self.bbox_min)) * 0.5, 1e-3)
        self.default_distance = self.radius / max(math.sin(math.radians(float(args.viewer_fov_deg)) * 0.5), 1e-3)
        self.default_distance *= float(args.viewer_fit_padding)

        anchor_idx = max(0, min(int(args.anchor_camera_index), len(self.views) - 1))
        anchor = self.views[anchor_idx]
        anchor_center = anchor.camera_center.detach().cpu().numpy().astype(np.float32)
        camera_to_world = anchor.world_view_transform.T.inverse().detach().cpu().numpy()
        self.up_hint = normalize(camera_to_world[:3, 1])
        self.base_direction = normalize(anchor_center - self.center)
        if np.linalg.norm(self.base_direction) < 1e-6:
            self.base_direction = normalize(-camera_to_world[:3, 2])

    def resolve_grid_level(self, query):
        level = parse_int(query, "grid_level", self.args.grid_level)
        n = len(self.grid_spec["all_grid_resolutions"])
        if level < 0:
            level = n + level
        level = max(0, min(level, n - 1))
        return level, self.grid_spec["all_grid_resolutions"][level]

    def make_camera_from_query(self, query):
        yaw = math.radians(parse_float(query, "yaw", 0.0))
        pitch = math.radians(parse_float(query, "pitch", self.args.viewer_pitch_deg))
        distance_scale = parse_float(query, "distance", 1.0)
        fov = parse_float(query, "fov", self.args.viewer_fov_deg)
        timestamp = parse_float(query, "time", self.args.viewer_timestamp)

        direction = rotate_axis(self.base_direction, self.up_hint, yaw)
        direction = normalize(direction * math.cos(pitch) + self.up_hint * math.sin(pitch))
        target_offset = np.asarray(
            [
                parse_float(query, "tx", 0.0),
                parse_float(query, "ty", 0.0),
                parse_float(query, "tz", 0.0),
            ],
            dtype=np.float32,
        )
        target = self.center + target_offset
        distance = self.default_distance * max(distance_scale, 0.02)
        eye = target + direction * distance

        return make_lookat_camera(
            eye,
            target,
            self.up_hint,
            parse_int(query, "width", self.args.viewer_width),
            parse_int(query, "height", self.args.viewer_height),
            fov,
            timestamp,
        )

    def render_png(self, query):
        camera = self.make_camera_from_query(query)
        level, resolution = self.resolve_grid_level(query)
        scale_modifier = parse_float(query, "scale", self.args.viewer_scale_modifier)
        overlay_enabled = parse_int(query, "overlay", 1) != 0
        draw_grid = parse_int(query, "draw_grid", self.args.draw_grid_lines)

        with torch.no_grad():
            pkg = self.render_fn(
                camera,
                self.gaussians,
                self.pipeline,
                self.background,
                scaling_modifier=scale_modifier,
                basicfunction=trbfunction,
                GRsetting=self.gr_setting,
                GRzer=self.gr_zero,
            )
            rendering = torch.clamp(pkg["render"], 0.0, 1.0)
            if overlay_enabled:
                overlay_args = SimpleNamespace(
                    draw_grid_lines=int(draw_grid),
                    grid_cell_line_color=self.args.grid_cell_line_color,
                    grid_cell_line_thickness=int(self.args.grid_cell_line_thickness),
                    grid_cell_line_halo=int(self.args.grid_cell_line_halo),
                )
                image, stats = draw_bbox_overlay(
                    rendering,
                    camera,
                    self.bbox_min,
                    self.bbox_max,
                    self.args.grid_line_color,
                    int(self.args.grid_line_thickness),
                    halo=int(self.args.grid_line_halo),
                    grid_resolution=resolution,
                    args=overlay_args,
                )
            else:
                image = Image.fromarray(tensor_to_uint8_image(rendering))
                stats = {"drawn_edges": 0, "drawn_grid_lines": 0}

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        meta = {
            "grid_level": int(level),
            "grid_resolution": [int(v) for v in resolution],
            "camera_center": camera.camera_center.detach().cpu().numpy().tolist(),
            "bbox_overlay": stats,
        }
        return buffer.getvalue(), meta

    def meta(self):
        return {
            "loaded_iteration": int(self.scene.loaded_iter),
            "num_test_cameras": int(len(self.views)),
            "bbox_min": self.bbox_min.tolist(),
            "bbox_max": self.bbox_max.tolist(),
            "bbox_center": self.center.tolist(),
            "bbox_radius": float(self.radius),
            "default_distance": float(self.default_distance),
            "grid_level": int(self.grid_spec["grid_level"]),
            "grid_resolution": [int(v) for v in self.grid_spec["grid_resolution"]],
            "all_grid_resolutions": [[int(v) for v in r] for r in self.grid_spec["all_grid_resolutions"]],
        }


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>STEGF Grid Viewer</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font: 14px sans-serif; display: flex; height: 100vh; overflow: hidden; }
    #panel { width: 310px; padding: 14px; background: #1d1d1d; box-sizing: border-box; overflow-y: auto; }
    #stage { flex: 1; display: flex; align-items: center; justify-content: center; background: #050505; }
    #view { max-width: 100%; max-height: 100%; image-rendering: auto; cursor: grab; }
    label { display: block; margin-top: 10px; color: #bbb; }
    input[type=range] { width: 100%; }
    input[type=number] { width: 85px; background: #2b2b2b; color: #eee; border: 1px solid #555; }
    button { margin-top: 10px; padding: 6px 10px; background: #333; color: #eee; border: 1px solid #555; cursor: pointer; }
    pre { white-space: pre-wrap; color: #aaa; font-size: 12px; }
  </style>
</head>
<body>
  <div id="panel">
    <h3>STEGF Grid Viewer</h3>
    <div>Drag: rotate. Wheel: zoom.</div>
    <label>time <input id="time" type="range" min="0" max="1" step="0.01" value="0.5"></label>
    <label>grid level <input id="grid" type="number" value="2"></label>
    <label>distance <input id="distance" type="range" min="0.05" max="4" step="0.01" value="1"></label>
    <label>fov <input id="fov" type="range" min="20" max="100" step="1" value="55"></label>
    <label>scale modifier <input id="scale" type="range" min="0.5" max="8" step="0.1" value="3"></label>
    <label><input id="overlay" type="checkbox" checked> render grid overlay</label>
    <label><input id="draw_grid" type="checkbox" checked> draw cell grid lines</label>
    <button id="reset">Reset View</button>
    <button id="save">Download Camera State</button>
    <pre id="info">Loading...</pre>
  </div>
  <div id="stage"><img id="view"></div>
  <script>
    const state = { yaw: 0, pitch: 45, distance: 1, fov: 55, time: 0.5, grid: 2, scale: 3, tx: 0, ty: 0, tz: 0 };
    let busy = false, pending = false, dragging = false, lastX = 0, lastY = 0;
    const img = document.getElementById('view');
    const info = document.getElementById('info');
    function syncControls() {
      document.getElementById('time').value = state.time;
      document.getElementById('grid').value = state.grid;
      document.getElementById('distance').value = state.distance;
      document.getElementById('fov').value = state.fov;
      document.getElementById('scale').value = state.scale;
    }
    function query() {
      const p = new URLSearchParams();
      Object.entries(state).forEach(([k,v]) => p.set(k, v));
      p.set('overlay', document.getElementById('overlay').checked ? 1 : 0);
      p.set('draw_grid', document.getElementById('draw_grid').checked ? 1 : 0);
      p.set('nonce', Date.now());
      return p.toString();
    }
    async function refresh() {
      if (busy) { pending = true; return; }
      busy = true; pending = false;
      const url = '/render?' + query();
      const res = await fetch(url);
      if (!res.ok) {
        info.textContent = await res.text();
        busy = false;
        return;
      }
      const meta = res.headers.get('X-STEGF-Meta');
      if (meta) info.textContent = JSON.stringify(JSON.parse(meta), null, 2);
      const blob = await res.blob();
      const old = img.src;
      img.src = URL.createObjectURL(blob);
      if (old.startsWith('blob:')) URL.revokeObjectURL(old);
      busy = false;
      if (pending) refresh();
    }
    function schedule() { refresh(); }
    ['time','distance','fov','scale'].forEach(id => {
      document.getElementById(id).addEventListener('input', e => { state[id] = parseFloat(e.target.value); schedule(); });
    });
    document.getElementById('grid').addEventListener('input', e => { state.grid = parseInt(e.target.value || '0'); schedule(); });
    document.getElementById('overlay').addEventListener('change', schedule);
    document.getElementById('draw_grid').addEventListener('change', schedule);
    document.getElementById('reset').onclick = () => {
      Object.assign(state, { yaw: 0, pitch: 45, distance: 1, fov: 55, time: 0.5, grid: 2, scale: 3, tx: 0, ty: 0, tz: 0 });
      syncControls(); schedule();
    };
    document.getElementById('save').onclick = () => {
      const blob = new Blob([JSON.stringify(state, null, 2)], {type: 'application/json'});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob); a.download = 'stegf_view_camera.json'; a.click();
      URL.revokeObjectURL(a.href);
    };
    img.addEventListener('pointerdown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; img.setPointerCapture(e.pointerId); });
    img.addEventListener('pointerup', e => { dragging = false; img.releasePointerCapture(e.pointerId); });
    img.addEventListener('pointermove', e => {
      if (!dragging) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      state.yaw += dx * 0.25;
      state.pitch = Math.max(-80, Math.min(80, state.pitch + dy * 0.25));
      schedule();
    });
    img.addEventListener('wheel', e => {
      e.preventDefault();
      state.distance = Math.max(0.05, Math.min(4, state.distance * Math.exp(e.deltaY * 0.001)));
      document.getElementById('distance').value = state.distance;
      schedule();
    }, { passive: false });
    fetch('/meta').then(r => r.json()).then(m => {
      info.textContent = JSON.stringify(m, null, 2);
      state.grid = m.grid_level;
      syncControls();
      refresh();
    });
  </script>
</body>
</html>
"""


def make_handler(viewer_state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if not viewer_state.args.viewer_quiet_http:
                super().log_message(fmt, *args)

        def send_bytes(self, status, content_type, data, headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            if headers:
                for key, value in headers.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_bytes(200, "text/html; charset=utf-8", HTML.encode("utf-8"))
                return
            if parsed.path == "/meta":
                data = json.dumps(viewer_state.meta(), indent=2).encode("utf-8")
                self.send_bytes(200, "application/json", data)
                return
            if parsed.path == "/render":
                try:
                    query = parse_qs(parsed.query)
                    png, meta = viewer_state.render_png(query)
                    self.send_bytes(
                        200,
                        "image/png",
                        png,
                        headers={"X-STEGF-Meta": json.dumps(meta, separators=(",", ":"))},
                    )
                except Exception:
                    error = traceback.format_exc()
                    self.send_bytes(500, "text/plain; charset=utf-8", error.encode("utf-8"))
                return
            self.send_bytes(404, "text/plain; charset=utf-8", b"not found")

    return Handler


def get_viewer_parse():
    parser = ArgumentParser(description="Local browser viewer for STEGF render + static grid overlay.")
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

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--viewer_width", type=int, default=960)
    parser.add_argument("--viewer_height", type=int, default=540)
    parser.add_argument("--viewer_fov_deg", type=float, default=55.0)
    parser.add_argument("--viewer_pitch_deg", type=float, default=45.0)
    parser.add_argument("--viewer_timestamp", type=float, default=0.5)
    parser.add_argument("--viewer_fit_padding", type=float, default=1.25)
    parser.add_argument("--viewer_scale_modifier", type=float, default=3.0)
    parser.add_argument("--viewer_quiet_http", action="store_true", default=True)
    parser.add_argument("--anchor_camera_index", type=int, default=0)

    parser.add_argument("--grid_level", type=int, default=2)
    parser.add_argument("--draw_grid_lines", type=int, default=1)
    parser.add_argument("--grid_line_color", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--grid_line_thickness", type=int, default=2)
    parser.add_argument("--grid_line_halo", type=int, default=1)
    parser.add_argument("--grid_cell_line_color", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--grid_cell_line_thickness", type=int, default=1)
    parser.add_argument("--grid_cell_line_halo", type=int, default=1)

    defaults = vars(parser.parse_args([]))
    args = get_combined_args(parser)
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

    return args, model.extract(args), pipeline.extract(args), multiview


def main():
    args, dataset, pipeline, multiview = get_viewer_parse()
    print(f"[STEGF] Loading viewer model from {dataset.model_path}")
    state = ViewerState(dataset, pipeline, args, multiview)
    server = HTTPServer((args.host, int(args.port)), make_handler(state))
    print(f"[STEGF] Viewer running at http://{args.host}:{args.port}")
    print("[STEGF] Open this URL in a local browser. Press Ctrl+C here to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
