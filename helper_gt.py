"""One lossless, reusable ground-truth image set per scene output.

No training/rendering math lives here. Compact exports keep a relative reference
to their source scene, so evaluation and photometric fitting share its GT.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image


def scene_output_root(model_path):
    root = Path(model_path).resolve()
    reference = root / "gt_source.json"
    if reference.is_file():
        data = json.loads(reference.read_text(encoding="utf-8"))
        if data.get("schema") != "stegf_shared_gt_source_v1":
            raise ValueError("Unsupported GT reference: {}".format(reference))
        source = (root / data["scene_root"]).resolve()
        if source.is_dir() and ((source / "cfg_args").is_file() or (source / "gt").is_dir()):
            return source
        # A compact moved into compact-candidates may have a stale relative
        # reference. Try its actual ancestor scene before treating it as a
        # standalone copy; an arbitrary existing parent is not a scene root.
    # Older compact exports predate gt_source.json. Only join an ancestor
    # scene, never an unrelated sibling experiment with the same scene name.
    if (root / "compact_config.json").is_file():
        for parent in root.parents:
            if (parent / "cfg_args").is_file() and not (parent / "compact_config.json").is_file():
                return parent
    return root


def shared_gt_path(model_path, split="test"):
    if split not in ("test", "train"):
        raise ValueError("Unsupported GT split: {}".format(split))
    return scene_output_root(model_path) / "gt" / split


def write_gt_reference(source_root, output_root):
    source = scene_output_root(source_root)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = {"schema": "stegf_shared_gt_source_v1", "scene_root": os.path.relpath(source, output)}
    _write_json(output / "gt_source.json", data)


def _write_json(path, data):
    descriptor, temporary = tempfile.mkstemp(prefix=".gt-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class SharedGroundTruth:
    """Save missing frames only; reject reuse with different pixels/order.

    The manifest caches decoded-pixel hashes against PNG hashes. Later
    checkpoints still validate input and saved bytes, but do not decode or
    rewrite an unchanged PNG. Metrics continue to use the original tensors.
    """

    def __init__(self, model_path, split="test", views=None):
        self.path = shared_gt_path(model_path, split)
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.path / "manifest.json"
        if self.manifest_path.is_file():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest.get("schema") != "stegf_shared_gt_v1":
                raise ValueError("Unsupported GT manifest: {}".format(self.manifest_path))
        else:
            self.manifest = {"schema": "stegf_shared_gt_v1", "frames": {}}
        self.changed = False
        if views is not None:
            identities = [{
                "image_name": str(view.image_name),
                "timestamp": float(view.timestamp),
                "width": int(view.image_width),
                "height": int(view.image_height),
            } for view in views]
            previous = self.manifest.get("views")
            if previous is not None and previous != identities:
                raise RuntimeError("GT view order/resolution differs from the scene cache: {}".format(self.path))
            self.manifest["views"] = identities
            self.changed = previous != identities
        self.saved = 0
        self.reused = 0

    def ensure_frame(self, filename, tensor):
        import torch

        if Path(filename).name != filename or not filename.endswith(".png"):
            raise ValueError("Expected a plain PNG filename: {}".format(filename))
        # Exactly torchvision.save_image's float -> uint8 rounding/clamping;
        # all operations are on a new tensor, never the metric input.
        pixels = tensor.detach().mul(255).add_(0.5).clamp_(0, 255)
        pixels = pixels.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        pixels = np.ascontiguousarray(pixels)
        rgb_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
        path = self.path / filename
        record = self.manifest["frames"].get(filename)
        png_hash = None
        if path.exists():
            png_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            cached = record is not None and record.get("png_sha256") == png_hash
            if cached:
                matches = record["rgb_sha256"] == rgb_hash and record["shape"] == list(pixels.shape)
            else:
                with Image.open(path) as image:
                    matches = image.mode == "RGB" and np.array_equal(np.asarray(image), pixels)
            if not matches:
                raise RuntimeError("GT pixels differ; refusing to overwrite or reuse {}".format(path))
            self.reused += 1
        else:
            descriptor, temporary = tempfile.mkstemp(prefix=".gt-", suffix=".png", dir=str(self.path))
            os.close(descriptor)
            try:
                Image.fromarray(pixels).save(temporary, format="PNG")
                try:
                    # Publish without overwriting a file from a concurrent
                    # evaluator that may have reached this frame first.
                    os.link(temporary, path)
                except FileExistsError:
                    with Image.open(path) as image:
                        if image.mode != "RGB" or not np.array_equal(np.asarray(image), pixels):
                            raise RuntimeError("Concurrent GT pixels differ: {}".format(path))
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            self.saved += 1
        info = path.stat()
        if png_hash is None:
            png_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        current = {"rgb_sha256": rgb_hash, "png_sha256": png_hash, "shape": list(pixels.shape),
                   "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
        if current != record:
            self.manifest["frames"][filename] = current
            self.changed = True
        return path

    def finish(self):
        if self.changed or not self.manifest_path.exists():
            _write_json(self.manifest_path, self.manifest)
        return self.path
