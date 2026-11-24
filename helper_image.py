"""Lossless CPU image storage helpers, independent of Gaussian CUDA modules."""

import torch


def cpu_uint8_to_float32(image):
    """Normalize on CPU, matching the original PILtoTorch arithmetic order."""
    if image.device.type != "cpu" or image.dtype != torch.uint8:
        raise ValueError("Expected a CPU uint8 image")
    return image.to(dtype=torch.float32).div_(255.0)


@torch.no_grad()
def temporal_median_mad_cpu(images, chunk_pixels=32768):
    """Compute the existing temporal median/MAD with a bounded spatial tile.

    Keep every time sample in each tile: temporal subsampling would change the
    prior. torch.median's lower-middle convention for even frame counts and
    the channel-mean MAD definition match train.py's original full stack.
    Cached uint8 pixels are never modified or converted to full float copies.
    """
    images = tuple(images)
    chunk_pixels = int(chunk_pixels)
    if not images or chunk_pixels < 1:
        raise ValueError("Images must be nonempty and chunk_pixels positive")
    first = images[0]
    if first.ndim != 3 or first.shape[0] != 3:
        raise ValueError("Expected nonempty RGB CHW images")
    shape = tuple(first.shape)
    if min(shape) < 1:
        raise ValueError("Expected nonempty RGB CHW images")
    for image in images:
        if (image.device.type != "cpu" or tuple(image.shape) != shape
                or image.dtype not in (torch.uint8, torch.float32)):
            raise ValueError("Images must have equal RGB shapes and CPU uint8/float32 storage")
    channels, height, width = shape
    # CHW views backed by HWC storage retain their strides when H/W collapse.
    flat = [image.detach().reshape(channels, -1) for image in images]
    pixels = height * width
    median = torch.empty((channels, pixels), dtype=torch.float32)
    mad = torch.empty(pixels, dtype=torch.float32)
    for start in range(0, pixels, chunk_pixels):
        stop = min(start + chunk_pixels, pixels)
        tiles = [cpu_uint8_to_float32(image[:, start:stop])
                 if image.dtype == torch.uint8 else image[:, start:stop]
                 for image in flat]
        stack = torch.stack(tiles, dim=0)
        del tiles
        tile_median = torch.median(stack, dim=0).values
        median[:, start:stop] = tile_median
        # stack owns its data; reusing it does not mutate the cached images.
        stack.sub_(tile_median.unsqueeze(0)).abs_()
        mad[start:stop] = torch.median(stack, dim=0).values.mean(dim=0)
        del stack, tile_median
    return median.reshape(shape), mad.reshape(height, width)
