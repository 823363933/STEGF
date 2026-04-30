import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EulerField(nn.Module):
    def __init__(
        self,
        bbox_min,
        bbox_max,
        base_resolution=4,
        num_levels=5,
        feature_dim=8,
        fourier_degree=10,
    ):
        super().__init__()
        self.base_resolution = base_resolution
        self.num_levels = num_levels
        self.feature_dim = feature_dim
        self.fourier_degree = fourier_degree

        bbox_min = torch.as_tensor(bbox_min, dtype=torch.float32, device="cuda").view(1, 3)
        bbox_max = torch.as_tensor(bbox_max, dtype=torch.float32, device="cuda").view(1, 3)
        bbox_span = torch.clamp(bbox_max - bbox_min, min=1e-6)
        self.register_buffer("bbox_min", bbox_min)
        self.register_buffer("bbox_max", bbox_max)
        self.register_buffer("bbox_span", bbox_span)

        dynamic_channels = feature_dim * 2 * fourier_degree
        self.static_grids = nn.ParameterList()
        self.dynamic_grids = nn.ParameterList()
        for level in range(num_levels):
            resolution = base_resolution * (2 ** level)
            static_grid = nn.Parameter(
                torch.empty(1, feature_dim, resolution, resolution, resolution, device="cuda")
            )
            dynamic_grid = nn.Parameter(
                torch.empty(1, dynamic_channels, resolution, resolution, resolution, device="cuda")
            )
            nn.init.normal_(static_grid, mean=0.0, std=1e-4)
            nn.init.normal_(dynamic_grid, mean=0.0, std=1e-4)
            self.static_grids.append(static_grid)
            self.dynamic_grids.append(dynamic_grid)

    def _normalize_points_unit(self, points):
        coords = (points - self.bbox_min) / self.bbox_span
        return coords.clamp(0.0, 1.0)

    def _normalize_points_grid(self, points):
        coords = self._normalize_points_unit(points) * 2.0 - 1.0
        return coords.view(1, -1, 1, 1, 3)

    def _sample_grid(self, grid, coords):
        sampled = F.grid_sample(
            grid,
            coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1).contiguous()

    def _fourier_basis(self, timestamp, device, dtype):
        if self.fourier_degree <= 0:
            return torch.empty(0, device=device, dtype=dtype)
        if not torch.is_tensor(timestamp):
            timestamp = torch.tensor(timestamp, device=device, dtype=dtype)
        timestamp = timestamp.reshape(1).to(device=device, dtype=dtype)
        harmonics = torch.arange(1, self.fourier_degree + 1, device=device, dtype=dtype)
        angles = 2.0 * math.pi * harmonics * timestamp
        basis = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
        return basis.reshape(-1)

    def _corner_indices_weights(self, points, resolution):
        coords = self._normalize_points_unit(points) * (resolution - 1)
        lower = torch.floor(coords).long()
        upper = torch.clamp(lower + 1, max=resolution - 1)
        frac = coords - lower.to(coords.dtype)

        x0, y0, z0 = lower[:, 0], lower[:, 1], lower[:, 2]
        x1, y1, z1 = upper[:, 0], upper[:, 1], upper[:, 2]
        wx1, wy1, wz1 = frac[:, 0], frac[:, 1], frac[:, 2]
        wx0, wy0, wz0 = 1.0 - wx1, 1.0 - wy1, 1.0 - wz1

        weights = torch.stack(
            (
                wx0 * wy0 * wz0,
                wx1 * wy0 * wz0,
                wx0 * wy1 * wz0,
                wx1 * wy1 * wz0,
                wx0 * wy0 * wz1,
                wx1 * wy0 * wz1,
                wx0 * wy1 * wz1,
                wx1 * wy1 * wz1,
            ),
            dim=1,
        )

        plane = resolution * resolution
        indices = torch.stack(
            (
                z0 * plane + y0 * resolution + x0,
                z0 * plane + y0 * resolution + x1,
                z0 * plane + y1 * resolution + x0,
                z0 * plane + y1 * resolution + x1,
                z1 * plane + y0 * resolution + x0,
                z1 * plane + y0 * resolution + x1,
                z1 * plane + y1 * resolution + x0,
                z1 * plane + y1 * resolution + x1,
            ),
            dim=1,
        )
        return indices, weights

    def _sample_dynamic_grid_nearest(self, grid, points):
        resolution = grid.shape[-1]
        coords = self._normalize_points_unit(points) * (resolution - 1)
        center = torch.round(coords).long().clamp_(0, resolution - 1)
        x, y, z = center[:, 0], center[:, 1], center[:, 2]
        return grid[0, :, z, y, x].transpose(0, 1).contiguous()

    def _sample_dynamic_grid_trilinear(self, grid, points):
        coords = self._normalize_points_grid(points)
        return self._sample_grid(grid, coords)

    def _global_view_direction(self, camera_center, device, dtype):
        if camera_center is None:
            return None
        if not torch.is_tensor(camera_center):
            camera_center = torch.tensor(camera_center, device=device, dtype=dtype)
        camera_center = camera_center.to(device=device, dtype=dtype).view(1, 3)
        bbox_center = 0.5 * (self.bbox_min + self.bbox_max).to(device=device, dtype=dtype)
        return F.normalize(camera_center - bbox_center, dim=1, eps=1e-6)

    def _view_condition_static_grid(self, static_grid, view_direction, view_mapper, view_scale):
        if view_direction is None or view_mapper is None or view_scale <= 0.0:
            return static_grid
        channels = static_grid.shape[1]
        resolution = static_grid.shape[-1]
        features = static_grid.permute(0, 2, 3, 4, 1).reshape(-1, channels)
        view_dirs = view_direction.expand(features.shape[0], -1)
        feature_delta = view_mapper(torch.cat((features, view_dirs), dim=1))
        feature_delta = feature_delta.view(1, resolution, resolution, resolution, channels)
        feature_delta = feature_delta.permute(0, 4, 1, 2, 3).contiguous()
        return static_grid + view_scale * feature_delta

    def query_static_level_features(self, points, camera_center=None, view_mapper=None, view_scale=0.0):
        coords = self._normalize_points_grid(points)
        view_direction = self._global_view_direction(camera_center, points.device, points.dtype)
        level_features = []
        for static_grid in self.static_grids:
            static_grid = self._view_condition_static_grid(
                static_grid,
                view_direction,
                view_mapper,
                view_scale,
            )
            level_features.append(self._sample_grid(static_grid, coords))
        return torch.stack(level_features, dim=1)

    def query_dynamic_level_features(self, points, timestamp, mode="nearest"):
        basis = self._fourier_basis(timestamp, points.device, points.dtype)
        temporal_channels = 2 * self.fourier_degree
        level_features = []
        for dynamic_grid in self.dynamic_grids:
            if mode == "trilinear":
                dynamic_coeff = self._sample_dynamic_grid_trilinear(dynamic_grid, points)
            else:
                dynamic_coeff = self._sample_dynamic_grid_nearest(dynamic_grid, points)
            dynamic_coeff = dynamic_coeff.view(-1, self.feature_dim, temporal_channels)
            dynamic_feature = torch.einsum("nck,k->nc", dynamic_coeff, basis)
            level_features.append(dynamic_feature)
        return torch.stack(level_features, dim=1)

    def blend_level_features(self, level_features, level_logits):
        weights = torch.softmax(level_logits, dim=1).unsqueeze(-1)
        return torch.sum(level_features * weights, dim=1)


class EulerResidualDecoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-2)

    def forward(self, field_feature):
        return self.net(field_feature)


class EulerLevelRouter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_levels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_levels, bias=False),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)

    def forward(self, router_input):
        return self.net(router_input)


class EulerQueryFusionGate(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias_init=-2.0, motion_scale_init=2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.net[-1].bias, bias_init)
        motion_scale_unconstrained = math.log(math.expm1(motion_scale_init))
        self.motion_scale = nn.Parameter(torch.tensor([motion_scale_unconstrained], device="cuda", dtype=torch.float32))

    def forward(self, gate_input, motion_strength=None):
        gate_logits = self.net(gate_input)
        if motion_strength is not None:
            gate_logits = gate_logits + F.softplus(self.motion_scale) * motion_strength
        return gate_logits
