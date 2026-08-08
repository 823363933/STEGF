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
        level_resolutions=None,
        enable_dynamic_grid=True,
        enable_static_temporal_residual=False,
        static_temporal_frames=50,
        static_temporal_scale=1.0,
    ):
        super().__init__()
        self.base_resolution = base_resolution
        self.level_resolutions = self._resolve_level_resolutions(base_resolution, num_levels, level_resolutions)
        self.num_levels = len(self.level_resolutions)
        self.feature_dim = feature_dim
        self.fourier_degree = fourier_degree
        self.enable_dynamic_grid = enable_dynamic_grid
        self.enable_static_temporal_residual = enable_static_temporal_residual
        self.static_temporal_frames = max(int(static_temporal_frames), 1)
        self.static_temporal_scale = float(static_temporal_scale)

        bbox_min = torch.as_tensor(bbox_min, dtype=torch.float32, device="cuda").view(1, 3)
        bbox_max = torch.as_tensor(bbox_max, dtype=torch.float32, device="cuda").view(1, 3)
        bbox_span = torch.clamp(bbox_max - bbox_min, min=1e-6)
        self.register_buffer("bbox_min", bbox_min)
        self.register_buffer("bbox_max", bbox_max)
        self.register_buffer("bbox_span", bbox_span)

        dynamic_channels = feature_dim * 2 * fourier_degree
        self.static_grids = nn.ParameterList()
        self.dynamic_grids = nn.ParameterList()
        self.static_temporal_grids = nn.ParameterList()
        self.static_temporal_bins = self._resolve_static_temporal_bins(
            self.num_levels,
            self.static_temporal_frames,
        )
        for resolution in self.level_resolutions:
            level_index = len(self.static_grids)
            res_x, res_y, res_z = resolution
            static_grid = nn.Parameter(
                torch.empty(1, feature_dim, res_z, res_y, res_x, device="cuda")
            )
            nn.init.normal_(static_grid, mean=0.0, std=1e-4)
            self.static_grids.append(static_grid)
            if enable_static_temporal_residual:
                temporal_bins = self.static_temporal_bins[level_index]
                temporal_grid = nn.Parameter(
                    torch.zeros(temporal_bins, feature_dim, res_z, res_y, res_x, device="cuda")
                )
                self.static_temporal_grids.append(temporal_grid)
            if enable_dynamic_grid:
                dynamic_grid = nn.Parameter(
                    torch.empty(1, dynamic_channels, res_z, res_y, res_x, device="cuda")
                )
                nn.init.normal_(dynamic_grid, mean=0.0, std=1e-4)
                self.dynamic_grids.append(dynamic_grid)

    @staticmethod
    def _resolve_static_temporal_bins(num_levels, max_frames):
        max_frames = max(int(max_frames), 1)
        bins = []
        for level in range(max(int(num_levels), 1)):
            divisor = 2 ** max(int(num_levels) - 1 - level, 0)
            bins.append(max(int(math.ceil(max_frames / divisor)), 1))
        return bins

    @staticmethod
    def _resolve_level_resolutions(base_resolution, num_levels, level_resolutions):
        if level_resolutions is None:
            return [
                (base_resolution * (2 ** level),) * 3
                for level in range(num_levels)
            ]

        resolved = []
        for resolution in level_resolutions:
            if isinstance(resolution, int):
                item = (resolution, resolution, resolution)
            else:
                if len(resolution) == 1:
                    item = (int(resolution[0]),) * 3
                elif len(resolution) == 3:
                    item = tuple(int(v) for v in resolution)
                else:
                    raise ValueError(f"Invalid Euler grid resolution entry: {resolution}")
            if min(item) < 2:
                raise ValueError(f"Euler grid resolution must be >= 2 on every axis, got {item}")
            resolved.append(item)
        if not resolved:
            raise ValueError("EulerField requires at least one grid level")
        return resolved

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

    @staticmethod
    def _grid_resolution(grid):
        return int(grid.shape[-1]), int(grid.shape[-2]), int(grid.shape[-3])

    def _corner_indices_weights(self, points, resolution):
        if isinstance(resolution, int):
            res_x = res_y = res_z = resolution
        else:
            res_x, res_y, res_z = resolution
        resolution_tensor = torch.tensor(
            [res_x, res_y, res_z],
            device=points.device,
            dtype=points.dtype,
        )
        coords = self._normalize_points_unit(points) * (resolution_tensor - 1)
        lower = torch.floor(coords).long()
        max_index = torch.tensor(
            [res_x - 1, res_y - 1, res_z - 1],
            device=points.device,
            dtype=torch.long,
        )
        upper = torch.minimum(lower + 1, max_index)
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

        plane = res_x * res_y
        indices = torch.stack(
            (
                z0 * plane + y0 * res_x + x0,
                z0 * plane + y0 * res_x + x1,
                z0 * plane + y1 * res_x + x0,
                z0 * plane + y1 * res_x + x1,
                z1 * plane + y0 * res_x + x0,
                z1 * plane + y0 * res_x + x1,
                z1 * plane + y1 * res_x + x0,
                z1 * plane + y1 * res_x + x1,
            ),
            dim=1,
        )
        return indices, weights

    def _sample_dynamic_grid_nearest(self, grid, points):
        resolution = torch.tensor(
            self._grid_resolution(grid),
            device=points.device,
            dtype=points.dtype,
        )
        coords = self._normalize_points_unit(points) * (resolution - 1)
        center = torch.round(coords).long()
        min_index = torch.zeros(3, device=points.device, dtype=torch.long)
        max_index = (resolution.long() - 1)
        center = torch.maximum(torch.minimum(center, max_index), min_index)
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
        depth, height, width = static_grid.shape[-3:]
        features = static_grid.permute(0, 2, 3, 4, 1).reshape(-1, channels)
        view_dirs = view_direction.expand(features.shape[0], -1)
        feature_delta = view_mapper(torch.cat((features, view_dirs), dim=1))
        feature_delta = feature_delta.view(1, depth, height, width, channels)
        feature_delta = feature_delta.permute(0, 4, 1, 2, 3).contiguous()
        return static_grid + view_scale * feature_delta

    def _static_temporal_bin_index(self, timestamp, temporal_bins, device):
        if not torch.is_tensor(timestamp):
            timestamp = torch.tensor(timestamp, device=device, dtype=torch.float32)
        timestamp = timestamp.detach().float().reshape(-1)[0].to(device=device)
        if timestamp > 1.0:
            timestamp = timestamp / max(float(self.static_temporal_frames - 1), 1.0)
        timestamp = torch.clamp(timestamp, 0.0, 1.0)
        index = torch.floor(timestamp * float(temporal_bins)).long()
        return torch.clamp(index, 0, temporal_bins - 1)

    def query_static_level_features(self, points, camera_center=None, view_mapper=None, view_scale=0.0, timestamp=None):
        coords = self._normalize_points_grid(points)
        view_direction = self._global_view_direction(camera_center, points.device, points.dtype)
        level_features = []
        for level_index, static_grid in enumerate(self.static_grids):
            static_grid = self._view_condition_static_grid(
                static_grid,
                view_direction,
                view_mapper,
                view_scale,
            )
            feature = self._sample_grid(static_grid, coords)
            if (
                self.enable_static_temporal_residual
                and timestamp is not None
                and level_index < len(self.static_temporal_grids)
                and self.static_temporal_scale != 0.0
            ):
                temporal_grid = self.static_temporal_grids[level_index]
                bin_index = self._static_temporal_bin_index(
                    timestamp,
                    temporal_grid.shape[0],
                    points.device,
                )
                temporal_feature = self._sample_grid(temporal_grid[bin_index].unsqueeze(0), coords)
                feature = feature + self.static_temporal_scale * temporal_feature
            level_features.append(feature)
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


class EulerVelocityField(nn.Module):
    """A shared Eulerian velocity field used to transport Gaussian particles.

    The trainable state is spatial (multi-resolution feature grids). Time is
    supplied to a shared decoder, so every Gaussian that reaches the same
    space-time location receives the same velocity.  This is deliberately
    separate from ``EulerField``: the latter can keep serving the radiance
    branch while this module has an unambiguous transport-only meaning.
    """

    def __init__(
        self,
        bbox_min,
        bbox_max,
        level_resolutions,
        feature_dim=8,
        hidden_dim=32,
        fourier_degree=4,
        max_normalized_speed=0.25,
    ):
        super().__init__()
        self.level_resolutions = EulerField._resolve_level_resolutions(
            base_resolution=4,
            num_levels=len(level_resolutions),
            level_resolutions=level_resolutions,
        )
        self.feature_dim = max(int(feature_dim), 1)
        self.fourier_degree = max(int(fourier_degree), 0)
        self.max_normalized_speed = max(float(max_normalized_speed), 0.0)

        bbox_min = torch.as_tensor(bbox_min, dtype=torch.float32).view(1, 3)
        bbox_max = torch.as_tensor(bbox_max, dtype=torch.float32).view(1, 3)
        bbox_span = torch.clamp(bbox_max - bbox_min, min=1e-6)
        self.register_buffer("bbox_min", bbox_min)
        self.register_buffer("bbox_max", bbox_max)
        self.register_buffer("bbox_span", bbox_span)

        self.feature_grids = nn.ParameterList()
        for res_x, res_y, res_z in self.level_resolutions:
            grid = nn.Parameter(
                torch.empty(1, self.feature_dim, res_z, res_y, res_x)
            )
            nn.init.normal_(grid, mean=0.0, std=1e-3)
            self.feature_grids.append(grid)

        time_dim = 1 + 2 * self.fourier_degree
        input_dim = (
            len(self.level_resolutions) * self.feature_dim
            + 3
            + time_dim
        )
        hidden_dim = max(int(hidden_dim), 4)
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3, bias=False),
        )
        # The H2 experiment must begin as the identity map.  The final layer
        # learns first; gradients then open the spatial grids and earlier MLP.
        nn.init.zeros_(self.decoder[-1].weight)

    def _normalize_points_unit(self, points):
        return ((points - self.bbox_min) / self.bbox_span).clamp(0.0, 1.0)

    def _sample_grid(self, grid, unit_points):
        coords = (unit_points * 2.0 - 1.0).view(1, -1, 1, 1, 3)
        sampled = F.grid_sample(
            grid,
            coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return (
            sampled.squeeze(0)
            .squeeze(-1)
            .squeeze(-1)
            .transpose(0, 1)
            .contiguous()
        )

    def _prepare_times(self, timestamp, count, device, dtype):
        if not torch.is_tensor(timestamp):
            timestamp = torch.tensor(timestamp, device=device, dtype=dtype)
        timestamp = timestamp.to(device=device, dtype=dtype)
        if timestamp.numel() == 1:
            return timestamp.reshape(1, 1).expand(count, 1)
        timestamp = timestamp.reshape(-1, 1)
        if timestamp.shape[0] != count:
            raise ValueError(
                "EulerVelocityField timestamp must be scalar or have one "
                f"value per point, got {timestamp.shape[0]} for {count}"
            )
        return timestamp

    def _time_features(self, timestamp):
        features = [2.0 * timestamp - 1.0]
        if self.fourier_degree > 0:
            harmonics = torch.arange(
                1,
                self.fourier_degree + 1,
                device=timestamp.device,
                dtype=timestamp.dtype,
            ).view(1, -1)
            angles = 2.0 * math.pi * timestamp * harmonics
            features.extend((torch.cos(angles), torch.sin(angles)))
        return torch.cat(features, dim=1)

    def query_velocity(self, points, timestamp, return_normalized=False):
        unit_points = self._normalize_points_unit(points)
        level_features = [
            self._sample_grid(grid, unit_points)
            for grid in self.feature_grids
        ]
        time_features = self._time_features(
            self._prepare_times(
                timestamp,
                points.shape[0],
                points.device,
                points.dtype,
            )
        )
        decoder_input = torch.cat(
            level_features + ([(unit_points * 2.0 - 1.0), time_features]),
            dim=1,
        )
        normalized_velocity = self.max_normalized_speed * torch.tanh(
            self.decoder(decoder_input)
        )
        if return_normalized:
            return normalized_velocity
        return normalized_velocity * self.bbox_span.to(
            device=points.device,
            dtype=points.dtype,
        )

    def transport(
        self,
        points,
        source_times,
        target_time,
        steps=2,
        method="midpoint",
    ):
        """Transport points from individual source times to one target time."""
        steps = max(int(steps), 1)
        method = str(method).strip().lower()
        if method not in {"euler", "midpoint"}:
            raise ValueError(
                "EulerVelocityField integration method must be euler or "
                f"midpoint, got {method!r}"
            )
        source_times = self._prepare_times(
            source_times,
            points.shape[0],
            points.device,
            points.dtype,
        )
        target_times = self._prepare_times(
            target_time,
            points.shape[0],
            points.device,
            points.dtype,
        )
        step_dt = (target_times - source_times) / float(steps)
        current_points = points
        current_times = source_times
        normalized_velocity_energy = []
        bbox_span = self.bbox_span.to(
            device=points.device,
            dtype=points.dtype,
        )

        for _ in range(steps):
            if method == "midpoint":
                velocity_start_norm = self.query_velocity(
                    current_points,
                    current_times,
                    return_normalized=True,
                )
                midpoint = (
                    current_points
                    + 0.5 * step_dt * velocity_start_norm * bbox_span
                )
                midpoint_time = current_times + 0.5 * step_dt
                velocity_norm = self.query_velocity(
                    midpoint,
                    midpoint_time,
                    return_normalized=True,
                )
            else:
                velocity_norm = self.query_velocity(
                    current_points,
                    current_times,
                    return_normalized=True,
                )
            current_points = current_points + step_dt * velocity_norm * bbox_span
            current_times = current_times + step_dt
            normalized_velocity_energy.append(velocity_norm.square().mean())

        aux = {
            "normalized_velocity_energy": torch.stack(
                normalized_velocity_energy
            ).mean(),
            "displacement": current_points - points,
        }
        return current_points, aux


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
