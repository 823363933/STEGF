"""Opt-in wide-support / narrow-child refinement. No extra training objective.

Proposal and confirmation use different training cameras, one configured interval apart.
The optional growth path observes ordinary backward passes without extra renders.
"""
import json
import importlib
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from helper_image import cpu_uint8_to_float32
from helper_narrow_growth import GROWTH_DEFAULTS, GrowthState, validate_growth


DEFAULTS = {
    'field_narrow_birth': 0,
    'field_narrow_birth_start': 10000,
    'field_narrow_birth_until': 30000,
    'field_narrow_birth_interval': 500,
    'field_narrow_birth_window_frames': 10.0,
    'field_narrow_birth_min_contrib': 0.03,
    'field_narrow_birth_min_ratio': 0.10,
    'field_narrow_birth_error_quantile': 0.90,
    'field_narrow_birth_error_floor': 0.03,
    'field_narrow_birth_activity_floor': 0.02,
    'field_narrow_birth_activity_offset': 3,
    'field_narrow_birth_blur_sigma': 2.0,
    'field_narrow_birth_dilate_radius': 3,
    'field_narrow_birth_event_ratio': 0.005,
    'field_narrow_birth_event_cap': 4096,
    'field_narrow_birth_total_ratio': 0.20,
    'field_narrow_birth_total_cap': 131072,
    'field_narrow_birth_opacity': 0.01,
    'field_narrow_birth_parent_width_ratio': 2.0,
    'field_narrow_birth_parent_kernel_min': 1.0 / 3,
    'field_narrow_birth_time_separation': 5,
    'field_narrow_birth_spatial_radius_ratio': 0.5,
    'field_narrow_birth_cleanup_until': 40000,
    'field_narrow_birth_cleanup_age': 2000,
    'field_narrow_birth_cleanup_opacity': 0.005,
    # Opt-in: old runs retain their exact single-pass/history policy.
    'field_narrow_birth_schedule': 'single_pass',
    'field_narrow_birth_phase1_until': 30000,
    'field_narrow_birth_phase1_budget_ratio': 2.0 / 3.0,
    'field_narrow_birth_phase2_seed': 0,
    'field_narrow_birth_dedup_mode': 'history',
}
# Keep old schema-2 policy signatures byte-for-byte equivalent when new knobs are off.
LEGACY_POLICY_KEYS = tuple(DEFAULTS)
DEFAULTS.update({
    'field_narrow_birth_opacity_mode': 'fixed',
    'field_narrow_birth_stratify_phase1': 0,
    'field_narrow_birth_cleanup_interval': 0,  # 0 keeps the legacy birth cadence.
    **GROWTH_DEFAULTS,
})


def options(source):
    get = source.get if isinstance(source, dict) else lambda k, d: getattr(source, k, d)
    result = {}
    for key, default in DEFAULTS.items():
        raw = get(key, default)
        if raw is None:  # Unspecified sentinel argument in an old checkpoint.
            raw = default
        value = str(raw).strip().lower() if isinstance(default, str) else type(default)(raw)
        if isinstance(default, int) and float(raw) != value:
            raise ValueError('Narrow birth integer option is fractional: ' + key)
        result[key] = value
    if not result['field_narrow_birth']:
        if result['field_narrow_birth_growth']:
            raise ValueError('Narrow spatial growth requires enabled narrow birth')
        return result
    short = {k[len('field_narrow_birth_'):]: v for k, v in result.items()
             if k != 'field_narrow_birth'}
    for k, v in short.items():
        if k in ['schedule', 'dedup_mode', 'opacity_mode']:
            continue
        if not math.isfinite(v) or v < 0:
            raise ValueError('Invalid narrow birth option: ' + k)
    for k in ['interval', 'event_cap', 'total_cap', 'time_separation', 'activity_offset']:
        if short[k] < 1:
            raise ValueError('Narrow birth requires positive ' + k)
    if not (0 < short['window_frames'] and short['start'] < short['until'] <= short['cleanup_until']):
        raise ValueError('Invalid narrow birth phase order/window')
    if (short['until'] - short['start']) % short['interval'] != 0:
        raise ValueError('Narrow birth confirmation window must contain whole event intervals')
    for k in ['error_quantile', 'min_ratio', 'event_ratio', 'total_ratio', 'parent_kernel_min']:
        if not 0 < short[k] <= 1:
            raise ValueError('Narrow birth option must be in (0,1]: ' + k)
    if not 0 < short['opacity'] < 1 or short['blur_sigma'] <= 0:
        raise ValueError('Invalid narrow birth opacity/blur')
    if short['schedule'] not in ['single_pass', 'two_stage']:
        raise ValueError('Unknown narrow birth schedule: ' + short['schedule'])
    if short['dedup_mode'] not in ['history', 'live']:
        raise ValueError('Unknown narrow birth dedup mode: ' + short['dedup_mode'])
    if short['opacity_mode'] not in ['fixed', 'parent_effective']:
        raise ValueError('Unknown narrow birth opacity mode: ' + short['opacity_mode'])
    if short['schedule'] == 'two_stage':
        middle = short['phase1_until']
        if not (short['start'] < middle < short['until']) or (middle - short['start']) % short['interval']:
            raise ValueError('Narrow birth phase1_until must be an aligned interior boundary')
        if not 0 < short['phase1_budget_ratio'] < 1:
            raise ValueError('Narrow birth phase1 budget ratio must be in (0,1)')
    validate_growth(result)
    return result


def _uniform_time_plan(duration, count):
    if duration < 1 or count < 1:
        raise ValueError('Narrow birth needs frames and at least one confirmation opportunity')
    bins = min(duration, count)
    result = []
    for index in range(count):
        bin_index = index % bins
        # Integer nearest rounding is deterministic on Python 3.8+.
        frame = ((duration - 1) // 2 if bins == 1 else
                 (bin_index * (duration - 1) + (bins - 1) // 2) // (bins - 1))
        left = bin_index * duration // bins
        right = (bin_index + 1) * duration // bins
        frame = min(max(frame, left), right - 1)
        result.append(dict(frame=frame, bin_index=bin_index,
                           bin_start=left, bin_end=right))
    return result


def proposal_time_plan(duration, config):
    """Full-sequence targets; opt-in two-stage revisit uses its own local RNG.

    Default output dictionaries are unchanged, including old checkpoint plans.
    Target coverage is not a guarantee of successful births or visibility.
    """
    duration = int(duration)
    start, until, interval = [config['field_narrow_birth_' + key] for key in ['start', 'until', 'interval']]
    count = (until - start) // interval
    if config.get('field_narrow_birth_schedule', 'single_pass') == 'single_pass':
        return _uniform_time_plan(duration, count)
    first = (config['field_narrow_birth_phase1_until'] - start) // interval
    result = []
    for phase, slots in enumerate([first, count - first], 1):
        targets = _uniform_time_plan(duration, slots)
        order = list(range(slots))
        if phase == 2 or config.get('field_narrow_birth_stratify_phase1', 0):
            # Each block visits all five temporal strata in a seeded order;
            # neither Python's global RNG nor training sampling is advanced.
            rng = random.Random(config['field_narrow_birth_phase2_seed'] + (1 if phase == 1 else 0))
            strata = min(5, slots)
            groups = [list(range(i * slots // strata, (i + 1) * slots // strata)) for i in range(strata)]
            for group in groups:
                rng.shuffle(group)
            order = []
            while any(groups):
                active = [i for i, group in enumerate(groups) if group]
                rng.shuffle(active)
                order.extend(groups[i].pop() for i in active)
        for slot, index in enumerate(order):
            result.append(dict(targets[index], phase=phase, phase_slot=slot, phase_slots=slots))
    return result


def fair_event_budget(event_cap, remaining, remaining_slots):
    """Keep the unchanged hard caps, reserving quota for later target times."""
    if remaining_slots < 1 or event_cap < 0 or remaining < 0:
        raise ValueError('Invalid narrow birth remaining budget/slots')
    return min(event_cap, remaining // remaining_slots)


def staged_event_budget(event_cap, total_cap, total_created, plan, slot, config):
    """Reserve phase 2; unused phase-1 quota rolls forward without refunds."""
    remaining = max(0, total_cap - total_created)
    target = plan[slot]
    if config.get('field_narrow_birth_schedule', 'single_pass') == 'single_pass':
        return dict(allowance=fair_event_budget(event_cap, remaining, len(plan) - slot))
    phase = target['phase']
    limit = (int(total_cap * config['field_narrow_birth_phase1_budget_ratio']) if phase == 1 else total_cap)
    phase_remaining = min(remaining, max(0, limit - total_created))
    slots = target['phase_slots'] - target['phase_slot']
    return dict(allowance=fair_event_budget(event_cap, phase_remaining, slots), phase=phase,
                phase_budget_limit=limit, phase_remaining_budget=phase_remaining,
                phase_remaining_slots=slots, cumulative_budget_limit=total_cap)


def policy_signature(config, duration):
    """Legacy checkpoints stay schema 1; opt-in policies require exact resume."""
    modern = (config.get('field_narrow_birth_growth', 0)
              or config.get('field_narrow_birth_stratify_phase1', 0)
              or config.get('field_narrow_birth_cleanup_interval', 0)
              or config.get('field_narrow_birth_opacity_mode', 'fixed') != 'fixed')
    if (not modern and config.get('field_narrow_birth_schedule', 'single_pass') == 'single_pass'
            and config.get('field_narrow_birth_dedup_mode', 'history') == 'history'):
        return None
    keys = DEFAULTS if modern else LEGACY_POLICY_KEYS
    return dict(duration=int(duration), options={key: config.get(key, DEFAULTS[key]) for key in keys})


class BirthState:
    """Point-aligned identities plus bounded pending candidates and birth ledger."""
    def __init__(self, count, device):
        self.ids = torch.arange(count, dtype=torch.long, device=device)
        self.family = self.ids.clone()
        self.born = torch.full((count,), -1, dtype=torch.int32, device=device)
        self.next_id = count
        self.reference = 0
        self.total_created = 0
        self.events = 0
        self.pending = None
        self.history = []  # (family_id, target_frame); never refunded on prune
        self.camera_groups = None
        self.last_event = -1
        self.time_plan = None
        self.policy = None
        self.growth = None

    def check(self, count):
        if any(x.shape != (count,) for x in [self.ids, self.family, self.born]):
            raise RuntimeError('Narrow birth identity/topology mismatch')
        if self.growth is not None:
            self.growth.check(count)

    def prune(self, keep):
        self.check(len(keep))
        if self.growth is not None:
            self.growth.prune(keep)
        self.ids, self.family, self.born = [x[keep] for x in [self.ids, self.family, self.born]]

    def append(self, count, parents=None, birth_iteration=None, allow_descendants=False):
        device = self.ids.device
        ids = torch.arange(self.next_id, self.next_id + count, device=device)
        family = self.family[parents] if parents is not None else ids
        if parents is not None and torch.any(self.born[parents] >= 0) and not allow_descendants:
            raise RuntimeError('New narrow children cannot clone/split or create descendants')
        if allow_descendants and (self.growth is None or birth_iteration is None
                                  or parents is None or not bool((self.born[parents] >= 0).all())):
            raise RuntimeError('Explicit narrow growth requires narrow parents and birth time')
        born = torch.full((count,), -1 if birth_iteration is None else birth_iteration,
                          dtype=torch.int32, device=device)
        self.ids = torch.cat((self.ids, ids))
        self.family = torch.cat((self.family, family))
        self.born = torch.cat((self.born, born))
        self.next_id += count
        if self.growth is not None:
            self.growth.append(count, -1 if birth_iteration is None else birth_iteration)

    def state_dict(self):
        result = dict(schema=1, ids=self.ids.cpu(), family=self.family.cpu(), born=self.born.cpu(),
                    next_id=self.next_id, reference=self.reference, total_created=self.total_created,
                    events=self.events, pending=self.pending, history=self.history,
                    camera_groups=self.camera_groups, last_event=self.last_event,
                    time_plan=self.time_plan)
        if self.policy is not None:
            result.update(schema=2, policy=self.policy)
        if self.growth is not None:
            result.update(schema=3, growth=self.growth.state_dict())
        return result

    @classmethod
    def load(cls, payload, device, mask=None):
        if payload.get('schema') not in [1, 2, 3]:
            raise RuntimeError('Unsupported narrow birth state schema')
        if payload['schema'] in [2, 3] and not isinstance(payload.get('policy'), dict):
            raise RuntimeError('Narrow birth state has no policy signature')
        if payload['schema'] == 3 and not isinstance(payload.get('growth'), dict):
            raise RuntimeError('Narrow growth state is missing')
        value = cls(0, device)
        for key in ['ids', 'family', 'born']:
            tensor = payload[key]
            if mask is not None:
                tensor = tensor[torch.as_tensor(mask, dtype=torch.bool, device=tensor.device)]
            setattr(value, key, tensor.to(device=device))
        for key in ['next_id', 'reference', 'total_created', 'events', 'pending', 'history', 'camera_groups', 'last_event']:
            setattr(value, key, payload[key])
        value.time_plan = payload.get('time_plan')
        value.policy = payload.get('policy')
        if payload['schema'] == 3:
            value.growth = GrowthState.load(payload['growth'], device, mask=mask)
        value.check(len(value.ids))
        if len(value.ids) and (not bool(torch.all(value.ids[1:] > value.ids[:-1]))
                               or value.next_id <= int(value.ids[-1])):
            raise RuntimeError('Invalid stable point IDs in narrow birth checkpoint')
        if value.total_created != len(value.history):
            raise RuntimeError('Narrow birth cumulative ledger does not match budget')
        return value


def ensure_state(model):
    if not model._narrow_birth_config['field_narrow_birth']:
        return None
    if model._narrow_birth_state is None:
        model._narrow_birth_state = BirthState(len(model.get_xyz), model.get_xyz.device)
    state = model._narrow_birth_state
    if model._narrow_birth_config.get('field_narrow_birth_growth', 0) and state.growth is None:
        if state.last_event >= 0 or state.total_created:
            raise RuntimeError('Cannot add narrow growth to an existing birth-only run')
        state.growth = GrowthState(len(model.get_xyz), model.get_xyz.device)
    model._narrow_birth_state.check(len(model.get_xyz))
    return model._narrow_birth_state


def validate_model(model):
    if not (model.field_motion_model == 'couptest_polynomial'
            and model.field_couptest_mode == 'coupled_detached'
            and model.field_existence_single_expert == 'transient'
            and not model._trbf_center.requires_grad
            and not model.field_existence_moe and not model.field_carrier_initialization
            and model.field_disable_legacy_aux and model.field_disable_global_omega_split
            and not model.field_v23_compat and not model.field_staged_training
            and model.field_disable_dynamic_grid
            and model.field_static_motion_scale == 0 and model.field_static_opacity_scale == 0):
        raise ValueError('Narrow birth requires the fixed-center, detached polynomial, appearance-only scaffold model')


def kernel_mask(model, frame, duration, config):
    width = model._trbf_scale.detach().exp().reshape(-1)
    center = model.get_trbfcenter.detach().reshape(-1)
    narrow = (config['field_narrow_birth_window_frames'] / (2 * duration)) / math.sqrt(math.log(255))
    kernel = torch.exp(-((frame / duration - center) / width).square())
    state = ensure_state(model)
    return ((width >= config['field_narrow_birth_parent_width_ratio'] * narrow)
            & (kernel >= config['field_narrow_birth_parent_kernel_min'])
            & (state.born < 0) & torch.isfinite(width) & torch.isfinite(kernel))


def group_score(total, masked, config):
    """Second-largest qualified contribution: >=2/3 independent cameras."""
    ratios = masked / total.clamp_min(1e-8)
    valid = ((masked >= config['field_narrow_birth_min_contrib'])
             & (ratios >= config['field_narrow_birth_min_ratio'])
             & torch.isfinite(total) & torch.isfinite(masked))
    return torch.where(valid, masked, torch.zeros_like(masked)).topk(2, dim=0).values[1]


def temporal_error_mask(gt, neighbours, prediction, config):
    from kornia.filters import gaussian_blur2d
    sigma = config['field_narrow_birth_blur_sigma']
    # Diagnostic sigma=2 uses the same 13x13 kernel as this runtime path.
    def smooth(image):
        return gaussian_blur2d(image[None], (13, 13), (sigma, sigma))[0]
    current = smooth(gt)
    activity = torch.zeros_like(gt[0])
    for other in neighbours:
        activity = torch.maximum(activity, (current - smooth(other)).abs().mean(0))
    active = (activity > config['field_narrow_birth_activity_floor']).float()[None, None]
    radius = config['field_narrow_birth_dilate_radius']
    if radius:
        active = F.max_pool2d(active, radius * 2 + 1, stride=1, padding=radius)
    error = (prediction - gt).abs().mean(0)
    threshold = max(float(torch.quantile(error, config['field_narrow_birth_error_quantile'])),
                    config['field_narrow_birth_error_floor'])
    return (error > threshold) & active[0, 0].bool()


@torch.no_grad()
def birth_attributes(model, parents, frame, duration, config):
    """Re-anchor position and tangent; appearance inheritance is approximate."""
    timestamp = frame / duration
    terms = model._couptest_motion_terms(timestamp, point_indices=parents)
    dt = timestamp - model._motion_time_anchor[parents]
    coefficients = model._motion[parents]
    velocity = (coefficients[:, :3] + terms['residual_coefficient'] * (
        2 * coefficients[:, 3:6] * dt + 3 * coefficients[:, 6:9] * dt.square()))
    center = torch.full_like(dt, timestamp)
    width = (config['field_narrow_birth_window_frames'] / (2 * duration)) / math.sqrt(math.log(255))
    opacity = config['field_narrow_birth_opacity']
    if config.get('field_narrow_birth_opacity_mode', 'fixed') == 'parent_effective':
        kernel = torch.exp(-((timestamp - model.get_trbfcenter[parents]) / model._trbf_scale[parents].exp()).square())
        probability = (model.get_opacity[parents] * kernel).clamp(1e-6, 1 - 1e-6)
        new_opacity = torch.log(probability / (1 - probability))
    else:
        new_opacity = torch.full_like(dt, math.log(opacity / (1 - opacity)))
    return dict(
        new_xyz=model.get_xyz[parents] + terms['offset'],
        new_features_dc=model._features_dc[parents],
        new_opacities=new_opacity,
        new_scaling=model._scaling[parents],
        # Keep the raw quaternion path, not its normalized value, to preserve omega.
        new_rotation=model._rotation[parents] + dt * model._omega[parents],
        new_trbf_center=center, new_trbfscale=torch.full_like(dt, math.log(width)),
        new_motion=torch.cat((velocity, torch.zeros_like(coefficients[:, 3:])), dim=1),
        new_omega=model._omega[parents], new_featuret=model._features_t[parents],
        new_static_level_logits=model._static_level_logits[parents] if model.use_euler_field else None,
        new_motion_time_anchor=center, narrow_birth_parents=parents,
    )


@torch.no_grad()
def deduplicate(model, parents, scores, frame, duration, config, limit, return_stats=False):
    """Fine NMS; old immutable history or opt-in live/active family separation."""
    from scipy.spatial import cKDTree
    state = ensure_state(model)
    stats = dict(input_candidates=len(parents), limit=int(limit))
    def finish(selected):
        stats['selected'] = len(selected)
        return (selected, stats) if return_stats else selected
    order = scores.argsort(descending=True)
    parents = parents[order]
    families = state.family[parents].cpu().tolist()
    live_mode = config.get('field_narrow_birth_dedup_mode', 'history') == 'live'
    active_children = None
    history = {}
    if live_mode:
        child_kernel = torch.exp(-((frame / duration - model.get_trbfcenter.detach().reshape(-1))
                                   / model._trbf_scale.detach().exp().reshape(-1)).square())
        active_children = torch.nonzero((state.born >= 0) & (child_kernel >= 1 / 3), as_tuple=False).reshape(-1)
        # Use current centers only from surviving, temporally active children.
        # The immutable history remains intact solely as the cumulative ledger.
        history_records = zip(state.family[active_children].cpu().tolist(),
                              (model.get_trbfcenter.detach().reshape(-1)[active_children] * duration).round().long().cpu().tolist())
        stats.update(dedup_mode='live', live_active_children=len(active_children),
                     historical_ledger_entries=len(state.history))
    else:
        history_records = state.history
    for family, time_index in history_records:
        history.setdefault(family, []).append(time_index)
    separation = config['field_narrow_birth_time_separation']
    allowed = [i for i, family in enumerate(families)
               if all(abs(frame - previous) >= separation for previous in history.get(family, []))]
    parents = parents[torch.tensor(allowed, dtype=torch.long, device=parents.device)]
    candidate_families = [families[i] for i in allowed]
    stats.update(time_rejected=stats['input_candidates'] - len(parents), after_time=len(parents))
    if not len(parents) or limit <= 0:
        return finish(parents[:0])
    xyz = birth_attributes(model, parents, frame, duration, config)['new_xyz'].cpu().numpy()
    radii = (model.get_scaling[parents].min(dim=1).values
             * config['field_narrow_birth_spatial_radius_ratio']).cpu().numpy()
    if not np.isfinite(xyz).all() or not np.isfinite(radii).all():
        raise RuntimeError('Non-finite narrow birth candidate geometry')
    tree = cKDTree(xyz)
    blocked = np.zeros(len(parents), dtype=bool)
    # Also avoid planting over an existing narrow child active at this time.
    if active_children is None:
        child_kernel = torch.exp(-((frame / duration - model.get_trbfcenter.detach().reshape(-1))
                                   / model._trbf_scale.detach().exp().reshape(-1)).square())
        active_children = torch.nonzero((state.born >= 0) & (child_kernel >= 1 / 3), as_tuple=False).reshape(-1)
    if len(active_children):
        child_terms = model._couptest_motion_terms(frame / duration, point_indices=active_children)
        child_xyz = (model.get_xyz[active_children] + child_terms['offset']).cpu().numpy()
        child_radius = (model.get_scaling[active_children].min(1).values
                        * config['field_narrow_birth_spatial_radius_ratio']).cpu().numpy()
        child_tree = cKDTree(child_xyz)
        for i, nearby in enumerate(child_tree.query_ball_point(xyz, radii)):
            if nearby:
                distance = np.linalg.norm(child_xyz[nearby] - xyz[i], axis=1)
                blocked[i] = np.any(distance < np.minimum(radii[i], child_radius[nearby]))
    stats['existing_active_child_rejected'] = int(blocked.sum())
    accepted = []
    used_families = set()
    stats['same_event_family_rejected'] = 0
    for i in range(len(parents)):
        if blocked[i]:
            continue
        if candidate_families[i] in used_families:
            stats['same_event_family_rejected'] += 1
            continue
        accepted.append(i)
        used_families.add(candidate_families[i])
        if len(accepted) >= limit:
            break
        nearby = np.asarray(tree.query_ball_point(xyz[i], radii[i]), dtype=np.int64)
        distance = np.linalg.norm(xyz[nearby] - xyz[i], axis=1)
        blocked[nearby[distance < np.minimum(radii[i], radii[nearby])]] = True
    stats['footprint_blocked_during_scan'] = int(blocked.sum())
    stats['budget_limit_reached'] = len(accepted) >= limit
    return finish(parents[torch.tensor(accepted, dtype=torch.long, device=parents.device)])


class NarrowBirthController:
    def __init__(self, model, cameras_by_time, duration, background, settings, rasterizer, kernel, output, extent=1.0):
        validate_model(model)
        self.model, self.cameras, self.duration = model, cameras_by_time, int(duration)
        self.config = model._narrow_birth_config
        self.state = ensure_state(model)
        self.extent = float(extent)
        if not math.isfinite(self.extent) or self.extent <= 0:
            raise ValueError('Narrow growth requires a positive scene extent')
        self.background, self.settings, self.rasterizer, self.kernel = background, settings, rasterizer, kernel
        raster_module = importlib.import_module(rasterizer.__module__)
        backend = getattr(raster_module, '_C', None)
        if not hasattr(rasterizer, 'contribution') or not hasattr(backend, 'rasterize_gaussians_contribution'):
            raise RuntimeError('Narrow birth requires the existing ch9 contribution extension; rebuild it before training')
        if duration < self.config['field_narrow_birth_window_frames']:
            raise ValueError('Narrow birth window must not exceed sequence length')
        signature = policy_signature(self.config, self.duration)
        if self.state.policy is None:
            if signature is not None and (self.state.last_event >= 0 or self.state.total_created):
                raise RuntimeError('Cannot resume an old narrow-birth run with a new policy; start this independent experiment from scratch')
            self.state.policy = signature
        elif self.state.policy != signature:
            raise RuntimeError('Narrow birth checkpoint policy differs from configuration')
        plan = proposal_time_plan(self.duration, self.config)
        if self.state.time_plan is None:
            if self.state.last_event >= self.get('start'):
                raise RuntimeError('Old narrow-birth checkpoint has no full-sequence time plan; start this experiment from scratch')
            self.state.time_plan = plan
        elif self.state.time_plan != plan:
            raise RuntimeError('Narrow birth checkpoint time plan differs from the configured sequence/budget')
        self.time_plan = self.state.time_plan
        first = sorted(cameras_by_time[0], key=lambda camera: camera.image_name)
        if len(first) < 6:
            raise ValueError('Narrow birth requires six training cameras')
        if self.state.camera_groups is None:
            centers = np.stack([camera.camera_center.detach().cpu().numpy() for camera in first])
            chosen = [0]
            while len(chosen) < 6:
                distance = np.linalg.norm(centers[:, None] - centers[chosen][None], axis=2).min(1)
                distance[chosen] = -1
                chosen.append(int(distance.argmax()))
            chosen = [chosen[j] for j in [0, 2, 4, 1, 3, 5]]
            self.state.camera_groups = [[first[j].image_name for j in chosen[:3]],
                                        [first[j].image_name for j in chosen[3:]]]
        required = set(sum(self.state.camera_groups, []))
        for frame in range(self.duration):
            if required - {camera.image_name for camera in cameras_by_time.get(frame, [])}:
                raise ValueError('Narrow birth requires synchronized training cameras at every frame')
        self.lookup = {frame: {camera.image_name: camera for camera in views}
                       for frame, views in cameras_by_time.items()}
        self.log = Path(output) / 'narrow_birth_events.jsonl'
        self.log.parent.mkdir(parents=True, exist_ok=True)
        if self.log.exists() and self.log.stat().st_size and self.state.last_event < 0:
            raise FileExistsError('Narrow birth output already contains an experiment; use a new output directory')
        growth_log = self.log.parent / 'narrow_growth_events.jsonl'
        if (self.get('growth') and growth_log.exists() and growth_log.stat().st_size
                and self.state.growth.last_event < 0):
            raise FileExistsError('Narrow growth output already contains an experiment')
        print('[STEGF] Narrow birth: proposal={} confirmation={}, interval={}, birth={}..{}, cleanup_until={}'.format(
            *self.state.camera_groups, self.get('interval'), self.get('start') + self.get('interval'),
            self.get('until'), self.get('cleanup_until')), flush=True)
        print('[STEGF] Narrow birth time plan: {} opportunities, frames={}..{}, '
              'window={} frames at kernel=1/255, family separation={} frames'.format(
                  len(self.time_plan), min(x['frame'] for x in self.time_plan), max(x['frame'] for x in self.time_plan),
                  self.get('window_frames'), self.get('time_separation')), flush=True)
        if signature is not None:
            print('[STEGF] Narrow birth policy: schedule={}, phase1_until={}, phase1_budget_ratio={:.6f}, '
                  'phase2_seed={}, dedup={}'.format(self.get('schedule'), self.get('phase1_until'),
                    self.get('phase1_budget_ratio'), self.get('phase2_seed'), self.get('dedup_mode')), flush=True)

        if self.get('growth'):
            print('[STEGF] Narrow seeds: opacity={}, phase1_stratified={}, '
                  'event=min({:.1%} N_ref,{}), total=min({:.1%} N_ref,{})'.format(
                      self.get('opacity_mode'), self.get('stratify_phase1'), self.get('event_ratio'),
                      self.get('event_cap'), self.get('total_ratio'), self.get('total_cap')), flush=True)
            print('[STEGF] Narrow spatial growth: {}..{}, interval={}, age={}, cooldown={}, '
                  'event=min({:.1%} N_ref,{}), total_net=min({:.1%} N_ref,{})'.format(
                      self.get('growth_start'), self.get('growth_until'), self.get('growth_interval'),
                      self.get('growth_min_age'), self.get('growth_cooldown'),
                      self.get('growth_event_ratio'), self.get('growth_event_cap'),
                      self.get('growth_total_ratio'), self.get('growth_total_cap')), flush=True)

    def get(self, key):
        return self.config['field_narrow_birth_' + key]

    def gt(self, frame, name):
        camera = self.lookup[frame][name]
        image = camera.original_image
        if image.dtype == torch.uint8:
            image = cpu_uint8_to_float32(image)
        return image.to(self.model.get_xyz.device)

    @torch.no_grad()
    def measure(self, frame, group):
        masked, totals, pixels = [], [], []
        for name in self.state.camera_groups[group]:
            camera = self.lookup[frame][name]
            attributes = self.model.compose_time_conditioned_attributes(
                camera.timestamp, self.kernel, camera_center=camera.camera_center)
            means, opacity, rotation, colors = attributes
            raster = self.rasterizer(raster_settings=self.settings(
                image_height=camera.image_height, image_width=camera.image_width,
                tanfovx=math.tan(camera.FoVx / 2), tanfovy=math.tan(camera.FoVy / 2),
                bg=self.background, scale_modifier=1., viewmatrix=camera.world_view_transform,
                projmatrix=camera.full_proj_transform, sh_degree=self.model.active_sh_degree,
                campos=camera.camera_center, prefiltered=False))
            kwargs = dict(means3D=means, shs=None, colors_precomp=colors,
                          opacities=opacity, scales=self.model.get_scaling, rotations=rotation,
                          cov3D_precomp=None)
            image, _, _ = raster(means2D=torch.zeros_like(means), **kwargs)
            prediction = self.model.rgbdecoder(image[None], camera.rays, camera.timestamp)[0]
            prediction = self.model.apply_content_exposure(prediction)
            offset = self.get('activity_offset')
            mask = temporal_error_mask(self.gt(frame, name),
                [self.gt(max(0, frame - offset), name), self.gt(min(self.duration - 1, frame + offset), name)],
                prediction, self.config)
            if not hasattr(raster, 'contribution'):
                raise RuntimeError('Installed ch9 rasterizer lacks contribution(); rebuild the existing extension')
            total, partial, _ = raster.contribution(unreliable_mask=mask.contiguous(), **kwargs)
            totals.append(total.reshape(-1)); masked.append(partial.reshape(-1)); pixels.append(int(mask.sum()))
        return group_score(torch.stack(totals), torch.stack(masked), self.config), pixels

    def budgets(self):
        reference = self.state.reference
        event = min(int(reference * self.get('event_ratio')), self.get('event_cap'))
        total = min(int(reference * self.get('total_ratio')), self.get('total_cap'))
        return event, max(0, total - self.state.total_created)

    @torch.no_grad()
    def tick(self, iteration):
        result = self._tick_birth(iteration)
        if self.get('growth'):
            from helper_narrow_growth import grow_narrow
            grow_narrow(self, iteration)
        return result

    @torch.no_grad()
    def _tick_birth(self, iteration):
        interval = (self.get('cleanup_interval') or self.get('interval')) if iteration > self.get('until') else self.get('interval')
        if (iteration < self.get('start') or iteration > self.get('cleanup_until')
                or (iteration - self.get('start')) % interval != 0):
            return None
        state = self.state
        if not len(self.model.get_xyz):
            raise RuntimeError('Narrow birth cannot operate on an empty Gaussian scene')
        if iteration <= state.last_event:
            raise RuntimeError('Narrow birth clock moved backwards or repeated an event')
        torch.cuda.synchronize()
        begin = time.perf_counter()
        state.check(len(self.model.get_xyz))
        if state.reference == 0:
            state.reference = len(self.model.get_xyz)
        event_cap, remaining = self.budgets()
        total_cap = remaining + state.total_created
        record = dict(iteration=iteration, reference_points=state.reference,
                      event_cap=event_cap, remaining_before=remaining, generated=0, pruned=0,
                      points_before=len(self.model.get_xyz))
        if iteration <= self.get('until'):
            pending = state.pending
            state.pending = None
            if pending is not None and remaining > 0:
                if iteration - pending['iteration'] != self.get('interval'):
                    raise RuntimeError('Pending narrow birth proposal has an invalid confirmation age')
                frame = pending['frame']
                proposal_slot = (pending['iteration'] - self.get('start')) // self.get('interval')
                remaining_slots = len(self.time_plan) - proposal_slot
                quota = staged_event_budget(event_cap, total_cap,
                                           state.total_created, self.time_plan, proposal_slot, self.config)
                event_allowance = quota.pop('allowance')
                record.update(quota)
                confirmed, pixels = self.measure(frame, 1)
                ids = pending['ids'].to(state.ids.device)
                positions = torch.searchsorted(state.ids, ids)
                in_range = positions < len(state.ids)
                safe_positions = positions.clamp(max=max(len(state.ids) - 1, 0))
                exists = in_range & (state.ids[safe_positions] == ids)
                previous = pending['scores'].to(confirmed.device)
                good = exists & (confirmed[safe_positions] > 0) & kernel_mask(
                    self.model, frame, self.duration, self.config)[safe_positions]
                parents = safe_positions[good]
                scores = torch.minimum(previous[good], confirmed[parents])
                record.update(confirm_frame=frame, pending=len(ids), survived=int(exists.sum()),
                              confirmed=len(parents), confirmation_pixels=pixels,
                              proposal_iteration=pending['iteration'], confirmation_age=iteration - pending['iteration'],
                              remaining_target_slots=remaining_slots, fair_event_budget=event_allowance)
                parents, dedup_stats = deduplicate(self.model, parents, scores, frame, self.duration,
                                      self.config, event_allowance, return_stats=True)
                record['deduplication'] = dedup_stats
                if len(parents):
                    families = state.family[parents].cpu().tolist()
                    parent_ids = state.ids[parents].cpu().tolist()
                    archive = self.log.parent / 'narrow_birth' / ('iteration_{}.npz'.format(iteration))
                    if archive.exists():
                        raise FileExistsError('Refusing to overwrite narrow birth archive: ' + str(archive))
                    attributes = birth_attributes(self.model, parents, frame, self.duration, self.config)
                    canonical_parent = self.model.get_xyz[parents].detach().cpu().numpy()
                    self.model.densification_postfix(**attributes)
                    state.born[-len(parents):] = iteration
                    state.history.extend((family, frame) for family in families)
                    state.total_created += len(parents)
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(archive, parent_ids=np.asarray(parent_ids),
                        child_ids=state.ids[-len(parents):].cpu().numpy(), families=np.asarray(families),
                        frame=np.asarray(frame), parent_canonical=canonical_parent,
                        xyz=attributes['new_xyz'].cpu().numpy(), velocity=attributes['new_motion'][:, :3].cpu().numpy(),
                        center=attributes['new_trbf_center'].cpu().numpy(),
                        width=attributes['new_trbfscale'].exp().cpu().numpy(),
                        scale=attributes['new_scaling'].exp().cpu().numpy(),
                        rotation=attributes['new_rotation'].cpu().numpy())
                    record.update(generated=len(parents), parent_ids=parent_ids,
                                  child_ids=state.ids[-len(parents):].cpu().tolist(),
                                  birth_archive=str(archive.relative_to(self.log.parent)))
                del confirmed
            # Pipeline the next target: at most six extra no-grad views per event.
            if iteration < self.get('until') and remaining > record['generated'] and event_cap > 0:
                slot = (iteration - self.get('start')) // self.get('interval')
                target = self.time_plan[slot]
                frame = target['frame']
                available = True
                if self.get('schedule') == 'two_stage':
                    quota = staged_event_budget(event_cap, total_cap,
                                               state.total_created, self.time_plan, slot, self.config)
                    available = quota['allowance'] > 0
                    record.update(proposal_phase=target['phase'], proposal_phase_slot=target['phase_slot'],
                                  proposal_budget_available=available)
                candidates = torch.empty(0, dtype=torch.long, device=state.ids.device)
                take, pixels = 0, []
                if available:
                    proposed, pixels = self.measure(frame, 0)
                    proposed = torch.where(kernel_mask(self.model, frame, self.duration, self.config),
                                           proposed, torch.zeros_like(proposed))
                    candidates = torch.nonzero(proposed > 0, as_tuple=False).reshape(-1)
                    take = min(len(candidates), event_cap * 8, 32768)
                    selected = candidates[proposed[candidates].topk(take).indices]
                if take:
                    state.pending = dict(frame=frame, iteration=iteration, ids=state.ids[selected].cpu(),
                                         scores=proposed[selected].cpu())
                record.update(proposal_frame=frame, proposal_slot=slot,
                              proposal_bin=target['bin_index'],
                              proposal_bin_bounds=[target['bin_start'], target['bin_end']],
                              time_schedule=('two_stage_stratified_v1' if self.get('schedule') == 'two_stage'
                                             else 'budget_stratified_v1'),
                              proposed=len(candidates), stored=take, proposal_pixels=pixels)
        else:
            state.pending = None
            weak = ((state.born >= 0) & (iteration - state.born >= self.get('cleanup_age'))
                    & (self.model.get_opacity.reshape(-1) < self.get('cleanup_opacity')))
            record['pruned'] = int(weak.sum())
            if record['pruned']:
                self.model.prune_points(weak)
        state.events += 1
        state.last_event = iteration
        torch.cuda.synchronize()
        record.update(points_after=len(self.model.get_xyz), total_created=state.total_created,
                      alive_children=int((state.born >= 0).sum()), seconds=time.perf_counter() - begin)
        with self.log.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')
        print('[STEGF][NarrowBirth] iter={} frame={} proposed={} confirmed={} added={} '
              'pruned={} total_created={} alive={} cost={:.2f}s'.format(
                  iteration, record.get('confirm_frame', '-'), record.get('proposed', 0),
                  record.get('confirmed', 0), record['generated'], record['pruned'],
                  state.total_created, record['alive_children'], record['seconds']), flush=True)
        return record
