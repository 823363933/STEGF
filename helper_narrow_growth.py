"""Opt-in spatial refinement of narrow seeds and descendants; no extra render."""
import json
import time

import numpy as np
import torch


GROWTH_DEFAULTS = {
    'field_narrow_birth_growth': 0,
    'field_narrow_birth_growth_start': 12000,
    'field_narrow_birth_growth_until': 60000,
    'field_narrow_birth_growth_interval': 500,
    'field_narrow_birth_growth_phase1_until': 40000,
    'field_narrow_birth_growth_phase1_budget_ratio': 0.7,
    'field_narrow_birth_growth_event_ratio': 0.01,
    'field_narrow_birth_growth_event_cap': 4096,
    'field_narrow_birth_growth_total_ratio': 0.8,
    'field_narrow_birth_growth_total_cap': 262144,
    'field_narrow_birth_growth_min_age': 2000,
    'field_narrow_birth_growth_cooldown': 1000,
    'field_narrow_birth_growth_min_observations': 5,
    'field_narrow_birth_growth_grad_threshold': 0.0002,
    'field_narrow_birth_growth_kernel_min': 1.0 / 3,
    'field_narrow_birth_growth_seed': 0,
}


def validate_growth(config):
    if not config['field_narrow_birth_growth']:
        return
    get = lambda key: config['field_narrow_birth_growth_' + key]
    start, middle, until, interval = [get(key) for key in ['start', 'phase1_until', 'until', 'interval']]
    if not (config['field_narrow_birth_start'] < start <= middle < until
            <= config['field_narrow_birth_cleanup_until']):
        raise ValueError('Invalid narrow growth phase order')
    for key in ['interval', 'event_cap', 'total_cap', 'min_age', 'cooldown', 'min_observations']:
        if get(key) < 1:
            raise ValueError('Narrow growth requires positive ' + key)
    if (middle - start) % interval or (until - start) % interval:
        raise ValueError('Narrow growth boundaries must lie on events')
    if not 0 < get('phase1_budget_ratio') < 1 or not 0 < get('kernel_min') <= 1:
        raise ValueError('Invalid narrow growth quota/kernel ratio')
    if not 0 < get('event_ratio') <= 1 or get('total_ratio') <= 0 or get('grad_threshold') <= 0:
        raise ValueError('Invalid narrow growth budget/gradient threshold')


class GrowthState:
    """Point-aligned window statistics plus a non-refundable net-growth ledger."""
    def __init__(self, count, device):
        self.gradient_sum = torch.zeros(count, device=device)
        self.observations = torch.zeros(count, device=device)
        self.last_growth = torch.full((count,), -1, dtype=torch.int32, device=device)
        self.net_added = 0
        self.created = 0
        self.cloned = 0
        self.split_parents = 0
        self.last_event = -1
        self.history = []

    def check(self, count):
        if any(value.shape != (count,) for value in [self.gradient_sum, self.observations, self.last_growth]):
            raise RuntimeError('Narrow growth statistics/topology mismatch')

    def prune(self, keep):
        self.check(len(keep))
        for key in ['gradient_sum', 'observations', 'last_growth']:
            setattr(self, key, getattr(self, key)[keep])

    def append(self, count, iteration):
        device = self.gradient_sum.device
        self.gradient_sum = torch.cat((self.gradient_sum, torch.zeros(count, device=device)))
        self.observations = torch.cat((self.observations, torch.zeros(count, device=device)))
        self.last_growth = torch.cat((self.last_growth, torch.full(
            (count,), iteration, dtype=torch.int32, device=device)))

    def state_dict(self):
        return dict(gradient_sum=self.gradient_sum.cpu(), observations=self.observations.cpu(),
                    last_growth=self.last_growth.cpu(), net_added=self.net_added, created=self.created,
                    cloned=self.cloned, split_parents=self.split_parents,
                    last_event=self.last_event, history=self.history)

    @classmethod
    def load(cls, payload, device, mask=None):
        value = cls(0, device)
        for key in ['gradient_sum', 'observations', 'last_growth']:
            tensor = payload[key]
            if mask is not None:
                tensor = tensor[torch.as_tensor(mask, device=tensor.device, dtype=torch.bool)]
            setattr(value, key, tensor.to(device))
        for key in ['net_added', 'created', 'cloned', 'split_parents', 'last_event', 'history']:
            setattr(value, key, payload[key])
        value.check(len(value.gradient_sum))
        if (value.net_added != value.cloned + value.split_parents
                or value.created != value.cloned + 2 * value.split_parents
                or value.net_added != sum(item['net_added'] for item in value.history)
                or value.created != sum(item['created'] for item in value.history)):
            raise RuntimeError('Invalid narrow growth cumulative ledger')
        if any(getattr(value, key) < 0 for key in ['net_added', 'created', 'cloned', 'split_parents']):
            raise RuntimeError('Negative narrow growth cumulative count')
        if (not bool(torch.isfinite(value.gradient_sum).all())
                or not bool(torch.isfinite(value.observations).all())
                or bool((value.gradient_sum < 0).any()) or bool((value.observations < 0).any())):
            raise RuntimeError('Invalid narrow growth gradient statistics')
        return value


def growth_budget(config, reference, iteration, net_added):
    get = lambda key: config['field_narrow_birth_growth_' + key]
    start, middle, until, interval = [get(key) for key in ['start', 'phase1_until', 'until', 'interval']]
    if not start <= iteration <= until or (iteration - start) % interval:
        return None
    total = min(int(reference * get('total_ratio')), get('total_cap'))
    event = min(int(reference * get('event_ratio')), get('event_cap'))
    phase = 1 if iteration <= middle else 2
    limit = int(total * get('phase1_budget_ratio')) if phase == 1 else total
    slots = ((middle if phase == 1 else until) - iteration) // interval + 1
    remaining = max(0, min(total - net_added, limit - net_added))
    return dict(phase=phase, allowance=min(event, remaining // slots), event_cap=event,
                phase_budget_limit=limit, remaining_phase_slots=slots,
                remaining_before=max(0, total - net_added), cumulative_limit=total)


@torch.no_grad()
def observe_growth(model, iteration, gradient, visibility, kernel, opacity):
    config = model._narrow_birth_config
    if (not config.get('field_narrow_birth_growth', 0) or gradient is None
            or iteration < config['field_narrow_birth_start']
            or iteration > config['field_narrow_birth_growth_until']):
        return
    state = model._narrow_birth_state
    if state is None or state.growth is None:
        raise RuntimeError('Narrow growth observation has no identity state')
    count = len(model.get_xyz)
    state.check(count)
    norm = torch.linalg.vector_norm(gradient.detach()[:, :2], dim=1)
    valid = ((state.born >= 0) & visibility.detach().reshape(-1).bool()
             & (kernel.detach().reshape(-1) >= config['field_narrow_birth_growth_kernel_min'])
             & (opacity.detach().reshape(-1) >= 1 / 255)
             & torch.isfinite(norm))
    # Only visible, temporally supported observations enter the denominator.
    state.growth.gradient_sum[valid] += norm[valid]
    state.growth.observations[valid] += 1


@torch.no_grad()
def grow_narrow(controller, iteration):
    model, state, config = controller.model, controller.state, controller.config
    if not config.get('field_narrow_birth_growth', 0):
        return None
    growth = state.growth
    info = growth_budget(config, state.reference, iteration, growth.net_added)
    if info is None:
        return None
    if state.reference <= 0 or iteration <= growth.last_event:
        raise RuntimeError('Narrow growth reference/clock is invalid')
    torch.cuda.synchronize()
    begin = time.perf_counter()
    from helper_narrow_birth import validate_model
    validate_model(model)
    state.check(len(model.get_xyz))
    get = lambda key: config['field_narrow_birth_growth_' + key]
    score = growth.gradient_sum / growth.observations.clamp_min(1)
    eligible = ((state.born >= 0) & (iteration - state.born >= get('min_age'))
                & (iteration - growth.last_growth >= get('cooldown'))
                & (growth.observations >= get('min_observations'))
                & (score >= get('grad_threshold')) & torch.isfinite(score)
                & (model.get_opacity.detach().reshape(-1) >= config['field_narrow_birth_cleanup_opacity']))
    scales = model.get_scaling.detach()
    eligible &= torch.isfinite(scales).all(dim=1) & (scales > 0).all(dim=1)
    indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    selected = indices[score[indices].argsort(descending=True, stable=True)[:info['allowance']]]
    split = scales[selected].amax(dim=1) > model.percent_dense * controller.extent
    clone_parents, split_parents = selected[~split], selected[split]
    clones, splits = len(clone_parents), len(split_parents)
    parents = torch.cat((clone_parents, split_parents.repeat_interleave(2)))
    before = len(model.get_xyz)
    record = dict(info, iteration=int(iteration), eligible=len(indices), selected=len(selected),
                  cloned=clones, split_parents=splits, created=len(parents), net_added=len(selected),
                  points_before=before, parent_ids=state.ids[selected].cpu().tolist())
    if len(selected):
        xyz = model.get_xyz.detach()[parents].clone()
        raw_scale = model._scaling.detach()[parents].clone()
        if splits:
            from utils.general_utils import build_rotation
            repeated = split_parents.repeat_interleave(2)
            # Local event RNG: new split sampling does not advance training RNG.
            generator = torch.Generator(device=xyz.device)
            generator.manual_seed(get('seed') + int(iteration))
            noise = torch.randn((len(repeated), 3), device=xyz.device, dtype=xyz.dtype, generator=generator)
            samples = noise * scales[repeated]
            rotations = build_rotation(model._rotation.detach()[repeated])
            xyz[clones:] += torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1)
            raw_scale[clones:] = torch.log(scales[repeated] / 1.6)
        archive = controller.log.parent / 'narrow_growth' / ('iteration_{}.npz'.format(iteration))
        if archive.exists():
            raise FileExistsError('Refusing to overwrite narrow growth archive: ' + str(archive))
        selected_ids = state.ids[selected].detach().cpu().numpy()
        split_ids = state.ids[split_parents].detach().cpu().numpy()
        parent_ids = state.ids[parents].detach().cpu().numpy()
        centers = model._trbf_center[parents].detach().cpu().numpy()
        widths = model._trbf_scale[parents].detach().exp().cpu().numpy()
        motion = model._motion[parents].detach().cpu().numpy()
        anchors = model._motion_time_anchor[parents].detach().cpu().numpy()
        omega = model._omega[parents].detach().cpu().numpy()
        rotations_raw = model._rotation[parents].detach().cpu().numpy()
        # Preserve learned temporal support, polynomial, anchor and raw opacity.
        # Do NOT apply effective-opacity inheritance again: centers are unchanged.
        model.densification_postfix(
            new_xyz=xyz, new_features_dc=model._features_dc[parents].detach(),
            new_opacities=model._opacity[parents].detach(), new_scaling=raw_scale,
            new_rotation=model._rotation[parents].detach(),
            new_trbf_center=model._trbf_center[parents].detach(),
            new_trbfscale=model._trbf_scale[parents].detach(),
            new_motion=model._motion[parents].detach(), new_omega=model._omega[parents].detach(),
            new_featuret=model._features_t[parents].detach(),
            new_static_level_logits=model._static_level_logits[parents].detach() if model.use_euler_field else None,
            new_motion_time_anchor=model._motion_time_anchor[parents].detach(),
            new_existence_parent_indices=parents, narrow_growth_parents=parents,
            narrow_growth_iteration=iteration)
        growth.last_growth[selected] = iteration
        child_ids = state.ids[before:].detach().cpu().numpy()
        family_ids = state.family[before:].detach().cpu().numpy()
        remove = torch.zeros(len(model.get_xyz), device=xyz.device, dtype=torch.bool)
        remove[split_parents] = True
        if splits:
            model.prune_points(remove)
        archive.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(archive, parent_ids=parent_ids, selected_parent_ids=selected_ids,
                            child_ids=child_ids, families=family_ids,
                            split_parent_ids=split_ids, center=centers, width=widths,
                            motion=motion, anchor=anchors, omega=omega, rotation=rotations_raw,
                            xyz=xyz.cpu().numpy(), scale=raw_scale.exp().cpu().numpy())
        record['child_ids'] = child_ids.tolist()
        record['archive'] = str(archive.relative_to(controller.log.parent))
    if len(model.get_xyz) != before + clones + splits:
        raise RuntimeError('Narrow spatial refinement net-growth mismatch')
    growth.net_added += clones + splits
    growth.created += clones + 2 * splits
    growth.cloned += clones
    growth.split_parents += splits
    growth.history.append(dict(iteration=int(iteration), net_added=clones + splits, created=clones + 2 * splits))
    growth.last_event = iteration
    # A 500-step decision clock is not 500 valid observations. Keep sparse
    # temporal samples until enough exist to evaluate a point; reset evaluated
    # points so old high gradients cannot drive repeated growth indefinitely.
    evaluated = growth.observations >= get('min_observations')
    growth.gradient_sum[evaluated] = 0
    growth.observations[evaluated] = 0
    state.check(len(model.get_xyz))
    torch.cuda.synchronize()
    record.update(total_net_added=growth.net_added, total_created=growth.created,
                  points_after=len(model.get_xyz), alive_narrow=int((state.born >= 0).sum()),
                  under_sampled_retained=int(((growth.observations > 0) & ~evaluated).sum()),
                  seconds=time.perf_counter() - begin)
    with (controller.log.parent / 'narrow_growth_events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')
    print('[STEGF][NarrowGrowth] iter={} eligible={} clone={} split={} net={} total_net={} points={}'.format(
        iteration, len(indices), clones, splits, clones + splits, growth.net_added, len(model.get_xyz)), flush=True)
    return record
