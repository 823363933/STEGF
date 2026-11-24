"""Opt-in, unlimited multi-view refinement; no change to legacy policies."""

import math
import random

import torch


class RefinementBirthState:
    """Birth ages for every row, not a narrow-Gaussian identity or growth budget."""

    def __init__(self, count, device):
        self.born = torch.zeros(count, dtype=torch.int32, device=device)
        self.revision = 0

    def check(self, count):
        if self.born.ndim != 1 or len(self.born) != count:
            raise RuntimeError("Multi-view refinement birth rows are misaligned")

    def append(self, count, iteration):
        if count:
            self.born = torch.cat((self.born, torch.full(
                (count,), int(iteration), dtype=torch.int32, device=self.born.device)))
            self.revision += 1

    def prune(self, keep):
        self.check(len(keep))
        self.born = self.born[keep]
        self.revision += 1

    def snapshot(self, iteration):
        return dict(born=self.born.detach().cpu(), revision=self.revision,
                    iteration=int(iteration))

    @classmethod
    def load(cls, payload, device, count, mask=None):
        if payload is None:
            raise RuntimeError("Unlimited refinement checkpoint is missing birth ages")
        born = payload['born'].to(device=device, dtype=torch.int32).reshape(-1)
        if mask is not None:
            born = born[torch.as_tensor(mask, device=device, dtype=torch.bool)]
        state = cls(0, device)
        state.born = born
        state.revision = int(payload['revision'])
        state.check(count)
        return state


class ShuffledTimeSampler:
    """Each cycle visits every training time once; never uses test losses."""

    def __init__(self, duration):
        if duration < 1:
            raise ValueError("Time sampler requires a positive duration")
        self.duration = int(duration)
        self.remaining = []
        self.cycles = 0

    def sample(self):
        if not self.remaining:
            self.remaining = list(range(self.duration))
            random.shuffle(self.remaining)
            self.cycles += 1
        return self.remaining.pop()


def refinement_schedule(opt):
    def integer(name, minimum=1):
        value = getattr(opt, name)
        number = int(value)
        if isinstance(value, bool) or number < minimum or number != value:
            raise ValueError(name + " must be an integer >= " + str(minimum))
        return number

    if not opt.field_late_prune or not opt.field_tail_lr_decay:
        raise ValueError("Unlimited refinement requires independent pruning and LR decay")
    start = integer('field_late_prune_start')
    stop = integer('field_late_prune_until')
    interval = integer('field_late_prune_interval')
    age = integer('field_late_prune_min_birth_age')
    join = integer('field_refinement_xyz_join')
    until = integer('field_refinement_xyz_until')
    lr_start = integer('field_tail_lr_start')
    lr_until = integer('field_tail_lr_until')
    factor = float(opt.field_tail_lr_final_factor)
    threshold = float(opt.opthr)
    if (start >= stop or (stop - start) % interval
            or start < int(opt.densify_until_iter) or stop >= lr_until
            or not 0 < join < int(opt.position_lr_max_steps) < until
            or not join < lr_start < until <= stop
            or not math.isfinite(factor) or not 0 < factor <= 1
            or not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError("Invalid unlimited-refinement clocks, threshold or LR factor")
    return dict(unlimited=True,
                prune=dict(start=start, until=stop, interval=interval,
                           min_birth_age=age, threshold=threshold),
                lr=dict(start=lr_start, until=lr_until, final_factor=factor),
                xyz=dict(join=join, until=until, old_until=int(opt.position_lr_max_steps)))


def validate_refinement(schedule, opt, model, densify):
    required = (densify == 1 and model.field_mvstruct_unlimited
                and model.field_mvstruct and model.field_mvstruct_densify
                and model.field_mvstruct_time_sampling == 'shuffle_cycle'
                and model.field_motion_model == 'couptest_polynomial'
                and model.field_couptest_mode == 'coupled_detached'
                and model.field_existence_single_expert == 'transient'
                and model.field_disable_global_omega_split
                and model.field_disable_legacy_aux and model.field_disable_ems_main)
    if not required:
        raise ValueError("Unlimited refinement requires the opt-in N3D detached model")
    for name in ('field_soft_geometry_lr', 'field_appearance_only_train',
                 'field_bg_dense_add', 'field_bg_prior', 'field_temporal_refine',
                 'field_carrier_initialization', 'field_existence_moe',
                 'field_mvstruct_oversize_split', 'field_mvstruct_conflict_split',
                 'field_mvstruct_hard_time', 'field_staged_training', 'field_global_reset'):
        if getattr(model, name, False):
            raise ValueError("Unlimited refinement is incompatible with " + name)
    if (model._narrow_birth_config.get('field_narrow_birth', 0)
            or getattr(opt, 'field_n3d_quality_schedule', 0)
            or getattr(opt, 'field_n3d_control_time_scale', 1) != 1):
        raise ValueError("Unlimited refinement preserves the original ordinary clock")
    if not (model.field_mvstruct_start < model.field_mvstruct_densify_start
            <= model.field_mvstruct_densify_until == model.field_mvstruct_until
            == schedule['lr']['start'] < schedule['xyz']['until']):
        raise ValueError("Refinement growth must finish before the XYZ terminal rate")
    if model.field_mvstruct_interval < 1 or model.field_mvstruct_densify_interval < 1:
        raise ValueError("Refinement event intervals must be positive")
    if ((model.field_mvstruct_until - model.field_mvstruct_start) % model.field_mvstruct_interval
            or (model.field_mvstruct_densify_until - model.field_mvstruct_densify_start)
            % model.field_mvstruct_densify_interval
            or (model.field_mvstruct_densify_start - model.field_mvstruct_start)
            % model.field_mvstruct_interval
            or model.field_mvstruct_densify_interval % model.field_mvstruct_interval):
        raise ValueError("Refinement windows must end on their event boundaries")


def apply_learning_rate(schedule, model, iteration):
    model._refinement_iteration = int(iteration)
    model._refinement_topology_until = schedule['prune']['until']
    progress = min(max((iteration - schedule['lr']['start']) /
                       (schedule['lr']['until'] - schedule['lr']['start']), 0.), 1.)
    factor = math.exp(math.log(schedule['lr']['final_factor']) * progress)
    clock = schedule['xyz']
    if iteration > clock['join']:
        fraction = min((iteration - clock['join']) / (clock['until'] - clock['join']), 1.)
        effective_step = clock['join'] + fraction * (clock['old_until'] - clock['join'])
    else:
        effective_step = iteration
    for group in model.optimizer.param_groups:
        if group['name'] == 'xyz':
            group['lr'] = model.xyz_scheduler_args(effective_step)
        else:
            # Immutable base rates prevent compounded decay. Disabled groups stay off.
            base = group['_stegf_base_lr']
            group['lr'] = base * factor if any(p.requires_grad for p in group['params']) else 0.
    return factor


@torch.no_grad()
def prune_mature_points(schedule, model, iteration):
    clock = schedule['prune']
    if not (clock['start'] <= iteration <= clock['until']
            and (iteration - clock['start']) % clock['interval'] == 0):
        return None
    state = model._refinement_birth_state
    if state is None:
        raise RuntimeError("Missing universal refinement birth state")
    count = len(model.get_xyz)
    state.check(count)
    opacity = model.get_opacity.detach().reshape(-1)
    if len(opacity) != count:
        raise RuntimeError("Refinement opacity rows are misaligned")
    low = torch.isfinite(opacity) & (opacity < clock['threshold'])
    young = iteration - state.born < clock['min_birth_age']
    mask = low & ~young
    removed = int(mask.sum().item())
    if removed and removed == count:
        raise RuntimeError("Low-opacity pruning would delete the entire scene")
    if removed:
        model.prune_points(mask)
    if len(model.get_xyz) != count - removed:
        raise RuntimeError("Refinement pruning produced an unexpected point count")
    cohorts, counts = torch.unique(state.born, return_counts=True)
    return dict(iteration=int(iteration), threshold=clock['threshold'],
                points_before=count, points_after=len(model.get_xyz),
                low_opacity_candidates=int(low.sum().item()),
                young_protected=int((low & young).sum().item()),
                young_narrow_protected=int((low & young).sum().item()),
                background_protected=0, pruned=removed,
                surviving_birth_cohorts=dict(zip(
                    map(str, cohorts.cpu().tolist()), counts.cpu().tolist())))


def check_fixed_topology(schedule, model, iteration):
    if iteration <= schedule['prune']['until']:
        return False
    state = model._refinement_birth_state
    state.check(len(model.get_xyz))
    signature = (len(model.get_xyz), state.revision)
    previous = getattr(model, '_refinement_fixed_topology_signature', None)
    if previous is None:
        model._refinement_fixed_topology_signature = signature
        return True
    if signature != previous:
        raise RuntimeError("Topology changed inside the fixed refinement tail")
    return False
