"""
Flat, whole-batch-as-one-group counterpart to offpolicy_trainer.OffPolicyTribeTrainer — a DEGENERATE-DATASET
WORKAROUND, not TRIBE's actual method. New, additive module: offpolicy_trainer.py and every existing
trainer are untouched.

TRIBE's real method is group-relative: Stage 1's advantage `A = reward - group_mean` needs genuine
within-group reward variance (a mix of correct/incorrect completions to the SAME question) to produce
anything but a degenerate rho*=1 (see tribe.stage1's own docstring, and this project's own investigation of
PRM800K/OpenR1-Math-220k, where most per-question groups turned out to be homogeneous — all-correct or
all-incorrect — leaving Stage 1 nothing to react to). Passing `group_size = <the whole gathered batch>`
instead of `self.args.group_size` (the per-question sub-group size) to the SAME, unmodified
`compute_rho_star_offpolicy` every step sidesteps that requirement entirely: the "group mean" becomes a
single per-batch constant, which — because the normalization constraint forces E[rho]=1 exactly — is
provably a constant term in Stage 1's own objective and therefore doesn't change Stage 1's argmax at all
(verified both algebraically and numerically: max abs diff vs. plugging in raw reward directly is 2.2e-16,
machine precision). So this trainer is mathematically EXACTLY equivalent to replacing Stage 1's
advantage-based objective with a raw-reward objective, re-solved fresh every step exactly like
OffPolicyTribeTrainer's own per-batch solve — a genuinely different regime (closer to trust-region-constrained
REINFORCE on raw reward than to GRPO/RLOO-style group-relative advantage), useful specifically when a
dataset's own per-question sample composition can't supply real within-group variance.

Matches OffPolicyTribeTrainer's established per-batch convention exactly (same trust region re-solved every
step from that step's own gathered rewards, not a one-time whole-dataset solve) — this is a full duplicate
of that class (per this project's own duplication-over-abstraction convention: trainers are self-contained,
shared logic is copied not abstracted), with two substitutions in compute_loss, both the same idea applied
consistently: `self.args.group_size` -> `rewards_global.numel()` (the whole gathered batch) wherever the
original assumed a per-question sub-group — once for Stage 1's own solve (see above), and once more for
negative_fraction's own per-group ranked-selection (_negative_sample_weight_flat below, a batch-as-one-group
counterpart to OffPolicyTrainer._negative_sample_weight_global, which reshapes via the ORIGINAL
self.args.group_size and so can't be reused directly here). Every other line — loss branches, logging — is
identical.

Use TRIBE's actual group-relative trainers when real within-group diversity exists — this flat variant is a
fallback for when it doesn't.
"""

import math
import warnings

import torch
from torch.utils.data import RandomSampler

from offpolicy_trainer import OffPolicyTrainer, _PolyakEMACallback
from tribe.offpolicy_stage1 import compute_rho_star_offpolicy


def _negative_sample_weight_flat(is_negative: torch.Tensor, negative_fraction: float, seed: int) -> torch.Tensor:
    """Batch-as-one-group counterpart to OffPolicyTrainer._negative_sample_weight_global — identical
    per-group ranked-selection mechanics (see that method's own docstring for the full negative_fraction
    contract: positives always weight 1, negatives get a weight built from negative_fraction's integer and
    fractional parts so the *expected* weight of each negative is exactly negative_fraction), just with the
    whole batch as the single group (no `.view(G, group_size)` reshape needed — G=1 already) instead of
    self.args.group_size-sized per-question sub-groups, matching this trainer's own Stage 1 substitution."""
    if negative_fraction == 1.0:
        return is_negative.new_ones(is_negative.shape, dtype=torch.float)
    floor_f = math.floor(negative_fraction)
    frac = negative_fraction - floor_f
    weight = torch.where(
        is_negative,
        torch.full_like(is_negative, float(floor_f), dtype=torch.float),
        torch.ones_like(is_negative, dtype=torch.float),
    )
    if frac > 0:
        n_neg = is_negative.sum()
        n_bonus = (n_neg.float() * frac).round().long()
        gen = torch.Generator(device=is_negative.device).manual_seed(seed)
        rand_vals = torch.rand(is_negative.shape, generator=gen, device=is_negative.device)
        rand_vals = rand_vals.masked_fill(~is_negative, float("inf"))
        rank = rand_vals.argsort().argsort()
        bonus = is_negative & (rank < n_bonus)
        weight = weight + bonus.float()
    return weight


class OffPolicyTribeFlatTrainer(OffPolicyTrainer):
    """See this module's own docstring for the full rationale — full duplicate of OffPolicyTribeTrainer
    with one substitution in compute_loss (group_size = the whole gathered batch, not self.args.group_size)."""

    def __init__(
        self,
        model,
        args,
        train_dataset,
        processing_class,
        beta: float = 0.1,
        trust_region_eps: float = 0.05,
        use_unlikelihood: bool = False,
        divergence: str = "kl_new_old",
        use_polyak_ema: bool = False,
        polyak_ema_tau: float = 0.99,
        rho_weight_offset: float = 1.0,
        negative_fraction: float = 1.0,
        min_pos_neg_ratio: float | None = None,
        negative_logp_floor: float = math.log(1e-8),
        negative_switch_threshold: float | None = None,
        ref_model_name_or_path: str | None = None,
        **kwargs,
    ):
        super().__init__(model, args, train_dataset, processing_class, **kwargs)
        self.beta = beta
        self.trust_region_eps = trust_region_eps
        self.divergence = divergence
        self.negative_logp_floor = negative_logp_floor
        self.rho_weight_offset = rho_weight_offset
        # Mutually exclusive with negative_fraction — see OffPolicyTrainer._pos_neg_floor_sample_weight's
        # own docstring for why they're two different gating modes, not stackable.
        if min_pos_neg_ratio is not None and negative_fraction != 1.0:
            raise ValueError(
                "min_pos_neg_ratio and negative_fraction are mutually exclusive — leave negative_fraction "
                "at its default (1.0) when setting min_pos_neg_ratio."
            )
        self.min_pos_neg_ratio = min_pos_neg_ratio
        self.negative_fraction = negative_fraction
        self.use_unlikelihood = use_unlikelihood
        self.negative_switch_threshold = negative_switch_threshold
        if negative_switch_threshold is not None:
            assert not use_unlikelihood, "negative_switch_threshold and use_unlikelihood are mutually exclusive"
        self.mu_model = self._prepare_frozen_ref_model(ref_model_name_or_path)
        self._warned_f_div_mu_overflow = False
        self._stage1_inaccurate_count = 0
        self._stage1_skip_count = 0
        self.use_polyak_ema = use_polyak_ema
        if use_polyak_ema:
            self.add_callback(_PolyakEMACallback(polyak_ema_tau))

    def _get_train_sampler(self, train_dataset=None):
        # OffPolicyTrainer's own default sampler (_GroupShuffledSampler) shuffles self.args.group_size-sized
        # blocks to keep each question's group contiguous within a batch — irrelevant here, since this
        # trainer treats whatever's in a batch as one flat group regardless of composition (no group
        # boundary to preserve), and would otherwise force the dataset length to be divisible by whatever
        # args.group_size happens to be set to for no reason. A plain per-example shuffle is both correct
        # and simpler.
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        return RandomSampler(dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        rewards_global = self._gather_grouped(inputs["reward"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # The one substitution vs. OffPolicyTribeTrainer.compute_loss: group_size = the whole gathered
            # batch (rewards_global.numel()), not self.args.group_size — see class/module docstring.
            rho_global, lam = compute_rho_star_offpolicy(
                rewards_global, rewards_global.numel(), self.beta, self.trust_region_eps, divergence=self.divergence
            )
        if any("optimal_inaccurate" in str(w.message) for w in caught):
            self._stage1_inaccurate_count += 1
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["tribe/stage1_inaccurate_total"].append(self._stage1_inaccurate_count)
        if rho_global is None:
            self._stage1_skip_count += 1
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["tribe/stage1_skipped_total"].append(self._stage1_skip_count)
            loss = 0.0 * per_token_logps.sum()
            return (loss, None) if return_outputs else loss
        weight_global = rho_global - self.rho_weight_offset

        # The other substitution vs. OffPolicyTribeTrainer.compute_loss (see module docstring): the
        # batch-as-one-group counterpart, not the original self.args.group_size-based helper. Note
        # min_pos_neg_ratio's own gate (_pos_neg_floor_sample_weight) needs NO such substitution at all —
        # it's already whole-batch, unmodified, shared by every TRIBE trainer.
        if self.min_pos_neg_ratio is not None:
            sample_weight_global = self._pos_neg_floor_sample_weight(
                weight_global < 0, self.min_pos_neg_ratio, seed=self.state.global_step
            )
        else:
            sample_weight_global = _negative_sample_weight_flat(
                weight_global < 0, self.negative_fraction, seed=self.state.global_step
            )

        weight = self._local_slice(weight_global, inputs["reward"].numel())
        sample_weight = self._local_slice(sample_weight_global, inputs["reward"].numel())

        if self.negative_switch_threshold is not None:
            positive_mask = (weight >= 0).float().unsqueeze(-1)
            negative_mask = (weight < 0).float().unsqueeze(-1)
            positive_loss = -weight.clamp(min=0).unsqueeze(-1) * per_token_logps
            p = per_token_logps.float().exp().clamp(max=1 - 1e-6)
            high_p_loss = -per_token_logps.float()
            low_p_loss = -torch.log1p(-p)
            switch_mask = (p > self.negative_switch_threshold).float()
            negative_token_loss = switch_mask * high_p_loss + (1 - switch_mask) * low_p_loss
            negative_loss = weight.clamp(max=0).abs().unsqueeze(-1) * negative_token_loss
            per_token_loss = (positive_loss * positive_mask + negative_loss * negative_mask) * completion_mask
        elif self.use_unlikelihood:
            positive_mask = (weight >= 0).float().unsqueeze(-1)
            negative_mask = (weight < 0).float().unsqueeze(-1)
            positive_loss = -weight.clamp(min=0).unsqueeze(-1) * per_token_logps
            p = per_token_logps.float().exp().clamp(max=1 - 1e-6)
            unlikelihood = -torch.log1p(-p)
            negative_loss = weight.clamp(max=0).abs().unsqueeze(-1) * unlikelihood
            per_token_loss = (positive_loss * positive_mask + negative_loss * negative_mask) * completion_mask
        else:
            per_token_logps_floored = per_token_logps.clamp(min=self.negative_logp_floor)
            per_token_loss = -weight.unsqueeze(-1) * per_token_logps_floored * completion_mask

        per_seq_loss = per_token_loss.sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        loss = (per_seq_loss * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["tribe/lambda"].append(lam.item())
        self._metrics[mode]["tribe/rho_star_mean"].append(rho_global.mean().item())
        self._metrics[mode]["tribe/rho_star_std"].append(rho_global.std().item())
        self._metrics[mode]["tribe/rho_star_min"].append(rho_global.min().item())
        self._metrics[mode]["tribe/rho_star_max"].append(rho_global.max().item())

        # pos:neg ratio AFTER negative_fraction gating — i.e., counting only what actually carries nonzero
        # weight into the gradient this step, not the raw batch composition. Positives (weight>=0) always
        # carry sample_weight=1 by construction (only negatives ever get gated), so n_positive is just their
        # count; n_negative_kept sums sample_weight over weight<0 rows (not just a boolean count) since
        # negative_fraction's ranked selection can bump a negative's weight above 1 for negative_fraction>1
        # (see _negative_sample_weight_flat's own docstring) — summing the weight itself is the honest
        # "how many negative-example-equivalents actually contributed" figure, not just how many were
        # nonzero. Logged for debugging (this project's own PRM800K investigation kept mis-tracking this by
        # eye from training logs, see prior debugging in this conversation).
        is_negative_global = weight_global < 0
        n_positive_global = (~is_negative_global).sum().item()
        n_negative_kept_global = sample_weight_global[is_negative_global].sum().item()
        self._metrics[mode]["tribe/n_positive"].append(n_positive_global)
        self._metrics[mode]["tribe/n_negative_kept"].append(n_negative_kept_global)
        self._metrics[mode]["tribe/pos_neg_kept_ratio"].append(
            n_positive_global / max(n_negative_kept_global, 1e-8)
        )

        if self.state.global_step % max(int(self.args.logging_steps), 1) == 0:
            with torch.no_grad():
                mu_per_token_logps = self._get_per_token_logps(self.mu_model, input_ids, attention_mask, logits_to_keep)
                seq_logps_theta = (per_token_logps.detach().float() * completion_mask).sum(-1)
                seq_logps_mu = (mu_per_token_logps.float() * completion_mask).sum(-1)
                rho_theta = torch.exp(seq_logps_theta - seq_logps_mu)
                from tribe.stage1 import TRUST_REGION_F

                f_val = TRUST_REGION_F["chi_squared"](rho_theta)
            if not self._warned_f_div_mu_overflow and torch.isinf(f_val).any():
                warnings.warn(
                    "tribe/f_div_mu overflowed to inf at step "
                    f"{self.state.global_step}: pi_theta has diverged from the frozen mu enough for exp() to "
                    "overflow float range. This metric will read as null/None in wandb from here on.",
                    stacklevel=2,
                )
                self._warned_f_div_mu_overflow = True
            self._metrics[mode]["tribe/f_div_mu"].append(self._gather_grouped(f_val).mean().item())

        self._log_logp_stats(per_token_logps, completion_mask, prefix="tribe")
        self._log_reward_metric(inputs["reward"])

        return (loss, None) if return_outputs else loss
