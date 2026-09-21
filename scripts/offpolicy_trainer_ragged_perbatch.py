"""
Per-batch counterpart to offpolicy_trainer_ragged.OffPolicyTribeGlobalRaggedTrainer — solves Stage 1 fresh
each step (like OffPolicyTribeTrainer, scripts/offpolicy_trainer.py) instead of once over the whole dataset.
New, additive module: offpolicy_trainer.py and offpolicy_trainer_ragged.py are untouched.

Why this needs its own machinery instead of just reusing OffPolicyTribeTrainer with ragged group_sizes:
that trainer's sampler (_GroupShuffledSampler) and DDP-gather reshape (`.view(G, group_size)`, in both
_gather_grouped's callers and its own compute_loss) hard-assume every group has the SAME size, so every
step's batch — after gathering across ranks — splits cleanly into equal-size blocks. PRM800K's groups
genuinely vary (1 to 467 rollouts per problem), so that assumption breaks.

Rather than building a new distributed batch-sampler from scratch (real complexity: packing variable-size
groups into a DataLoader batch, keeping ranks in sync, no group split across ranks), this reuses the
EXISTING, battle-tested _GroupShuffledSampler/DDP-gather machinery unchanged by first turning every group
into a fixed-size `max_group_size` block: every positive is kept (rare — see
scripts/convert_prm800k_offpolicy.py's own --group_size mode for the same "never drop a positive"
rationale), negatives are randomly subsampled down to fit if there are more than needed, and the block is
padded out to exactly max_group_size with dummy rows if the (capped) group has fewer real rows than that.
Padding rows carry a trivial completion with completion_mask forced to 0 by the collator below (same
mechanism as mask_truncated_completions's own all-zero-mask pattern) — one cheap extra forward pass per
padding row, but they contribute exactly nothing to the loss and are excluded from Stage 1's own
advantage/rho* computation entirely (real group sizes, not the padded block size, are what get passed to
compute_rho_star_offpolicy_ragged).
"""

import math
import random
import warnings

import torch
from torch import Tensor

from offpolicy_trainer import OffPolicyTrainer, _GroupShuffledSampler, _OffPolicyCollator
from tribe.offpolicy_stage1_ragged import compute_rho_star_offpolicy_ragged
from tribe.stage1 import TRUST_REGION_F


def pad_and_cap_ragged_dataset(dataset, max_group_size: int, seed: int):
    """See this module's own docstring for the full rationale. Returns a new flat dataset (no more
    'group_id' column — replaced by fixed-size `max_group_size` blocks, laid out contiguously per group in
    dataset row order, one block per original group) with an added 'is_real' column (False for padding
    rows). Groups whose OWN positive count alone exceeds max_group_size (not expected to occur at this
    dataset's ~9% overall positive rate, but not impossible) fall back to capping positives too, since a
    block can never exceed max_group_size regardless.

    Args:
        dataset (`Dataset`):
            Must have a 'group_id' column (e.g. scripts/convert_prm800k_offpolicy.py's --ragged output) —
            one integer per row, identical within a question's group, laid out in contiguous per-group
            blocks.
        max_group_size (`int`):
            Fixed block size every group gets padded/capped to. Should match the trainer's own
            per-step-gathered group size (`per_device_train_batch_size * num_processes`, matching
            OffPolicyTribeTrainer's own "one full group per step" convention).
        seed (`int`):
            Random seed for the negative subsample.
    """
    if "group_id" not in dataset.column_names:
        raise ValueError("pad_and_cap_ragged_dataset requires a 'group_id' column (see docstring).")
    rng = random.Random(seed)
    group_ids = dataset["group_id"]
    rewards = dataset["reward"]

    groups = {}
    prev = object()
    for i, gid in enumerate(group_ids):
        if gid != prev:
            groups[gid] = []
            prev = gid
        groups[gid].append(i)

    rows = list(dataset)
    n_capped_positives = 0
    out_rows = []
    for idx_list in groups.values():
        positives = [i for i in idx_list if rewards[i] == 1.0]
        negatives = [i for i in idx_list if rewards[i] == 0.0]
        if len(positives) > max_group_size:
            n_capped_positives += 1
            positives = rng.sample(positives, max_group_size)
        n_neg_slots = max_group_size - len(positives)
        if len(negatives) > n_neg_slots:
            negatives = rng.sample(negatives, n_neg_slots)
        real_idx = positives + negatives
        for i in real_idx:
            row = dict(rows[i])
            row["is_real"] = True
            out_rows.append(row)
        n_pad = max_group_size - len(real_idx)
        if n_pad > 0:
            # Padding rows reuse a real row's own (prompt, completion) — any valid, tokenizable text works
            # since completion_mask forces their loss contribution to exactly 0 regardless of content; reward
            # is irrelevant too (never read for is_real=False rows) but set to 0.0 for a well-typed column.
            template = rows[real_idx[0]]
            for _ in range(n_pad):
                row = dict(template)
                row["reward"] = 0.0
                row["is_real"] = False
                out_rows.append(row)

    if n_capped_positives:
        warnings.warn(
            f"{n_capped_positives} group(s) had more positives than max_group_size={max_group_size} — "
            "capped positives too (randomly subsampled), since a block can never exceed max_group_size.",
            stacklevel=2,
        )

    from datasets import Dataset

    out = Dataset.from_list(out_rows)
    if "group_id" in out.column_names:
        out = out.remove_columns("group_id")
    return out


class _OffPolicyRealMaskCollator(_OffPolicyCollator):
    """Same as _OffPolicyCollator, plus zeroing completion_mask for padding rows (is_real=False) — same
    mechanism/rationale as mask_truncated_completions's own all-zero-mask pattern, applied here
    unconditionally to padding rows regardless of that flag."""

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        batch = super().__call__(examples)
        is_real = torch.tensor([example["is_real"] for example in examples], dtype=torch.bool)
        batch["completion_mask"] = batch["completion_mask"] * is_real.unsqueeze(-1).long()
        batch["is_real"] = is_real
        return batch


def _negative_sample_weight_ragged_perstep(
    is_negative: Tensor, group_sizes: list[int], negative_fraction: float, seed: int
) -> Tensor:
    """Per-step-reseeded counterpart to offpolicy_trainer_ragged._negative_sample_weight_ragged — identical
    per-group ranked-selection mechanics (see that function's docstring for the full negative_fraction
    contract), just reseeded fresh every call instead of once in __init__, matching
    OffPolicyTrainer._negative_sample_weight_global's own per-training-step reseeding (via
    self.state.global_step) rather than a single fixed seed. Kept as a separate function (not a shared
    helper) because the two callers' seeding lifecycles are genuinely different — see this module's and
    offpolicy_trainer_ragged.py's own docstrings."""
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
        gen = torch.Generator(device=is_negative.device).manual_seed(seed)
        bonus_parts = []
        for g in torch.split(is_negative, group_sizes):
            n_neg = g.sum()
            n_bonus = (n_neg.float() * frac).round().long()
            rand_vals = torch.rand(g.shape, generator=gen, device=g.device)
            rand_vals = rand_vals.masked_fill(~g, float("inf"))
            rank = rand_vals.argsort().argsort()
            bonus_parts.append(g & (rank < n_bonus))
        weight = weight + torch.cat(bonus_parts).float()
    return weight


class OffPolicyTribeRaggedTrainer(OffPolicyTrainer):
    """
    Per-batch ragged-group TRIBE — see this module's own docstring for why the dataset must first be
    padded/capped to fixed `max_group_size` blocks (pad_and_cap_ragged_dataset) rather than trained on
    directly. Otherwise a straight duplicate of OffPolicyTribeTrainer (scripts/offpolicy_trainer.py) — same
    Stage 1 solve-every-step, same Stage 2 loss branches (floored NLL / unlikelihood / negative_switch) —
    with two differences, both purely about excluding padding rows from Stage 1's own math:
      - compute_rho_star_offpolicy_ragged (variable per-block REAL sizes) instead of compute_rho_star_offpolicy
        (fixed group_size) — real sizes come from each block's own is_real count, not max_group_size.
      - _negative_sample_weight_ragged_perstep instead of _negative_sample_weight_global — same reasoning.
    Padding rows still flow through the forward pass (needed for a rectangular batch) but contribute
    exactly 0 to the loss via completion_mask, already zeroed by _OffPolicyRealMaskCollator.
    """

    def __init__(
        self,
        model,
        args,
        train_dataset,
        processing_class,
        max_group_size: int,
        beta: float = 0.1,
        trust_region_eps: float = 0.05,
        use_unlikelihood: bool = False,
        divergence: str = "kl_new_old",
        negative_fraction: float = 1.0,
        min_pos_neg_ratio: float | None = None,
        negative_logp_floor: float = math.log(1e-8),
        negative_switch_threshold: float | None = None,
        ref_model_name_or_path: str | None = None,
        **kwargs,
    ):
        super().__init__(model, args, train_dataset, processing_class, **kwargs)
        self.data_collator = _OffPolicyRealMaskCollator(
            processing_class, args.max_completion_length, args.max_prompt_length, args.mask_truncated_completions
        )
        self.max_group_size = max_group_size
        self.beta = beta
        self.trust_region_eps = trust_region_eps
        self.divergence = divergence
        self.negative_logp_floor = negative_logp_floor
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

    def _get_train_sampler(self, train_dataset=None) -> _GroupShuffledSampler:
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        return _GroupShuffledSampler(dataset, self.max_group_size, self.args.seed)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        rewards_global = self._gather_grouped(inputs["reward"])
        is_real_global = self._gather_grouped(inputs["is_real"])
        G = rewards_global.numel() // self.max_group_size
        real_mask_blocks = is_real_global.view(G, self.max_group_size)
        group_sizes = real_mask_blocks.sum(dim=1).tolist()
        # Skip a block that padded down to zero real rows — can't happen for a genuine group (every group
        # keeps at least its own positives, or is dropped entirely upstream by data prep if it has none;
        # this is purely a defensive floor), but a block-level group_size of 0 would break the ragged
        # solver's per-group normalization constraint.
        if any(gs == 0 for gs in group_sizes):
            raise RuntimeError("A padded block had zero real rows — check pad_and_cap_ragged_dataset's output.")

        rewards_real = rewards_global[is_real_global]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rho_real, lam = compute_rho_star_offpolicy_ragged(
                rewards_real, group_sizes, self.beta, self.trust_region_eps, divergence=self.divergence
            )
        if any("optimal_inaccurate" in str(w.message) for w in caught):
            self._stage1_inaccurate_count += 1
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["tribe/stage1_inaccurate_total"].append(self._stage1_inaccurate_count)
        if rho_real is None:
            self._stage1_skip_count += 1
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["tribe/stage1_skipped_total"].append(self._stage1_skip_count)
            loss = 0.0 * per_token_logps.sum()
            return (loss, None) if return_outputs else loss

        # Scatter the real-only rho* back into full (padding-included) block positions — padding rows get
        # a placeholder rho*=1 (weight=0), irrelevant since completion_mask already zeroes their loss
        # contribution, but keeps every downstream tensor shape aligned with the gathered batch.
        rho_global = rewards_global.new_ones(rewards_global.shape, dtype=rho_real.dtype)
        rho_global[is_real_global] = rho_real
        weight_global = rho_global - 1.0

        # Same real-only-then-scatter-back pattern as rho_global above: both gating functions must only
        # ever see REAL rows — padding rows aren't meaningfully "positive" or "negative" at all, and
        # _negative_sample_weight_ragged_perstep's per-group torch.split needs its input's shape to match
        # group_sizes (the REAL per-block counts) specifically.
        if self.min_pos_neg_ratio is not None:
            sample_weight_real = self._pos_neg_floor_sample_weight(
                weight_global[is_real_global] < 0, self.min_pos_neg_ratio, seed=self.state.global_step
            )
        else:
            sample_weight_real = _negative_sample_weight_ragged_perstep(
                weight_global[is_real_global] < 0, group_sizes, self.negative_fraction, seed=self.state.global_step
            )
        sample_weight_global = weight_global.new_zeros(weight_global.shape)
        sample_weight_global[is_real_global] = sample_weight_real.to(sample_weight_global.dtype)

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
        self._metrics[mode]["tribe/rho_star_mean"].append(rho_real.mean().item())
        self._metrics[mode]["tribe/rho_star_std"].append(rho_real.std().item())
        self._metrics[mode]["tribe/rho_star_min"].append(rho_real.min().item())
        self._metrics[mode]["tribe/rho_star_max"].append(rho_real.max().item())

        # pos:neg ratio AFTER gating (negative_fraction or min_pos_neg_ratio) — see
        # OffPolicyTrainer._pos_neg_floor_sample_weight's docstring. Real rows only — padding rows aren't
        # meaningfully "positive" or "negative" at all.
        is_negative_real = weight_global[is_real_global] < 0
        n_positive_real = (~is_negative_real).sum().item()
        n_negative_kept_real = sample_weight_global[is_real_global][is_negative_real].sum().item()
        self._metrics[mode]["tribe/n_positive"].append(n_positive_real)
        self._metrics[mode]["tribe/n_negative_kept"].append(n_negative_kept_real)
        self._metrics[mode]["tribe/pos_neg_kept_ratio"].append(n_positive_real / max(n_negative_kept_real, 1e-8))

        if self.state.global_step % max(int(self.args.logging_steps), 1) == 0:
            with torch.no_grad():
                mu_per_token_logps = self._get_per_token_logps(self.mu_model, input_ids, attention_mask, logits_to_keep)
                seq_logps_theta = (per_token_logps.detach().float() * completion_mask).sum(-1)
                seq_logps_mu = (mu_per_token_logps.float() * completion_mask).sum(-1)
                rho_theta = torch.exp(seq_logps_theta - seq_logps_mu)
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
