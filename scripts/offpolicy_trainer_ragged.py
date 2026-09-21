"""
Ragged-group counterpart to offpolicy_trainer.OffPolicyTribeGlobalTrainer — for datasets where the number
of completions per question genuinely varies (PRM800K: 1 to 470, median 5), instead of every trainer in
this project's own fixed `--num_generations`. New, additive module: offpolicy_trainer.py is untouched, so
every existing off-policy trainer's behavior is unaffected — see tribe/stage1_ragged.py's own docstring for
why variable group sizes don't actually require different math, just different indexing.

`OffPolicyTribeGlobalRaggedTrainer` subclasses `OffPolicyTribeGlobalTrainer` directly rather than
duplicating it (unlike this project's usual duplication-over-abstraction convention for the big, complex
on-policy trainers) because the difference is genuinely small and localized: only `__init__` changes (the
ragged Stage 1 solve, group_sizes read from the dataset's own 'group_id' column instead of a fixed
`args.group_size`, a plain random sampler instead of the group-preserving `_GroupShuffledSampler`) —
`compute_loss` is inherited completely unchanged, since it only ever consumes each row's own precomputed
`rho_star` and has no group_size dependency at all (the group-size-dependent work all happens once in
`__init__`, before training starts — see the class docstring below for why the sampler swap is safe).
"""

import math

import torch
from torch.utils.data import RandomSampler

from offpolicy_trainer import OffPolicyTrainer, OffPolicyTribeGlobalTrainer, _OffPolicyGlobalRhoCollator
from tribe.offpolicy_stage1_ragged import compute_rho_star_offpolicy_ragged


def _negative_sample_weight_ragged(
    is_negative: torch.Tensor, group_sizes: list[int], negative_fraction: float, seed: int
) -> torch.Tensor:
    """
    Ragged-group counterpart to OffPolicyTrainer._negative_sample_weight_global — see that method's own
    docstring for the full negative_fraction contract (per-group-count rounding + ranked random selection
    so the *expected* weight of each of a group's own negatives is exactly negative_fraction). Generalized
    to variable-size groups via `group_sizes` (`torch.split` instead of a fixed `.view(G, group_size)`
    reshape, which requires uniform groups). Computed ONCE over the whole dataset in __init__ (this
    trainer's Stage 1 is also a one-time global solve, not per-batch, unlike OffPolicyTribeTrainer), so
    takes a fixed `seed` instead of reseeding from `self.state.global_step` every training step.
    """
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
        gen = torch.Generator().manual_seed(seed)
        bonus_parts = []
        for g in torch.split(is_negative, group_sizes):
            n_neg = g.sum()
            n_bonus = int((n_neg.float() * frac).round().item())
            rand_vals = torch.rand(g.shape, generator=gen)
            rand_vals = rand_vals.masked_fill(~g, float("inf"))
            rank = rand_vals.argsort().argsort()
            bonus_parts.append(g & (rank < n_bonus))
        weight = weight + torch.cat(bonus_parts).float()
    return weight


class _OffPolicyGlobalRhoSampleWeightCollator(_OffPolicyGlobalRhoCollator):
    """Same as _OffPolicyGlobalRhoCollator, plus passing through a precomputed 'sample_weight' column
    (the ragged trainer's negative_fraction weight, baked in once in __init__ — see
    _negative_sample_weight_ragged)."""

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        batch = super().__call__(examples)
        batch["sample_weight"] = torch.tensor([example["sample_weight"] for example in examples], dtype=torch.float32)
        return batch


class OffPolicyTribeGlobalRaggedTrainer(OffPolicyTribeGlobalTrainer):
    """
    Same as OffPolicyTribeGlobalTrainer, but the one-time Stage 1 solve runs over VARIABLE-size groups
    (`tribe.offpolicy_stage1_ragged.compute_rho_star_offpolicy_ragged`) instead of assuming a fixed
    `args.group_size` for every question. Requires the dataset to carry a `'group_id'` column: an integer
    per row, identical for every row belonging to the same question's group, laid out in CONTIGUOUS blocks
    (all of one group_id's rows adjacent) — matching scripts/convert_prm800k_offpolicy.py's `--ragged`
    output. Group sizes are inferred directly from run-lengths of this column, no separate size list needed.

    `_get_train_sampler` is overridden to a plain `RandomSampler` instead of the parent's group-preserving
    `_GroupShuffledSampler`: safe specifically because `compute_loss` only ever reads each row's own
    precomputed `rho_star`/`sample_weight` columns — nothing in a training step's forward/backward pass
    needs a batch to contain an intact group, unlike OffPolicyTribeTrainer's per-batch Stage 1 solve (which
    this global variant never does at all; Stage 1 runs exactly once, in __init__, before training starts).

    `compute_loss` is OVERRIDDEN (not inherited from OffPolicyTribeGlobalTrainer, unlike everything else):
    that parent class never implemented `negative_fraction` at all (only the per-batch OffPolicyTribeTrainer
    has it, via OffPolicyTrainer._negative_sample_weight_global, which assumes a uniform `self.args.group_size`
    reshape and so can't be reused here either) — its compute_loss just does plain `(rho_star - 1)`-weighted
    NLL over every sample. Since this trainer's negative_fraction is also meaningful and its Stage 1 is
    already a one-time global solve, negative_fraction's sample-weighting is likewise baked in ONCE in
    __init__ (`_negative_sample_weight_ragged`, this module) as a `sample_weight` dataset column, and
    compute_loss just applies it in the final average — see that helper's docstring for the exact contract.
    """

    def __init__(
        self,
        model,
        args,
        train_dataset,
        processing_class,
        beta: float = 0.1,
        trust_region_eps: float = 0.05,
        use_unlikelihood: bool = False,
        divergence: str = "chi_squared",
        negative_fraction: float = 1.0,
        min_pos_neg_ratio: float | None = None,
        negative_logp_floor: float = math.log(1e-8),
        **kwargs,
    ):
        # Mutually exclusive with negative_fraction — see OffPolicyTrainer._pos_neg_floor_sample_weight's
        # own docstring for why they're two different gating modes, not stackable.
        if min_pos_neg_ratio is not None and negative_fraction != 1.0:
            raise ValueError(
                "min_pos_neg_ratio and negative_fraction are mutually exclusive — leave negative_fraction "
                "at its default (1.0) when setting min_pos_neg_ratio."
            )
        if "group_id" not in train_dataset.column_names:
            raise ValueError(
                "OffPolicyTribeGlobalRaggedTrainer requires a 'group_id' column (e.g. scripts/"
                "convert_prm800k_offpolicy.py's --ragged output) — one integer per row, identical within a "
                "question's group, laid out in contiguous per-group blocks."
            )
        group_ids = train_dataset["group_id"]
        group_sizes = []
        prev = object()  # sentinel, never equal to a real group_id
        for gid in group_ids:
            if gid == prev:
                group_sizes[-1] += 1
            else:
                group_sizes.append(1)
                prev = gid
        if sum(group_sizes) != len(train_dataset):
            raise AssertionError("Unreachable: run-length sum must equal dataset length.")

        rewards = torch.tensor(train_dataset["reward"], dtype=torch.float64)
        rho_star_all, lam = compute_rho_star_offpolicy_ragged(
            rewards, group_sizes, beta, trust_region_eps, divergence=divergence
        )
        if rho_star_all is None:
            # Same reasoning as OffPolicyTribeGlobalTrainer's own uniform-group-size version: this is a
            # ONE-TIME solve over the whole dataset before training starts, no per-batch fallback exists.
            raise RuntimeError(
                "Stage 1's global ragged cvxpy solve failed (see the warning above for the cvxpy status) — "
                "OffPolicyTribeGlobalRaggedTrainer has no per-batch fallback for this."
            )
        train_dataset = train_dataset.add_column("rho_star", rho_star_all.tolist())

        # negative_fraction's (or min_pos_neg_ratio's) sample-weighting, baked in ONCE here (see
        # _negative_sample_weight_ragged's docstring) — "negative" is TRIBE's own weight<0 definition
        # (rho* - 1 < 0), same as OffPolicyTribeGlobalTrainer.compute_loss's `weight = inputs["rho_star"] - 1`.
        # A fixed seed (args.seed), not a per-step one, since this whole solve happens once before training
        # starts — same rationale as _negative_sample_weight_ragged's own docstring.
        is_negative_all = (rho_star_all - 1) < 0
        if min_pos_neg_ratio is not None:
            sample_weight_all = self._pos_neg_floor_sample_weight(is_negative_all, min_pos_neg_ratio, seed=args.seed)
        else:
            sample_weight_all = _negative_sample_weight_ragged(is_negative_all, group_sizes, negative_fraction, seed=args.seed)
        train_dataset = train_dataset.add_column("sample_weight", sample_weight_all.tolist())

        # Deliberately calls the GRANDPARENT's __init__ (OffPolicyTrainer, not OffPolicyTribeGlobalTrainer)
        # — the immediate parent's __init__ would redo the (wrong, uniform-group-size) Stage 1 solve above.
        # compute_loss is OVERRIDDEN below (not inherited from OffPolicyTribeGlobalTrainer — see class
        # docstring for why).
        OffPolicyTrainer.__init__(self, model, args, train_dataset, processing_class, **kwargs)
        self.data_collator = _OffPolicyGlobalRhoSampleWeightCollator(
            processing_class, args.max_completion_length, args.max_prompt_length, args.mask_truncated_completions
        )
        self.global_lambda = lam.item()
        self.use_unlikelihood = use_unlikelihood
        self.negative_fraction = negative_fraction
        self.min_pos_neg_ratio = min_pos_neg_ratio
        self.negative_logp_floor = negative_logp_floor

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        return RandomSampler(dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Copy of OffPolicyTribeGlobalTrainer.compute_loss (offpolicy_trainer.py), with `sample_weight`
        # (this trainer's negative_fraction weighting, precomputed in __init__) applied in the final
        # average — see class docstring for why this can't just be inherited.
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        weight = inputs["rho_star"] - 1
        sample_weight = inputs["sample_weight"]

        if self.use_unlikelihood:
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

        # sample_weight.sum() in the denominator, not a plain .mean() — same rationale as
        # OffPolicyTribeTrainer.compute_loss: dropped (negative_fraction<1) examples must not silently
        # shrink every kept example's effective weight, and this reduces to an exact .mean() when
        # negative_fraction=1.0 (sample_weight is all-ones).
        per_seq_loss = per_token_loss.sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        loss = (per_seq_loss * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)

        mode = "train" if self.model.training else "eval"
        # pos:neg ratio AFTER gating — see OffPolicyTrainer._pos_neg_floor_sample_weight's own docstring.
        # Gathered across ranks (unlike weight/sample_weight above, which are this rank's own local slice)
        # to match the same whole-batch view every other TRIBE trainer logs this metric over.
        weight_gathered = self._gather_grouped(weight)
        sample_weight_gathered = self._gather_grouped(sample_weight)
        is_negative_gathered = weight_gathered < 0
        n_positive = (~is_negative_gathered).sum().item()
        n_negative_kept = sample_weight_gathered[is_negative_gathered].sum().item()
        self._metrics[mode]["tribe/n_positive"].append(n_positive)
        self._metrics[mode]["tribe/n_negative_kept"].append(n_negative_kept)
        self._metrics[mode]["tribe/pos_neg_kept_ratio"].append(n_positive / max(n_negative_kept, 1e-8))
        self._metrics[mode]["tribe/global_lambda"].append(self.global_lambda)
        self._log_logp_stats(per_token_logps, completion_mask, prefix="tribe_global")
        self._log_reward_metric(inputs["reward"])

        return (loss, None) if return_outputs else loss
