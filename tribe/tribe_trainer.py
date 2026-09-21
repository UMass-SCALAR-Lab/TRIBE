"""
TRIBE trainer: a GRPOTrainer subclass wiring the Stage 1 closed form (tribe.stage1) into GRPO's rollout
loop, plus a plain (rho*-1)-weighted NLL Stage 2 loss (aggregation over tokens/batch controlled by
TribeConfig.stage2_loss_type: "sum", "grpo", or "dapo" — see that field's docstring).

See TRIBE-plan.tex. This is a correctness-first prototype, scoped to text-only tasks (GSM8K/MATH): no
vLLM, no multimodal, no tools, no environments. Everything not specific to TRIBE (generation, reward
computation, padding, ref-model bookkeeping) is inherited from GRPOTrainer as-is.

Known simplification: backfill (see `_generate_and_score_completions`) regenerates the whole local batch
via `super()` on each retry rather than surgically re-generating only the replaced groups, which wastes
compute re-running already-good groups. Fine for a correctness-first prototype at small batch sizes;
worth revisiting (e.g. splicing individual groups into the padded batch, as
`GRPOWithReplayBufferTrainer.update_with_replay_buffer` already does for a similar problem) if this
becomes a bottleneck at larger batch sizes. It also means `super()`'s own side effects (appending to
`self._metrics`/`self._logs` for reward stats, prompt/completion logging) fire on every retry, not just
the final one, so a backfilled step's logged reward/completions stats include some now-discarded,
since-replaced groups alongside the ones that actually end up in the batch.

Stage 2 uses the (rho*-1)-weighted loss as originally specified (TRIBE-plan.tex Section 2.2), run for a
small fixed number of steps, with per-token log-probs floored (TribeConfig.negative_logp_floor, matching
the off-policy trainer's own default) to bound the gradient magnitude on negative-weight tokens — see
TRIBE-plan.tex's "Open question" paragraph for the unboundedness this addresses. Settable to `None` to
recover the original plain unbounded loss. Also supports the off-policy trainer's own `negative_fraction`
ablation (TribeConfig.negative_fraction, `_negative_sample_weight_global`), gating what fraction of each
group's negative-weight examples participate in the loss; defaults to `1.0` (every negative kept, this
trainer's original behavior).
"""

import math
import random

import torch
from trl.trainer.grpo_trainer import GRPOTrainer

from .stage1 import TRUST_REGION_F, compute_rho_star, filter_zero_variance_groups
from .tribe_config import TribeConfig


class TribeTrainer(GRPOTrainer):
    def __init__(self, args: TribeConfig | None = None, **kwargs):
        super().__init__(args=args, **kwargs)
        # Cache of the previous training cycle's (tokens, old_per_token_logps), used to measure how far
        # that whole cycle's step(s) actually moved the policy — see _generate_and_score_completions.
        self._prev_cycle_snapshot = None

    def _negative_sample_weight_global(self, is_negative_global: torch.Tensor, group_size: int) -> torch.Tensor:
        # Ported from scripts.offpolicy_trainer.OffPolicyTrainer._negative_sample_weight_global, unchanged
        # mechanics -- only the group_size source differs (num_generations here, not a fixed config field,
        # since eval can use a different num_generations_eval). See that method's own docstring for the
        # full subsampling/oversampling rationale; summarized: returns a non-negative per-example WEIGHT
        # (not a boolean mask) -- positives always 1.0; negatives get a weight built from
        # negative_fraction's integer and fractional parts so the *expected* weight of each of a group's
        # own negatives is exactly negative_fraction, for any negative_fraction >= 0. Caller must use this
        # as both the numerator coefficient and the denominator in its own loss's averaging, and must NOT
        # also fold it into `weight` (rho*-1) upstream, or an example weighted `w` contributes w^2, not w.
        negative_fraction = self.args.negative_fraction
        if negative_fraction == 1.0:
            return is_negative_global.new_ones(is_negative_global.shape, dtype=torch.float)
        floor_f = math.floor(negative_fraction)
        frac = negative_fraction - floor_f
        G = is_negative_global.numel() // group_size
        is_neg_grouped = is_negative_global.view(G, group_size)
        weight = torch.where(
            is_neg_grouped,
            torch.full_like(is_neg_grouped, float(floor_f), dtype=torch.float),
            torch.ones_like(is_neg_grouped, dtype=torch.float),
        )
        if frac > 0:
            n_neg = is_neg_grouped.sum(dim=1, keepdim=True)
            n_bonus = (n_neg.float() * frac).round().long()
            gen = torch.Generator(device=is_neg_grouped.device).manual_seed(self.state.global_step)
            rand_vals = torch.rand(is_neg_grouped.shape, generator=gen, device=is_neg_grouped.device)
            rand_vals = rand_vals.masked_fill(~is_neg_grouped, float("inf"))
            rank = rand_vals.argsort(dim=1).argsort(dim=1)
            bonus = is_neg_grouped & (rank < n_bonus)
            weight = weight + bonus.float()
        return weight.view(-1)

    def _sample_backfill_example(self) -> dict:
        """Draw one fresh raw example from the training dataset to replace a zero-variance group."""
        return self.train_dataset[random.randrange(len(self.train_dataset))]

    def _global_sum_and_count(self, local_sum: torch.Tensor, local_count: torch.Tensor) -> float:
        totals = self.accelerator.reduce(torch.stack([local_sum, local_count]), reduction="sum")
        return (totals[0] / totals[1].clamp(min=1.0)).item()

    def _global_masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> float:
        return self._global_sum_and_count((x * mask).sum(), mask.sum().float())

    def _global_mean(self, x: torch.Tensor) -> float:
        return self._global_sum_and_count(x.sum(), torch.tensor(float(x.numel()), device=x.device))

    def _generate_and_score_completions(self, inputs: list[dict]) -> dict[str, torch.Tensor]:
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # This method only runs at cycle boundaries (every num_iterations steps), so by the time we get
        # here, the *previous* cycle's step(s) have already been applied via optimizer.step() — _compute_
        # loss itself only ever sees the state BEFORE its own step's update (loss.backward()/optimizer.
        # step() happen after it returns), so it can't measure this on its own; at num_iterations=1 that
        # makes tribe/f_div_old always ~0 there (see its docstring). Score the previous cycle's cached
        # tokens with the CURRENT (now post-update) model before they're discarded, to get the divergence
        # actually caused by that whole cycle's step(s) — informative at any num_iterations, including 1.
        if mode == "train" and self._prev_cycle_snapshot is not None:
            prev = self._prev_cycle_snapshot
            prev_input_ids = torch.cat([prev["prompt_ids"], prev["completion_ids"]], dim=1)
            prev_attention_mask = torch.cat([prev["prompt_mask"], prev["completion_mask"]], dim=1)
            with torch.no_grad():
                new_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prev_input_ids,
                    prev_attention_mask,
                    prev["completion_ids"].size(1),
                    self.args.per_device_train_batch_size,
                )
            seq_new_logps = (new_per_token_logps * prev["completion_mask"]).sum(-1)
            seq_prev_logps = (prev["old_per_token_logps"] * prev["completion_mask"]).sum(-1)
            rho_end_of_cycle = torch.exp(seq_new_logps - seq_prev_logps)
            f_val = TRUST_REGION_F[self.args.divergence](rho_end_of_cycle)
            self._metrics[mode]["tribe/f_div_old_end_of_cycle"].append(self._global_mean(f_val))

        # Zero-variance groups (TRIBE-plan.tex Section 2.3) carry no reward signal — they still get a
        # reference-term-driven rho* out of the joint Stage-1 solve below (nothing is excluded from it),
        # but no reward-driven signal. If enabled (backfill_zero_variance_prompts, off by default) and
        # only during training (this is a dynamic-sampling concept, not an eval-time one): replace such
        # groups' raw examples with freshly-sampled prompts and regenerate, up to `max_backfill_attempts`
        # rounds, so fewer groups end up reward-uninformative in the first place.
        inputs = list(inputs)
        output = super()._generate_and_score_completions(inputs)
        attempts = 0
        while (
            mode == "train"
            and self.args.backfill_zero_variance_prompts
            and attempts < self.args.max_backfill_attempts
        ):
            keep_mask_local = filter_zero_variance_groups(output["advantages"], num_generations)
            if keep_mask_local.all():
                break
            for g in (~keep_mask_local).nonzero(as_tuple=True)[0].tolist():
                replacement = self._sample_backfill_example()
                inputs[g * num_generations : (g + 1) * num_generations] = [replacement] * num_generations
            output = super()._generate_and_score_completions(inputs)
            attempts += 1
        self._metrics[mode]["tribe/backfill_attempts"].append(attempts)

        completion_mask = output["completion_mask"]

        # c(x,y) = log(pi_old(y|x) / pi_ref(y|x)), sequence-level (summed over completion tokens).
        old_per_token_logps = output.get("old_per_token_logps")
        if old_per_token_logps is None:
            # Steps aligned (the common case): the current model IS pi_old at this point in the outer
            # loop, before any Stage 2 update this iteration, so its logps serve as pi_old's.
            prompt_ids, prompt_mask = output["prompt_ids"], output["prompt_mask"]
            completion_ids = output["completion_ids"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)
            batch_size = (
                self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
            )
            with torch.no_grad():
                old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                    self.model, input_ids, attention_mask, logits_to_keep, batch_size
                )
            # _compute_loss's tribe/f_div_old metric needs this in `inputs` (== this dict) regardless of
            # which branch computed it; the parent only sets this key itself when it wasn't None above.
            output["old_per_token_logps"] = old_per_token_logps

        seq_old_logps = (old_per_token_logps * completion_mask).sum(-1)
        ref_per_token_logps = output.get("ref_per_token_logps")
        if ref_per_token_logps is not None:
            seq_ref_logps = (ref_per_token_logps * completion_mask).sum(-1)
        else:
            # beta == 0.0: no reference model was loaded, and c is multiplied by beta in Stage 1 anyway.
            seq_ref_logps = torch.zeros_like(seq_old_logps)
        c_local = seq_old_logps - seq_ref_logps

        # Whatever GRPOTrainer's own advantage computation produced for this batch. TribeConfig defaults
        # scale_rewards to "none" (overriding GRPOConfig's own "group" default), so this is normally the
        # raw R(x,y) - mean_K(R) quantity TRIBE-plan.tex's derivation is built around — but "group"/"batch"
        # remain fully valid if passed explicitly (see TribeConfig.scale_rewards' own docstring for the
        # tradeoff). multi_objective_aggregation is still forced to "sum_then_normalize" (see
        # TribeConfig.__post_init__): "normalize_then_sum" always std-normalizes regardless of
        # scale_rewards, which would remove the ability to run Stage 1 on raw advantage at all.
        A_local = output["advantages"]

        # Stage 1's trust region is a single global scalar (TRIBE-plan.tex Section 2.1), so lambda must be
        # solved against the whole batch, not each process's local shard.
        A_global = self.accelerator.gather(A_local)
        c_global = self.accelerator.gather(c_local)

        # Every group (including zero-variance ones) goes into a single joint solve, exactly like GRPO
        # trains on every group regardless of its advantage: a zero-variance group's A is exactly 0, so
        # its reward term washes out of the objective on its own (same as GRPO's advantage-weighted PPO
        # term vanishing for it), while it still correctly participates in the shared lambda and gets a
        # reference-term-driven rho* out of the same computation. Excluding such groups from the solve
        # entirely is only a well-motivated efficiency move when paired with backfill (no point excluding
        # a group from the trust-region budget if you're not also about to replace it) — TRIBE-plan.tex
        # Section 2.3 frames "filter... and backfill" as one combined practice, not filtering on its own.
        rho_global, lam = compute_rho_star(
            A_global, c_global, self.beta, num_generations, self.args.trust_region_eps, divergence=self.args.divergence
        )

        # Group-relative A already has mean 0 within each group, so std(A) == std(raw reward) per group:
        # filtering on A directly is equivalent to filtering on the reward. Monitoring only here — doesn't
        # affect the solve above; see the backfill loop for where this mask actually gates anything.
        keep_mask = filter_zero_variance_groups(A_global, num_generations)

        process_slice = slice(
            self.accelerator.process_index * A_local.numel(),
            (self.accelerator.process_index + 1) * A_local.numel(),
        )
        rho_local = rho_global[process_slice]

        # negative_fraction gating (see _negative_sample_weight_global's docstring). "Negative" here means
        # weight<0 (below-group-average reward, TRIBE's own definition, same as the off-policy trainer's).
        sample_weight_global = self._negative_sample_weight_global((rho_global - 1) < 0, num_generations)
        sample_weight_local = sample_weight_global[process_slice]

        self._metrics[mode]["tribe/lambda"].append(lam.item())
        self._metrics[mode]["tribe/frac_zero_variance_groups"].append(1.0 - keep_mask.float().mean().item())

        # rho_global is already the full gathered batch (identical on every process, same as lam above),
        # so these need no further reduction. Mean should sit close to 1 by construction (rho* is a
        # per-group softmax-normalized weight), so std/min/max are what actually show how much Stage 1 is
        # reweighting this step — e.g. a max blowing up while mean stays ~1 indicates a few samples
        # dominating the reweighting even though the batch-average looks unremarkable.
        self._metrics[mode]["tribe/rho_star_mean"].append(rho_global.mean().item())
        self._metrics[mode]["tribe/rho_star_std"].append(rho_global.std().item())
        self._metrics[mode]["tribe/rho_star_min"].append(rho_global.min().item())
        self._metrics[mode]["tribe/rho_star_max"].append(rho_global.max().item())

        # Replace GRPO's own advantage with TRIBE's centered Stage 1 weight, reusing the same key so the
        # rest of the (inherited) plumbing and _compute_loss stay uniform.
        output["advantages"] = rho_local - 1
        output["sample_weight"] = sample_weight_local

        # Cache this cycle's (tokens, old_per_token_logps) so the *next* call to this method — after this
        # cycle's step(s) have been applied — can measure how far they actually moved the policy, before
        # this data is discarded. Train-only, matching the read side above.
        if mode == "train":
            self._prev_cycle_snapshot = {
                "prompt_ids": output["prompt_ids"].detach(),
                "prompt_mask": output["prompt_mask"].detach(),
                "completion_ids": output["completion_ids"].detach(),
                "completion_mask": completion_mask.detach(),
                "old_per_token_logps": old_per_token_logps.detach(),
            }

        return output

    def _compute_loss(self, model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep
        )
        weight = inputs["advantages"]  # (rho* - 1), computed once in _generate_and_score_completions
        # Floor per-token log-probs before weighting (kept as a separate variable -- per_token_logps
        # itself stays raw/unfloored below for tribe/f_div_old, which measures actual policy drift, not
        # the loss's own clamped view of it). None (see TribeConfig.negative_logp_floor) disables this
        # and falls back to the plain unbounded loss.
        if self.args.negative_logp_floor is not None:
            per_token_logps_for_loss = per_token_logps.clamp(min=self.args.negative_logp_floor)
        else:
            per_token_logps_for_loss = per_token_logps
        # Per-token contribution before aggregation: -(rho*-1) * log pi_theta(token), masked, times the
        # negative_fraction sample_weight (see _negative_sample_weight_global's docstring: this gates "how
        # many times this example counts in the batch average", never the magnitude of `weight` itself, so
        # it's applied here as its own separate factor, not folded into `weight` upstream).
        sample_weight = inputs["sample_weight"]
        per_token_loss = -weight.unsqueeze(-1) * per_token_logps_for_loss * completion_mask * sample_weight.unsqueeze(-1)

        mode = "train" if self.model.training else "eval"
        if self.args.stage2_loss_type == "sum":
            # Full-sequence sum of log-probs: TRIBE-plan.tex Section 2.2's literal M-projection quantity
            # (log pi(y|x) is a sum over tokens by the chain rule), length-dependent gradient scale.
            # sample_weight.sum() in the denominator instead of a plain .mean(): dropped (negative_fraction
            # ablation-zeroed) examples must not silently shrink every kept example's effective weight by
            # diluting the average, and oversampled (sample_weight>1) examples must count proportionally
            # more — this reduces to an exact .mean() when negative_fraction=1.0 (sample_weight is
            # all-ones), reproducing prior behavior.
            loss = per_token_loss.sum(-1).sum() / sample_weight.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.args.stage2_loss_type == "grpo":
            # Per-sequence token average, matching GRPOTrainer's own default ("grpo") loss type. Same
            # sample_weight.sum() denominator rationale as "sum" above.
            per_seq_loss = per_token_loss.sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            loss = per_seq_loss.sum() / sample_weight.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.args.stage2_loss_type == "dapo":
            # Batch-level token-count normalization, matching GRPOTrainer's "dapo" loss type: each token
            # gets equal weight across the whole batch, rather than each sequence. Unlike "sum"/"grpo",
            # the normalizer here is left as the raw (unweighted) token count: per_token_loss already has
            # sample_weight folded in (dropped examples contribute exactly 0), so every surviving token
            # keeps the same per-token weight this loss type is meant to guarantee, without needing to
            # recompute a weighted token count.
            normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes
            if mode == "train":  # in eval, the batch is neither split across steps nor accumulated
                normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation
            loss = per_token_loss.sum() / normalizer
        else:
            raise ValueError(f"Unknown stage2_loss_type: {self.args.stage2_loss_type}.")

        # Monitoring only — neither of these feeds into the loss above (TRIBE's reference-KL control is
        # already entirely inside Stage 1's rho*, no separate loss term needed, unlike GRPO's additive
        # beta*KL), but both are useful to watch alongside GRPO's own equivalents.
        if self.beta != 0.0:
            # KL(pi_theta || pi_ref), same k3 estimator and token-level global-mean convention as
            # GRPOTrainer's own "kl" metric (grpo_trainer.py's _compute_loss), for direct comparability.
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            self._metrics[mode]["tribe/kl_ref"].append(self._global_masked_mean(per_token_kl.detach(), completion_mask))

        # E_old[f(rho_theta)], rho_theta = pi_theta/pi_old, using whichever f Stage 1's trust region uses
        # (self.args.divergence) — the divergence of the CURRENT step's policy (before this step's own
        # update) from the pi_old this batch was sampled from, directly comparable to trust_region_eps.
        # Sequence-level (not token-level), matching Stage 1's own constraint granularity. At
        # num_iterations=1 this is always ~0 by construction (no update has happened yet between
        # pi_old's snapshot and this forward pass) — see tribe/f_div_old_end_of_cycle below for the
        # metric that's actually informative in that regime.
        old_per_token_logps = inputs["old_per_token_logps"]
        seq_logps_theta = (per_token_logps.detach() * completion_mask).sum(-1)
        seq_old_logps = (old_per_token_logps * completion_mask).sum(-1)
        rho_theta = torch.exp(seq_logps_theta - seq_old_logps)
        f_val = TRUST_REGION_F[self.args.divergence](rho_theta)
        self._metrics[mode]["tribe/f_div_old"].append(self._global_mean(f_val))

        return loss
