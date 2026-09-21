"""
Safe-TRIBE trainer: same GRPOTrainer/rollout wiring as `tribe.tribe_trainer.TribeTrainer`, with Stage 1
replaced by the cost-constrained solve (`tribe.stage1_safe.compute_rho_star_safe`) instead of the plain
reward-only one (`tribe.stage1.compute_rho_star`).

Per this repo's duplication convention (trainers are self-contained, shared logic copied rather than
abstracted — see AGENTS.md), this file is a near-verbatim copy of `tribe_trainer.py`; the only
Safe-TRIBE-specific pieces are (1) recovering a per-sample COST signal (RAFTTrainer's own pattern: GRPO's
`_generate_and_score_completions` doesn't expose the raw per-reward-function values, so `_calculate_rewards`
is called a second time) and (2) calling the constrained solver with that cost and `SafeTribeConfig.
cost_limit` instead of the unconstrained one. Stage 2 (`_compute_loss`) is untouched from `TribeTrainer` —
it only ever consumes `inputs["advantages"]`/`inputs["sample_weight"]`, which the constrained solve
produces in exactly the same shape as the unconstrained one, so it's inherited as-is.

Convention for `reward_funcs`: the LAST entry is always the cost function (e.g.
`reward_funcs=[reward_fn, cost_fn]`), and `reward_weights` for it should be `0.0` so it never contributes
to GRPO's own blended `rewards`/`advantages` (kept purely reward-based, matching plain TRIBE's Stage 1
input) — its score is recovered separately below for the cost constraint instead.

Unlike `TribeTrainer`, EVERY divergence choice here goes through cvxpy (`tribe.stage1_safe` has no closed
form even for `kl_new_old` — a second linear constraint has none), so an infeasible solve (trust-region
budget too tight to also satisfy the cost limit) is a real, non-negligible possibility, not just a
numerical-tolerance edge case. Handled the same way `OffPolicyTribeTrainer.compute_loss` handles a failed
cvxpy solve elsewhere in this project (see `feedback_skip_dont_overengineer_numerical_failures`): fall back
to `rho == 1` everywhere (no Stage-1 signal this step, matching `solve_lambda`'s own degenerate-batch
behavior) and count the skip, rather than crashing the run.
"""

import math
import random

import torch
from trl.trainer.grpo_trainer import GRPOTrainer

from .stage1 import TRUST_REGION_F, compute_rho_star, filter_zero_variance_groups
from .stage1_safe import compute_rho_star_safe
from .safe_tribe_config import SafeTribeConfig


class SafeTribeTrainer(GRPOTrainer):
    def __init__(self, args: SafeTribeConfig | None = None, **kwargs):
        super().__init__(args=args, **kwargs)
        # Cache of the previous training cycle's (tokens, old_per_token_logps), used to measure how far
        # that whole cycle's step(s) actually moved the policy — see _generate_and_score_completions.
        self._prev_cycle_snapshot = None
        self._stage1_infeasible_count = 0
        # Persistent dual-ascent Lagrange multiplier — only ever updated/used when
        # args.soft_cost_constraint is set (see compute_loss). Lives across the whole training run, not
        # recomputed per-batch, matching Safe-RLHF's own PPO-Lag convention.
        self._lambda_cost_dual = args.lambda_init if args is not None else 1.0

    def _negative_sample_weight_global(self, is_negative_global: torch.Tensor, group_size: int) -> torch.Tensor:
        # Ported from tribe.tribe_trainer.TribeTrainer, unchanged — see that method's own docstring for
        # the full subsampling/oversampling rationale.
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

        # Same end-of-cycle f-divergence bookkeeping as TribeTrainer — see that class's own comment.
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
            output["old_per_token_logps"] = old_per_token_logps

        seq_old_logps = (old_per_token_logps * completion_mask).sum(-1)
        ref_per_token_logps = output.get("ref_per_token_logps")
        if ref_per_token_logps is not None:
            seq_ref_logps = (ref_per_token_logps * completion_mask).sum(-1)
        else:
            seq_ref_logps = torch.zeros_like(seq_old_logps)
        c_local = seq_old_logps - seq_ref_logps

        A_local = output["advantages"]
        A_global = self.accelerator.gather(A_local)
        c_global = self.accelerator.gather(c_local)

        # Cost signal: the LAST registered reward function (see this module's docstring convention).
        # GRPOTrainer's own _generate_and_score_completions doesn't expose the raw per-reward-function
        # values (only the weighted-summed `advantages`), so recompute via _calculate_rewards a second
        # time — same pattern scripts/raft_trainer.py already uses for the analogous problem. This call's
        # own internal gather means cost_global is already the full batch, no separate gather needed.
        prompts = [x["prompt"] for x in inputs]
        completions_text = self.processing_class.batch_decode(output["completion_ids"], skip_special_tokens=True)
        completions = [[{"role": "assistant", "content": c}] for c in completions_text]
        completion_ids_list = [
            ids[: mask.sum()].tolist() for ids, mask in zip(output["completion_ids"], completion_mask)
        ]
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        cost_global = rewards_per_func[:, -1]

        if self.args.soft_cost_constraint:
            # Never hard-fail on an infeasible cost constraint: fold cost into the advantage via a
            # persistent dual-ascent multiplier (Safe-RLHF's own PPO-Lag convention) instead, then solve
            # the ordinary unconstrained TRIBE Stage 1 on the adjusted advantage. Always feasible (the
            # underlying solve has no cost constraint at all), so there is no rho==1 fallback path here.
            A_adjusted = A_global - self._lambda_cost_dual * cost_global
            rho_global, lam_tr = compute_rho_star(
                A_adjusted,
                c_global,
                self.beta,
                num_generations,
                self.args.trust_region_eps,
                divergence=self.args.divergence,
            )
            batch_cost_mean = cost_global.mean().item()
            self._lambda_cost_dual = min(
                self.args.lambda_max,
                max(0.0, self._lambda_cost_dual + self.args.lambda_lr * (batch_cost_mean - self.args.cost_limit)),
            )
            lam_cost = torch.tensor(self._lambda_cost_dual, dtype=A_global.dtype, device=A_global.device)
        else:
            rho_global, lam_tr, lam_cost = compute_rho_star_safe(
                A_global,
                c_global,
                cost_global,
                self.beta,
                num_generations,
                self.args.trust_region_eps,
                self.args.cost_limit,
                divergence=self.args.divergence,
            )
            if rho_global is None:
                # Infeasible this step (trust region too tight to also satisfy the cost limit) — no
                # Stage-1 signal, leave the policy where it already is, same fallback solve_lambda uses
                # for its own degenerate case. See this module's docstring for why this isn't a crash.
                self._stage1_infeasible_count += 1
                rho_global = torch.ones_like(A_global)
                lam_tr = torch.zeros((), dtype=A_global.dtype, device=A_global.device)
                lam_cost = torch.zeros((), dtype=A_global.dtype, device=A_global.device)
        self._metrics[mode]["tribe/stage1_infeasible_total"].append(self._stage1_infeasible_count)

        keep_mask = filter_zero_variance_groups(A_global, num_generations)

        process_slice = slice(
            self.accelerator.process_index * A_local.numel(),
            (self.accelerator.process_index + 1) * A_local.numel(),
        )
        rho_local = rho_global[process_slice]

        sample_weight_global = self._negative_sample_weight_global((rho_global - 1) < 0, num_generations)
        sample_weight_local = sample_weight_global[process_slice]

        self._metrics[mode]["tribe/lambda_trust_region"].append(lam_tr.item())
        self._metrics[mode]["tribe/lambda_cost"].append(lam_cost.item())
        self._metrics[mode]["tribe/frac_zero_variance_groups"].append(1.0 - keep_mask.float().mean().item())
        self._metrics[mode]["tribe/rho_star_mean"].append(rho_global.mean().item())
        self._metrics[mode]["tribe/rho_star_std"].append(rho_global.std().item())
        self._metrics[mode]["tribe/rho_star_min"].append(rho_global.min().item())
        self._metrics[mode]["tribe/rho_star_max"].append(rho_global.max().item())
        self._metrics[mode]["tribe/cost_mean"].append(cost_global.mean().item())
        self._metrics[mode]["tribe/expected_cost"].append((rho_global * cost_global).mean().item())

        output["advantages"] = rho_local - 1
        output["sample_weight"] = sample_weight_local

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
        # Identical to tribe.tribe_trainer.TribeTrainer._compute_loss — Stage 2 doesn't change at all
        # between plain and cost-constrained TRIBE, only Stage 1 (above) does.
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep
        )
        weight = inputs["advantages"]
        if self.args.negative_logp_floor is not None:
            per_token_logps_for_loss = per_token_logps.clamp(min=self.args.negative_logp_floor)
        else:
            per_token_logps_for_loss = per_token_logps
        sample_weight = inputs["sample_weight"]
        per_token_loss = -weight.unsqueeze(-1) * per_token_logps_for_loss * completion_mask * sample_weight.unsqueeze(-1)

        mode = "train" if self.model.training else "eval"
        if self.args.stage2_loss_type == "sum":
            loss = per_token_loss.sum(-1).sum() / sample_weight.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.args.stage2_loss_type == "grpo":
            per_seq_loss = per_token_loss.sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            loss = per_seq_loss.sum() / sample_weight.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.args.stage2_loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes
            if mode == "train":
                normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation
            loss = per_token_loss.sum() / normalizer
        else:
            raise ValueError(f"Unknown stage2_loss_type: {self.args.stage2_loss_type}.")

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            self._metrics[mode]["tribe/kl_ref"].append(self._global_masked_mean(per_token_kl.detach(), completion_mask))

        old_per_token_logps = inputs["old_per_token_logps"]
        seq_logps_theta = (per_token_logps.detach() * completion_mask).sum(-1)
        seq_old_logps = (old_per_token_logps * completion_mask).sum(-1)
        rho_theta = torch.exp(seq_logps_theta - seq_old_logps)
        f_val = TRUST_REGION_F[self.args.divergence](rho_theta)
        self._metrics[mode]["tribe/f_div_old"].append(self._global_mean(f_val))

        return loss
