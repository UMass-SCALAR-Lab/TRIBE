"""
RAFT (rejection-sampling SFT), reusing GRPOTrainer's own generation/reward/multi-GPU machinery rather than
hand-rolling a distributed loop. Shared by scripts/train_gsm8k_raft.py and scripts/train_math_raft.py.

Why this design instead of a custom Accelerate loop (which this project tried first): GRPOTrainer already
generates completions, computes rewards, and gathers them across processes correctly, on every GPU count —
that's already proven by every other baseline script here (GRPO/DAPO/RLOO/TRIBE). Subclassing it and only
overriding `_compute_loss` (the same pattern TribeTrainer already uses) gets multi-GPU support for free,
with a bonus: HF's own `Trainer.training_step()` calls `backward()`/`optimizer.step()` exactly once per
step no matter what `_compute_loss` computes internally, so there's no possibility of different processes
calling optimizer.step() a different number of times in the same step (the exact correctness trap that
made the hand-rolled Accelerate version hard to get right). It also fixes the OOM risk Online DPO hit:
`_get_per_token_logps_and_entropies` (used here, same as every other method) already internally
micro-batches the forward pass, unlike an unchunked full-vocab log_softmax call.

GRPOTrainer's own `_generate_and_score_completions` doesn't expose the raw per-sample reward in its
returned dict (only the group-centered `advantages`) — RAFT needs the raw reward directly (a binary 0/1
reward's advantage collapses to the same value for every sample in an all-correct or all-wrong group,
making "was this one actually correct" unrecoverable from the advantage alone). So `_generate_and_score_
completions` below calls `self._calculate_rewards(...)` a second time with locally-reconstructed
prompts/completions/completion_ids (the same private method GRPOTrainer's own advantage computation
already calls internally) — it's the exact function that already does the correct cross-process gather
(ends in `gather(rewards_per_func)`), so no custom distributed code is needed here either. The redundant
second reward-function call is a deliberate, minor tradeoff: the reward functions used here (regex
matching, math_verify) are cheap, so paying for them twice is simpler than plumbing the already-computed
value out of the parent's private locals.
"""

from dataclasses import dataclass, field

import torch
from trl import GRPOConfig, GRPOTrainer


@dataclass
class RAFTConfig(GRPOConfig):
    """
    Configuration for [`RAFTTrainer`].

    Args:
        penalize_incorrect_weight (`float`, *optional*, defaults to `0.0`):
            If > 0, also runs an unlikelihood-training phase on that step's INCORRECT completions (push
            DOWN their likelihood — Welleck et al., "Neural Text Degeneration with Unlikelihood
            Training", https://huggingface.co/papers/1908.04319 — bounded/saturating as p -> 1, unlike
            naive negative-NLL gradient ascent, which diverges as p -> 0), scaled by this weight. `0.0`
            (default) disables it, matching plain RAFT (positive-only, incorrect completions dropped).
    """

    penalize_incorrect_weight: float = field(
        default=0.0,
        metadata={
            "help": "If > 0, also runs an unlikelihood-training phase on that step's INCORRECT "
            "completions, scaled by this weight. 0.0 (default) disables it, matching plain RAFT."
        },
    )


class RAFTTrainer(GRPOTrainer):
    def _generate_and_score_completions(self, inputs: list[dict]) -> dict[str, torch.Tensor]:
        output = super()._generate_and_score_completions(inputs)

        prompts = [x["prompt"] for x in inputs]
        completion_mask = output["completion_mask"]
        completions_text = self.processing_class.batch_decode(output["completion_ids"], skip_special_tokens=True)
        completions = [[{"role": "assistant", "content": c}] for c in completions_text]
        completion_ids_list = [
            ids[: mask.sum()].tolist() for ids, mask in zip(output["completion_ids"], completion_mask)
        ]

        # Global (gathered across all processes) reward, one column per reward function.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        rewards_global = (rewards_per_func * self.reward_weights.to(rewards_per_func.device).unsqueeze(0)).nansum(
            dim=1
        )

        # Slice back down to this process's own local portion — same pattern TribeTrainer uses for its
        # Stage 1 gather/solve/slice.
        local_size = output["completion_ids"].size(0)
        start = self.accelerator.process_index * local_size
        output["rewards"] = rewards_global[start : start + local_size]
        return output

    def _compute_loss(self, model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, _, _ = self._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep)
        rewards = inputs["rewards"]
        is_correct = (rewards == 1.0).float()

        # Positive phase: masked NLL (per-sequence token average, like every other stage2_loss_type="grpo"
        # baseline here), restricted to correct completions via a multiplicative mask — never a
        # conditional branch on whether this process happens to have any correct examples locally, since
        # that would make different processes take different code paths (harmless here since there's no
        # manual backward()/step() loop left to desync, but multiplying through the mask is simplest and
        # keeps the loss numerically well-defined, including the all-zero-mask case, without a special case).
        seq_logps = (per_token_logps * completion_mask).sum(-1)
        token_counts = completion_mask.sum(-1).clamp(min=1)
        per_seq_nll = -seq_logps / token_counts
        loss = (per_seq_nll * is_correct).sum() / is_correct.sum().clamp(min=1)

        if self.args.penalize_incorrect_weight > 0:
            # per_token_logps is already the actual completion tokens' log-probs, aligned 1:1 with
            # completion_mask (see _get_per_token_logps_and_entropies) — no manual shift/gather needed.
            is_incorrect = (rewards == 0.0).float()  # None/unscorable (nan) rewards excluded from both sets
            token_probs = per_token_logps.exp()
            # log1p(-p), not log(1-p): the tokens unlikelihood is meant to push down are exactly the ones
            # the model already assigns low probability to, so p is typically tiny (e.g. ~1e-6 for an
            # undertrained model over a ~150k vocab) — computing `1.0 - p` directly in bf16 (~2-3
            # significant digits) underflows straight to 1.0, silently zeroing this whole loss term.
            # log1p is numerically stable for small arguments regardless of dtype, so it doesn't.
            unlikelihood = -torch.log1p(-token_probs.clamp(max=1 - 1e-6))
            seq_unlikelihood = (unlikelihood * completion_mask).sum(-1) / token_counts
            neg_loss = (seq_unlikelihood * is_incorrect).sum() / is_incorrect.sum().clamp(min=1)
            loss = loss + self.args.penalize_incorrect_weight * neg_loss

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["raft/accept_rate"].append(self.accelerator.gather(is_correct).mean().item())
        return loss
