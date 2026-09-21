"""
Shared base for the off-policy suite (TOPR's single-iteration, fully-offline regime,
https://huggingface.co/papers/2503.14286): a fixed dataset of `group_size` completions per question,
generated once from a frozen behavioral policy (scripts/generate_offpolicy_gsm8k.py et al.) and consumed
for exactly one epoch, no regeneration.

Structurally this is supervised training over a fixed (prompt, completion, reward) dataset — SFTTrainer's
own data shape — but subclasses transformers.Trainer directly rather than SFTTrainer: SFTTrainer's own
compute_loss is entangled with Liger kernel/MoE-aux-loss/VLM machinery irrelevant here and would need a
full override anyway, and its collator has no concept of a `reward` column. _get_per_token_logps below is
a lean, text-only reimplementation of the core of GRPOTrainer's _get_per_token_logps_and_entropies (chunked
forward pass, temperature-scaled log-softmax) — TribeTrainer/RAFTTrainer get that method for free via
inheriting GRPOTrainer directly, which this trainer deliberately doesn't (GRPOTrainer's own value-add is
its online generation loop, which off-policy training must NOT do).

Three methods share this base (see the *Trainer classes at the bottom), differing only in advantage/loss:
  - OffPolicyReinforceTrainer: naive REINFORCE with RLOO's leave-one-out baseline, no importance-sampling
    correction at all — deliberately: this is the unstable baseline TOPR's own paper contrasts itself
    against (Section 2.1's `naive REINFORCE`, literally R(tau)*grad(log pi(tau)) with no pi/mu ratio).
  - OffPolicyGRPOTrainer: GRPO's clipped importance-ratio objective, with `pi_old` redefined as the frozen
    behavioral policy mu (a frozen reference copy, loaded once, never updated) instead of GRPO's normal
    on-policy meaning ("the model right before this step").
  - OffPolicyTribeTrainer: TRIBE's Stage 1 (tribe/offpolicy_stage1.py) + the same (rho*-1)-weighted NLL
    Stage 2 loss as the on-policy TribeTrainer.

Multi-GPU correctness: a question's full group of `group_size` completions can be split across processes
in a batch (never across steps — see OffPolicyTrainer._get_train_sampler for why). Advantage/rho* must be
computed against the COMPLETE group, not a per-process shard — gathered via self.accelerator.gather() then
sliced back to the local portion, the exact pattern TribeTrainer already uses for the same reason
(tribe/tribe_trainer.py's _generate_and_score_completions).
"""

import contextlib
import copy
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from torch.utils.data import RandomSampler, Sampler
from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from tribe.offpolicy_stage1 import compute_rho_star_offpolicy
from tribe.stage1 import TRUST_REGION_F
from trl.trainer.utils import selective_log_softmax


@dataclass
class OffPolicyConfig(TrainingArguments):
    """
    Configuration shared by every off-policy trainer in this file.

    Args:
        group_size (`int`, *optional*, defaults to `16`):
            Number of completions per question in the pre-generated dataset (`n=16` for GSM8K, `n=32` for
            MATH, per TOPR's own setup). `per_device_train_batch_size * num_processes` must be a multiple
            of this, so a question's full group always lands within a single training step (never split
            across steps — only across processes within a step, which is handled via gather).
        max_completion_length (`int`, *optional*, defaults to `512`):
            Completions longer than this are truncated (from the pre-generated dataset, not re-generated).
        max_prompt_length (`int`, *optional*):
            If set, prompts longer than this keep only their last `max_prompt_length` tokens. `None`
            (default) means no truncation.
        temperature (`float`, *optional*, defaults to `1.0`):
            Sampling temperature used when the dataset was generated — logits are divided by this before
            the log-softmax, so log-probs computed here match the actual generation distribution (same
            reasoning as GRPOTrainer's own temperature scaling).
        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Overrides `TrainingArguments`' own default of `True`. The dataset's `prompt`/`completion`/
            `reward` columns don't match the model's `forward()` signature, so the default would silently
            strip all of them before `_OffPolicyCollator` ever sees a batch.
        val_fraction (`float`, *optional*, defaults to `0.1`):
            Fraction of questions (whole groups, never split) held out from the END of the dataset for
            hyperparameter-selection validation (see scripts/offpolicy_split.py) — never trained on.
            Driver scripts apply this themselves (`split_off_policy_dataset`) since it must run on the raw,
            still-group-aligned dataset before any trainer-specific filtering (e.g. RAFT's reward==1.0
            filter, which breaks the group alignment this split relies on).
        mask_truncated_completions (`bool`, *optional*, defaults to `False`):
            Zero out the loss entirely for completions whose ORIGINAL length (before this collator's own
            `max_completion_length` clipping) was already at or past that cap — same intent as trl's
            on-policy `GRPOTrainer`'s own `mask_truncated_completions`, which this repo's off-policy
            trainers never had an equivalent of before now. A completion this long almost certainly never
            reached a natural stop at generation time (cut off mid-derivation, not concluded) — its
            reward (near-always 0, since an interrupted derivation essentially never happens to state a
            correct final boxed answer) reflects "ran out of room," not "the reasoning was wrong," so
            training on it as an ordinary negative example teaches the model to suppress the (possibly
            perfectly reasonable) pattern of writing a long derivation, not to reason better. Off by
            default to preserve every existing run's exact behavior — see conversation for the suspected
            link to GRPO's MATH-specific confident-early-termination failure mode (short completions, 100%
            natural stop, only ~52% ever boxed) that GSM8K's off-policy data (rarely hits its own, shorter
            cap) doesn't show.
    """

    group_size: int = field(
        default=16, metadata={"help": "Completions per question in the pre-generated dataset."}
    )
    val_fraction: float = field(
        default=0.1,
        metadata={"help": "Fraction of questions held out from the end for validation. See scripts/offpolicy_split.py."},
    )
    max_completion_length: int = field(
        default=512, metadata={"help": "Completions longer than this are truncated."}
    )
    max_prompt_length: int | None = field(
        default=None, metadata={"help": "If set, keep only the last max_prompt_length prompt tokens."}
    )
    temperature: float = field(
        default=1.0, metadata={"help": "Sampling temperature used when the dataset was generated."}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Must stay False: prompt/completion/reward don't match the model's forward()."},
    )
    logging_steps: float = field(
        default=10,
        metadata={"help": "Overrides TrainingArguments' own default of 500, matching every other config in this repo (_BaseConfig)."},
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={"help": "Zero out the loss for completions that hit max_completion_length (see class docstring)."},
    )


class _OffPolicyCollator:
    """Tokenizes raw (prompt, completion, reward) rows into padded (prompt_ids, completion_ids, ...) tensors."""

    def __init__(
        self,
        tokenizer,
        max_completion_length: int,
        max_prompt_length: int | None,
        mask_truncated_completions: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_completion_length = max_completion_length
        self.max_prompt_length = max_prompt_length
        self.mask_truncated_completions = mask_truncated_completions

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        prompt_ids_list = []
        completion_ids_list = []
        truncated_list = []
        for example in examples:
            if self.tokenizer.chat_template is not None:
                prompt_text = self.tokenizer.apply_chat_template(
                    example["prompt"], tokenize=False, add_generation_prompt=True
                )
            else:
                # Base (non-instruct) models ship with no chat_template at all -- fall back to the
                # PKU-Alignment safe-rlhf raw prompt convention ("BEGINNING OF CONVERSATION: USER: {input}
                # ASSISTANT:", see safe_rlhf/configs/constants.py), the standard practice for SFT-ing a base
                # model in this line of work.
                instruction = example["prompt"][-1]["content"]
                prompt_text = f"BEGINNING OF CONVERSATION: USER: {instruction} ASSISTANT:"
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[-self.max_prompt_length :]
            completion_ids = self.tokenizer(example["completion"], add_special_tokens=False)["input_ids"]
            # Recorded BEFORE clipping to max_completion_length: a completion already at/past that length
            # was (almost certainly) cut off mid-generation, never reaching a natural stop — see
            # OffPolicyConfig.mask_truncated_completions's docstring for why that's not an ordinary
            # negative example.
            is_truncated = len(completion_ids) >= self.max_completion_length
            truncated_list.append(is_truncated)
            # BUG FIX: `example["completion"]` is generation's own `completion_output.text` (vLLM), which
            # strips the stop token before the string is ever saved to disk — confirmed directly: 0/500
            # completions in math-llama-boxed re-tokenize with eos_token_id as their last token. Every
            # off-policy training run in this repo has therefore never shown the model a single example of
            # "predict end-of-turn here" as a supervised target — nothing ever reinforces that decision,
            # which is a complete explanation for the longer-completions/lower-natural-stop-rate signature
            # seen across every method. A NON-truncated completion (didn't hit the length cap) genuinely
            # did stop there at generation time, so the missing token is unambiguously eos_token_id; a
            # truncated one didn't stop at all, so nothing is appended for it (appending eos there would
            # fabricate a stop that never happened, teaching the opposite lesson).
            if not is_truncated:
                completion_ids = completion_ids + [self.tokenizer.eos_token_id]
            completion_ids = completion_ids[: self.max_completion_length]
            prompt_ids_list.append(prompt_ids)
            completion_ids_list.append(completion_ids)

        pad_id = self.tokenizer.pad_token_id
        max_prompt_len = max(len(p) for p in prompt_ids_list)
        max_completion_len = max(len(c) for c in completion_ids_list)
        n = len(examples)

        prompt_ids = torch.full((n, max_prompt_len), pad_id, dtype=torch.long)
        prompt_mask = torch.zeros((n, max_prompt_len), dtype=torch.long)
        completion_ids = torch.full((n, max_completion_len), pad_id, dtype=torch.long)
        completion_mask = torch.zeros((n, max_completion_len), dtype=torch.long)

        for i, (p, c) in enumerate(zip(prompt_ids_list, completion_ids_list, strict=True)):
            # Left-pad the prompt (so every prompt ends right where its completion begins), right-pad the
            # completion — matching GRPOTrainer's own padding convention for the same prompt/completion split.
            prompt_ids[i, -len(p) :] = torch.tensor(p, dtype=torch.long)
            prompt_mask[i, -len(p) :] = 1
            completion_ids[i, : len(c)] = torch.tensor(c, dtype=torch.long)
            completion_mask[i, : len(c)] = 1
            if self.mask_truncated_completions and truncated_list[i]:
                # Every per_seq_loss computation in this file divides by
                # completion_mask.sum(-1).clamp(min=1.0), so an all-zero row here contributes exactly 0 to
                # the loss rather than a division-by-zero — same effect as trl's on-policy GRPOTrainer's
                # own mask_truncated_completions, applied at the collator level so it's automatic for every
                # off-policy trainer that shares this collator, not just one.
                completion_mask[i, :] = 0

        rewards = torch.tensor([example["reward"] for example in examples], dtype=torch.float32)
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "reward": rewards,
        }


class _OffPolicyGlobalRhoCollator(_OffPolicyCollator):
    """Same as _OffPolicyCollator, plus passing through a precomputed 'rho_star' column."""

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        batch = super().__call__(examples)
        batch["rho_star"] = torch.tensor([example["rho_star"] for example in examples], dtype=torch.float32)
        return batch


class _GroupShuffledSampler(Sampler):
    """Shuffles the ORDER of `group_size`-sized question-blocks — each block's own completions stay
    contiguous and in their original relative order — instead of shuffling individual examples. Satisfies
    the same "every step's batch contains only whole groups" requirement a plain SequentialSampler does
    (see OffPolicyTrainer._get_train_sampler), without inheriting the source dataset's own row order. Some
    datasets (MATH-lighteval's train split is stored as one contiguous block per problem `type`, verified
    directly) would otherwise turn a single training epoch into an unshuffled, topic-sorted curriculum with
    no interleaving between blocks — confirmed to be the cause of MATH off-policy training regressing below
    the base model (loss curves track the type-block boundaries almost exactly; grad_norm stays flat across
    the whole epoch, ruling out an LR/instability explanation), while GSM8K (whose native order isn't
    grouped this way) trains normally under the same sequential sampler.
    """

    def __init__(self, dataset, group_size: int, seed: int):
        if len(dataset) % group_size != 0:
            raise ValueError(f"dataset length {len(dataset)} is not a multiple of group_size {group_size}")
        self.num_groups = len(dataset) // group_size
        self.group_size = group_size
        self.seed = seed

    def __len__(self):
        return self.num_groups * self.group_size

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        for g in torch.randperm(self.num_groups, generator=generator).tolist():
            yield from range(g * self.group_size, (g + 1) * self.group_size)


class OffPolicyTrainer(Trainer):
    def __init__(self, model, args: OffPolicyConfig, train_dataset, processing_class, **kwargs):
        collator = _OffPolicyCollator(
            processing_class, args.max_completion_length, args.max_prompt_length, args.mask_truncated_completions
        )
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            data_collator=collator,
            **kwargs,
        )
        # Populated by subclasses' compute_loss (e.g. OffPolicyTribeTrainer's "tribe/lambda"), flushed into
        # the actual logged/wandb output by log() below — same pattern GRPOTrainer uses for its own
        # self._metrics, which plain Trainer has no equivalent of on its own.
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def training_step(self, *args, **kwargs):
        # Same defrag call safe-rlhf's own training loop makes after every micro-batch
        # (trainers/supervised_trainer.py's train()) — this dataset's completions vary a lot in length, so
        # the CUDA caching allocator's freed blocks fragment across a gradient-accumulation window instead
        # of being reusable, which is what was actually behind the "reserved by PyTorch but unallocated"
        # OOMs at gradient_accumulation_steps>1, not accumulation having some inherent memory cost of its
        # own. Plain Trainer never calls this between micro-batches.
        loss = super().training_step(*args, **kwargs)
        torch.cuda.empty_cache()
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {}
        for key, val in self._metrics[mode].items():
            valid = [v for v in val if not math.isnan(v)]
            metrics[key] = sum(valid) / len(valid) if valid else None
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

    def _get_train_sampler(self, train_dataset=None) -> _GroupShuffledSampler:
        # Trainer's default sampler shuffles individual examples every epoch, which would scramble the
        # dataset's (question, group_size)-contiguous blocks that advantage/rho* computation relies on. The
        # dataset is generated already grouped by question (scripts/generate_offpolicy_gsm8k.py), so
        # _GroupShuffledSampler shuffles which group lands where in the epoch while keeping each group's own
        # group_size completions contiguous within it (see OffPolicyConfig.group_size's docstring for the
        # batch-size-divisibility requirement this depends on, and _GroupShuffledSampler's own docstring for
        # why a plain SequentialSampler over the source dataset's own order is NOT safe for every dataset).
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        return _GroupShuffledSampler(dataset, self.args.group_size, self.args.seed)

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None):
        # BUG FIX: prompts are left-padded (_OffPolicyCollator), so batches mixing questions of different
        # prompt lengths have real, non-padding tokens sitting at different absolute offsets into the
        # tensor. Without an explicit position_ids, transformers' LlamaModel.forward falls back to plain
        # torch.arange(seq_len) — it does NOT look at attention_mask — so every real token in a
        # shorter-than-max-in-batch prompt gets assigned a position index inflated by however much left
        # padding precedes it. Deriving position_ids from attention_mask (standard left-padding pattern)
        # fixes this regardless of how much padding precedes a given row.
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.clamp_(min=0)
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]
            position_ids_batch = position_ids[start : start + batch_size]
            outputs = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                position_ids=position_ids_batch,
                use_cache=False,
            )
            logits = outputs.logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.args.temperature
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            all_logps.append(selective_log_softmax(logits, completion_ids))
        return torch.cat(all_logps, dim=0)

    def _gather_grouped(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1 / advantage computation needs the COMPLETE group even when it's split across processes —
        # same gather-then-slice-back pattern as TribeTrainer's _generate_and_score_completions.
        return self.accelerator.gather(x)

    def _negative_sample_weight_global(self, is_negative_global: torch.Tensor, negative_fraction: float) -> torch.Tensor:
        # Shared negative_fraction ablation, used by every off-policy trainer that has one (each defines its
        # own "negative" differently — TRIBE's weight<0, TOPR/TIS's signed_reward<0, GRPO/RLOO's
        # advantage<0 — but the subsampling/oversampling mechanics below are identical, so it lives here
        # once rather than being re-derived per trainer). Returns a non-negative per-example WEIGHT (float),
        # not a boolean keep/drop mask: positive examples always get weight 1.0; negatives get a weight
        # built from negative_fraction's integer and fractional parts so the *expected* weight of each of a
        # group's own negatives is exactly negative_fraction, for ANY negative_fraction >= 0 (not just the
        # original [0, 1] "subsample" regime):
        #   - negative_fraction in [0, 1): floor=0, so every negative starts at weight 0 ("dropped") and
        #     exactly round(negative_fraction * n_neg) of each group's OWN negatives (chosen via a
        #     step-seeded per-group ranked random selection, not an independent per-example coin flip — an
        #     independent flip could, by bad luck, drop every negative in a small/unlucky group; rounding a
        #     per-group COUNT guarantees every group gets its fair, proportional share) get bumped to weight
        #     1 ("kept"). Bit-for-bit the old boolean mask, just represented as 0.0/1.0 floats.
        #   - negative_fraction == 1.0: every negative deterministically gets weight 1 (short-circuited
        #     below, both for speed and to dodge the frac==0 floating-point edge case). Note this checks
        #     ==1.0, NOT >=1.0 like the original code did — >=1.0 would silently swallow any oversampling
        #     request (negative_fraction=1.5 would wrongly short-circuit to "return all-ones").
        #   - negative_fraction > 1.0 ("oversample"): floor=k>=1, so every negative starts at weight k
        #     (already counted k times), and round(frac * n_neg) of each group's own negatives (same
        #     per-group ranked selection) get bumped to weight k+1 — e.g. negative_fraction=1.2 gives every
        #     negative weight >=1, plus a random 20% of each group's negatives weight 2, for an expected
        #     per-group average weight of exactly 1.2.
        # Only WHICH negatives get the bonus is randomized (via a step-seeded generator, so every process
        # computes the identical weight tensor without an extra collective call), not how many.
        # Callers must use this weight as BOTH the numerator coefficient and the denominator in their own
        # loss's averaging (`(per_seq_loss * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)`) —
        # and must NOT ALSO multiply anything upstream (e.g. advantages, or TRIBE's rho-based weight) by
        # this same weight a second time, or an example weighted `w` would contribute w^2 (not the intended
        # linear w) to the loss. For negative_fraction<=1 this was invisible before (0^2=0, 1^2=1,
        # idempotent for a boolean), but is a real bug once weight can exceed 1 — so it must be applied
        # exactly once, at the final average (see every compute_loss below for the fixed pattern).
        assert negative_fraction >= 0, "negative_fraction must be non-negative"
        if negative_fraction == 1.0:
            return is_negative_global.new_ones(is_negative_global.shape, dtype=torch.float)
        floor_f = math.floor(negative_fraction)
        frac = negative_fraction - floor_f
        G = is_negative_global.numel() // self.args.group_size
        is_neg_grouped = is_negative_global.view(G, self.args.group_size)
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

    def _pos_neg_floor_sample_weight(
        self, is_negative_global: torch.Tensor, min_pos_neg_ratio: float, seed: int
    ) -> torch.Tensor:
        """
        Deterministic pos:neg LOWER-BOUND gate — a mutually-exclusive alternative to negative_fraction's
        own gating (see _negative_sample_weight_global's docstring for that one). negative_fraction targets
        an EXPECTED per-group keep-fraction, so the REALIZED pos:neg ratio in any given batch still varies
        with whatever happens to land in it; this instead directly targets a MINIMUM realized ratio for
        the batch as a whole. Unlike negative_fraction, this has no per-trainer group_size dependence at
        all — "how many positives vs. negatives are in this batch" is a property of the whole gathered
        batch regardless of any per-question sub-grouping, so this one method is shared unmodified by every
        TRIBE trainer (no ragged/flat-specific adaptation needed, unlike _negative_sample_weight_global).

        Never touches positives — they always get weight 1, exactly like negative_fraction. If the batch's
        current pos:neg ratio already meets or exceeds min_pos_neg_ratio, every negative is ALSO kept
        (weight 1, same as negative_fraction=1.0) — this is a floor, not a target, so an already
        positive-heavy-enough batch is left alone. Otherwise, keeps exactly
        floor(n_positive / min_pos_neg_ratio) negatives (a per-step-reseeded random ranking, same mechanism
        negative_fraction's own selection uses) and drops (weight 0) the rest, so the REALIZED ratio in this
        specific batch is deterministically >= min_pos_neg_ratio — not just an expected value the way
        negative_fraction's is.

        Args:
            is_negative_global (`Tensor` of shape `(N,)`):
                Boolean, True for this trainer's own definition of "negative" (TRIBE's weight<0).
            min_pos_neg_ratio (`float`):
                Minimum acceptable positives-per-negative in the batch. E.g. 1.0 means "at least as many
                positives as kept negatives"; 2.0 means "at least twice as many positives as kept
                negatives".
            seed (`int`):
                Caller supplies this (self.state.global_step for a per-batch trainer re-solving every
                step, or a fixed seed like args.seed for a one-time global solve) — this method has no
                opinion on which, unlike _negative_sample_weight_global which always uses global_step.
        """
        n_positive = (~is_negative_global).sum()
        n_negative = is_negative_global.sum()
        if n_negative == 0 or n_positive.float() / n_negative.float() >= min_pos_neg_ratio:
            return is_negative_global.new_ones(is_negative_global.shape, dtype=torch.float)
        n_negative_keep = int((n_positive.float() / min_pos_neg_ratio).floor().item())
        gen = torch.Generator(device=is_negative_global.device).manual_seed(seed)
        rand_vals = torch.rand(is_negative_global.shape, generator=gen, device=is_negative_global.device)
        rand_vals = rand_vals.masked_fill(~is_negative_global, float("inf"))
        rank = rand_vals.argsort().argsort()
        keep = (~is_negative_global) | (is_negative_global & (rank < n_negative_keep))
        return keep.float()

    def _prepare_frozen_ref_model(self, ref_model_name_or_path: str | None = None):
        # Every off-policy trainer needing a frozen behavioral-policy copy (GRPO's ref_model, TOPR/TRIBE's
        # mu_model) builds it the same way — shared here rather than duplicated three times because a
        # DeepSpeed-correctness fix missed in one copy would be a silent, hard-to-notice bug in the others.
        # Owns the deepcopy itself (callers used to deepcopy before calling this) — under ZeRO-3 the copy
        # must happen while parameters are gathered, see below, so this can't be handed an already-made copy.
        #
        # ref_model_name_or_path: only needed when the model being TRAINED is not the model that actually
        # GENERATED the offline dataset (e.g. a weak-learner setup: training Llama-3.2-1B-Instruct on data
        # generated by Llama-3.2-3B-Instruct). Default None preserves the original assumption — mu/ref IS
        # the model's own initialization, correct whenever generator and learner are the same model — by
        # deep-copying self.model exactly as before. When given, loads that model fresh from disk/hub
        # instead of copying self.model, since self.model is a different (and differently-shaped) network
        # from the actual behavioral policy in that case.
        if ref_model_name_or_path is not None:
            dtype = torch.bfloat16 if self.args.bf16 else torch.float32
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name_or_path, dtype=dtype)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad_(False)
            if is_deepspeed_zero3_enabled():
                import deepspeed

                ref_model, *_ = deepspeed.initialize(
                    model=ref_model,
                    config={
                        "train_batch_size": None,
                        "train_micro_batch_size_per_gpu": 1,
                        "gradient_accumulation_steps": 1,
                        "zero_optimization": {"stage": 3},
                        "bf16": {"enabled": self.args.bf16},
                    },
                )
                return ref_model
            return self.accelerator.prepare_model(ref_model, evaluation_mode=True)

        if is_deepspeed_zero3_enabled():
            # self.model was loaded via from_pretrained() while ZeRO-3 was already globally active (set the
            # moment OffPolicyConfig was constructed, before the model load), so each rank only ever
            # materialized ITS OWN SHARD — the full tensor was never assembled anywhere. A plain deepcopy in
            # this state clones whatever fragment each rank currently holds (confirmed: a `ds_numel: 0`
            # assertion — a genuinely empty shard), not the true parameter, and no amount of wrapping that
            # broken copy in deepspeed.initialize() afterward fixes data that was already garbage at copy
            # time. Fix: gather every parameter to its full form first, deepcopy while gathered, then give
            # the copy its own DeepSpeed engine (accelerator.prepare_model(..., evaluation_mode=True) leaves
            # a second model's parameters stuck partitioned — confirmed separately: `embed_tokens.weight`
            # came back a flat 1-D shard, not a 2-D matrix, on its first forward call — because it never
            # gets its own ZeRO-3 gather/scatter hooks). Same overall pattern safe-rlhf's own
            # algorithms/dpo_pop_fixed_ema/trainer.py uses for its reference model.
            import deepspeed
            from deepspeed.runtime.zero.partition_parameters import GatheredParameters

            with GatheredParameters(list(self.model.parameters()), modifier_rank=None):
                ref_model = copy.deepcopy(self.model)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad_(False)
            ref_model, *_ = deepspeed.initialize(
                model=ref_model,
                config={
                    "train_batch_size": None,
                    "train_micro_batch_size_per_gpu": 1,
                    "gradient_accumulation_steps": 1,
                    "zero_optimization": {"stage": 3},
                    "bf16": {"enabled": self.args.bf16},
                },
            )
            return ref_model
        ref_model = copy.deepcopy(self.model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)
        return self.accelerator.prepare_model(ref_model, evaluation_mode=True)

    def _log_reward_metric(self, rewards_local: torch.Tensor) -> None:
        # Gathered (not just this process's local shard) for an accurate global-batch mean under DDP —
        # same reasoning as _gather_grouped, applied to a plain monitoring metric instead of a loss input.
        mode = "train" if self.model.training else "eval"
        rewards_global = self._gather_grouped(rewards_local)
        self._metrics[mode]["reward"].append(rewards_global.mean().item())

    def _log_logp_stats(self, per_token_logps: torch.Tensor, completion_mask: torch.Tensor, prefix: str) -> None:
        # min/mean/std/max of this step's per-token log-probs over valid (non-padding) positions, gathered
        # across processes — the direct diagnostic for whether a method's negative-example treatment is
        # driving log-probs to extreme values (TRIBE's static reward-only weight) vs. staying contained
        # (TOPR's ratio-based self-attenuating weight). Same accelerator.reduce(sum) pattern as the
        # on-policy TribeTrainer's own _global_sum_and_count, for exact consistency.
        mode = "train" if self.model.training else "eval"
        valid = per_token_logps.detach()[completion_mask.bool()]
        local_sum = valid.sum()
        local_sumsq = (valid * valid).sum()
        local_count = torch.tensor(float(valid.numel()), device=valid.device)
        local_min = valid.min() if valid.numel() > 0 else torch.full((), float("inf"), device=per_token_logps.device)
        local_max = valid.max() if valid.numel() > 0 else torch.full((), float("-inf"), device=per_token_logps.device)

        global_sum, global_sumsq, global_count = self.accelerator.reduce(
            torch.stack([local_sum, local_sumsq, local_count]), reduction="sum"
        )
        global_min = self._gather_grouped(local_min.unsqueeze(0)).min()
        global_max = self._gather_grouped(local_max.unsqueeze(0)).max()

        mean = (global_sum / global_count.clamp(min=1)).item()
        var = (global_sumsq / global_count.clamp(min=1) - mean**2).clamp(min=0)
        self._metrics[mode][f"{prefix}/logp_mean"].append(mean)
        self._metrics[mode][f"{prefix}/logp_std"].append(var.sqrt().item())
        self._metrics[mode][f"{prefix}/logp_min"].append(global_min.item())
        self._metrics[mode][f"{prefix}/logp_max"].append(global_max.item())

    def _local_slice(self, x_global: torch.Tensor, local_size: int) -> torch.Tensor:
        start = self.accelerator.process_index * local_size
        return x_global[start : start + local_size]

    def _group_relative_advantage(self, rewards_local: torch.Tensor) -> torch.Tensor:
        """GRPO/TRIBE-style: A_i = reward_i - mean(rewards in its group), group mean includes itself."""
        rewards_global = self._gather_grouped(rewards_local)
        group_size = self.args.group_size
        G = rewards_global.numel() // group_size
        rewards_grouped = rewards_global.view(G, group_size)
        A_global = (rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)).reshape(-1)
        return self._local_slice(A_global, rewards_local.numel())

    def _leave_one_out_advantage(self, rewards_local: torch.Tensor) -> torch.Tensor:
        """RLOO-style: A_i = reward_i - mean(other rewards in its group), excluding itself."""
        rewards_global = self._gather_grouped(rewards_local)
        group_size = self.args.group_size
        G = rewards_global.numel() // group_size
        rewards_grouped = rewards_global.view(G, group_size)
        group_sum = rewards_grouped.sum(dim=1, keepdim=True)
        leave_one_out_mean = (group_sum - rewards_grouped) / (group_size - 1)
        A_global = (rewards_grouped - leave_one_out_mean).reshape(-1)
        return self._local_slice(A_global, rewards_local.numel())


class OffPolicyReinforceTrainer(OffPolicyTrainer):
    """
    REINFORCE + RLOO's leave-one-out baseline, no importance-sampling correction. NOT literally TOPR's own
    "naive REINFORCE" baseline (https://huggingface.co/papers/2503.14286, Section 2.1's R(tau)*grad(log
    pi(tau)) has no baseline at all, raw signed reward) — RLOO's baseline is itself a real
    variance-reduction ingredient TOPR's literal naive formula doesn't have. Kept as its own baseline
    (does RLOO's baseline alone, with no ratio correction, already help?) alongside
    OffPolicyNaiveReinforceTrainer below, which reproduces TOPR's Eq. 6 exactly.
    """

    def __init__(self, model, args: OffPolicyConfig, train_dataset, processing_class, negative_fraction: float = 1.0, **kwargs):
        super().__init__(model, args, train_dataset, processing_class, **kwargs)
        # See OffPolicyTrainer._negative_sample_weight_global's docstring. "Negative" here means
        # advantage<0 (below-group-leave-one-out-mean reward).
        self.negative_fraction = negative_fraction

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        advantages_global = self._gather_grouped(self._leave_one_out_advantage(inputs["reward"]))
        sample_weight_global = self._negative_sample_weight_global(advantages_global < 0, self.negative_fraction)
        advantages = self._local_slice(advantages_global, inputs["reward"].numel())
        sample_weight = self._local_slice(sample_weight_global, inputs["reward"].numel())

        # Naive: -A * log pi_theta(completion), no pi_theta/mu ratio anywhere.
        per_token_loss = -advantages.unsqueeze(-1) * per_token_logps * completion_mask
        per_seq_loss = per_token_loss.sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        loss = (per_seq_loss * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)

        self._log_reward_metric(inputs["reward"])

        return (loss, None) if return_outputs else loss


class OffPolicyNaiveReinforceTrainer(OffPolicyTrainer):
    """
    TOPR's own "naive REINFORCE" baseline, literally (https://huggingface.co/papers/2503.14286, Section
    2.1: R(tau)*grad(log pi(tau))) — raw SIGNED reward (+1/-1, not the 0/1 stored on disk), no baseline
    subtraction of any kind (not even RLOO's, unlike OffPolicyReinforceTrainer above), no
    importance-sampling ratio. The purest possible strawman: the on-policy REINFORCE update applied
    directly to off-policy data with zero correction for the mismatch.
    """

    def __init__(self, model, args: OffPolicyConfig, train_dataset, processing_class, negative_fraction: float = 1.0, **kwargs):
        super().__init__(model, args, train_dataset, processing_class, **kwargs)
        # See OffPolicyTrainer._negative_sample_weight_global's docstring. "Negative" here means
        # signed_reward<0 (i.e. every incorrect completion, since reward is binary).
        self.negative_fraction = negative_fraction

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        signed_reward_global = self._gather_grouped(2 * inputs["reward"] - 1)
        sample_weight_global = self._negative_sample_weight_global(signed_reward_global < 0, self.negative_fraction)
        signed_reward = self._local_slice(signed_reward_global, inputs["reward"].numel())
        sample_weight = self._local_slice(sample_weight_global, inputs["reward"].numel())

        per_token_loss = -signed_reward.unsqueeze(-1) * per_token_logps * completion_mask
        per_seq_loss = per_token_loss.sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        loss = (per_seq_loss * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)

        self._log_reward_metric(inputs["reward"])

        return (loss, None) if return_outputs else loss


class OffPolicyGRPOTrainer(OffPolicyTrainer):
    """
    GRPO's clipped importance-ratio objective, off-policy: `pi_old` is the frozen behavioral policy mu (a
    frozen reference copy loaded once at init, never updated) instead of GRPO's normal on-policy meaning.
    """

    def __init__(self, model, args: OffPolicyConfig, train_dataset, processing_class, epsilon: float = 0.2, negative_fraction: float = 1.0, ref_model_name_or_path: str | None = None, **kwargs):
        super().__init__(model, args, train_dataset, processing_class, **kwargs)
        self.epsilon = epsilon
        # mu == the model's own initialization (it's the behavioral policy that generated this dataset) —
        # frozen for the whole run, never synced to the training model again. ref_model_name_or_path
        # overrides this when the model being trained differs from the model that generated the dataset
        # (see _prepare_frozen_ref_model's docstring).
        self.ref_model = self._prepare_frozen_ref_model(ref_model_name_or_path)
        # See OffPolicyTrainer._negative_sample_weight_global's docstring. "Negative" here means
        # advantage<0 (below-group-average reward).
        self.negative_fraction = negative_fraction

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        with torch.no_grad():
            mu_per_token_logps = self._get_per_token_logps(
                self.ref_model, input_ids, attention_mask, logits_to_keep
            )
        advantages_global = self._gather_grouped(self._group_relative_advantage(inputs["reward"]))
        sample_weight_global = self._negative_sample_weight_global(advantages_global < 0, self.negative_fraction)
        advantages = self._local_slice(advantages_global, inputs["reward"].numel())
        sample_weight = self._local_slice(sample_weight_global, inputs["reward"].numel())

        ratio = torch.exp(per_token_logps - mu_per_token_logps)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss = -torch.min(ratio * advantages.unsqueeze(-1), clipped_ratio * advantages.unsqueeze(-1))
        per_token_loss = per_token_loss * completion_mask
        per_seq_loss = per_token_loss.sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        loss = (per_seq_loss * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)

        self._log_reward_metric(inputs["reward"])

        return (loss, None) if return_outputs else loss


class _PolyakEMACallback(TrainerCallback):
    """After every real optimizer step, folds the live model's parameters into their own running Polyak
    average and writes that average straight back into the model — the trained policy itself becomes a
    Polyak/EMA blend of its past iterates (`p <- tau*ema + (1-tau)*p`, then `ema <- p`), not a passive
    shadow copy tracked on the side. Fires on `on_step_end`, which transformers.Trainer only calls once
    per real optimizer update (i.e. after gradient-accumulation resolves), not once per micro-batch.

    Correct under any ZeRO stage, including 3: guards every parameter access with the same `ds_status`/
    `GatheredParameters` pattern safe-rlhf's own EMA trainer uses
    (safe_rlhf/algorithms/dpo_pop_fixed_ema/trainer.py's sync_model_to_ema). Under ZeRO-3 a parameter is
    only a valid full tensor while gathered — touching `p.data` outside that context (as a naive version
    of this callback did, and was confirmed unsafe: OOM'd even under plain DDP for unrelated reasons, but
    would have silently blended a partial/placeholder tensor under ZeRO-3) reads/writes the wrong thing.
    `GatheredParameters(p, modifier_rank=None)` materializes the full tensor identically on every rank for
    the duration of the block and re-partitions whatever was written back into `p.data` into shards on
    exit. Under ZeRO-0/1/2 (no parameter partitioning), `ds_status` is never NOT_AVAILABLE, so this falls
    straight through to the same direct-access path plain DDP uses.
    """

    def __init__(self, tau: float):
        self.tau = tau
        self.ema: list[torch.Tensor] | None = None

    @staticmethod
    def _gathered(p):
        from deepspeed.runtime.zero.partition_parameters import GatheredParameters, ZeroParamStatus

        if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            return GatheredParameters(p, modifier_rank=None)
        return contextlib.nullcontext()

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        ema = []
        with torch.no_grad():
            for p in model.parameters():
                with self._gathered(p):
                    ema.append(p.data.clone())
        self.ema = ema

    def on_step_end(self, args, state, control, model=None, **kwargs):
        with torch.no_grad():
            for p, ema_p in zip(model.parameters(), self.ema):
                with self._gathered(p):
                    ema_p.mul_(self.tau).add_(p.data, alpha=1 - self.tau)
                    p.data.copy_(ema_p)


class OffPolicyTribeTrainer(OffPolicyTrainer):
    """TRIBE's Stage 1 (tribe/offpolicy_stage1.py) + the same (rho*-1)-weighted NLL Stage 2 loss as on-policy."""

    def __init__(self, model, args: OffPolicyConfig, train_dataset, processing_class, beta: float = 0.1, trust_region_eps: float = 0.05, use_unlikelihood: bool = False, divergence: str = "kl_new_old", use_polyak_ema: bool = False, polyak_ema_tau: float = 0.99, rho_weight_offset: float = 1.0, negative_fraction: float = 1.0, min_pos_neg_ratio: float | None = None, negative_logp_floor: float = math.log(1e-8), negative_switch_threshold: float | None = None, ref_model_name_or_path: str | None = None, log_f_div_mu: bool = True, use_chi_squared_beta0_closed_form: bool = False, **kwargs):
        super().__init__(model, args, train_dataset, processing_class, **kwargs)
        self.beta = beta
        self.trust_region_eps = trust_region_eps
        self.divergence = divergence
        self.use_chi_squared_beta0_closed_form = use_chi_squared_beta0_closed_form
        # Mutually exclusive with negative_fraction (see OffPolicyTrainer._pos_neg_floor_sample_weight's
        # own docstring for why they're two different gating modes, not stackable) — pick one per run.
        if min_pos_neg_ratio is not None and negative_fraction != 1.0:
            raise ValueError(
                "min_pos_neg_ratio and negative_fraction are mutually exclusive — leave negative_fraction "
                "at its default (1.0) when setting min_pos_neg_ratio."
            )
        self.min_pos_neg_ratio = min_pos_neg_ratio
        # Floor (in log-prob space) the floored-NLL branch clamps a token's log-prob at before the loss
        # stops applying further downward pressure on it (see compute_loss below). Default log(1e-8) ~=
        # -18.42 was found (by directly comparing training-time logp_mean/logp_min against DPO/SimPO's own
        # chosen/rejected log-probs at negative_fraction=1.0) to be far too permissive: the batch-average
        # token log-prob crosses this floor by ~30% through the epoch and never stabilizes, while the
        # worst tokens drift to logp ~= -85 (4x past the floor) with no sign of leveling off — unlike DPO's
        # rejected log-prob, which plateaus. A tighter (less negative) floor stops the loss from licensing
        # that much downward pressure in the first place.
        self.negative_logp_floor = negative_logp_floor
        # Stage 2's weight is normally rho*-1 (offset=1, TRIBE's own baseline-centered convention: reward
        # exactly at the group mean gets weight 0). offset=0 tests the raw-rho* variant instead (weight is
        # never negative for chi_squared, since its rho* is clamped at 0) — every example gets some
        # non-negative push toward its own completion, only scaled by how far above/below average its
        # reward was, instead of below-average examples getting an actively negative (penalizing) weight.
        self.rho_weight_offset = rho_weight_offset
        # Ablation: fraction of negative (below-group-average) examples to actually keep in Stage 2's loss
        # each step — 1.0 (default) keeps all of them, reproducing prior behavior exactly. See compute_loss.
        self.negative_fraction = negative_fraction
        # If True, weight<0 (below-group-average-reward) samples use the unlikelihood trick
        # (-log1p(-p), self-attenuating as p->0, same mechanism RAFT's own negative-example fix uses)
        # instead of the plain floored NLL below. Uses only pi_theta's own probability, no mu/ratio needed
        # — keeps TRIBE's "no behavioral logits required" property, unlike a ratio-based taper would.
        self.use_unlikelihood = use_unlikelihood
        # Alternative to both use_unlikelihood and the floored branch: per-token switch between plain NLL
        # and unlikelihood for weight<0 samples, based on that token's OWN current probability p (not a
        # fixed threshold on training progress). Plain NLL's gradient magnitude is |weight|*(1-p) — weak
        # while p is still high, but grows toward its max as p->0, which is why the floored branch needs an
        # artificial hard floor to ever stop (confirmed empirically: swept negative_logp_floor across many
        # orders of magnitude, none of them stabilized logp_mean/logp_min the way DPO/SimPO's rejected
        # logp naturally plateaus). Unlikelihood's gradient magnitude is |weight|*p — the opposite profile:
        # dangerous while p is still high (this is why the pure use_unlikelihood mode above collapses
        # catastrophically under pass@k, see results_pass_at_k.tex's discussion), but naturally vanishes as
        # p->0. Using plain NLL for p > negative_switch_threshold and unlikelihood for p <=
        # negative_switch_threshold uses each one exactly where it's safe and avoids each one exactly where
        # it's dangerous. At threshold=0.5 the two branches' gradient magnitudes are equal (both 0.5) at the
        # switch point, so there's no discontinuity — the combined gradient magnitude as a function of p is
        # simply min(p, 1-p): zero at both p->0 and p->1, peaking in the middle. This makes
        # negative_logp_floor's artificial clamp unnecessary (self-attenuation replaces it), so the two are
        # mutually exclusive with use_unlikelihood and with each other in compute_loss below.
        self.negative_switch_threshold = negative_switch_threshold
        if negative_switch_threshold is not None:
            assert not use_unlikelihood, "negative_switch_threshold and use_unlikelihood are mutually exclusive"
        # mu == the model's own initialization (the behavioral policy that generated this dataset), frozen
        # for the whole run — not used by Stage 1/2's math at all (c is fixed at 0, no reference term in
        # the loss), only to monitor how far pi_theta has actually drifted from it (tribe/f_div_mu below).
        # Same frozen-copy pattern as OffPolicyGRPOTrainer's own ref_model. ref_model_name_or_path overrides
        # this when the model being trained differs from the model that generated the dataset (see
        # _prepare_frozen_ref_model's docstring).
        # log_f_div_mu=False skips building mu_model at all (not just skipping the forward pass) — purely
        # diagnostic, and both the static memory of holding a second full frozen model copy and its
        # periodic forward pass (see compute_loss below) have been confirmed to eat enough memory margin to
        # OOM a ZeRO-3 run with long completions (UltraFeedback's max_completion_length=1024) even though
        # nothing about the actual loss depends on it.
        self.log_f_div_mu = log_f_div_mu
        self.mu_model = self._prepare_frozen_ref_model(ref_model_name_or_path) if log_f_div_mu else None
        self._warned_f_div_mu_overflow = False
        # Cumulative counts for Stage 1's cvxpy solve (any divergence other than "kl_new_old") not fully
        # succeeding on a batch's reward configuration — see compute_loss's own handling below and
        # tribe.stage1.compute_rho_star's docstring. Only relevant when self.divergence != "kl_new_old"
        # (that path is the closed form, never fails). "inaccurate" = accepted an approximate solution and
        # used it; "skip" = the solve genuinely failed (infeasible/unbounded/solver_error/...) and this
        # batch contributed no gradient at all.
        self._stage1_inaccurate_count = 0
        self._stage1_skip_count = 0
        self.use_polyak_ema = use_polyak_ema
        if use_polyak_ema:
            self.add_callback(_PolyakEMACallback(polyak_ema_tau))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Kept raw (unfloored) here — used as-is below for diagnostics (f_div_mu, logp stats) so they
        # reflect the model's true behavior; any flooring/clamping happens only inside the loss branches.
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        rewards_global = self._gather_grouped(inputs["reward"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rho_global, lam = compute_rho_star_offpolicy(
                rewards_global,
                self.args.group_size,
                self.beta,
                self.trust_region_eps,
                divergence=self.divergence,
                use_chi_squared_beta0_closed_form=self.use_chi_squared_beta0_closed_form,
            )
        if any("optimal_inaccurate" in str(w.message) for w in caught):
            # Stage 1's cvxpy solve (see compute_rho_star's own docstring) accepted an approximate
            # solution for this batch rather than a fully-converged one — still used (rho_global is not
            # None here), just tracked separately from genuine skips below for later analysis.
            self._stage1_inaccurate_count += 1
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["tribe/stage1_inaccurate_total"].append(self._stage1_inaccurate_count)
        if rho_global is None:
            # Stage 1 failed to solve this batch (see compute_rho_star's own warning for why) — skip it
            # rather than crash the run. rewards_global was gathered identically on every process, so the
            # solve (and this skip decision) is identical on every rank too — no risk of ranks disagreeing
            # about whether to skip. `0.0 * per_token_logps.sum()` keeps every parameter that participated
            # in the forward pass connected to the loss (still contributes a zero gradient) instead of
            # returning a fresh disconnected zero tensor, which DDP/DeepSpeed would otherwise flag as
            # unused parameters.
            self._stage1_skip_count += 1
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["tribe/stage1_skipped_total"].append(self._stage1_skip_count)
            loss = 0.0 * per_token_logps.sum()
            return (loss, None) if return_outputs else loss
        weight_global = rho_global - self.rho_weight_offset

        # See OffPolicyTrainer._negative_sample_weight_global's docstring for the full mechanics/rationale.
        # "Negative" here means weight<0 (below-group-average reward, TRIBE's own definition). Deliberately
        # NOT folded into weight_global itself (unlike the old code) — sample_weight only ever gates "how
        # many times this example counts in the batch average", it must never touch the magnitude of
        # TRIBE's own reward-derived weight (rho* - offset), which is a separate quantity.
        if self.min_pos_neg_ratio is not None:
            sample_weight_global = self._pos_neg_floor_sample_weight(
                weight_global < 0, self.min_pos_neg_ratio, seed=self.state.global_step
            )
        else:
            sample_weight_global = self._negative_sample_weight_global(weight_global < 0, self.negative_fraction)

        weight = self._local_slice(weight_global, inputs["reward"].numel())
        sample_weight = self._local_slice(sample_weight_global, inputs["reward"].numel())

        if self.negative_switch_threshold is not None:
            # Positive (weight>=0): plain NLL, unchanged — pushing p->1 is bounded, no self-attenuation
            # needed. Negative (weight<0): per-token switch between plain NLL (while p is still above
            # negative_switch_threshold) and unlikelihood (once p has dropped to/below it) — see
            # negative_switch_threshold's own docstring in __init__ for the full gradient-shape rationale.
            # Same fp32-upcast requirement as the use_unlikelihood branch below (bf16 can't distinguish 1.0
            # from 1.0-1e-6).
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
            # Positive (weight>=0): plain NLL, unchanged — pushing p->1 is bounded, no self-attenuation
            # needed. Negative (weight<0): unlikelihood (-log1p(-p)) instead of raw -logp — self-attenuates
            # toward 0 as p->0 (the penalty backs off once it's succeeded), unlike a fixed-strength penalty
            # that keeps pushing regardless of how far p has already dropped. selective_log_softmax computes
            # log_softmax directly in bf16 when the model runs in bf16 (trl/trainer/utils.py) rather than
            # upcasting — bf16 can't distinguish 1.0 from 1.0-1e-6 (verified: both round to exactly 1.0), so
            # p must be explicitly upcast to float32 before the clamp, or the clamp is silently a no-op.
            positive_mask = (weight >= 0).float().unsqueeze(-1)
            negative_mask = (weight < 0).float().unsqueeze(-1)
            positive_loss = -weight.clamp(min=0).unsqueeze(-1) * per_token_logps
            p = per_token_logps.float().exp().clamp(max=1 - 1e-6)
            unlikelihood = -torch.log1p(-p)
            negative_loss = weight.clamp(max=0).abs().unsqueeze(-1) * unlikelihood
            per_token_loss = (positive_loss * positive_mask + negative_loss * negative_mask) * completion_mask
        else:
            # Floor probability at self.negative_logp_floor: once a token's probability is pushed below
            # this, the clamp makes the loss flat w.r.t. further decreases (zero gradient beyond the
            # floor), stopping compounding downward pressure on already-very-unlikely tokens instead of
            # the unbounded -logp pressure a negative weight would otherwise keep applying. Crude compared
            # to the unlikelihood mode above, but a cheap default that doesn't need the fp32 upcast.
            per_token_logps_floored = per_token_logps.clamp(min=self.negative_logp_floor)
            per_token_loss = -weight.unsqueeze(-1) * per_token_logps_floored * completion_mask

        # sample_weight.sum() in the denominator instead of a plain .mean(): dropped (ablation-zeroed)
        # examples must not silently shrink every kept example's effective weight by diluting the average,
        # and oversampled (sample_weight>1) examples must count proportionally more — this reduces to an
        # exact .mean() when negative_fraction=1.0 (sample_weight is all-ones), reproducing prior behavior.
        per_seq_loss = per_token_loss.sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        loss = (per_seq_loss * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["tribe/lambda"].append(lam.item())
        # rho_global is already the full gathered batch (identical on every process, same as lam above), so
        # these need no further reduction — same rationale as the on-policy TribeTrainer's own rho_star
        # logging. Mean should sit close to 1 by construction; std/min/max show how much Stage 1 is actually
        # reweighting this step.
        self._metrics[mode]["tribe/rho_star_mean"].append(rho_global.mean().item())
        self._metrics[mode]["tribe/rho_star_std"].append(rho_global.std().item())
        self._metrics[mode]["tribe/rho_star_min"].append(rho_global.min().item())
        self._metrics[mode]["tribe/rho_star_max"].append(rho_global.max().item())

        # pos:neg ratio AFTER negative_fraction gating — i.e., counting only what actually carries nonzero
        # weight into the gradient this step, not the raw batch composition. Positives (weight>=0) always
        # carry sample_weight=1 by construction (only negatives ever get gated), so n_positive is just their
        # count; n_negative_kept sums sample_weight over weight<0 rows (not just a boolean count) since
        # negative_fraction's ranked selection can bump a negative's weight above 1 for negative_fraction>1
        # (see _negative_sample_weight_global's own docstring) — summing the weight itself is the honest
        # "how many negative-example-equivalents actually contributed" figure, not just how many were
        # nonzero. Purely additive logging — does not affect the loss/gradient computed above.
        is_negative_global = weight_global < 0
        n_positive_global = (~is_negative_global).sum().item()
        n_negative_kept_global = sample_weight_global[is_negative_global].sum().item()
        self._metrics[mode]["tribe/n_positive"].append(n_positive_global)
        self._metrics[mode]["tribe/n_negative_kept"].append(n_negative_kept_global)
        self._metrics[mode]["tribe/pos_neg_kept_ratio"].append(
            n_positive_global / max(n_negative_kept_global, 1e-8)
        )

        # How far pi_theta has actually drifted from the frozen mu so far — the off-policy equivalent of
        # on-policy TribeTrainer's tribe/f_div_old, but simpler: mu never refreshes here (unlike pi_old,
        # which resets every cycle on-policy), so there's no separate "end of cycle" variant needed — this
        # one running metric already captures the whole accumulating-drift story. Sequence-level (not
        # token-level), matching Stage 1's own constraint granularity.
        #
        # Purely diagnostic — not used by Stage 1/2's math at all (c is fixed at 0, no reference term in
        # the loss) — so mu_model's full second forward pass only needs to run often enough to have a fresh
        # value at each log, not every single step. That forward pass was eating memory margin for no
        # training-relevant reason (confirmed: it's the reason this trainer OOMs the moment
        # gradient_accumulation_steps>1 is used, unlike safe-rlhf's DPO trainer, whose own reference-model
        # forward pass is load-bearing for its loss and so can't be made this cheap).
        if self.log_f_div_mu and self.state.global_step % max(int(self.args.logging_steps), 1) == 0:
            with torch.no_grad():
                mu_per_token_logps = self._get_per_token_logps(self.mu_model, input_ids, attention_mask, logits_to_keep)
                seq_logps_theta = (per_token_logps.detach().float() * completion_mask).sum(-1)
                seq_logps_mu = (mu_per_token_logps.float() * completion_mask).sum(-1)
                rho_theta = torch.exp(seq_logps_theta - seq_logps_mu)
                # chi_squared (0.5*(rho-1)**2), not kl_new_old (rho*log(rho)): rho_theta underflowing to
                # exactly 0.0 (pi_theta assigns ~0 probability relative to mu, common once training has
                # moved on) makes kl_new_old's rho*log(rho) evaluate 0.0*(-inf) = NaN in floating point,
                # not the mathematical limit of 0 — silently poisoning this metric long before pi_theta has
                # actually diverged far enough to matter. chi_squared is a plain polynomial in rho (no
                # log()), so it stays finite and meaningful at rho=0 and only goes to inf on genuine
                # overflow (rho_theta -> inf), the actually-informative failure mode.
                f_val = TRUST_REGION_F["chi_squared"](rho_theta)
            if not self._warned_f_div_mu_overflow and torch.isinf(f_val).any():
                # exp() overflows once the log-prob sum vs mu exceeds ~88 — pi_theta has diverged from the
                # frozen behavioral policy enough to blow past float range, not just drifted. tribe/f_div_mu
                # will silently read as None from here on (wandb serializes inf as null), so warn once rather
                # than let that look like a logging bug instead of a real divergence signal.
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


class OffPolicyTOPRTrainer(OffPolicyTrainer):
    """
    TOPR (canonical), https://huggingface.co/papers/2503.14286: SFT-style update (no ratio) on
    positive-reward examples, truncated importance sampling (ratio clipped to [0,1]) on negative-reward
    examples. Reward here is signed (+1/-1), converted from this repo's stored 0.0/1.0 convention.
    """

    def __init__(self, model, args: OffPolicyConfig, train_dataset, processing_class, negative_fraction: float = 1.0, ref_model_name_or_path: str | None = None, **kwargs):
        super().__init__(model, args, train_dataset, processing_class, **kwargs)
        # mu == the model's own initialization (the behavioral policy that generated this dataset), frozen
        # for the whole run — same frozen-copy pattern as OffPolicyGRPOTrainer's own ref_model.
        # ref_model_name_or_path overrides this when the model being trained differs from the model that
        # generated the dataset (see _prepare_frozen_ref_model's docstring).
        self.mu_model = self._prepare_frozen_ref_model(ref_model_name_or_path)
        # See OffPolicyTrainer._negative_sample_weight_global's docstring. "Negative" here means
        # signed_reward<0 (every incorrect completion).
        self.negative_fraction = negative_fraction

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        seq_logps_theta = (per_token_logps.float() * completion_mask).sum(-1)
        token_counts = completion_mask.sum(-1).clamp(min=1)

        with torch.no_grad():
            mu_per_token_logps = self._get_per_token_logps(self.mu_model, input_ids, attention_mask, logits_to_keep)
            seq_logps_mu = (mu_per_token_logps.float() * completion_mask).sum(-1)
            ratio = torch.exp(seq_logps_theta.detach() - seq_logps_mu)
            signed_reward = 2 * inputs["reward"] - 1
            is_positive = signed_reward >= 0
            # Positive: weight=1 (plain SFT gradient, no ratio at all). Negative: ratio clipped to [0,1]
            # (TIS) — ratio is always >=0, so clamp(max=1.0) alone implements clip(ratio, 0, 1).
            weight = torch.where(is_positive, torch.ones_like(ratio), ratio.clamp(max=1.0))

        sample_weight_global = self._negative_sample_weight_global(
            self._gather_grouped(signed_reward) < 0, self.negative_fraction
        )
        sample_weight = self._local_slice(sample_weight_global, inputs["reward"].numel())

        per_seq_loss = -weight * signed_reward * (seq_logps_theta / token_counts)
        loss = (per_seq_loss * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)

        self._log_logp_stats(per_token_logps, completion_mask, prefix="topr")
        self._log_reward_metric(inputs["reward"])

        return (loss, None) if return_outputs else loss


class OffPolicyTribeGlobalTrainer(OffPolicyTrainer):
    """
    TRIBE off-policy, global Stage 1 variant. rho* depends only on reward in this regime (c=0 always, see
    tribe/offpolicy_stage1.py) — it never depends on the model at all. So instead of re-solving it noisily
    from each small mini-batch (OffPolicyTribeTrainer's approach), solve it ONCE over the full dataset's
    rewards — one shared lambda across all groups, matching Stage 1's own single-global-trust-region
    derivation properly instead of an ad-hoc per-step approximation of it — and bake the result into the
    dataset as a fixed per-example weight. Stage 2 then becomes a plain weighted-SFT pass: no per-step
    Stage 1 solve, no cross-process gather/slice needed at all, since every process already holds the
    correct precomputed rho* for its own rows.

    Does not track tribe/f_div_mu (no frozen mu_model here — nothing in this variant's loss needs one,
    unlike OffPolicyTribeTrainer's diagnostic-only use of it); can be added later if useful for comparison.
    """

    def __init__(self, model, args: OffPolicyConfig, train_dataset, processing_class, beta: float = 0.1, trust_region_eps: float = 0.05, use_unlikelihood: bool = False, divergence: str = "kl_new_old", negative_logp_floor: float = math.log(1e-8), use_chi_squared_beta0_closed_form: bool = False, **kwargs):
        rewards = torch.tensor(train_dataset["reward"], dtype=torch.float64)
        rho_star_all, lam = compute_rho_star_offpolicy(
            rewards,
            args.group_size,
            beta,
            trust_region_eps,
            divergence=divergence,
            use_chi_squared_beta0_closed_form=use_chi_squared_beta0_closed_form,
        )
        if rho_star_all is None:
            # Unlike OffPolicyTribeTrainer's per-batch solve, this is a ONE-TIME solve over the whole
            # dataset before any training starts — there's no coherent "skip and keep going" here (the
            # entire dataset either gets a global rho* or it doesn't), so fail loudly and immediately
            # instead of letting the None flow into add_column() below and crash with a confusing
            # AttributeError on .tolist(). See compute_rho_star's own warning (already emitted) for why
            # the solve failed.
            raise RuntimeError(
                "Stage 1's global cvxpy solve failed (see the warning above for the cvxpy status) — "
                "OffPolicyTribeGlobalTrainer has no per-batch fallback for this, unlike "
                "OffPolicyTribeTrainer's per-step skip."
            )
        train_dataset = train_dataset.add_column("rho_star", rho_star_all.tolist())

        super().__init__(model, args, train_dataset, processing_class, **kwargs)
        # Replaces OffPolicyTrainer.__init__'s own collator (which doesn't know about rho_star) with one
        # that passes it through — pure addition on top of it, not a modification of the shared collator.
        self.data_collator = _OffPolicyGlobalRhoCollator(
            processing_class, args.max_completion_length, args.max_prompt_length, args.mask_truncated_completions
        )
        self.global_lambda = lam.item()
        # Same trick, same rationale as OffPolicyTribeTrainer's own use_unlikelihood — see that class's
        # __init__/compute_loss for the full explanation (self-attenuating -log1p(-p) for weight<0 samples
        # instead of a fixed-strength penalty, no mu/ratio needed).
        self.use_unlikelihood = use_unlikelihood
        # See OffPolicyTribeTrainer's own negative_logp_floor for the full rationale.
        self.negative_logp_floor = negative_logp_floor

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Kept raw (unfloored) here — used as-is below for logp stats so they reflect the model's true
        # behavior; any flooring/clamping happens only inside the loss branches.
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        weight = inputs["rho_star"] - 1

        if self.use_unlikelihood:
            # Positive (weight>=0): plain NLL, unchanged. Negative (weight<0): unlikelihood (-log1p(-p)),
            # self-attenuating toward 0 as p->0 instead of a fixed-strength penalty. p must be explicitly
            # upcast to float32 before the clamp — selective_log_softmax computes log_softmax directly in
            # bf16 for a bf16 model (trl/trainer/utils.py), and bf16 can't distinguish 1.0 from 1.0-1e-6
            # (verified: both round to exactly 1.0), so clamping in bf16 would silently be a no-op.
            positive_mask = (weight >= 0).float().unsqueeze(-1)
            negative_mask = (weight < 0).float().unsqueeze(-1)
            positive_loss = -weight.clamp(min=0).unsqueeze(-1) * per_token_logps
            p = per_token_logps.float().exp().clamp(max=1 - 1e-6)
            unlikelihood = -torch.log1p(-p)
            negative_loss = weight.clamp(max=0).abs().unsqueeze(-1) * unlikelihood
            per_token_loss = (positive_loss * positive_mask + negative_loss * negative_mask) * completion_mask
        else:
            # Floor probability at self.negative_logp_floor, same rationale as OffPolicyTribeTrainer's
            # default path: stops compounding downward pressure on already-very-unlikely tokens without
            # needing the fp32 upcast the unlikelihood mode above requires.
            per_token_logps_floored = per_token_logps.clamp(min=self.negative_logp_floor)
            per_token_loss = -weight.unsqueeze(-1) * per_token_logps_floored * completion_mask

        loss = (per_token_loss.sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["tribe/global_lambda"].append(self.global_lambda)
        self._log_logp_stats(per_token_logps, completion_mask, prefix="tribe_global")
        self._log_reward_metric(inputs["reward"])

        return (loss, None) if return_outputs else loss


class OffPolicyRaftTrainer(OffPolicyTrainer):
    """
    Rejection-sampling SFT, offline: filters the dataset down to reward==1 examples ONCE at init (the
    incorrect ones never contribute to this loss, so there's no reason to pay for their forward pass at
    all), then plain NLL on what remains. No Stage 1, no mu/ratio, no frozen reference model needed — the
    simplest possible baseline in this suite, for isolating how much of TRIBE/TOPR's improvement actually
    comes from doing anything with the negative (incorrect) examples, versus just plain positive-only SFT.

    max_positive_per_question caps how many reward==1 completions survive per question (ReST-EM,
    https://huggingface.co/papers/2312.06585, caps this at 10 for MATH; STaR, Zelikman et al. 2022, uses
    the same design) — without it, easier questions (higher solve rate across the fixed per-question
    sample budget) contribute far more positive examples than harder ones, skewing training composition
    toward easy questions relative to the raw/test question distribution (measured directly on this
    dataset: solve rate 75.6% on MATH Level 1 vs 35.3% on Level 5).
    """

    def __init__(
        self, model, args: OffPolicyConfig, train_dataset, processing_class,
        max_positive_per_question: int | None = None, **kwargs
    ):
        if max_positive_per_question is not None:
            question_idx = [i // args.group_size for i in range(len(train_dataset))]
            train_dataset = train_dataset.add_column("_question_idx", question_idx)
        train_dataset = train_dataset.filter(lambda example: example["reward"] == 1.0)
        if max_positive_per_question is not None:
            kept_indices = []
            seen_per_question = defaultdict(int)
            for i, q in enumerate(train_dataset["_question_idx"]):
                if seen_per_question[q] < max_positive_per_question:
                    kept_indices.append(i)
                    seen_per_question[q] += 1
            train_dataset = train_dataset.select(kept_indices).remove_columns("_question_idx")
        super().__init__(model, args, train_dataset, processing_class, **kwargs)

    def _get_train_sampler(self, train_dataset=None) -> RandomSampler:
        # The reward==1.0 filter above breaks group_size-alignment (each group keeps a different number of
        # its own completions, so the surviving dataset generally isn't even a multiple of group_size
        # anymore) — but compute_loss below has no grouped math at all (plain per-example NLL, no
        # advantage/rho*/ratio computed against a group), so unlike every other trainer in this file, RAFT
        # doesn't need _GroupShuffledSampler's group-contiguity guarantee. A plain per-example shuffle is
        # both correct and sufficient (and necessary — see _GroupShuffledSampler's docstring for why the
        # dataset's own row order can't be trusted un-shuffled).
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        return RandomSampler(dataset, generator=generator)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        seq_logps = (per_token_logps * completion_mask).sum(-1)
        token_counts = completion_mask.sum(-1).clamp(min=1)
        loss = (-seq_logps / token_counts).mean()

        self._log_logp_stats(per_token_logps, completion_mask, prefix="raft")
        self._log_reward_metric(inputs["reward"])

        return (loss, None) if return_outputs else loss


class OffPolicyTISTrainer(OffPolicyTrainer):
    """
    Plain truncated importance sampling on top of TOPR's own reward convention: raw SIGNED reward (+1/-1,
    no baseline of any kind — same as OffPolicyNaiveReinforceTrainer, deliberately NOT RLOO's
    leave-one-out baseline, to keep the reward representation identical to TOPR's so the only thing that
    varies is how the ratio is used), times a sequence-level ratio clipped to [0, 1] (TOPR's own clip
    range, https://huggingface.co/papers/2503.14286) applied UNIFORMLY to every example, positive and
    negative alike — unlike TOPR, which only truncates the ratio for negative-reward examples and gives
    positive ones a fixed weight of 1 (no ratio dependence at all). This isolates exactly that one design
    choice: does TOPR's asymmetric treatment (uncapped SFT push on positives, truncated-IS pushback on
    negatives) actually matter, or does a fully symmetric truncated-IS correction do just as well?
    """

    def __init__(self, model, args: OffPolicyConfig, train_dataset, processing_class, negative_fraction: float = 1.0, ref_model_name_or_path: str | None = None, **kwargs):
        super().__init__(model, args, train_dataset, processing_class, **kwargs)
        # mu == the model's own initialization (the behavioral policy that generated this dataset), frozen
        # for the whole run — same frozen-copy pattern as OffPolicyGRPOTrainer/OffPolicyTOPRTrainer's own
        # ref_model/mu_model. ref_model_name_or_path overrides this when the model being trained differs
        # from the model that generated the dataset (see _prepare_frozen_ref_model's docstring).
        self.mu_model = self._prepare_frozen_ref_model(ref_model_name_or_path)
        # See OffPolicyTrainer._negative_sample_weight_global's docstring. "Negative" here means
        # signed_reward<0 (every incorrect completion).
        self.negative_fraction = negative_fraction

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        signed_reward = 2 * inputs["reward"] - 1

        with torch.no_grad():
            mu_per_token_logps = self._get_per_token_logps(self.mu_model, input_ids, attention_mask, logits_to_keep)
            seq_logps_theta = (per_token_logps.detach().float() * completion_mask).sum(-1)
            seq_logps_mu = (mu_per_token_logps.float() * completion_mask).sum(-1)
            # Sequence-level ratio (TOPR's own choice, not GRPO's token-level ratio), single-sided clip at
            # 1 — ratio is always >=0, so clamp(max=1.0) alone implements clip(ratio, 0, 1). Applied here
            # regardless of the reward's sign, unlike TOPR's positive branch (fixed weight=1, no clip).
            ratio = torch.exp(seq_logps_theta - seq_logps_mu).clamp(max=1.0)

        sample_weight_global = self._negative_sample_weight_global(
            self._gather_grouped(signed_reward) < 0, self.negative_fraction
        )
        sample_weight = self._local_slice(sample_weight_global, inputs["reward"].numel())

        per_token_loss = -(ratio * signed_reward).unsqueeze(-1) * per_token_logps * completion_mask
        per_seq_loss = per_token_loss.sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        loss = (per_seq_loss * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)

        self._log_logp_stats(per_token_logps, completion_mask, prefix="tis")
        self._log_reward_metric(inputs["reward"])

        return (loss, None) if return_outputs else loss
