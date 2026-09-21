# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""
Plain GRPO baseline on GSM8K, for comparison against TRIBE (scripts/train_gsm8k.py).

Identical to scripts/train_gsm8k.py's dataset/prompt/reward setup (SYSTEM_PROMPT, extract_answer,
correctness_reward, make_conversation kept verbatim), swapping TribeTrainer/TribeConfig for stock
GRPOTrainer/GRPOConfig, so the comparison isolates the algorithm rather than incidental setup
differences. Left on GRPOConfig's own default scale_rewards="group" — TribeConfig deliberately defaults
to "none" instead (see scripts/train_gsm8k.py's docstring for why: Stage 1 solves one shared lambda
jointly across every group, so a per-group rescaling doesn't get absorbed the way a global one would).
This is the one axis intentionally NOT matched between the two scripts. beta is set explicitly to 0.1
(GRPOConfig's own default is 0.0) so every RL baseline in this suite shares the same KL anchor strength —
see the DAPO block below for the one deliberate exception on that axis.

Usage: shared hyperparams (beta, learning_rate, num_generations, num_train_epochs, batch size, max
completion length) are kept identical across every baseline script for a fair comparison; omit
TRIBE-only flags (trust_region_eps, divergence, backfill_zero_variance_prompts, max_backfill_attempts)
since GRPOConfig doesn't have them.

python scripts/train_gsm8k_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir grpo-gsm8k-baseline \
    --num_generations 8 \
    --per_device_train_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --bf16 True \
    --log_completions

DAPO baseline: same script, no code changes, just a different flag combination on GRPOConfig plus the
`--soft_overlong_cache_length` script arg below. Of the DAPO paper's four components, two map directly
onto existing GRPOConfig flags — Clip-Higher (asymmetric clip bounds, `epsilon`/`epsilon_high`) and the
Token-level Policy Gradient Loss (`loss_type="dapo"`, global token-count normalization instead of
per-sequence averaging) — and `loss_type="dapo"` is already GRPOConfig's own default, so every run of
this script so far has been using it unless overridden. Overlong Reward Shaping is implemented here as
`--soft_overlong_cache_length`: past `max_completion_length - soft_overlong_cache_length` tokens, the
correctness reward is linearly reduced by up to 1.0 as the completion approaches `max_completion_length`,
reaching exactly -1.0 at the cap (paper's L_cache buffer). Use this INSTEAD of
`--mask_truncated_completions`, not alongside it: masking zeroes out the completion's loss contribution
entirely regardless of reward, so it would silence the very gradient the soft penalty is meant to steer
with — this was in fact the actual mechanism behind the DAPO collapse observed in the on-policy suite
(mask_truncated_completions + a lucky-then-unlucky non-terminating rollout gave the model zero signal to
correct course; the soft penalty gives it a real gradient to shorten instead). DAPO's fourth component,
Dynamic Sampling (oversample, filter out zero-reward-variance prompts, resample until the batch is full
of informative prompts), has no equivalent in stock GRPOTrainer — the same gap TRIBE fills, opt-in, via
`backfill_zero_variance_prompts`. So this run is Clip-Higher + Token-level Loss + soft overlong shaping,
not the full DAPO recipe. `beta` is deliberately kept at 0.1 here too (matching every other baseline),
even though the DAPO paper's own recipe drops the KL term entirely (`beta=0`) — this isolates Clip-Higher
+ the token-level loss + overlong shaping as DAPO's delta from plain GRPO under a shared KL budget, rather
than reproducing the paper's exact recipe (which also removes the anchor to the reference model):

python scripts/train_gsm8k_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir dapo-gsm8k-baseline \
    --num_generations 8 \
    --per_device_train_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --loss_type dapo \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --soft_overlong_cache_length 64 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --bf16 True \
    --log_completions
"""

import re
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


SYSTEM_PROMPT_HASH = (
    "You are a helpful math tutor. Solve the problem step by step, then provide the final "
    "numeric answer on the last line in the format: #### <number>"
)
SYSTEM_PROMPT_BOXED = (
    "You are a helpful math tutor. Solve the problem step by step, then put your final answer in "
    "\\boxed{}."
)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    soft_overlong_cache_length: int | None = field(
        default=None,
        metadata={
            "help": "DAPO's Overlong Reward Shaping buffer (L_cache in the paper). None (default) disables "
            "it. When set, completions longer than max_completion_length - soft_overlong_cache_length have "
            "their correctness reward linearly reduced, reaching -1.0 at max_completion_length, instead of "
            "the reward being left untouched (or the completion silently masked out of the loss via "
            "--mask_truncated_completions)."
        },
    )
    answer_format: str = field(
        default="hash",
        metadata={
            "help": "'hash' (default): prompt for '#### <number>' (this suite's original GSM8K format). "
            "'boxed': prompt for '\\boxed{}' instead, matching MATH's format and this model's own tendency "
            "to answer GSM8K in \\boxed{} regardless of what's asked — avoids the mixed-format mislabeling "
            "'#### '-only grading produces when the model doesn't follow the hash instruction."
        },
    )


def extract_answer_hash(text: str) -> str | None:
    match = re.search(r"####\s*([\d,]+)", text)
    return match.group(1).replace(",", "") if match else None


def extract_answer_boxed(text: str) -> str | None:
    match = re.search(r"\\boxed\{([-\d,]+)\}", text)
    return match.group(1).replace(",", "") if match else None


def make_correctness_reward(max_completion_length, soft_overlong_cache_length, extract_answer):
    def correctness_reward(completions, completion_ids, reference_answer, **kwargs):
        rewards = []
        for completion, ids, ref in zip(completions, completion_ids, reference_answer, strict=False):
            predicted = extract_answer(completion if isinstance(completion, str) else completion[-1]["content"])
            reward = 1.0 if predicted is not None and predicted == ref else 0.0
            if soft_overlong_cache_length is not None:
                soft_max_length = max_completion_length - soft_overlong_cache_length
                if len(ids) > soft_max_length:
                    reward += (soft_max_length - len(ids)) / soft_overlong_cache_length
            rewards.append(reward)
        return rewards

    return correctness_reward


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    system_prompt = SYSTEM_PROMPT_BOXED if script_args.answer_format == "boxed" else SYSTEM_PROMPT_HASH
    extract_answer = extract_answer_boxed if script_args.answer_format == "boxed" else extract_answer_hash

    ################
    # Model
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )

    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name or "openai/gsm8k",
        script_args.dataset_config or "main",
        split=script_args.dataset_train_split,
    )

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["question"]},
            ],
            "reference_answer": example["answer"].split("####")[-1].strip().replace(",", ""),
        }

    dataset = dataset.map(make_conversation, remove_columns=dataset.column_names)

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=make_correctness_reward(
            training_args.max_completion_length, script_args.soft_overlong_cache_length, extract_answer
        ),
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
