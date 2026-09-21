# /// script
# dependencies = [
#     "trl",
#     "math-verify",
# ]
# ///

"""
Plain GRPO baseline on MATH, for comparison against TRIBE (scripts/train_math.py).

Same GRPOTrainer/GRPOConfig wiring as scripts/train_gsm8k_grpo.py; see scripts/train_math.py for why the
dataset/reward differ from the GSM8K scripts (MATH answers are LaTeX expressions, checked via
`trl.rewards.accuracy_reward` instead of GSM8K's regex extraction). Left on GRPOConfig's own default
scale_rewards="group" (loss_type="dapo" is also GRPOConfig's own default — see
scripts/train_gsm8k_grpo.py's docstring for the DAPO flag combination on this same script); TribeConfig
deliberately defaults to "none" instead — the one axis intentionally not matched, see
scripts/train_gsm8k_grpo.py's docstring for why. beta=0.1 overrides GRPOConfig's own default of 0.0,
matching every other baseline's KL anchor strength.

Usage: shared hyperparams are kept identical to scripts/train_math.py's for a fair comparison, including
max_completion_length=1024 (not GSM8K's 512 — see that script's docstring for why).

python scripts/train_math_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir grpo-math-baseline \
    --num_generations 8 \
    --per_device_train_batch_size 16 \
    --max_completion_length 1024 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --bf16 True \
    --log_completions

DAPO baseline: same script, same flag combination as scripts/train_gsm8k_grpo.py's DAPO block plus
--soft_overlong_cache_length below — see that script's docstring for what maps directly (Clip-Higher,
Token-level Loss), what implements the paper's actual Overlong Reward Shaping mechanism
(--soft_overlong_cache_length, use INSTEAD of --mask_truncated_completions, not alongside it), what's
missing entirely (Dynamic Sampling), and why beta is kept at 0.1 here rather than the paper's own beta=0:

python scripts/train_math_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir dapo-math-baseline \
    --num_generations 8 \
    --per_device_train_batch_size 16 \
    --max_completion_length 1024 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --loss_type dapo \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --soft_overlong_cache_length 128 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --bf16 True \
    --log_completions
"""

from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.rewards import accuracy_reward


SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem step by step, then give your final answer "
    "wrapped as \\boxed{answer}."
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
            "--mask_truncated_completions). Same mechanism as scripts/train_gsm8k_grpo.py's flag of the "
            "same name."
        },
    )


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
        "solution": example["solution"],
    }


def make_accuracy_reward(max_completion_length, soft_overlong_cache_length):
    def reward_func(completions, completion_ids, solution, **kwargs):
        rewards = accuracy_reward(completions, solution, **kwargs)
        if soft_overlong_cache_length is not None:
            soft_max_length = max_completion_length - soft_overlong_cache_length
            for i, ids in enumerate(completion_ids):
                if rewards[i] is not None and len(ids) > soft_max_length:
                    rewards[i] += (soft_max_length - len(ids)) / soft_overlong_cache_length
        return rewards

    return reward_func


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

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
        script_args.dataset_name or "DigitalLearningGmbH/MATH-lighteval",
        script_args.dataset_config or "default",
        split=script_args.dataset_train_split,
    )
    dataset = dataset.map(make_conversation, remove_columns=dataset.column_names)

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=make_accuracy_reward(
            training_args.max_completion_length, script_args.soft_overlong_cache_length
        ),
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
