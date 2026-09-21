# /// script
# dependencies = [
#     "trl",
#     "math-verify",
# ]
# ///

"""
RAFT (rejection-sampling SFT) baseline on MATH, for comparison against TRIBE (scripts/train_math.py).

Same RAFTTrainer/RAFTConfig wiring as scripts/train_gsm8k_raft.py — see that script's docstring and
scripts/raft_trainer.py's own docstring for why this reuses GRPOTrainer's generation/reward/multi-GPU
machinery instead of a hand-rolled training loop. See scripts/train_math.py for why the dataset/reward
differ from the GSM8K script (MATH answers are LaTeX expressions, checked via `trl.rewards.accuracy_reward`
instead of GSM8K's regex extraction — a completion is kept only when `accuracy_reward` returns exactly
`1.0`; both `0.0` (wrong) and `None` (gold solution unparseable) contribute nothing to the positive phase,
and only `0.0` counts as "incorrect" for the optional negative/unlikelihood phase).

Usage: same shared flags as train_gsm8k_raft.py apply, except max_completion_length=1024 (not GSM8K's
512 — see scripts/train_math.py's docstring for why).

python scripts/train_math_raft.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir raft-math-baseline \
    --num_generations 8 \
    --per_device_train_batch_size 16 \
    --max_completion_length 1024 \
    --learning_rate 5e-6 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --bf16 True
"""

import torch
from datasets import load_dataset
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.rewards import accuracy_reward

from raft_trainer import RAFTConfig, RAFTTrainer


SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem step by step, then give your final answer "
    "wrapped as \\boxed{answer}."
)


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
        "solution": example["solution"],
    }


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, RAFTConfig, ModelConfig))
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
    trainer = RAFTTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
