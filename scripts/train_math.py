# /// script
# dependencies = [
#     "trl",
#     "math-verify",
# ]
# ///

"""
TRIBE trial run on MATH (Hendrycks et al.), for comparison against the GSM8K runs (scripts/train_gsm8k.py).

Same TribeTrainer/TribeConfig wiring as train_gsm8k.py; only the dataset and reward differ. MATH answers
are LaTeX expressions (not bare integers), so instead of GSM8K's regex-based extract_answer/
correctness_reward this uses `trl.rewards.accuracy_reward`, TRL's own math-verify-backed answer-
equivalence checker (LaTeX-aware: extracts \\boxed{...}, compares fractions/decimals/expressions for
mathematical equivalence rather than string equality). Requires `pip install math-verify`.

Usage: shared hyperparams are kept identical to scripts/train_gsm8k.py's for a fair GSM8K-vs-MATH
comparison — see that script's docstring for why beta=0.1, stage2_loss_type=grpo, and scale_rewards=none
(TribeConfig's own default) specifically. One deliberate exception: max_completion_length is 1024 here,
not GSM8K's 512 — MATH's harder, competition-style problems need noticeably more reasoning tokens before
reaching \\boxed{...}, and 512 was truncating completions before the boxed answer appeared (visible as a
high completions/clipped_ratio and frequent answer_parsed="[unparseable]" in the logged completions table).

python scripts/train_math.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir tribe-math-baseline \
    --num_generations 8 \
    --per_device_train_batch_size 16 \
    --max_completion_length 1024 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --trust_region_eps 0.05 \
    --stage2_loss_type grpo \
    --scale_rewards none \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --bf16 True \
    --log_completions
"""

import torch
from datasets import load_dataset
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.rewards import accuracy_reward

from tribe.tribe_config import TribeConfig
from tribe.tribe_trainer import TribeTrainer


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
    parser = TrlParser((ScriptArguments, TribeConfig, ModelConfig))
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
    trainer = TribeTrainer(
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
