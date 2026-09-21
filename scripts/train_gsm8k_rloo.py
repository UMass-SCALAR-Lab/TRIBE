# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""
Plain RLOO baseline on GSM8K, for comparison against TRIBE (scripts/train_gsm8k.py).

Identical dataset/prompt/reward setup (SYSTEM_PROMPT, extract_answer, correctness_reward,
make_conversation kept verbatim from scripts/train_gsm8k_grpo.py), swapping in RLOOTrainer/RLOOConfig
so the comparison isolates the algorithm rather than incidental setup differences. RLOO drops the
learned/estimated value baseline GRPO's advantage still shares the shape of (it also group-mean-centers
rewards), using a leave-one-out baseline instead — see RLOOConfig's own defaults for the rest (PPO-style
clipped surrogate loss). `--num_generations 8` below overrides RLOOConfig's own default of `2` — kept
identical to every other baseline's group size for a fair comparison, since RLOO's own smaller default
would otherwise give it a noisier leave-one-out baseline estimate purely from having fewer samples per
group, unrelated to the algorithm itself.

Usage: shared hyperparams (beta, learning_rate, num_generations, num_train_epochs, batch size, max
completion length) are kept identical across every baseline script for a fair comparison.

python scripts/train_gsm8k_rloo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir rloo-gsm8k-baseline \
    --num_generations 8 \
    --per_device_train_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --bf16 True \
    --log_completions
"""

import re
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from trl import ModelConfig, RLOOConfig, RLOOTrainer, ScriptArguments, TrlParser, get_peft_config


SYSTEM_PROMPT_HASH = (
    "You are a helpful math tutor. Solve the problem step by step, then provide the final "
    "numeric answer on the last line in the format: #### <number>"
)
SYSTEM_PROMPT_BOXED = (
    "You are a helpful math tutor. Solve the problem step by step, then put your final answer in "
    "\\boxed{}."
)


@dataclass
class GSM8KScriptArguments(ScriptArguments):
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


def make_correctness_reward(extract_answer):
    def correctness_reward(completions, reference_answer, **kwargs):
        rewards = []
        for completion, ref in zip(completions, reference_answer, strict=False):
            predicted = extract_answer(completion if isinstance(completion, str) else completion[-1]["content"])
            rewards.append(1.0 if predicted is not None and predicted == ref else 0.0)
        return rewards

    return correctness_reward


if __name__ == "__main__":
    parser = TrlParser((GSM8KScriptArguments, RLOOConfig, ModelConfig))
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
    trainer = RLOOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=make_correctness_reward(extract_answer),
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
