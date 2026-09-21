# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""
TRIBE trial run on GSM8K.

Adapted from trl/examples/scripts/grpo_continuous_batching.py's GSM8K setup (SYSTEM_PROMPT,
extract_answer, correctness_reward, make_conversation kept as-is) — swaps GRPOTrainer/GRPOConfig for
TribeTrainer/TribeConfig and drops continuous batching (not relevant to validating TRIBE's own pieces).

Usage: shared hyperparams (beta, learning_rate, num_generations, num_train_epochs, batch size, max
completion length) are kept identical across every baseline script for a fair comparison — beta=0.1 in
particular so no baseline is either over- or under-anchored to its reference model relative to the
others (see scripts/train_gsm8k_grpo.py's docstring for the one deliberate exception, DAPO). beta and
trust_region_eps happen to already match TribeConfig's own defaults; stage2_loss_type is explicitly set
to "grpo" (not the tex-literal "sum" default) since "sum"'s full-sequence-sum aggregation was found to
have a length-dependent gradient scale ~250-300x larger than GRPO's own per-token-averaged loss — see
TRIBE-plan.tex's "Open question: bounding Stage 2 in practice" note. scale_rewards="none" is TribeConfig's
own default (overriding GRPOConfig's "group") — this is the one axis deliberately left NOT matching the
GRPO baseline, since "group" divides each group by its own std, a per-group rescaling that Stage 1's
single shared lambda (solved jointly across every group in the batch) can't cleanly absorb the way it can
a single global rescale, unlike beta which is forced uniform across every baseline. See
tribe/tribe_config.py's scale_rewards field for the full reasoning.

python scripts/train_gsm8k.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir tribe-gsm8k-baseline \
    --num_generations 8 \
    --per_device_train_batch_size 16 \
    --max_completion_length 512 \
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

import re
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from tribe.tribe_config import TribeConfig
from tribe.tribe_trainer import TribeTrainer


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
    parser = TrlParser((GSM8KScriptArguments, TribeConfig, ModelConfig))
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
    trainer = TribeTrainer(
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
