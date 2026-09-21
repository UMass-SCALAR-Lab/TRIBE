# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""
RAFT (rejection-sampling SFT) baseline on GSM8K, for comparison against TRIBE (scripts/train_gsm8k.py).

Same dataset/prompt/reward setup as scripts/train_gsm8k_grpo.py (SYSTEM_PROMPT, extract_answer,
correctness_reward, make_conversation kept verbatim), swapping in RAFTTrainer/RAFTConfig
(scripts/raft_trainer.py) — a GRPOTrainer subclass that keeps GRPO's own generation/reward/multi-GPU
machinery entirely as-is and only overrides the loss: masked NLL on completions the verifier scores
correct (reward == 1.0), incorrect ones simply contributing zero (dropped), matching plain
rejection-sampling SFT — no advantage/baseline, no KL term, no clipping. See scripts/raft_trainer.py's
own docstring for why this reuses GRPOTrainer rather than a hand-rolled training loop (an earlier version
of this script did that, and hit real correctness/OOM problems this design avoids).

Optional mode (--penalize_incorrect_weight, default 0.0 = off): also runs an unlikelihood-training phase
on that step's incorrect completions (push DOWN their likelihood, Welleck et al., bounded/saturating
unlike naive negative-NLL gradient ascent) — see scripts/raft_trainer.py's RAFTConfig docstring.

Usage: same shared flags as train_gsm8k_grpo.py apply (this is now structurally the same kind of script —
same num_generations/per_device_train_batch_size/num_train_epochs conventions, same multi-GPU launch via
accelerate launch --num_processes=N).

python scripts/train_gsm8k_raft.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir raft-gsm8k-baseline \
    --num_generations 8 \
    --per_device_train_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --bf16 True
"""

import re
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from raft_trainer import RAFTConfig, RAFTTrainer


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
    parser = TrlParser((GSM8KScriptArguments, RAFTConfig, ModelConfig))
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
    trainer = RAFTTrainer(
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
