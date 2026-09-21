# /// script
# dependencies = [
#     "trl",
#     "math-verify",
# ]
# ///

"""
Online DPO baseline on MATH, for comparison against TRIBE (scripts/train_math.py).

Same OnlineDPOTrainer/OnlineDPOConfig wiring as scripts/train_gsm8k_online_dpo.py; see
scripts/train_math.py for why the dataset/reward differ from the GSM8K scripts (MATH answers are LaTeX
expressions, checked via `trl.rewards.accuracy_reward` instead of GSM8K's regex extraction). As in the
GSM8K script: exactly 2 completions per prompt, chosen/rejected picked via `first_half >= second_half`
(ties break to completion #1) — that's Online DPO's own nature, not something this script works around.
`ref_model` is left unset so OnlineDPOTrainer creates the frozen reference copy itself.

Usage: OnlineDPOConfig uses `max_new_tokens`/`max_length` rather than GRPO's `max_completion_length`
(max_length is the *total* prompt+completion cap and must exceed max_new_tokens); no `num_generations`
field exists here at all (see scripts/train_gsm8k_online_dpo.py's docstring for why). max_new_tokens is
1024 here, not GSM8K's 512 (see scripts/train_math.py's docstring for why), with max_length raised to
2048 to keep the same 1:1 prompt:completion headroom as the GSM8K script's 512/1024.

python scripts/train_math_online_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir online-dpo-math-baseline \
    --per_device_train_batch_size 16 \
    --max_new_tokens 1024 \
    --max_length 2048 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --bf16 True \
    --missing_eos_penalty 1.0
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer
from trl.rewards import accuracy_reward


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
    parser = TrlParser((ScriptArguments, OnlineDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        # OnlineDPOTrainer defaults device_map to "auto" (model-parallel sharding within one process)
        # whenever it's absent from model_init_kwargs — incompatible with accelerate launch
        # --num_processes>1 (DDP expects each process to own one whole model copy on one GPU). Setting
        # it explicitly to None here stops that default from kicking in.
        device_map=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, revision=model_args.model_revision, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    trainer = OnlineDPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
