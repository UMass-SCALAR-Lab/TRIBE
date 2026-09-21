# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""
Online DPO baseline on GSM8K, for comparison against TRIBE (scripts/train_gsm8k.py).

Identical dataset/prompt/reward setup (SYSTEM_PROMPT, extract_answer, correctness_reward,
make_conversation kept verbatim from scripts/train_gsm8k_grpo.py) passed as `reward_funcs` directly
(OnlineDPOTrainer accepts a callable reward, same as GRPO/RLOO — no reward model needed here). Note
OnlineDPOTrainer hardcodes exactly 2 completions per prompt and picks chosen/rejected via
`first_half_reward >= second_half_reward`, so ties (both completions equally right or equally wrong,
common with a binary correctness reward) always break toward completion #1 — that's this method's own
nature, not something this script works around. `ref_model` is left unset so OnlineDPOTrainer creates
the frozen reference copy itself (same role as GRPO's implicit ref model under beta > 0).

Usage: same shared beta/learning_rate/num_train_epochs/batch size as the other baselines apply; note
OnlineDPOConfig uses `max_new_tokens`/`max_length` rather than GRPO's `max_completion_length` (max_length
is the *total* prompt+completion cap and must exceed max_new_tokens), and has no `num_generations` field
at all — unlike RLOO's overridable default, OnlineDPOTrainer hardcodes exactly 2 completions per prompt
internally, so there's no flag to set here to match the other baselines' group size of 8.

python scripts/train_gsm8k_online_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir online-dpo-gsm8k-baseline \
    --per_device_train_batch_size 16 \
    --max_new_tokens 512 \
    --max_length 1024 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --bf16 True \
    --missing_eos_penalty 1.0
"""

import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer


SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem step by step, then provide the final "
    "numeric answer on the last line in the format: #### <number>"
)


def extract_answer(text: str) -> str | None:
    match = re.search(r"####\s*([\d,]+)", text)
    return match.group(1).replace(",", "") if match else None


def correctness_reward(completions, reference_answer, **kwargs):
    rewards = []
    for completion, ref in zip(completions, reference_answer, strict=False):
        predicted = extract_answer(completion if isinstance(completion, str) else completion[-1]["content"])
        rewards.append(1.0 if predicted is not None and predicted == ref else 0.0)
    return rewards


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
        script_args.dataset_name or "openai/gsm8k",
        script_args.dataset_config or "main",
        split=script_args.dataset_train_split,
    )

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "reference_answer": example["answer"].split("####")[-1].strip().replace(",", ""),
        }

    dataset = dataset.map(make_conversation, remove_columns=dataset.column_names)

    ################
    # Training
    ################
    # OnlineDPOTrainer's own default (ref_model=None) builds the reference via create_reference_model(),
    # a deepcopy-based approach that explicitly refuses to run under DeepSpeed ZeRO-3 (parameters are
    # sharded, not a real tensor to copy -- same class of issue documented in
    # scripts/offpolicy_trainer.py's _prepare_frozen_ref_model). Under ZeRO-3, load it fresh instead, same
    # weights as the policy's own starting point.
    ref_model = None
    if is_deepspeed_zero3_enabled():
        ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **training_args.model_init_kwargs)

    trainer = OnlineDPOTrainer(
        model=model_args.model_name_or_path,
        ref_model=ref_model,
        reward_funcs=correctness_reward,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
