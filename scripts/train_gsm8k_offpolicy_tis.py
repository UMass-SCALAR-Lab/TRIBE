# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""
Truncated importance sampling (TIS) baseline on GSM8K, off-policy / fully-offline (TOPR's single-iteration
regime, https://huggingface.co/papers/2503.14286): trains for ONE epoch over a FIXED dataset pre-generated
by scripts/generate_offpolicy_gsm8k.py, same as every other script in this suite.

Uses OffPolicyTISTrainer (scripts/offpolicy_trainer.py): REINFORCE+RLOO's leave-one-out advantage
(scripts/train_gsm8k_offpolicy_reinforce.py), plus a sequence-level ratio clipped to [0, 1] applied
UNIFORMLY to both positive and negative examples — unlike TOPR, which gives positive examples a fixed
weight of 1 (no ratio at all) and only truncates the ratio for negative examples. Isolates whether TOPR's
asymmetric treatment is actually necessary, or whether a fully symmetric truncated-IS correction on top of
plain REINFORCE+RLOO does just as well.

Usage:
python scripts/train_gsm8k_offpolicy_tis.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/gsm8k \
    --output_dir offpolicy-tis-gsm8k-baseline \
    --group_size 16 \
    --per_device_train_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 1e-6 \
    --gradient_checkpointing \
    --bf16 True
"""

from dataclasses import dataclass, field

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, TrlParser, get_peft_config

from offpolicy_split import split_off_policy_dataset
from offpolicy_trainer import OffPolicyConfig, OffPolicyTISTrainer


@dataclass
class OffPolicyScriptArguments:
    dataset_path: str = field(
        metadata={"help": "Path to the pre-generated dataset (scripts/generate_offpolicy_gsm8k.py's --output_dir)."}
    )
    negative_fraction: float = field(
        default=1.0,
        metadata={
            "help": "Fraction of negative (signed_reward<0) examples to keep in the loss each step; all "
            "positive examples are always kept. Values above 1.0 oversample instead of subsampling: e.g. "
            "1.2 gives every negative weight at least 1 plus a random fifth of each group's negatives "
            "weight 2, for an expected per-group average weight of 1.2. Default 1.0 keeps all of them "
            "exactly once (prior behavior)."
        },
    )
    ref_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Model that actually GENERATED the offline dataset, if different from "
            "--model_name_or_path (the model being trained). Only needed for a weak-learner setup (e.g. "
            "training a smaller model on a larger model's generations) -- default None assumes the model "
            "being trained is also the one that generated the data (the ordinary same-model case)."
        },
    )


if __name__ == "__main__":
    parser = TrlParser((OffPolicyScriptArguments, OffPolicyConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    peft_config = get_peft_config(model_args)
    if peft_config is not None:
        from peft import get_peft_model

        model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_from_disk(script_args.dataset_path)
    dataset = split_off_policy_dataset(dataset, training_args.group_size, training_args.val_fraction)

    ################
    # Training
    ################
    trainer = OffPolicyTISTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        negative_fraction=script_args.negative_fraction,
        ref_model_name_or_path=script_args.ref_model_name_or_path,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
