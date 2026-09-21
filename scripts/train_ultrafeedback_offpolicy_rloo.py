# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""
RLOO baseline on UltraFeedback, off-policy / fully-offline (TOPR's single-iteration regime,
https://huggingface.co/papers/2503.14286): trains for ONE epoch over UltraFeedback's own fixed 4-completions-
per-prompt data (scripts/convert_ultrafeedback_offpolicy.py) — same dataset scripts/train_ultrafeedback_
offpolicy_tribe.py trains on (all 4 completions per group, continuous reward), unlike RAFT which discards 3
of the 4.

Uses OffPolicyReinforceTrainer (scripts/offpolicy_trainer.py): RLOO's leave-one-out advantage baseline, no
importance-sampling correction for pi_theta/mu drift over the single epoch — this is TOPR's own "naive
REINFORCE" ablation baseline (paper Section 2.1), same as scripts/train_gsm8k_offpolicy_reinforce.py, just
pointed at UltraFeedback's data.

Subclasses plain transformers.Trainer (not GRPOTrainer/SFTTrainer) — see scripts/offpolicy_trainer.py's own
docstring for why. Model and tokenizer are loaded explicitly here (plain Trainer, unlike GRPOTrainer/
TribeTrainer, doesn't auto-load a model from a string name_or_path).

Usage:
python scripts/train_ultrafeedback_offpolicy_rloo.py \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/ultrafeedback-v2 \
    --output_dir offpolicy-rloo-ultrafeedback-baseline \
    --group_size 4 \
    --per_device_train_batch_size 16 \
    --max_completion_length 1024 \
    --learning_rate 1e-7 \
    --gradient_checkpointing \
    --bf16 True
"""

from dataclasses import dataclass, field

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, TrlParser, get_peft_config

from offpolicy_split import split_off_policy_dataset
from offpolicy_trainer import OffPolicyConfig, OffPolicyReinforceTrainer


@dataclass
class OffPolicyScriptArguments:
    dataset_path: str = field(
        metadata={"help": "Path to the pre-converted dataset (scripts/convert_ultrafeedback_offpolicy.py's --output_dir)."}
    )
    negative_fraction: float = field(
        default=1.0,
        metadata={
            "help": "Fraction of negative (advantage<0) examples to keep in the loss each step; all "
            "positive examples are always kept. Default 1.0 keeps all of them exactly once."
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
    trainer = OffPolicyReinforceTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        negative_fraction=script_args.negative_fraction,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
