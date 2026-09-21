# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""
TRIBE, off-policy / fully-offline on GSM8K, GLOBAL Stage 1 variant (see scripts/offpolicy_trainer.py's
OffPolicyTribeGlobalTrainer docstring): rho* depends only on reward in this regime (c=0 always — mu and
the KL anchor are the same frozen base model, see tribe/offpolicy_stage1.py), so it's solved ONCE over the
full dataset (one shared lambda across all groups) instead of being re-solved noisily from each small
mini-batch like scripts/train_gsm8k_offpolicy_tribe.py does. Stage 2 is then a plain weighted-SFT pass
using that fixed, precomputed rho* per example.

Usage:
python scripts/train_gsm8k_offpolicy_tribe_global.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/gsm8k \
    --output_dir offpolicy-tribe-global-gsm8k-baseline \
    --group_size 16 \
    --per_device_train_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 1e-7 \
    --trust_region_eps 0.05 \
    --gradient_checkpointing \
    --bf16 True
"""

from dataclasses import dataclass, field

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, TrlParser, get_peft_config

from offpolicy_split import split_off_policy_dataset
from offpolicy_trainer import OffPolicyConfig, OffPolicyTribeGlobalTrainer


@dataclass
class OffPolicyScriptArguments:
    dataset_path: str = field(
        metadata={"help": "Path to the pre-generated dataset (scripts/generate_offpolicy_gsm8k.py's --output_dir)."}
    )
    beta: float = field(default=0.1, metadata={"help": "No effect here (c is always 0) — kept for flag parity."})
    trust_region_eps: float = field(default=0.05, metadata={"help": "Stage 1's trust-region budget."})
    use_unlikelihood: bool = field(
        default=False,
        metadata={
            "help": "Use the unlikelihood trick (-log1p(-p), self-attenuating) for weight<0 samples instead "
            "of the plain floored NLL. See OffPolicyTribeGlobalTrainer's compute_loss."
        },
    )
    divergence: str = field(
        default="kl_new_old",
        metadata={
            "help": "Stage 1's trust-region divergence, one of 'kl_new_old' (closed form, fast), "
            "'kl_old_new', 'chi_squared' (both solved via cvxpy — fine here since Stage 1 runs once over "
            "the whole dataset before training, not per step). See tribe/stage1.py."
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
    trainer = OffPolicyTribeGlobalTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        beta=script_args.beta,
        trust_region_eps=script_args.trust_region_eps,
        use_unlikelihood=script_args.use_unlikelihood,
        divergence=script_args.divergence,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
