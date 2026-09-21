# /// script
# dependencies = [
#     "trl",
#     "cvxpy",
# ]
# ///

"""
TRIBE, off-policy / fully-offline, on PRM800K (openai/prm800k, see scripts/convert_prm800k_offpolicy.py),
PER-BATCH Stage 1 variant (scripts/offpolicy_trainer_ragged_perbatch.OffPolicyTribeRaggedTrainer) — Stage 1
re-solved fresh every step from that step's own gathered rewards, matching how the established GSM8K/MATH
sweeps actually trained (scripts/train_gsm8k_offpolicy_tribe.py), unlike
scripts/train_prm800k_offpolicy_tribe_ragged.py's global (solve-once) variant.

Dataset must come from scripts/convert_prm800k_offpolicy.py's --ragged output (needs a 'group_id' column) —
this script pads/caps it to fixed max_group_size blocks itself (see offpolicy_trainer_ragged_perbatch.py's
own docstring for why), so pass the RAW ragged output here, not a --group_size-downsampled one.

Usage:
python scripts/train_prm800k_offpolicy_tribe_perbatch_ragged.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/prm800k-ragged \
    --output_dir offpolicy-tribe-perbatch-prm800k-negfrac0.6 \
    --max_group_size 16 \
    --negative_fraction 0.6 \
    --per_device_train_batch_size 8 \
    --max_completion_length 1024 \
    --learning_rate 5e-7 \
    --trust_region_eps 2.0 \
    --divergence chi_squared \
    --gradient_checkpointing \
    --bf16 True
"""

from dataclasses import dataclass, field

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, TrlParser, get_peft_config

from offpolicy_trainer import OffPolicyConfig
from offpolicy_trainer_ragged_perbatch import OffPolicyTribeRaggedTrainer, pad_and_cap_ragged_dataset


@dataclass
class OffPolicyScriptArguments:
    dataset_path: str = field(
        metadata={"help": "Path to scripts/convert_prm800k_offpolicy.py's --ragged --output_dir (RAW, not "
        "--group_size-downsampled or --balance_negatives'd — this script does its own capping)."}
    )
    max_group_size: int = field(
        default=16,
        metadata={"help": "Fixed block size every question's group gets padded/capped to (see "
        "offpolicy_trainer_ragged_perbatch.pad_and_cap_ragged_dataset's docstring). Should match "
        "per_device_train_batch_size * (number of processes), matching OffPolicyTribeTrainer's own "
        "established 'one full group per step' convention."},
    )
    beta: float = field(default=0.1, metadata={"help": "No effect here (c is always 0) — kept for flag parity."})
    trust_region_eps: float = field(default=0.05, metadata={"help": "Stage 1's trust-region budget, applied fresh every step."})
    divergence: str = field(default="kl_new_old", metadata={"help": "Stage 1's trust-region divergence."})
    negative_fraction: float = field(
        default=1.0,
        metadata={"help": "Fraction of each step's negative (rho*<1) examples kept in the loss. See "
        "offpolicy_trainer.OffPolicyTrainer's own docstring. Mutually exclusive with --min_pos_neg_ratio."},
    )
    min_pos_neg_ratio: float | None = field(
        default=None,
        metadata={"help": "Mutually exclusive with --negative_fraction (leave that at 1.0 when using "
        "this). Directly enforces a MINIMUM realized positives-per-kept-negative ratio for the whole "
        "batch instead of an expected fraction — see OffPolicyTrainer._pos_neg_floor_sample_weight's own "
        "docstring. None (default) leaves --negative_fraction's own gating in effect."},
    )
    negative_logp_floor: float | None = field(
        default=None, metadata={"help": "None (default) uses OffPolicyTribeRaggedTrainer's own default floor."}
    )
    use_unlikelihood: bool = field(default=False)


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
    dataset = pad_and_cap_ragged_dataset(dataset, script_args.max_group_size, seed=training_args.seed)

    ################
    # Training
    ################
    trainer_kwargs = {}
    if script_args.negative_logp_floor is not None:
        trainer_kwargs["negative_logp_floor"] = script_args.negative_logp_floor

    trainer = OffPolicyTribeRaggedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        max_group_size=script_args.max_group_size,
        beta=script_args.beta,
        trust_region_eps=script_args.trust_region_eps,
        divergence=script_args.divergence,
        negative_fraction=script_args.negative_fraction,
        min_pos_neg_ratio=script_args.min_pos_neg_ratio,
        use_unlikelihood=script_args.use_unlikelihood,
        **trainer_kwargs,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
