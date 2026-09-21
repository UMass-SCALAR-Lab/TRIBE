# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""
TRIBE, off-policy / fully-offline, on PRM800K, FLAT (whole-BATCH-as-one-group) Stage 1 variant
(scripts/offpolicy_trainer_flat.OffPolicyTribeFlatTrainer) — see that module's own docstring for why this
exists: PRM800K's per-prompt groups turned out to be mostly homogeneous (all-correct or all-incorrect),
leaving TRIBE's real, group-relative Stage 1 nothing to react to. This variant is mathematically equivalent
to maximizing raw reward instead of group-relative advantage — a degenerate-dataset WORKAROUND, not TRIBE's
actual method (use scripts/train_prm800k_offpolicy_tribe_ragged.py or
scripts/train_prm800k_offpolicy_tribe_perbatch_ragged.py when real within-group diversity exists).

Per-batch, matching OffPolicyTribeTrainer's own established convention exactly (Stage 1 re-solved fresh
every step from that step's own gathered rewards) — NOT a one-time whole-dataset solve. Only
negative_fraction=1.0 is supported (see trainer docstring for why).

Dataset must have a flat (prompt, completion, reward) schema, no group_id needed at all — the FLAT (no
--ragged, no --group_size, no --balance_negatives) output of scripts/convert_prm800k_offpolicy.py.

Usage:
python scripts/train_prm800k_offpolicy_tribe_flat.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/prm800k \
    --output_dir offpolicy-tribe-flat-prm800k-eps0.5 \
    --trust_region_eps 0.5 \
    --divergence chi_squared \
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

from offpolicy_trainer import OffPolicyConfig
from offpolicy_trainer_flat import OffPolicyTribeFlatTrainer


@dataclass
class OffPolicyScriptArguments:
    dataset_path: str = field(
        metadata={"help": "Path to scripts/convert_prm800k_offpolicy.py's FLAT output (no --ragged/--group_size)."}
    )
    beta: float = field(default=0.1, metadata={"help": "No effect here (c is always 0) — kept for flag parity."})
    trust_region_eps: float = field(
        default=0.5,
        metadata={"help": "Stage 1's trust-region budget, re-solved fresh every step over that step's own "
        "gathered batch (matching OffPolicyTribeTrainer's own established convention) — the 'group' is the "
        "whole batch, not a per-question sub-group."},
    )
    divergence: str = field(
        default="chi_squared",
        metadata={"help": "Stage 1's trust-region divergence, matching this project's established "
        "off-policy TRIBE convention. cvxpy-based (chi_squared included) is fine here since a batch is "
        "only per_device_train_batch_size * num_processes samples, not the whole dataset."},
    )
    negative_fraction: float = field(
        default=1.0,
        metadata={"help": "Fraction of each step's negative (rho*<1) examples kept in the loss — see "
        "offpolicy_trainer_flat.OffPolicyTribeFlatTrainer's own docstring for the batch-as-one-group "
        "ranked-selection this uses. Mutually exclusive with --min_pos_neg_ratio."},
    )
    min_pos_neg_ratio: float | None = field(
        default=None,
        metadata={"help": "Mutually exclusive with --negative_fraction (leave that at 1.0 when using "
        "this). Directly enforces a MINIMUM realized positives-per-kept-negative ratio for the whole "
        "batch instead of an expected fraction — see OffPolicyTrainer._pos_neg_floor_sample_weight's own "
        "docstring. None (default) leaves --negative_fraction's own gating in effect."},
    )
    negative_logp_floor: float | None = field(
        default=None, metadata={"help": "None (default) uses OffPolicyTribeFlatTrainer's own default floor."}
    )
    use_unlikelihood: bool = field(default=False)
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

    ################
    # Training
    ################
    trainer_kwargs = {}
    if script_args.negative_logp_floor is not None:
        trainer_kwargs["negative_logp_floor"] = script_args.negative_logp_floor

    trainer = OffPolicyTribeFlatTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        beta=script_args.beta,
        trust_region_eps=script_args.trust_region_eps,
        divergence=script_args.divergence,
        negative_fraction=script_args.negative_fraction,
        min_pos_neg_ratio=script_args.min_pos_neg_ratio,
        use_unlikelihood=script_args.use_unlikelihood,
        ref_model_name_or_path=script_args.ref_model_name_or_path,
        **trainer_kwargs,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
