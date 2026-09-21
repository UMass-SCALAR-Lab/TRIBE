# /// script
# dependencies = [
#     "trl",
#     "cvxpy",
# ]
# ///

"""
TRIBE, off-policy / fully-offline, on PRM800K (openai/prm800k, see scripts/convert_prm800k_offpolicy.py),
GLOBAL Stage 1 variant with RAGGED (variable-size) groups: PRM800K has a genuinely variable number of GPT-4
rollouts per problem (1 to 470, median 5), unlike every other dataset in this project (a fixed
`--num_generations`), so this uses scripts/offpolicy_trainer_ragged.OffPolicyTribeGlobalRaggedTrainer
(tribe/stage1_ragged.py's variable-group-size Stage 1 solve) instead of
scripts/train_gsm8k_offpolicy_tribe_global.py's uniform-group-size one — see that module's own docstring
for why this needed a new solver rather than downsampling every problem to a common group size.

Dataset must come from scripts/convert_prm800k_offpolicy.py's --ragged output (needs a 'group_id' column,
contiguous per-group blocks, no downsampling — every rollout for every problem is used, no positive or
negative discarded).

No train/val split here (scripts/offpolicy_split.py also assumes uniform group_size) — this is a quick
comparison run, evaluated against scripts/eval_prm800k.py's held-out PRM800K test set, not a val-based
hyperparameter sweep.

Usage:
python scripts/train_prm800k_offpolicy_tribe_ragged.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/prm800k-ragged \
    --output_dir offpolicy-tribe-prm800k-negfrac0.6 \
    --negative_fraction 0.6 \
    --per_device_train_batch_size 16 \
    --max_completion_length 1024 \
    --learning_rate 1e-7 \
    --trust_region_eps 0.5 \
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
from offpolicy_trainer_ragged import OffPolicyTribeGlobalRaggedTrainer


@dataclass
class OffPolicyScriptArguments:
    dataset_path: str = field(
        metadata={"help": "Path to scripts/convert_prm800k_offpolicy.py's --ragged --output_dir."}
    )
    beta: float = field(default=0.1, metadata={"help": "No effect here (c is always 0) — kept for flag parity."})
    trust_region_eps: float = field(default=0.5, metadata={"help": "Stage 1's trust-region budget."})
    divergence: str = field(
        default="chi_squared",
        metadata={
            "help": "Stage 1's trust-region divergence. 'chi_squared' matches this project's established "
            "best off-policy TRIBE config (see results_llama_final.tex); 'kl_new_old' uses the fast ragged "
            "closed form, any other goes through the ragged cvxpy solve (fine here — Stage 1 runs once over "
            "the whole dataset before training, not per step)."
        },
    )
    negative_fraction: float = field(
        default=1.0,
        metadata={
            "help": "Fraction of each group's negative-weight (rho*<1) examples kept in the loss. 1.0 "
            "(default) keeps every negative. See offpolicy_trainer.OffPolicyTrainer's own docstring. "
            "Mutually exclusive with --min_pos_neg_ratio."
        },
    )
    min_pos_neg_ratio: float | None = field(
        default=None,
        metadata={
            "help": "Mutually exclusive with --negative_fraction (leave that at 1.0 when using this). "
            "Directly enforces a MINIMUM realized positives-per-kept-negative ratio for the whole dataset "
            "(this trainer's Stage 1 is a one-time solve, so this is baked in once too) instead of an "
            "expected fraction — see OffPolicyTrainer._pos_neg_floor_sample_weight's own docstring. None "
            "(default) leaves --negative_fraction's own gating in effect."
        },
    )
    negative_logp_floor: float | None = field(
        default=None,
        metadata={"help": "None (default) uses OffPolicyTribeGlobalRaggedTrainer's own default floor."},
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

    trainer = OffPolicyTribeGlobalRaggedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        beta=script_args.beta,
        trust_region_eps=script_args.trust_region_eps,
        divergence=script_args.divergence,
        negative_fraction=script_args.negative_fraction,
        min_pos_neg_ratio=script_args.min_pos_neg_ratio,
        **trainer_kwargs,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
