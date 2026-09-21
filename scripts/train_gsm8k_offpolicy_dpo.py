# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""
Offline DPO baseline on GSM8K, off-policy / fully-offline (TOPR's single-iteration regime,
https://huggingface.co/papers/2503.14286): trains on a FIXED preference dataset built once by
scripts/build_offpolicy_dpo_pairs.py from the same underlying (prompt, completion, reward) data every
other off-policy baseline in this suite trains on — no online generation, no regeneration mid-epoch.

Unlike scripts/train_gsm8k_offpolicy_*.py's custom OffPolicyTrainer subclasses, this uses trl's own
DPOTrainer/DPOConfig directly (not scripts/offpolicy_trainer.py) — DPO already only trains on a fixed
paired dataset with no generation step of its own, so there's no online-generation machinery to route
around here (the reason the other off-policy scripts subclass plain Trainer instead of GRPOTrainer/
SFTTrainer, see scripts/offpolicy_trainer.py's docstring, doesn't apply to DPO).

Usage:
python scripts/build_offpolicy_dpo_pairs.py \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/gsm8k \
    --output_dir /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/gsm8k-dpo-pairs \
    --group_size 16

python scripts/train_gsm8k_offpolicy_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/gsm8k-dpo-pairs \
    --output_dir offpolicy-dpo-gsm8k-baseline \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-7 \
    --beta 0.1 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --bf16 True
"""

from dataclasses import dataclass, field

import torch
from datasets import load_from_disk
from trl import DPOConfig, DPOTrainer, ModelConfig, TrlParser, get_peft_config
from trl.trainer.utils import create_model_from_path


@dataclass
class OffPolicyScriptArguments:
    dataset_path: str = field(
        metadata={"help": "Path to the paired dataset built by scripts/build_offpolicy_dpo_pairs.py."}
    )
    ref_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Model that actually GENERATED the offline dataset, if different from "
            "--model_name_or_path (the model being trained). Only needed for a weak-learner setup (e.g. "
            "training a smaller model on a larger model's generations) -- default None lets DPOTrainer "
            "build its own reference from --model_name_or_path, correct only when the model being trained "
            "is also the one that generated the data (the ordinary same-model case)."
        },
    )


if __name__ == "__main__":
    parser = TrlParser((OffPolicyScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    training_args.model_init_kwargs = model_init_kwargs

    # DPOTrainer's own default (ref_model=None) builds the reference by reloading whatever model
    # --model_name_or_path points at — correct only when that's also the model that generated the offline
    # dataset. For a weak-learner setup those differ, so the actual generator must be loaded explicitly.
    ref_model = None
    if script_args.ref_model_name_or_path is not None:
        ref_model_init_kwargs = dict(model_init_kwargs)
        # Same guard DPOTrainer itself applies before its own create_model_from_path calls (both the main
        # model and its own auto-built reference) — device_map="auto" is incompatible with DeepSpeed
        # Zero-3/multi-GPU and from_pretrained raises if one ends up set, so it must be forced off here too.
        if training_args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
            ref_model_init_kwargs["device_map"] = None
        ref_model = create_model_from_path(script_args.ref_model_name_or_path, **ref_model_init_kwargs)

    ################
    # Dataset
    ################
    dataset = load_from_disk(script_args.dataset_path)

    ################
    # Training
    ################
    trainer = DPOTrainer(
        model=model_args.model_name_or_path,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
