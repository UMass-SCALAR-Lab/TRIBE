# /// script
# dependencies = [
#     "trl",
#     "datasets",
# ]
# ///

"""
Offline RAFT (rejection-sampling SFT) baseline on UltraFeedback: plain, UNWEIGHTED NLL on each prompt's
single best-scored completion (of 4), the rest dropped entirely. No Stage 1, no rho*, no mu/ratio, no
frozen reference model — the simplest possible "does plain positive-only SFT on this data work at all"
baseline, requested to isolate whether TRIBE negative_fraction=0.0's underperformance vs base (see
ultrafeedback_eval/) is specific to TRIBE's own (rho*-1)-weighted loss or a property of positive-only SFT
on this off-policy, cross-model UltraFeedback data more generally.

scripts/offpolicy_trainer.py's OffPolicyRaftTrainer hardcodes a binary `reward == 1.0` filter (GSM8K/MATH's
own 0/1 correctness reward) — UltraFeedback's reward is continuous (overall_score/10, see
scripts/convert_ultrafeedback_offpolicy.py), so there is no natural reward==1.0 subset. Adapting for
continuous, per-group reward instead of modifying that trainer: pre-select, once here, the single
highest-reward completion in each group_size=4 block and relabel ONLY those rows' `reward` to 1.0 (the
exact value RAFT's own filter/loss need — its loss is unweighted NLL on survivors, so the literal reward
value never enters the loss, only survival/non-survival does) — then hand this already-filtered dataset to
the unmodified OffPolicyRaftTrainer, same as every other off-policy driver script in this suite reuses that
trainer as-is.

Usage:
python scripts/train_ultrafeedback_offpolicy_raft.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/ultrafeedback \
    --output_dir offpolicy-raft-ultrafeedback-baseline \
    --group_size 4 \
    --per_device_train_batch_size 16 \
    --max_completion_length 1024 \
    --learning_rate 1e-7 \
    --lr_scheduler_type constant \
    --deepspeed configs/deepspeed_zero3.json \
    --gradient_checkpointing \
    --bf16 True
"""

from dataclasses import dataclass, field

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, TrlParser, get_peft_config

from offpolicy_split import split_off_policy_dataset
from offpolicy_trainer import OffPolicyConfig, OffPolicyRaftTrainer


@dataclass
class OffPolicyScriptArguments:
    dataset_path: str = field(
        metadata={"help": "Path to the pre-converted dataset (scripts/convert_ultrafeedback_offpolicy.py's --output_dir)."}
    )


def select_best_of_group(dataset: Dataset, group_size: int) -> Dataset:
    """One row per group_size-block: the row with the highest `reward`, relabeled to 1.0 so
    OffPolicyRaftTrainer's own `reward == 1.0` filter (written for GSM8K/MATH's binary reward) keeps it
    unchanged."""
    rows = []
    for start in range(0, len(dataset), group_size):
        group = dataset[start : start + group_size]
        best_i = max(range(group_size), key=lambda i: group["reward"][i])
        rows.append({"prompt": group["prompt"][best_i], "completion": group["completion"][best_i], "reward": 1.0})
    return Dataset.from_list(rows)


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
    dataset = select_best_of_group(dataset, training_args.group_size)

    ################
    # Training
    ################
    trainer = OffPolicyRaftTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
