# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""
RAFT (plain positive-only SFT, OffPolicyRaftTrainer) on HuggingFaceH4/ultrafeedback_binarized's `train_sft`
split (already 1 `chosen` completion per prompt, reward=1.0 for every row -- see
scripts/convert_ultrafeedback_binarized_offpolicy.py) -- a diagnostic control paired with
scripts/train_ultrafeedback_offpolicy_raft.py: same trainer, same hyperparameters, different (already
externally-validated) data, to isolate whether that script's RAFT-underperforms-base result traces to this
suite's `overall_score`-based best-of-4 data selection or to a bug in the shared trainer/collator itself.

No group_size/val split here (unlike every other off-policy script in this suite): this dataset has no
group structure to begin with (one completion per prompt already), and HuggingFace ships its own official
`test_sft` held-out split, used directly instead of scripts/offpolicy_split.py's carved-off tail.

Usage:
python scripts/train_ultrafeedback_binarized_raft.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/ultrafeedback-binarized/train \
    --output_dir offpolicy-raft-ultrafeedback-binarized \
    --group_size 1 \
    --per_device_train_batch_size 16 \
    --max_completion_length 1024 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --deepspeed configs/deepspeed_zero3.json \
    --gradient_checkpointing \
    --bf16 True
"""

from dataclasses import dataclass, field

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, TrlParser, get_peft_config

from offpolicy_trainer import OffPolicyConfig, OffPolicyRaftTrainer


@dataclass
class OffPolicyScriptArguments:
    dataset_path: str = field(
        metadata={"help": "Path to the pre-converted train split (scripts/convert_ultrafeedback_binarized_offpolicy.py's --output_dir/train)."}
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
