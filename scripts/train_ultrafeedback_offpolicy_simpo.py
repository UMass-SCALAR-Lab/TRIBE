# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""
SimPO baseline on UltraFeedback, off-policy / fully-offline: trains on the FIXED preference dataset built
once by scripts/build_ultrafeedback_dpo_pairs.py (chosen = highest-reward, rejected = lowest-reward
completion within each prompt's group of 4) from the same underlying (prompt, completion, reward) data
scripts/train_ultrafeedback_offpolicy_raft.py and scripts/train_ultrafeedback_offpolicy_tribe.py train on.

SimPO (https://huggingface.co/papers/2405.14734) is REFERENCE-FREE by design -- no frozen mu/ref model,
length-normalized average log-probability as its own implicit reward instead of a log-ratio against a
reference policy. See scripts/train_gsm8k_offpolicy_simpo.py's own docstring for why this is the relevant
baseline family for this suite (no behavioral-policy access needed, same premise as TRIBE's off-policy
Stage 1/2).

Uses trl.experimental.cpo.CPOTrainer with loss_type="simpo" (TRL doesn't have a standalone SimPOTrainer) --
identical mechanics to train_gsm8k_offpolicy_simpo.py, just pointed at UltraFeedback's data.

Usage:
python scripts/build_ultrafeedback_dpo_pairs.py \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/ultrafeedback-v2 \
    --output_dir /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/ultrafeedback-v2-dpo-pairs \
    --group_size 4

python scripts/train_ultrafeedback_offpolicy_simpo.py \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/ultrafeedback-v2-dpo-pairs \
    --output_dir offpolicy-simpo-ultrafeedback-baseline \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-7 \
    --beta 2.0 \
    --simpo_gamma 0.5 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --bf16 True
"""

from dataclasses import dataclass, field

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import ModelConfig, TrlParser, get_peft_config
from trl.experimental.cpo import CPOConfig, CPOTrainer


@dataclass
class OffPolicyScriptArguments:
    dataset_path: str = field(
        metadata={"help": "Path to the paired dataset built by scripts/build_ultrafeedback_dpo_pairs.py."}
    )


if __name__ == "__main__":
    parser = TrlParser((OffPolicyScriptArguments, CPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # SimPO's paper loss (no extra NLL term on the chosen response, unlike CPO's own default).
    training_args.loss_type = "simpo"
    training_args.cpo_alpha = 0.0

    ################
    # Model
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        # device_map defaults to "auto" (model-parallel sharding) inside CPOTrainer whenever it's absent
        # from model_init_kwargs -- incompatible with accelerate launch --num_processes>1 (DDP expects each
        # process to own one whole model copy on one GPU). Same fix as train_gsm8k_offpolicy_simpo.py.
        device_map=None,
    )

    ################
    # Dataset
    ################
    dataset = load_from_disk(script_args.dataset_path)

    ################
    # Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
    if tokenizer.pad_token is None:
        # Base (non-instruct) and some instruct tokenizers alike ship without a pad token -- CPOTrainer
        # requires one since padding is enabled. eos_token as pad_token is the standard fallback used
        # elsewhere in this suite.
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        # CPOTrainer's own preprocessing calls apply_chat_template unconditionally (crashes outright with
        # no template set) -- base models ship with none. Same PKU-Alignment raw-prompt convention as
        # scripts/offpolicy_trainer.py's _OffPolicyCollator and every other base-model script in this suite
        # ("BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:{output}"), just expressed as a real Jinja
        # template since CPOTrainer needs an actual tokenizer.chat_template, not a Python-side bypass.
        tokenizer.chat_template = (
            "{%- for message in messages -%}"
            "{%- if message['role'] == 'user' -%}BEGINNING OF CONVERSATION: USER: {{ message['content'] }} ASSISTANT:"
            "{%- elif message['role'] == 'assistant' -%}{{ message['content'] }}"
            "{%- endif -%}"
            "{%- endfor -%}"
        )

    ################
    # Training
    ################
    trainer = CPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
