# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""
SimPO baseline on GSM8K, off-policy / fully-offline (TOPR's single-iteration regime,
https://huggingface.co/papers/2503.14286): trains on the FIXED preference dataset built once by
scripts/build_offpolicy_dpo_pairs.py from the same underlying (prompt, completion, reward) data every
other off-policy baseline in this suite trains on — same pairing, same data, as
scripts/train_gsm8k_offpolicy_dpo.py.

SimPO (https://huggingface.co/papers/2405.14734) is REFERENCE-FREE by design — no frozen mu/ref model
anywhere, using length-normalized average log-probability as its own implicit reward instead of a
log-ratio against a reference policy. Directly relevant to this suite's own "no behavioral-policy access
needed" framing (see TRIBE's off-policy Stage 1/2, scripts/offpolicy_trainer.py), as an existing baseline
built on the same premise.

Uses trl.experimental.cpo.CPOTrainer with loss_type="simpo" — TRL doesn't have a standalone SimPOTrainer;
SimPO is implemented as a CPOConfig loss_type variant. `cpo_alpha=0.0` disables CPO's own extra NLL
regularization term on the chosen response (SimPO's own paper loss doesn't have it) to get pure SimPO.

Usage:
python scripts/build_offpolicy_dpo_pairs.py \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/gsm8k \
    --output_dir /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/gsm8k-dpo-pairs \
    --group_size 16

python scripts/train_gsm8k_offpolicy_simpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/gsm8k-dpo-pairs \
    --output_dir offpolicy-simpo-gsm8k-baseline \
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
        metadata={"help": "Path to the paired dataset built by scripts/build_offpolicy_dpo_pairs.py."}
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
        # from model_init_kwargs — incompatible with accelerate launch --num_processes>1 (DDP expects each
        # process to own one whole model copy on one GPU). Same fix as train_gsm8k_online_dpo.py.
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
        # Some tokenizers (e.g. Llama-3.2's) don't ship a pad token at all — CPOTrainer requires one since
        # padding is enabled. eos_token as pad_token is the standard fallback (same convention used
        # elsewhere in this suite for models without their own pad token).
        tokenizer.pad_token = tokenizer.eos_token

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
