# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""
TRIBE, off-policy / fully-offline on UltraFeedback (https://huggingface.co/papers/2310.01377): trains for
ONE epoch over a FIXED dataset pre-converted by scripts/convert_ultrafeedback_offpolicy.py (completions and
GPT-4 scores UltraFeedback itself ships, never regenerated) — same single-iteration off-policy regime as
scripts/train_gsm8k_offpolicy_tribe.py; see that script's own docstring for the shared OffPolicyTribeTrainer
mechanics (`c` fixed at 0, `beta` a no-op kept only for flag parity). The only real difference from the
GSM8K/MATH off-policy scripts: group_size is fixed at 4 (UltraFeedback ships exactly 4 completions per
prompt), not a --num_generations choice made at data-generation time.

Uses OffPolicyTribeTrainer (scripts/offpolicy_trainer.py): see scripts/train_gsm8k_offpolicy_tribe.py's
docstring for why this subclasses plain transformers.Trainer rather than GRPOTrainer/SFTTrainer.

Usage:
python scripts/train_ultrafeedback_offpolicy_tribe.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/ultrafeedback \
    --output_dir offpolicy-tribe-ultrafeedback-baseline \
    --group_size 4 \
    --per_device_train_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 1e-7 \
    --trust_region_eps 0.05 \
    --gradient_checkpointing \
    --bf16 True
"""

import math
from dataclasses import dataclass, field

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, TrlParser, get_peft_config

from offpolicy_split import split_off_policy_dataset
from offpolicy_trainer import OffPolicyConfig, OffPolicyTribeTrainer


@dataclass
class OffPolicyScriptArguments:
    dataset_path: str = field(
        metadata={"help": "Path to the pre-converted dataset (scripts/convert_ultrafeedback_offpolicy.py's --output_dir)."}
    )
    beta: float = field(default=0.1, metadata={"help": "No effect here (c is always 0) — kept for flag parity."})
    trust_region_eps: float = field(default=0.05, metadata={"help": "Stage 1's trust-region budget."})
    use_unlikelihood: bool = field(
        default=False,
        metadata={
            "help": "Use the unlikelihood trick (-log1p(-p), self-attenuating) for weight<0 samples instead "
            "of the plain floored NLL. See OffPolicyTribeTrainer's docstring/compute_loss."
        },
    )
    divergence: str = field(
        default="kl_new_old",
        metadata={
            "help": "Stage 1's trust-region divergence, one of 'kl_new_old' (closed form, fast), "
            "'kl_old_new', 'chi_squared' (both solved via cvxpy — called once per step here, ~10ms for a "
            "single group, negligible total overhead). See tribe/stage1.py."
        },
    )
    use_polyak_ema: bool = field(
        default=False,
        metadata={
            "help": "After every optimizer step, blend the live model's parameters into a running Polyak "
            "average and write that average back into the model (see offpolicy_trainer.py's "
            "_PolyakEMACallback), instead of training on the raw SGD/Adam trajectory."
        },
    )
    polyak_ema_tau: float = field(
        default=0.99, metadata={"help": "Decay for --use_polyak_ema: ema <- tau*ema + (1-tau)*p_new."}
    )
    rho_weight_offset: float = field(
        default=1.0,
        metadata={
            "help": "Stage 2's weight is rho* minus this offset. Default 1.0 is TRIBE's own convention "
            "(reward at the group mean gets weight 0). 0.0 tests the raw-rho* variant instead (weight is "
            "never negative for chi_squared, since its rho* is already clamped at 0)."
        },
    )
    negative_fraction: float = field(
        default=1.0,
        metadata={
            "help": "Fraction of negative (below-group-average) examples to keep in Stage 2's loss each "
            "step; all positive examples are always kept. Values above 1.0 oversample instead of "
            "subsampling: e.g. 1.2 gives every negative weight at least 1 plus a random fifth of each "
            "group's negatives weight 2, for an expected per-group average weight of 1.2. Default 1.0 "
            "keeps all of them exactly once (prior behavior). Mutually exclusive with --min_pos_neg_ratio."
        },
    )
    min_pos_neg_ratio: float | None = field(
        default=None,
        metadata={
            "help": "Mutually exclusive with --negative_fraction (leave that at 1.0 when using this). "
            "Instead of an EXPECTED per-group keep-fraction, directly enforces a MINIMUM realized "
            "positives-per-kept-negative ratio for the whole batch: never touches positives; if the "
            "batch's own pos:neg ratio already meets or exceeds this, no negatives are dropped either; "
            "otherwise drops just enough (randomly selected) negatives to reach it exactly. None (default) "
            "leaves --negative_fraction's own gating in effect."
        },
    )
    negative_logp_floor: float = field(
        default=math.log(1e-8),
        metadata={
            "help": "Floor (in log-prob space) the floored-NLL branch clamps a negative-weighted token's "
            "log-prob at before the loss stops applying further downward pressure on it. Default log(1e-8) "
            "~= -18.42 was found too permissive at negative_fraction=1.0: batch-average token log-prob "
            "crosses this floor ~30%% through the epoch and never stabilizes. Pass a less negative value "
            "(e.g. -3, -4, -5, -6) to stop pushing much earlier. Ignored when --negative_switch_threshold "
            "is set."
        },
    )
    negative_switch_threshold: float | None = field(
        default=None,
        metadata={
            "help": "If set (e.g. 0.5), replaces both the floored-NLL and use_unlikelihood branches for "
            "weight<0 samples with a per-token switch: plain NLL while that token's own probability p is "
            "above this threshold, unlikelihood once p drops to/below it. Self-attenuates without needing "
            "negative_logp_floor's artificial clamp (see OffPolicyTribeTrainer.negative_switch_threshold's "
            "docstring for the gradient-shape rationale). Mutually exclusive with --use_unlikelihood."
        },
    )
    ref_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Model that actually GENERATED the offline dataset, if different from "
            "--model_name_or_path (the model being trained). Only needed for a weak-learner setup (e.g. "
            "training a smaller model on a larger model's generations) -- default None assumes the model "
            "being trained is also the one that generated the data (the ordinary same-model case)."
        },
    )
    log_f_div_mu: bool = field(
        default=True,
        metadata={
            "help": "Log tribe/f_div_mu (purely diagnostic, not used by Stage 1/2's math). Building the "
            "frozen mu_model copy and its periodic forward pass this needs have been confirmed to eat "
            "enough memory margin to OOM a ZeRO-3 run with long completions -- pass --log_f_div_mu False "
            "to skip both if you hit that."
        },
    )
    use_chi_squared_beta0_closed_form: bool = field(
        default=False,
        metadata={
            "help": "Opt-in fast path for --divergence chi_squared --beta 0 batches: closed-form Stage 1 "
            "solve (tribe.stage1._solve_chi_squared_beta0_closed_form) instead of the CPU-bound cvxpy "
            "solve every step normally requires for chi_squared. Falls back to cvxpy automatically for "
            "any batch it can't handle exactly (nonnegativity constraint binding). No effect unless "
            "--divergence chi_squared and --beta 0 are both set."
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
    trainer = OffPolicyTribeTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        beta=script_args.beta,
        trust_region_eps=script_args.trust_region_eps,
        use_unlikelihood=script_args.use_unlikelihood,
        divergence=script_args.divergence,
        use_polyak_ema=script_args.use_polyak_ema,
        polyak_ema_tau=script_args.polyak_ema_tau,
        rho_weight_offset=script_args.rho_weight_offset,
        negative_fraction=script_args.negative_fraction,
        min_pos_neg_ratio=script_args.min_pos_neg_ratio,
        negative_logp_floor=script_args.negative_logp_floor,
        negative_switch_threshold=script_args.negative_switch_threshold,
        ref_model_name_or_path=script_args.ref_model_name_or_path,
        log_f_div_mu=script_args.log_f_div_mu,
        use_chi_squared_beta0_closed_form=script_args.use_chi_squared_beta0_closed_form,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
