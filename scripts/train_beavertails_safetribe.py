# /// script
# dependencies = [
#     "trl",
#     "cvxpy",
# ]
# ///

"""
Safe-TRIBE: on-policy, cost-constrained TRIBE on BeaverTails (https://huggingface.co/papers/2310.12773's
own dataset). Stage 1 maximizes the reward-advantage objective subject to an expected-cost constraint
(tribe.stage1_safe.compute_rho_star_safe) instead of TRIBE's plain reward-only trust region; Stage 2 is
the same (rho*-1)-weighted BC loss as plain TRIBE (tribe.tribe_trainer.TribeTrainer) — see
tribe/safe_tribe_trainer.py's own docstring for exactly what differs.

Setup: the reward and cost models are Safe-RLHF's own AutoModelForScore checkpoints (a custom score-head
architecture, not a plain AutoModelForSequenceClassification one). Rather than requiring the `safe_rlhf`
package importable from the `tribe` conda env, `tribe/score_model.py` ports just the Llama-specific score
model class directly into this repo (same class hierarchy/state_dict keys as Safe-RLHF's own checkpoints,
so `from_pretrained` loads them unchanged) — see that module's own docstring. Also requires
--model_name_or_path to be an SFT checkpoint already fine-tuned the same way the reward/cost models' own
base model was (this project: Llama-3.2-3B-Instruct SFT'd on Alpaca, matching the Safe-RLHF paper's own
pipeline — see the safe-rlhf repo's scripts/sft.sh, not this repo's).

Prompt format: reward/cost models score `PROMPT_INPUT.format(input=prompt) + " " + response`
(safe_rlhf.configs.constants.PROMPT_INPUT = "BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:"), the
exact non-chat-template plain-text format their SFT/RM/CM training all use — the policy model is prompted
the same way here (not this project's usual chat-template convention) so it's consistent with what the
reward/cost models were actually trained to score.

Usage:
python scripts/train_beavertails_safetribe.py \\
    --model_name_or_path /path/to/safe-rlhf-suite/llama3.2-3b/sft \\
    --reward_model_name_or_path /path/to/safe-rlhf-suite/llama3.2-3b/rm \\
    --cost_model_name_or_path /path/to/safe-rlhf-suite/llama3.2-3b/cm \\
    --output_dir safetribe-beavertails-baseline \\
    --dataset_train_split 30k_train \\
    --num_generations 8 \\
    --per_device_train_batch_size 8 \\
    --gradient_accumulation_steps 4 \\
    --max_completion_length 512 \\
    --learning_rate 1e-6 \\
    --beta 0 \\
    --trust_region_eps 0.05 \\
    --divergence kl_new_old \\
    --cost_limit 0.0 \\
    --num_train_epochs 1 \\
    --gradient_checkpointing \\
    --bf16 True
"""

from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from tribe.safe_tribe_config import SafeTribeConfig
from tribe.safe_tribe_trainer import SafeTribeTrainer


PROMPT_BEGIN = "BEGINNING OF CONVERSATION: "
PROMPT_USER = "USER: {input} "
PROMPT_ASSISTANT = "ASSISTANT:"
PROMPT_INPUT = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT


@dataclass
class SafeTribeScriptArguments(ScriptArguments):
    reward_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Safe-RLHF AutoModelForScore checkpoint (reward model), trained on the SFT model."},
    )
    cost_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Safe-RLHF AutoModelForScore checkpoint (cost model), trained on the SFT model."},
    )
    score_batch_size: int = field(
        default=8,
        metadata={"help": "Micro-batch size for the reward/cost model forward passes (separate from the policy's)."},
    )


def make_score_fn(model_name_or_path: str, score_batch_size: int):
    """
    Build a trl-reward-function-shaped callable (`(prompts, completions, **kwargs) -> list[float]`) around
    a Safe-RLHF AutoModelForScore checkpoint. Lazily loaded on first call so --help/arg-parsing doesn't
    need the safe_rlhf package importable.
    """
    state = {}

    def load():
        if not state:
            import deepspeed
            from transformers import AutoTokenizer

            from tribe.score_model import LlamaForScore

            # No `device_map=` here: DeepSpeed ZeRO-3 (used for the policy) refuses `device_map` on ANY
            # from_pretrained call once active in the accelerate state, even for this unrelated,
            # never-trained score model — see transformers/integrations/accelerate.py's
            # check_and_set_device_map.
            model = LlamaForScore.from_pretrained(model_name_or_path, dtype=torch.bfloat16).eval()
            # The `from_pretrained` call above also constructs this model's parameters already
            # ZeRO-3-partitioned (the same global zero.Init() hook `transformers` installs once any
            # ZeRO-3 TrainingArguments is parsed catches this "unrelated" model too) — exactly Safe-RLHF's
            # own situation with its reward/cost models (safe_rlhf/trainers/rl_trainer.py's
            # _init_eval_engine). Left unwrapped, a forward pass only sees this rank's local parameter
            # shard (e.g. a truncated embedding table) instead of the full model, which is what caused
            # "Padding_idx must be within num_embeddings". deepspeed.initialize() with no optimizer (a
            # pure eval engine, matching _init_eval_engine's own pattern) installs the all-gather/
            # re-partition hooks a correct forward pass needs.
            model, *_ = deepspeed.initialize(
                model=model,
                config={"bf16": {"enabled": True}, "zero_optimization": {"stage": 3}, "train_micro_batch_size_per_gpu": 1},
            )
            state["model"] = model
            state["tokenizer"] = AutoTokenizer.from_pretrained(model_name_or_path)
        return state["model"], state["tokenizer"]

    def score_fn(prompts, completions, **kwargs):
        model, tokenizer = load()
        texts = []
        for prompt, completion in zip(prompts, completions, strict=False):
            prompt_text = prompt if isinstance(prompt, str) else prompt[-1]["content"]
            completion_text = completion if isinstance(completion, str) else completion[-1]["content"]
            texts.append(PROMPT_INPUT.format(input=prompt_text) + " " + completion_text)

        scores = []
        with torch.no_grad():
            for i in range(0, len(texts), score_batch_size):
                batch = texts[i : i + score_batch_size]
                encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(
                    torch.cuda.current_device()
                )
                end_scores = model(encoded["input_ids"], attention_mask=encoded["attention_mask"]).end_scores
                scores.extend(end_scores.squeeze(dim=-1).float().cpu().tolist())
        return scores

    return score_fn


def make_conversation(example):
    # Plain-text PROMPT_INPUT format, not this project's usual chat-template convention — see this
    # script's own docstring for why (must match what the reward/cost models were trained to score).
    return {"prompt": PROMPT_INPUT.format(input=example["prompt"])}


if __name__ == "__main__":
    parser = TrlParser((SafeTribeScriptArguments, SafeTribeConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if script_args.reward_model_name_or_path is None or script_args.cost_model_name_or_path is None:
        parser.error("--reward_model_name_or_path and --cost_model_name_or_path are required")

    ################
    # Model
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )

    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name or "PKU-Alignment/BeaverTails",
        script_args.dataset_config,
        split=script_args.dataset_train_split,
    )
    # BeaverTails repeats the same prompt across many (response, category, is_safe) rows — dedupe to
    # unique prompts before generating our own on-policy completions for them.
    seen = set()
    unique_indices = []
    for i, p in enumerate(dataset["prompt"]):
        if p not in seen:
            seen.add(p)
            unique_indices.append(i)
    dataset = dataset.select(unique_indices)
    dataset = dataset.map(make_conversation, remove_columns=dataset.column_names)

    ################
    # Reward / cost functions
    ################
    reward_fn = make_score_fn(script_args.reward_model_name_or_path, script_args.score_batch_size)
    cost_fn = make_score_fn(script_args.cost_model_name_or_path, script_args.score_batch_size)

    # Convention (see tribe/safe_tribe_trainer.py's docstring): LAST reward_funcs entry is the cost
    # signal, weighted 0 so it never enters GRPO's own blended reward/advantage. reward_weights is a
    # GRPOConfig field (read from args), not a GRPOTrainer.__init__ parameter, so it's set here rather
    # than passed to the constructor below.
    training_args.reward_weights = [1.0, 0.0]

    ################
    # Training
    ################
    trainer = SafeTribeTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[reward_fn, cost_fn],
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
