# /// script
# dependencies = [
#     "vllm",
#     "datasets",
# ]
# ///

"""
Generate completions for the UltraFeedback off-policy TRIBE held-out validation prompts (the last
--val_fraction of groups scripts/offpolicy_split.py's split_off_policy_dataset carves off and every
train_ultrafeedback_offpolicy_tribe.py run excludes from training — see that module's docstring), from a
given model (base or a trained checkpoint). Paired with scripts/score_ultrarm.py: run this once per model
being compared, then score both completions files and diff the mean rewards.

Only the DISTINCT prompts are used (one per group_size=4 block), not UltraFeedback's own stored
completions — those came from other models and aren't what we're evaluating.

Usage:
python scripts/generate_ultrafeedback_eval_completions.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --name base \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/ultrafeedback \
    --output_file ultrafeedback_eval/base_completions.json

Debug mode: add --limit 8 to generate for only the first 8 held-out prompts.
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_from_disk
from vllm import LLM, SamplingParams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="HF model id or local checkpoint directory.")
    parser.add_argument("--name", required=True, help="Label for this model in the output JSON.")
    parser.add_argument("--dataset_path", required=True, help="Same --dataset_path training was pointed at.")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Must match the training run's own value.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--limit", type=int, default=200, help="Number of held-out prompts to evaluate.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)
    num_groups = len(dataset) // args.group_size
    num_val_groups = round(args.val_fraction * num_groups)
    val_start = (num_groups - num_val_groups) * args.group_size
    # One row per group (they share the same prompt) — the held-out tail split_off_policy_dataset excludes.
    val_indices = list(range(val_start, len(dataset), args.group_size))[: args.limit]
    prompts_structured = [dataset[i]["prompt"] for i in val_indices]

    llm = LLM(model=args.model_path, dtype="bfloat16")
    tokenizer = llm.get_tokenizer()
    if tokenizer.chat_template is None:
        chat_template_path = Path(args.model_path) / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer.chat_template = chat_template_path.read_text()

    if tokenizer.chat_template is not None:
        rendered = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts_structured
        ]
    else:
        # Base (non-instruct) models ship with no chat_template -- same PKU-Alignment raw-prompt fallback
        # as _OffPolicyCollator (scripts/offpolicy_trainer.py), so train/eval prompt formatting matches.
        rendered = [
            f"BEGINNING OF CONVERSATION: USER: {p[-1]['content']} ASSISTANT:" for p in prompts_structured
        ]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(rendered, sampling_params)

    rows = [
        {"prompt": p, "completion": o.outputs[0].text}
        for p, o in zip(prompts_structured, outputs, strict=True)
    ]
    result = {"name": args.name, "model_path": args.model_path, "num_prompts": len(rows), "rows": rows}
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {len(rows)} completions to {args.output_file}")
    # vLLM's EngineCore subprocess routinely fails to join cleanly on normal interpreter shutdown (NCCL/
    # multiprocessing teardown hang), leaving the SLURM job RUNNING indefinitely after all real work is
    # done -- output is already flushed to disk above, so skip Python's atexit/GC teardown entirely instead
    # of waiting on it.
    os._exit(0)
