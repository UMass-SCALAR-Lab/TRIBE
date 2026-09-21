# /// script
# dependencies = [
#     "vllm",
# ]
# ///

"""
Off-policy data generation for GSM8K — TOPR's single-iteration, fully-offline setup
(https://huggingface.co/papers/2503.14286): generate `n` completions per training question from the
UNTRAINED base model (the behavioral policy mu), label each with the same correctness reward used
everywhere else in this suite, and save the result as a fixed HF dataset. Generated ONCE, never touched
again — every off-policy trainer (REINFORCE, GRPO, DPO, TRIBE) trains for a single epoch over this same
frozen dataset, no regeneration mid-epoch (see this suite's off-policy design discussion for why).

Only (prompt, completion, reward) is saved — not behavioral log-probs. Every downstream method that needs
pi_old/pi_ref log-probs (GRPO's importance ratio, DPO's reference) can recompute them at train time from a
frozen copy of the same base model, same as GRPOTrainer/TribeTrainer already do internally for their own
reference model on-policy; TRIBE's off-policy Stage 1 (tribe/offpolicy_stage1.py) doesn't need them at all
(c is fixed at 0 by construction, not computed from log-probs).

`prompt` is saved as the structured message list (matching every on-policy script's own "prompt" column,
e.g. scripts/train_gsm8k.py's make_conversation), not a pre-rendered chat-template string — so downstream
training scripts apply the chat template themselves with whatever tokenizer they're using, rather than
being tied to vLLM's tokenizer version here.

Uses vLLM for generation (fast enough to matter: GSM8K's ~7473 training questions x n=16 completions is
~120k generations) — requires the `llm_gen` conda env, not `tribe` (same reasoning as
scripts/eval_gsm8k.py: vLLM isn't installed in `tribe`, and doesn't need to be just for this one-time
generation step). SYSTEM_PROMPT/extract_answer copied verbatim from scripts/train_gsm8k.py (not imported —
see scripts/eval_gsm8k.py's docstring for why: that module imports trl at the top level, not installed in
the vllm env this script runs in).

Usage:
python scripts/generate_offpolicy_gsm8k.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --num_generations 16 \
    --output_dir /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/gsm8k

Debug mode: add `--limit 8` to generate for only the first 8 training questions, to confirm the script/env
works end-to-end in under a minute instead of a full pass over all ~7473.
"""

import argparse
import re

from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams


SYSTEM_PROMPT_HASH = (
    "You are a helpful math tutor. Solve the problem step by step, then provide the final "
    "numeric answer on the last line in the format: #### <number>"
)
SYSTEM_PROMPT_BOXED = (
    "You are a helpful math tutor. Solve the problem step by step, then put your final answer in "
    "\\boxed{}."
)


def extract_answer_hash(text: str) -> str | None:
    match = re.search(r"####\s*([\d,]+)", text)
    return match.group(1).replace(",", "") if match else None


def extract_answer_boxed(text: str) -> str | None:
    match = re.search(r"\\boxed\{([-\d,]+)\}", text)
    return match.group(1).replace(",", "") if match else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--num_generations", type=int, default=16, help="TOPR uses n=16 for GSM8K, n=32 for MATH."
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--limit", type=int, default=None, help="Debug mode: only the first N training questions."
    )
    parser.add_argument(
        "--answer_format",
        choices=["hash", "boxed"],
        default="hash",
        help="'hash' (default): prompt for '#### <number>' (this suite's original GSM8K format). 'boxed': "
        "prompt for '\\boxed{}' instead, matching MATH's format and this model's own tendency to answer "
        "GSM8K in \\boxed{} regardless of what's asked — avoids the mixed-format mislabeling '#### '-only "
        "grading produces when the model doesn't follow the hash instruction.",
    )
    args = parser.parse_args()
    SYSTEM_PROMPT = SYSTEM_PROMPT_BOXED if args.answer_format == "boxed" else SYSTEM_PROMPT_HASH
    extract_answer = extract_answer_boxed if args.answer_format == "boxed" else extract_answer_hash

    split = "train" if args.limit is None else f"train[:{args.limit}]"
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    llm = LLM(model=args.model_name_or_path, dtype="bfloat16")
    tokenizer = llm.get_tokenizer()

    prompts_structured = []
    prompts_rendered = []
    reference_answers = []
    for example in dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        prompts_structured.append(messages)
        prompts_rendered.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        reference_answers.append(example["answer"].split("####")[-1].strip().replace(",", ""))

    sampling_params = SamplingParams(
        n=args.num_generations, temperature=args.temperature, max_tokens=args.max_new_tokens
    )
    outputs = llm.generate(prompts_rendered, sampling_params)

    rows = []
    for messages, ref, output in zip(prompts_structured, reference_answers, outputs, strict=True):
        for completion_output in output.outputs:
            completion_text = completion_output.text
            predicted = extract_answer(completion_text)
            reward = 1.0 if predicted is not None and predicted == ref else 0.0
            rows.append({"prompt": messages, "completion": completion_text, "reward": reward})

    dataset_out = Dataset.from_list(rows)
    dataset_out.save_to_disk(args.output_dir)
    print(f"Saved {len(rows)} (prompt, completion, reward) rows to {args.output_dir}")
    print(f"Mean reward: {sum(r['reward'] for r in rows) / len(rows):.4f}")
