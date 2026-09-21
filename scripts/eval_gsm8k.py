# /// script
# dependencies = [
#     "vllm",
# ]
# ///

"""
GSM8K test-set eval (pass@1 greedy by default, or pass@k with --num_generations/--temperature) for a
single checkpoint, via vLLM.

Evaluates exactly one model per process, deliberately — vLLM doesn't reliably release GPU memory between
successive LLM() instantiations in the same process, so evaluating multiple checkpoints means running
this script once per checkpoint (see scripts/eval_gsm8k_all.sh for the loop over all of them), not looping
over checkpoints in here.

SYSTEM_PROMPT/extract_answer below are copied verbatim from scripts/train_gsm8k.py (not imported — that
module imports trl at the top level for its actual training code, and trl isn't installed in the vllm env
this script runs in, nor should it need to be just to reuse two functions). Keep these in sync with
train_gsm8k.py's copies if that format ever changes — same "reproduce exactly" consistency rule this repo
applies to duplicated trainer code, not a reimplementation that could silently drift from what training
actually optimized the reward against.

Requires vLLM, which is NOT installed in the `tribe` conda env used for training (that env's dependency
pins are for training via the local trl/tribe packages, not inference serving) — run this with a separate
env that has vllm (e.g. `llm_gen`), not `tribe`.

Usage:
python scripts/eval_gsm8k.py \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --name base \
    --output_file gsm8k_eval/base.json

Debug mode: add `--limit 8` to run on just the first 8 test examples, to confirm the model loads and
generates correctly (and, on SLURM, that the whole env/job wiring works) in under a minute instead of a
full pass over all 1319.

Validation mode: add `--val_fraction 0.1` to evaluate against the held-out SEGMENT of the GSM8K TRAINING
split instead of the actual test set — the last val_fraction of `train`, matching the exact same
questions scripts/offpolicy_split.py holds out from every off-policy training script (and
scripts/build_offpolicy_dpo_pairs.py) via the same formula (round(val_fraction * num_questions)). Use this
for hyperparameter selection (e.g. SimPO's beta/gamma, TRIBE's trust_region_eps); reserve the actual test
split (no --val_fraction) for final reported numbers only, so hyperparameters are never chosen by looking
at the same data used for the final score.

Pass@k: add `--num_generations 4 --temperature 1.0 --top_k 50` to sample k completions per problem instead
of one greedy decode; a problem counts as solved if any of the k completions is correct (standard pass@k).
--temperature must be > 0 for --num_generations > 1 (greedy decoding gives k identical completions
otherwise). --top_k is recommended too: unrestricted (top_k=-1) temperature=1.0 sampling can draw from a
checkpoint's noisy low-probability tail and derail into degenerate, incoherent completions that greedy
pass@1 never encounters (see scripts/eval_math.py's docstring, same finding on the MATH negative_fraction
sweep checkpoints).
"""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
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


def build_prompts_and_references(
    tokenizer, system_prompt: str, limit: int | None = None, val_fraction: float | None = None
) -> tuple[list[str], list[str]]:
    if val_fraction is not None:
        # Held-out segment of TRAIN, not the test set — must match scripts/offpolicy_split.py's formula
        # exactly (round(val_fraction * num_questions), last that-many questions) so this evaluates the
        # same questions every off-policy training script excluded from its own train_dataset.
        num_train = load_dataset("openai/gsm8k", "main", split="train").num_rows
        num_val = round(val_fraction * num_train)
        split = f"train[-{num_val}:]"
    else:
        split = "test" if limit is None else f"test[:{limit}]"
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    prompts = []
    references = []
    for example in dataset:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        references.append(example["answer"].split("####")[-1].strip().replace(",", ""))
    return prompts, references


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="HF model id or local checkpoint directory.")
    parser.add_argument("--name", required=True, help="Label for this checkpoint in the output JSON.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Matches GSM8K's max_completion_length.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug mode: evaluate only the first N test examples instead of all 1319.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=None,
        help="Evaluate the held-out train-set validation segment instead of the test set (see module docstring).",
    )
    parser.add_argument(
        "--num_generations", type=int, default=1, help="k for pass@k. Defaults to 1 (greedy pass@1)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Must be > 0 when --num_generations > 1."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Truncate sampling to the top k tokens (-1 disables, vLLM's default — full-vocabulary "
        "sampling). Recommended when --num_generations > 1 (see module docstring).",
    )
    parser.add_argument(
        "--answer_format",
        choices=["hash", "boxed"],
        default="hash",
        help="'hash' (default): prompt for '#### <number>' (this suite's original GSM8K format). 'boxed': "
        "prompt for '\\boxed{}' instead — must match whatever format the checkpoint being evaluated was "
        "actually trained/generated with (see scripts/generate_offpolicy_gsm8k.py's own --answer_format).",
    )
    args = parser.parse_args()
    system_prompt = SYSTEM_PROMPT_BOXED if args.answer_format == "boxed" else SYSTEM_PROMPT_HASH
    extract_answer = extract_answer_boxed if args.answer_format == "boxed" else extract_answer_hash

    llm = LLM(model=args.model_path, dtype="bfloat16")
    tokenizer = llm.get_tokenizer()
    if tokenizer.chat_template is None:
        # Checkpoints saved by a newer transformers (in the `tribe` training env) write the chat template
        # as a standalone chat_template.jinja file rather than embedding it in tokenizer_config.json;
        # llm_gen's older transformers doesn't know to look for that file, so it loads with no template at
        # all. HF Hub model ids (e.g. the base model) already have it embedded and never hit this path.
        chat_template_path = Path(args.model_path) / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer.chat_template = chat_template_path.read_text()
    prompts, references = build_prompts_and_references(tokenizer, system_prompt, args.limit, args.val_fraction)

    sampling_params = SamplingParams(
        n=args.num_generations, temperature=args.temperature, top_k=args.top_k, max_tokens=args.max_new_tokens
    )
    outputs = llm.generate(prompts, sampling_params)

    num_correct = 0
    per_example = []
    for output, reference in zip(outputs, references, strict=True):
        samples = output.outputs
        correct = any(extract_answer(o.text) == reference for o in samples)
        num_correct += int(correct)
        lens = [len(o.token_ids) for o in samples]
        per_example.append({
            "correct": correct,
            "mean_completion_length": sum(lens) / len(lens),
            "frac_extracted": sum(1 for o in samples if extract_answer(o.text) is not None) / len(samples),
            "frac_finished_naturally": sum(1 for o in samples if o.finish_reason == "stop") / len(samples),
        })

    result = {
        "name": args.name,
        "model_path": args.model_path,
        "split": "val" if args.val_fraction is not None else "test",
        "num_generations": args.num_generations,
        "num_correct": num_correct,
        "num_total": len(references),
        "accuracy": num_correct / len(references),
        # Per-datapoint stats (correctness/length/extraction-rate/natural-stop-rate) for post-hoc analysis
        # without rerunning generation — same rationale as eval_math.py's own per_example field.
        "per_example": per_example,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_example"}, indent=2))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
