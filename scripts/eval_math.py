# /// script
# dependencies = [
#     "vllm",
#     "math-verify",
# ]
# ///

"""
MATH test-set eval (pass@1 greedy by default, or pass@k with --num_generations/--temperature) for a single
checkpoint, via vLLM. Mirrors scripts/eval_gsm8k.py's structure exactly; see that module's docstring for the
rationale behind evaluating one checkpoint per process and copying training-time constants here rather than
importing them.

SYSTEM_PROMPT/compute_reward below are copied verbatim from scripts/generate_offpolicy_math.py (not
imported — that module runs vLLM generation and this one does too, both in the `llm_gen` env, but keeping
them as independent copies matches the same "reproduce exactly, don't silently drift" rule this repo
applies to duplicated trainer code). Keep these in sync with generate_offpolicy_math.py's copies if that
reward definition ever changes.

Unlike GSM8K's "#### <number>" format, MATH answers are free-form LaTeX that can't be checked with a plain
string/regex match (e.g. `\\dfrac{1}{2}`, `0.5`, and `1/2` are the same answer) — this uses `math_verify`
for LaTeX-aware symbolic verification, same as generate_offpolicy_math.py's reward labeling. Requires
`math-verify` installed in whichever env this runs in: `pip install math-verify`.

Usage:
python scripts/eval_math.py \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --name base \
    --output_file math_eval/base.json

Pass@k: add `--num_generations 4 --temperature 1.0 --top_k 50` to sample k completions per problem instead
of one greedy decode; a problem counts as solved if any of the k completions is correct (standard pass@k).
--temperature must be > 0 for --num_generations > 1 (greedy decoding gives k identical completions
otherwise). --top_k is recommended too: unrestricted (top_k=-1) temperature=1.0 sampling can draw from a
checkpoint's noisy low-probability tail and derail into degenerate, incoherent completions that greedy
pass@1 never encounters — confirmed by direct inspection on the negative_fraction sweep checkpoints.

Debug mode: add `--limit 8` to run on just the first 8 test examples.

Validation mode: add `--val_fraction 0.1` to evaluate against the held-out SEGMENT of the MATH TRAINING
split instead of the actual test set — the last val_fraction of `train`, matching the exact same questions
scripts/offpolicy_split.py holds out from off-policy training (see scripts/eval_gsm8k.py's own docstring
for why: hyperparameter selection should never look at the actual test split).
"""

import argparse
import json
import logging
import threading
from pathlib import Path

from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem step by step, then put your final answer in "
    "\\boxed{}."
)

# Copied verbatim from scripts/generate_offpolicy_math.py — see that module's own FEW_SHOT_EXAMPLES
# docstring for provenance (Lewkowycz et al. 2022 / lm-evaluation-harness). Must stay in sync: evaluating
# a checkpoint trained with --few_shot using a mismatched (zero-shot) prompt here would silently measure
# the wrong thing.
FEW_SHOT_EXAMPLES = [
    {
        "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
        "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
    },
    {
        "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
    },
    {
        "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
    },
]


def build_messages(problem: str, few_shot: bool) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": ex["problem"]})
            messages.append({"role": "assistant", "content": ex["solution"]})
    messages.append({"role": "user", "content": problem})
    return messages


def compute_reward(gold_parsed, completion_text: str) -> float:
    # math_verify uses signal.alarm() for timeouts, which only works in the main thread.
    is_main_thread = threading.current_thread() is threading.main_thread()
    parsing_timeout = 10 if is_main_thread else None
    verify_timeout = 5 if is_main_thread else None
    answer_parsed = parse(
        completion_text,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(units=True),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
        parsing_timeout=parsing_timeout,
    )
    return float(verify(gold_parsed, answer_parsed, timeout_seconds=verify_timeout))


def build_prompts_and_references(
    tokenizer, limit: int | None = None, val_fraction: float | None = None, few_shot: bool = False
) -> tuple[list[str], list, list[dict]]:
    if val_fraction is not None:
        # Held-out segment of TRAIN, not the test set — must match scripts/offpolicy_split.py's formula
        # exactly (round(val_fraction * num_questions), last that-many questions) so this evaluates the
        # same questions scripts/generate_offpolicy_math.py's off-policy training excluded from training.
        num_train = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train").num_rows
        num_val = round(val_fraction * num_train)
        split = f"train[-{num_val}:]"
    else:
        split = "test" if limit is None else f"test[:{limit}]"
    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split=split)
    prompts = []
    references = []
    metadata = []
    num_skipped = 0
    for example in dataset:
        gold_parsed = parse(example["solution"], parsing_timeout=10)
        if len(gold_parsed) == 0:
            # Matches generate_offpolicy_math.py's own skip semantics: a gold solution with no parseable
            # boxed answer can't be checked against, so it's excluded here too (not counted as wrong).
            num_skipped += 1
            continue
        messages = build_messages(example["problem"], few_shot)
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        references.append(gold_parsed)
        metadata.append({"type": example["type"], "level": example["level"]})
    if num_skipped:
        print(f"Skipped {num_skipped}/{len(dataset)} problems with an unparseable gold solution")
    return prompts, references, metadata


if __name__ == "__main__":
    logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
    logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="HF model id or local checkpoint directory.")
    parser.add_argument("--name", required=True, help="Label for this checkpoint in the output JSON.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Matches MATH's max_completion_length.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug mode: evaluate only the first N test examples instead of the full test split.",
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
        "sampling). Recommended when --num_generations > 1: unrestricted temperature=1.0 sampling can "
        "draw from a checkpoint's noisy low-probability tail and produce degenerate completions that "
        "greedy pass@1 never encounters.",
    )
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Prepend the same 4-shot Minerva-style exemplars scripts/generate_offpolicy_math.py's own "
        "--few_shot uses. Must match whatever the checkpoint being evaluated was actually trained/generated "
        "with — a mismatch silently measures the wrong thing.",
    )
    args = parser.parse_args()

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
    prompts, references, metadata = build_prompts_and_references(
        tokenizer, args.limit, args.val_fraction, args.few_shot
    )

    sampling_params = SamplingParams(
        n=args.num_generations, temperature=args.temperature, top_k=args.top_k, max_tokens=args.max_new_tokens
    )
    outputs = llm.generate(prompts, sampling_params)

    num_correct = 0
    per_example = []
    for output, gold_parsed, meta in zip(outputs, references, metadata, strict=True):
        samples = output.outputs
        correct = any(compute_reward(gold_parsed, o.text) > 0 for o in samples)
        num_correct += int(correct)
        lens = [len(o.token_ids) for o in samples]
        per_example.append({
            "type": meta["type"],
            "level": meta["level"],
            "correct": correct,
            "mean_completion_length": sum(lens) / len(lens),
            "frac_boxed": sum(1 for o in samples if "\\boxed" in o.text) / len(samples),
            "frac_finished_naturally": sum(1 for o in samples if o.finish_reason == "stop") / len(samples),
        })

    # The writer/reader transformers-version pair is the actual variable behind a whole class of silent
    # checkpoint-corruption bugs (see scripts/overlay_base_config.py's docstring) and wasn't recorded
    # anywhere in this project's eval artifacts before now. reader_transformers_version is always this
    # env's own version; writer_transformers_version (and which base model's config was overlaid in, if
    # any) comes from overlay_manifest.json when scripts/overlay_base_config.py was run on this checkpoint.
    from transformers import __version__ as reader_transformers_version

    manifest_path = Path(args.model_path) / "overlay_manifest.json"
    overlay_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    result = {
        "name": args.name,
        "model_path": args.model_path,
        "split": "val" if args.val_fraction is not None else "test",
        "num_generations": args.num_generations,
        "num_correct": num_correct,
        "num_total": len(references),
        "accuracy": num_correct / len(references),
        "reader_transformers_version": reader_transformers_version,
        "overlay_manifest": overlay_manifest,
        # Per-datapoint stats (type/level/correctness/length/boxed-rate/natural-stop-rate) for post-hoc
        # analysis (e.g. accuracy broken down by topic or difficulty, or the length/termination-rate checks
        # this suite has repeatedly needed via one-off throwaway scripts) without rerunning generation.
        "per_example": per_example,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_example"}, indent=2))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
