# /// script
# dependencies = [
#     "vllm",
#     "math-verify",
# ]
# ///

"""
Off-policy data generation for MATH — same TOPR single-iteration, fully-offline setup
(https://huggingface.co/papers/2503.14286) as scripts/generate_offpolicy_gsm8k.py, applied to MATH instead
of GSM8K: generate `n` completions per training problem from the UNTRAINED base model (the behavioral
policy mu), label each with a correctness reward, and save the result as a fixed HF dataset. Generated
ONCE, never touched again — mirrors scripts/generate_offpolicy_gsm8k.py's docstring for the full rationale
(only (prompt, completion, reward) saved, no behavioral log-probs; prompt saved as a structured message
list, not pre-rendered).

Unlike GSM8K's "#### <number>" format, MATH answers are free-form LaTeX (fractions, expressions, sets, ...)
that can't be checked with a plain string/regex match — e.g. `\\dfrac{1}{2}` and `0.5` and `1/2` are the same
answer. Correctness here uses `math_verify` (https://github.com/huggingface/Math-Verify) for LaTeX-aware
symbolic verification, the same mechanism trl.rewards.accuracy_reward uses on-policy — replicated directly
here (not imported from trl) for the same reason scripts/eval_gsm8k.py/generate_offpolicy_gsm8k.py don't
import trl: this script runs in the `llm_gen` conda env (vLLM), not `tribe` (trl/transformers/torch), and
trl isn't installed there. Requires `math-verify` (pulls in `latex2sympy2_extended`) to be installed in
whichever env this runs in: `pip install math-verify`.

Dataset: DigitalLearningGmbH/MATH-lighteval (`problem`/`solution`/`level`/`type` columns, ungated, no
loading script) — the same MATH train split (~7500 problems) used by open-r1 and most TRL-based GRPO/MATH
recipes. Problems whose gold `solution` doesn't contain a parseable boxed answer are skipped entirely
(can't assign a reward), matching accuracy_reward's own `None`-skip semantics.

Uses vLLM for generation — requires the `llm_gen` conda env, not `tribe` (vLLM isn't installed in `tribe`).

Usage:
python scripts/generate_offpolicy_math.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --num_generations 32 \
    --output_dir /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/math

Debug mode: add `--limit 8` to generate for only the first 8 training problems.

Add `--few_shot` to prepend TOPR's own 4-shot Minerva-style exemplars (see FEW_SHOT_EXAMPLES below) —
recommended for weaker/less math-specialized base models.
"""

import argparse
import logging
import threading

from datasets import Dataset, load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem step by step, then put your final answer in "
    "\\boxed{}."
)

# TOPR's own MATH setup (https://huggingface.co/papers/2503.14286) uses "the 4-shot prompt from Lewkowycz
# et al. (2022)" (Minerva) when generating candidate solutions. Reproduced verbatim from EleutherAI's
# lm-evaluation-harness (lm_eval/tasks/minerva_math/utils.py's list_fewshot_samples()) — the canonical,
# widely-used source for this exact prompt — rather than paraphrased, so it matches what every other
# MATH-few-shot recipe in the literature actually uses. Injected as prior user/assistant turns (this
# repo's chat-template convention, see generate_offpolicy_gsm8k.py's own message-list docstring) rather
# than as raw pre-chat-template text, since Llama-3.2-3B-Instruct is an instruction-tuned chat model, not
# a base completion model — multi-turn few-shot is the idiomatic way to prompt it.
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


if __name__ == "__main__":
    logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
    logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--num_generations", type=int, default=32, help="TOPR uses n=16 for GSM8K, n=32 for MATH."
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1, help="TOPR uses top_k=500 (vLLM default: -1, unrestricted).")
    parser.add_argument("--top_p", type=float, default=1.0, help="TOPR uses top_p=1.0 (same as vLLM's own default).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--limit", type=int, default=None, help="Debug mode: only the first N training problems."
    )
    parser.add_argument(
        "--num_shards", type=int, default=1, help="Split the training set into this many contiguous shards."
    )
    parser.add_argument(
        "--shard_index", type=int, default=0, help="Which shard (0-indexed) this process generates."
    )
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Prepend TOPR's own 4-shot Minerva-style exemplars (Lewkowycz et al. 2022, reproduced from "
        "lm-evaluation-harness) as prior user/assistant turns before the actual problem. Off by default "
        "(matches this script's original zero-shot behavior); recommended for weaker/less math-specialized "
        "base models where zero-shot format compliance is poor.",
    )
    args = parser.parse_args()

    split = "train" if args.limit is None else f"train[:{args.limit}]"
    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split=split)
    if args.num_shards > 1:
        dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_index, contiguous=True)

    llm = LLM(model=args.model_name_or_path, dtype="bfloat16")
    tokenizer = llm.get_tokenizer()

    prompts_structured = []
    prompts_rendered = []
    gold_answers = []
    num_skipped = 0
    for example in dataset:
        gold_parsed = parse(example["solution"], parsing_timeout=10)
        if len(gold_parsed) == 0:
            # Gold solution has no parseable boxed answer — can't assign a reward, skip the problem.
            num_skipped += 1
            continue
        messages = build_messages(example["problem"], args.few_shot)
        prompts_structured.append(messages)
        prompts_rendered.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        gold_answers.append(gold_parsed)
    print(f"Skipped {num_skipped}/{len(dataset)} problems with an unparseable gold solution")

    sampling_params = SamplingParams(
        n=args.num_generations, temperature=args.temperature, max_tokens=args.max_new_tokens,
        top_k=args.top_k, top_p=args.top_p,
    )
    outputs = llm.generate(prompts_rendered, sampling_params)

    rows = []
    for messages, gold_parsed, output in zip(prompts_structured, gold_answers, outputs, strict=True):
        for completion_output in output.outputs:
            completion_text = completion_output.text
            reward = compute_reward(gold_parsed, completion_text)
            rows.append({"prompt": messages, "completion": completion_text, "reward": reward})

    dataset_out = Dataset.from_list(rows)
    dataset_out.save_to_disk(args.output_dir)
    print(f"Saved {len(rows)} (prompt, completion, reward) rows to {args.output_dir}")
    print(f"Mean reward: {sum(r['reward'] for r in rows) / len(rows):.4f}")
