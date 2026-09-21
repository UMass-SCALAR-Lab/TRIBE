# /// script
# dependencies = [
#     "vllm",
#     "math-verify",
# ]
# ///

"""
Eval a checkpoint on PRM800K's held-out test-set problems (scripts/convert_prm800k_offpolicy.py's
--test_output_file — unique problems from openai/prm800k's phase2_test.jsonl, ground_truth_answer only,
no full reference solution text). Otherwise mirrors scripts/eval_math.py's own generation/grading logic
exactly (same SYSTEM_PROMPT, same math_verify config) for direct comparability with every other MATH eval
in this project; a separate script rather than extending eval_math.py since that one is hardcoded to
DigitalLearningGmbH/MATH-lighteval, not a arbitrary local problem list.

Runs in the `llm_gen` conda env (vLLM), not `tribe`.

Usage:
python scripts/eval_prm800k.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --test_problems_file /scratch4/.../offpolicy-data/prm800k_test_problems.json \
    --name base-llama-prm800k \
    --output_file math_eval/base-llama-prm800k.json
"""

import argparse
import json
import logging
import threading
from pathlib import Path

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem step by step, then put your final answer in "
    "\\boxed{}."
)


def compute_reward(gold_parsed, completion_text: str) -> float:
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
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_problems_file", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=-1)
    args = parser.parse_args()

    llm = LLM(model=args.model_path, dtype="bfloat16")
    tokenizer = llm.get_tokenizer()
    if tokenizer.chat_template is None:
        chat_template_path = Path(args.model_path) / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer.chat_template = chat_template_path.read_text()

    test_problems = json.loads(Path(args.test_problems_file).read_text())
    if args.limit:
        test_problems = test_problems[: args.limit]

    prompts, references = [], []
    num_skipped = 0
    for item in test_problems:
        gold_parsed = parse(item["answer"], parsing_timeout=10)
        if len(gold_parsed) == 0:
            num_skipped += 1
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["problem"]},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        references.append(gold_parsed)
    if num_skipped:
        print(f"Skipped {num_skipped}/{len(test_problems)} problems with an unparseable gold answer")

    sampling_params = SamplingParams(
        n=args.num_generations, temperature=args.temperature, top_k=args.top_k, max_tokens=args.max_new_tokens
    )
    outputs = llm.generate(prompts, sampling_params)

    num_correct = 0
    per_example = []
    for output, gold_parsed in zip(outputs, references, strict=True):
        samples = output.outputs
        correct = any(compute_reward(gold_parsed, o.text) > 0 for o in samples)
        num_correct += int(correct)
        lens = [len(o.token_ids) for o in samples]
        per_example.append({
            "correct": correct,
            "mean_completion_length": sum(lens) / len(lens),
            "frac_boxed": sum(1 for o in samples if "\\boxed" in o.text) / len(samples),
            "frac_finished_naturally": sum(1 for o in samples if o.finish_reason == "stop") / len(samples),
        })

    result = {
        "name": args.name,
        "model_path": args.model_path,
        "split": "prm800k_test",
        "num_generations": args.num_generations,
        "num_correct": num_correct,
        "num_total": len(references),
        "accuracy": num_correct / len(references),
        "per_example": per_example,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_example"}, indent=2))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
