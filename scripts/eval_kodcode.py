# /// script
# dependencies = [
#     "vllm",
#     "datasets",
# ]
# ///

"""
KodCode-Light-RL-10K eval (pass@1 greedy by default, or pass@k with --num_generations/--temperature) for a
single checkpoint, via vLLM. Mirrors scripts/eval_math.py's structure; see that module's docstring for the
rationale behind evaluating one checkpoint per process.

KodCode/KodCode-Light-RL-10K (https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K) has no held-out
test split — only `train` (10,000 rows) — so this evaluates a random subsample of `train` by default (see
--limit/--seed). Each row's `test` field is a pytest-style test module that does `from solution import
<fn>`; correctness is checked by writing the model's extracted code to `solution.py` in a scratch dir next
to a copy of that test module, then running `pytest` as a subprocess with a timeout. This is the same
"generate code, execute it against hidden tests" protocol KodCode/HumanEval/MBPP-style benchmarks all use —
only ever runs the checkpoint's OWN generated code, in an isolated per-example temp directory, with a hard
wall-clock timeout per example (--exec_timeout) to bound runaway/hanging completions.

Usage:
python scripts/eval_kodcode.py \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --name base \
    --limit 200 \
    --output_file kodcode_eval/base.json
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write a correct, self-contained solution to the given problem. "
    "Return ONLY a single ```python code block containing the full solution — no explanation."
)


def build_prompt(example: dict) -> str:
    signatures = "\n".join(info["function_declaration"] for info in example["test_info"])
    return (
        f"{example['question']}\n\n"
        f"Your solution must define exactly this/these function signature(s) so the tests can import them:\n"
        f"{signatures}"
    )


def extract_code(completion_text: str) -> str:
    # Last block, not first: reasoning models (e.g. Qwen3's default thinking mode) can emit code inside
    # <think>...</think> before the actual final answer's code block.
    matches = CODE_BLOCK_RE.findall(completion_text)
    return matches[-1] if matches else completion_text


def run_test(solution_code: str, test_code: str, exec_timeout: float) -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "solution.py").write_text(solution_code)
        (tmp_path / "test_solution.py").write_text(test_code)
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "-q", "test_solution.py"],
                cwd=tmp_path,
                capture_output=True,
                timeout=exec_timeout,
            )
        except subprocess.TimeoutExpired:
            return False
        return result.returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="HF model id or local checkpoint directory.")
    parser.add_argument("--name", required=True, help="Label for this checkpoint in the output JSON.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of examples to evaluate (random subsample of `train`, the only split this dataset "
        "has). Defaults to 200; pass a larger value (up to 10000) for a less noisy estimate.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for the --limit subsample.")
    parser.add_argument("--num_generations", type=int, default=1, help="k for pass@k. Defaults to 1 (greedy pass@1).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Must be > 0 when --num_generations > 1.")
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument(
        "--exec_timeout", type=float, default=10.0, help="Wall-clock timeout (seconds) per test-suite execution."
    )
    args = parser.parse_args()

    if shutil.which("pytest") is None:
        parser.error("pytest not found on PATH — install it in this env (pip install pytest).")

    dataset = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
    dataset = dataset.shuffle(seed=args.seed).select(range(min(args.limit, len(dataset))))

    llm = LLM(model=args.model_path, dtype="bfloat16")
    tokenizer = llm.get_tokenizer()
    if tokenizer.chat_template is None:
        chat_template_path = Path(args.model_path) / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer.chat_template = chat_template_path.read_text()

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_prompt(ex)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in dataset
    ]

    sampling_params = SamplingParams(
        n=args.num_generations, temperature=args.temperature, top_k=args.top_k, max_tokens=args.max_new_tokens
    )
    outputs = llm.generate(prompts, sampling_params)

    num_correct = 0
    per_example = []
    for output, example in zip(outputs, dataset, strict=True):
        samples = output.outputs
        results = [run_test(extract_code(o.text), example["test"], args.exec_timeout) for o in samples]
        correct = any(results)
        num_correct += int(correct)
        per_example.append({
            "question_id": example["question_id"],
            "difficulty": example["gpt_difficulty"],
            "correct": correct,
        })

    result = {
        "name": args.name,
        "model_path": args.model_path,
        "dataset": "KodCode/KodCode-Light-RL-10K",
        "split": f"train (random {len(dataset)}-example subsample, seed={args.seed})",
        "num_generations": args.num_generations,
        "num_correct": num_correct,
        "num_total": len(dataset),
        "accuracy": num_correct / len(dataset),
        "per_example": per_example,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_example"}, indent=2))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
