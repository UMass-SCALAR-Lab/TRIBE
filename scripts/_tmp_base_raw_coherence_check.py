# /// script
# dependencies = [
#     "vllm",
#     "math-verify",
# ]
# ///
"""Throwaway diagnostic: check whether the RAW (non-instruction-tuned) meta-llama/Llama-3.2-3B base
checkpoint can even produce coherent, on-format MATH completions under a few-shot CoT prompt (raw base
models generally can't follow a zero-shot chat-style instruction the way an -Instruct model can) — before
considering it as an alternative starting point for the offline RAFT/DPO/etc. suite. Delete after use."""

from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from vllm import LLM, SamplingParams

FEWSHOT_PREFIX = """Problem: What is $1+2+3+\\cdots+10$?
Solution: This is the sum of the first 10 positive integers, which is $\\frac{10 \\cdot 11}{2} = 55$. The answer is $\\boxed{55}$.

Problem: Simplify $\\frac{6}{\\sqrt{12}}$.
Solution: We have $\\frac{6}{\\sqrt{12}} = \\frac{6}{2\\sqrt{3}} = \\frac{3}{\\sqrt{3}} = \\sqrt{3}$. The answer is $\\boxed{\\sqrt{3}}$.

Problem: If $f(x) = 2x + 3$, what is $f(5)$?
Solution: $f(5) = 2(5) + 3 = 13$. The answer is $\\boxed{13}$.

"""


def compute_reward(gold_parsed, completion_text):
    answer_parsed = parse(
        completion_text,
        extraction_config=[LatexExtractionConfig(normalization_config=NormalizationConfig(units=True), boxed_match_priority=0, try_extract_without_anchor=False)],
        extraction_mode="first_match",
    )
    return float(verify(gold_parsed, answer_parsed))


if __name__ == "__main__":
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="test")
    llm = LLM(model="meta-llama/Llama-3.2-3B", dtype="bfloat16")

    prompts = []
    golds = []
    for ex in ds:
        gold_parsed = parse(ex["solution"])
        if len(gold_parsed) == 0:
            continue
        prompts.append(FEWSHOT_PREFIX + f"Problem: {ex['problem']}\nSolution:")
        golds.append(gold_parsed)

    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512, stop=["\nProblem:"]))
    lens = [len(o.outputs[0].token_ids) for o in outputs]
    correct = [compute_reward(g, o.outputs[0].text) for g, o in zip(golds, outputs, strict=True)]
    has_boxed = [1 if "\\boxed" in o.outputs[0].text else 0 for o in outputs]

    print(f"n={len(lens)}  mean_len={sum(lens)/len(lens):.1f}  accuracy={sum(correct)/len(correct):.4f}  "
          f"frac_with_boxed={sum(has_boxed)/len(has_boxed):.4f}")
    print("\n--- 5 sample completions ---")
    for i in range(min(5, len(outputs))):
        print("=" * 80)
        print(f"[correct={correct[i]}] {outputs[i].outputs[0].text[:500]}")
