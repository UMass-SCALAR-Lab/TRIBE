# /// script
# dependencies = [
#     "vllm",
#     "math-verify",
# ]
# ///
"""Throwaway diagnostic: same length/termination check already run for DPO/RAFT, now for TOPR-MATH-Llama,
to test whether the termination-collapse signature generalizes beyond DPO/RAFT to the policy-gradient-style
methods too. Delete after use."""

import argparse
from pathlib import Path

from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem step by step, then put your final answer in "
    "\\boxed{}."
)

def compute_reward(gold_parsed, completion_text):
    answer_parsed = parse(
        completion_text,
        extraction_config=[LatexExtractionConfig(normalization_config=NormalizationConfig(units=True), boxed_match_priority=0, try_extract_without_anchor=False)],
        extraction_mode="first_match",
    )
    return float(verify(gold_parsed, answer_parsed))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="test[:300]")
    llm = LLM(model=args.model_path, dtype="bfloat16")
    tok = llm.get_tokenizer()
    if tok.chat_template is None:
        p = Path(args.model_path) / "chat_template.jinja"
        if p.exists():
            tok.chat_template = p.read_text()
    prompts = []
    golds = []
    for ex in ds:
        gold_parsed = parse(ex["solution"])
        if len(gold_parsed) == 0:
            continue
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ex["problem"]}]
        prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        golds.append(gold_parsed)
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1024))
    lens = [len(o.outputs[0].token_ids) for o in outputs]
    correct = [compute_reward(g, o.outputs[0].text) for g, o in zip(golds, outputs, strict=True)]
    has_boxed = [1 if "\\boxed" in o.outputs[0].text else 0 for o in outputs]
    finish_stop = [1 if o.outputs[0].finish_reason == "stop" else 0 for o in outputs]
    print(f"=== {args.name} ===")
    print(f"  n={len(lens)}  mean_len={sum(lens)/len(lens):.1f}  accuracy={sum(correct)/len(correct):.4f}")
    print(f"  frac_with_boxed={sum(has_boxed)/len(has_boxed):.4f}  frac_finished_naturally(stop)={sum(finish_stop)/len(finish_stop):.4f}")
