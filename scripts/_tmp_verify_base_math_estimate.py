# /// script
# dependencies = [
#     "vllm",
#     "math-verify",
# ]
# ///
"""Throwaway: verify eval_math.py's own extraction/reward path isn't hiding a bug like the one found in
eval_gsm8k.py's boxed regex (dollar signs). Reuses eval_math.py's exact build_messages/compute_reward via
direct import, same greedy zero-shot config, on a subset of the real test split. Delete after use."""
import sys
sys.path.insert(0, "scripts")
from eval_math import build_messages, compute_reward, SYSTEM_PROMPT
from datasets import load_dataset
from math_verify import parse
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="test[:100]")
    llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", dtype="bfloat16")
    tok = llm.get_tokenizer()
    prompts, golds = [], []
    for ex in ds:
        gold_parsed = parse(ex["solution"])
        if len(gold_parsed) == 0:
            continue
        messages = build_messages(ex["problem"], few_shot=False)
        prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        golds.append(gold_parsed)
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1024))
    rewards = [compute_reward(g, o.outputs[0].text) for g, o in zip(golds, outputs, strict=True)]
    has_boxed = [1 if "\\boxed" in o.outputs[0].text else 0 for o in outputs]
    print(f"n={len(rewards)} accuracy={sum(rewards)/len(rewards):.4f} frac_has_boxed_substring={sum(has_boxed)/len(has_boxed):.4f}")
    # flag cases with a \boxed{...} substring present but reward==0, to manually inspect for extraction misses
    print("\n--- boxed present but reward=0 (potential extraction misses) ---")
    shown = 0
    for i, o in enumerate(outputs):
        if has_boxed[i] and rewards[i] == 0.0 and shown < 8:
            text = o.outputs[0].text
            idx = text.rfind("\\boxed")
            print("="*80)
            print(text[max(0,idx-50):idx+120])
            shown += 1
    print(f"\ntotal boxed-but-reward0 cases: {sum(1 for i in range(len(rewards)) if has_boxed[i] and rewards[i]==0.0)}")
