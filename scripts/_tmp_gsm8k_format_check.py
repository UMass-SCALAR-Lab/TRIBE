# /// script
# dependencies = [
#     "vllm",
# ]
# ///
"""Throwaway: check whether zero-shot boxed-format GSM8K eval fails mostly on FORMAT COMPLIANCE
(no \\boxed{} produced) vs genuine wrong answers, to explain the surprisingly low 43.6% base GSM8K
number vs Meta's official 77.7% (8-shot, presumably #### format). Delete after use."""
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams

SYSTEM_PROMPT_BOXED = (
    "You are a helpful math tutor. Solve the problem step by step, then put your final answer in "
    "\\boxed{}."
)

def extract_answer_boxed(text):
    m = re.search(r"\\boxed\{([-\d,]+)\}", text)
    return m.group(1).replace(",", "") if m else None

if __name__ == "__main__":
    ds = load_dataset("openai/gsm8k", "main", split="test[:100]")
    llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", dtype="bfloat16")
    tok = llm.get_tokenizer()
    prompts = []
    refs = []
    for ex in ds:
        messages = [{"role": "system", "content": SYSTEM_PROMPT_BOXED}, {"role": "user", "content": ex["question"]}]
        prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        refs.append(ex["answer"].split("####")[-1].strip().replace(",", ""))
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512))
    extracted = [extract_answer_boxed(o.outputs[0].text) for o in outputs]
    correct = [1 if e == r else 0 for e, r in zip(extracted, refs)]
    has_boxed = [1 if e is not None else 0 for e in extracted]
    print(f"n={len(refs)} accuracy={sum(correct)/len(correct):.4f} frac_has_boxed={sum(has_boxed)/len(has_boxed):.4f}")
    print("\n--- 5 samples where boxed extraction FAILED ---")
    shown = 0
    for i, o in enumerate(outputs):
        if extracted[i] is None and shown < 5:
            print("="*80)
            print(o.outputs[0].text[-400:])
            shown += 1
