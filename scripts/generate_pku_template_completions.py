# /// script
# dependencies = [
#     "vllm",
# ]
# ///

"""
Generate completions using safe-rlhf's legacy PKU-Alignment prompt template ("BEGINNING OF CONVERSATION:
USER: {input} ASSISTANT:", see /work/pi_sniekum_umass_edu/ychittepu/Codes/safe-rlhf/safe_rlhf/configs/
constants.py and datasets/utils.py's format_prompt) instead of the model's own native chat template --
measures how much of the ultrafeedback_eval base-vs-sft reward gap is explained by that template mismatch,
by generating base's completions on the SAME held-out prompts under the SAME non-native template safe-rlhf's
arena.py evaluates both models with.

Usage:
python scripts/generate_pku_template_completions.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --name base_pku_template \
    --completions_file ultrafeedback_eval/base_completions.json \
    --output_file ultrafeedback_eval/base_pku_template_completions.json
"""

import argparse
import json
import os
from pathlib import Path

from vllm import LLM, SamplingParams

PROMPT_BEGIN = "BEGINNING OF CONVERSATION: "
PROMPT_USER = "USER: {input} "
PROMPT_ASSISTANT = "ASSISTANT:"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--completions_file", required=True, help="Existing completions file to reuse the exact same prompt set from."
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.completions_file).read_text())
    prompts_structured = [row["prompt"] for row in data["rows"]]
    instructions = [p[-1]["content"] if isinstance(p, list) else p for p in prompts_structured]

    llm = LLM(model=args.model_path, dtype="bfloat16")
    tokenizer = llm.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rendered = [PROMPT_BEGIN + PROMPT_USER.format(input=instr) + PROMPT_ASSISTANT for instr in instructions]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    outputs = llm.generate(rendered, sampling_params)

    rows = [
        {"prompt": p, "completion": o.outputs[0].text}
        for p, o in zip(prompts_structured, outputs, strict=True)
    ]
    result = {"name": args.name, "model_path": args.model_path, "num_prompts": len(rows), "rows": rows}
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {len(rows)} completions to {args.output_file}")
    os._exit(0)
