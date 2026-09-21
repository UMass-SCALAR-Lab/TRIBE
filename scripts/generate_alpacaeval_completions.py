# /// script
# dependencies = [
#     "vllm",
# ]
# ///

"""
Generate model outputs on the AlpacaEval 2.0 eval set (tatsu-lab/alpaca_eval, 805 prompts) in the format
`alpaca_eval` (https://github.com/tatsu-lab/alpaca_eval) expects: a JSON list of {instruction, output,
generator} dicts. Reads prompts from alpacaeval/alpaca_eval_instructions.json, a one-time local cache of
alpaca_eval's own `get_alpaca_eval_data()` loader -- the HF dataset ships a legacy loading script that
current `datasets` versions refuse to run, so the live loader only works in an env with `datasets<4`
installed (see .venv_evalharness in this repo); caching it locally keeps this generation script (which
needs vLLM, run in llm_gen_new) free of that conflicting dependency. Judge separately with:
    source .venv_evalharness/bin/activate
    alpaca_eval --model_outputs <output_file> --annotators_config 'alpaca_eval_gpt4_turbo_fn'

Prompt formatting matches whatever each model was actually trained/evaluated with elsewhere in this suite:
the model's own native chat template if it has one (Instruct checkpoints), otherwise the PKU-Alignment
raw-prompt fallback ("BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:") used for the 3B-base-trained
checkpoints (see scripts/offpolicy_trainer.py's _OffPolicyCollator and
scripts/generate_ultrafeedback_eval_completions.py, which apply the identical fallback).

Usage:
python scripts/generate_alpacaeval_completions.py \
    --model_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-suite-final/ultrafeedback-tribe/raft-3b-base \
    --generator_name raft-3b-base \
    --output_file alpacaeval/raft-3b-base/model_outputs_alpaca.json
"""

import argparse
import json
import os
from pathlib import Path

from vllm import LLM, SamplingParams

INSTRUCTIONS_CACHE = Path(__file__).parent.parent / "alpacaeval" / "alpaca_eval_instructions.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--generator_name", required=True, help="Name recorded in the 'generator' field.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--limit", type=int, default=None, help="Debug mode: only the first N prompts.")
    args = parser.parse_args()

    instructions = json.loads(INSTRUCTIONS_CACHE.read_text())
    if args.limit is not None:
        instructions = instructions[: args.limit]

    llm = LLM(model=args.model_path, dtype="bfloat16")
    tokenizer = llm.get_tokenizer()
    if tokenizer.chat_template is None:
        chat_template_path = Path(args.model_path) / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer.chat_template = chat_template_path.read_text()

    if tokenizer.chat_template is not None:
        rendered = [
            tokenizer.apply_chat_template([{"role": "user", "content": instr}], tokenize=False, add_generation_prompt=True)
            for instr in instructions
        ]
    else:
        rendered = [f"BEGINNING OF CONVERSATION: USER: {instr} ASSISTANT:" for instr in instructions]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(rendered, sampling_params)

    results = [
        {"instruction": instr, "output": o.outputs[0].text, "generator": args.generator_name}
        for instr, o in zip(instructions, outputs, strict=True)
    ]
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} outputs to {args.output_file}")
    os._exit(0)
