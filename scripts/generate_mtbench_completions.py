# /// script
# dependencies = [
#     "vllm",
# ]
# ///

"""
Generate model answers for MT-Bench (lm-sys/FastChat, 80 questions, 2 turns each) in the format
`fastchat.llm_judge.gen_judgment` expects: one JSONL line per question at
mtbench/data/mt_bench/model_answer/{model_id}.jsonl, {"question_id", "answer_id", "model_id", "choices":
[{"index": 0, "turns": [turn1_answer, turn2_answer]}], "tstamp"}.

Multi-turn: turn 2's prompt is built from turn 1's question AND the model's OWN generated turn-1 answer
(not a reference), so this generates in two passes -- turn 1 for every question first, then turn 2 once
every turn-1 answer is known. Prompt formatting matches the rest of this suite: native chat template if the
model has one, otherwise the PKU-Alignment raw-prompt fallback for base-model checkpoints.

Judge separately with (needs OPENAI_API_KEY):
    source .venv_evalharness/bin/activate
    cd mtbench && python -m fastchat.llm_judge.gen_judgment --model-list <model_id> --judge-model gpt-4

Usage:
python scripts/generate_mtbench_completions.py \
    --model_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-suite-final/ultrafeedback-tribe/raft-3b-base \
    --model_id raft-3b-base
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

from vllm import LLM, SamplingParams

MTBENCH_DIR = Path(__file__).parent.parent / "mtbench"
QUESTION_FILE = MTBENCH_DIR / "data" / "mt_bench" / "question.jsonl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_id", required=True, help="Recorded as 'model_id'; also the output filename.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    questions = [json.loads(line) for line in QUESTION_FILE.read_text().splitlines() if line.strip()]

    llm = LLM(model=args.model_path, dtype="bfloat16")
    tokenizer = llm.get_tokenizer()
    if tokenizer.chat_template is None:
        chat_template_path = Path(args.model_path) / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer.chat_template = chat_template_path.read_text()
    has_chat_template = tokenizer.chat_template is not None

    def render(messages):
        if has_chat_template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # PKU-Alignment raw-prompt fallback (base models with no chat_template) -- same convention as
        # scripts/offpolicy_trainer.py's _OffPolicyCollator and every other base-model script in this suite.
        text = "BEGINNING OF CONVERSATION: "
        for m in messages:
            if m["role"] == "user":
                text += f"USER: {m['content']} ASSISTANT:"
            else:
                text += m["content"]
        return text

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    # Turn 1
    turn1_prompts = [render([{"role": "user", "content": q["turns"][0]}]) for q in questions]
    turn1_outputs = llm.generate(turn1_prompts, sampling_params)
    turn1_answers = [o.outputs[0].text for o in turn1_outputs]

    # Turn 2 -- built from each question's own turn-1 answer, not a reference.
    turn2_prompts = [
        render(
            [
                {"role": "user", "content": q["turns"][0]},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": q["turns"][1]},
            ]
        )
        for q, a1 in zip(questions, turn1_answers, strict=True)
    ]
    turn2_outputs = llm.generate(turn2_prompts, sampling_params)
    turn2_answers = [o.outputs[0].text for o in turn2_outputs]

    output_dir = MTBENCH_DIR / "data" / "mt_bench" / "model_answer"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.model_id}.jsonl"
    with output_file.open("w") as f:
        for q, a1, a2 in zip(questions, turn1_answers, turn2_answers, strict=True):
            f.write(
                json.dumps(
                    {
                        "question_id": q["question_id"],
                        "answer_id": uuid.uuid4().hex,
                        "model_id": args.model_id,
                        "choices": [{"index": 0, "turns": [a1, a2]}],
                        "tstamp": time.time(),
                    }
                )
                + "\n"
            )
    print(f"Saved {len(questions)} MT-Bench answers to {output_file}")
    os._exit(0)
