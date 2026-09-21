# /// script
# dependencies = [
#     "transformers",
#     "torch",
# ]
# ///

"""
Score a scripts/generate_ultrafeedback_eval_completions.py output file with Skywork-Reward-V2-Llama-3.1-8B
(https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B) -- a second, independent judge for the same
"did off-policy TRIBE/RAFT training on UltraFeedback actually improve response quality" question
scripts/score_ultrarm.py answers with UltraRM-13b. UltraRM was itself trained on UltraFeedback (2023, mostly
markdown-light source models) and the base-vs-trained reward gap correlated strongly with markdown/bold
formatting (see ultrafeedback_eval/ writeup); Skywork-Reward-V2 is a much more recent (2025), independently
curated 26M-pair preference model near the top of RewardBench, used here to check whether that gap is
UltraRM-specific or holds under a differently-trained judge too.

Standard AutoModelForSequenceClassification reward-model interface (unlike UltraRM's custom architecture) --
see the model card's own usage example. No system prompt in the chat template per the card's guidance.

Usage:
python scripts/score_skyrm.py \
    --completions_file ultrafeedback_eval/base_completions.json \
    --output_file ultrafeedback_eval/base_skyrm_scored.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=4096)
    args = parser.parse_args()

    data = json.loads(Path(args.completions_file).read_text())
    rows = data["rows"]

    print("loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("loading model...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, num_labels=1
    )
    model = model.to("cuda").eval()
    print("ready, scoring...", flush=True)

    scores = []
    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]
            texts = []
            for row in batch:
                prompt_messages = row["prompt"] if isinstance(row["prompt"], list) else [
                    {"role": "user", "content": row["prompt"]}
                ]
                conv = prompt_messages + [{"role": "assistant", "content": row["completion"]}]
                formatted = tokenizer.apply_chat_template(conv, tokenize=False)
                if tokenizer.bos_token and formatted.startswith(tokenizer.bos_token):
                    formatted = formatted[len(tokenizer.bos_token) :]
                texts.append(formatted)
            encoded = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length
            ).to("cuda")
            batch_scores = model(**encoded).logits[:, 0]
            scores.extend(batch_scores.float().cpu().tolist())

    result = {
        "name": data["name"],
        "model_path": data["model_path"],
        "num_prompts": len(rows),
        "mean_reward": sum(scores) / len(scores),
        "per_example_reward": scores,
    }
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_example_reward"}, indent=2))
