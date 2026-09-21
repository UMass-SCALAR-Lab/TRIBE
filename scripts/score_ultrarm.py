# /// script
# dependencies = [
#     "transformers",
#     "torch",
# ]
# ///

"""
Score a scripts/generate_ultrafeedback_eval_completions.py output file with UltraRM-13b
(https://huggingface.co/openbmb/UltraRM-13b) — the reward model released alongside UltraFeedback itself
(trained on UltraFeedback + a mix of HH-RLHF/SHP/Summarization), used here as the judge for "did off-policy
TRIBE training on UltraFeedback actually improve response quality" (run this on both a base-model and a
trained-checkpoint completions file, then diff the mean rewards / compute win-rate).

UltraRM-13b ships a custom architecture (LlamaModel backbone + a linear regression head reading the last
non-padding token's hidden state) with no AutoModel mapping, so the class is ported directly here — copied
from the model card's own README usage example (https://huggingface.co/openbmb/UltraRM-13b#usage), NOT
from the repo's own checked-in modeling_llama_rm.py, which has two bugs the README's own inline version
already fixes (`self.model.model(...)` double-nesting that doesn't match `self.model = LlamaModel(...)`,
and `return reward_models` referencing an undefined name).

Plain transformers, no vLLM — runs in the same llm_gen/llm_gen_new env used for generation, or the tribe
env, anything with transformers+torch. 13B params in bf16 (~26GB) fits comfortably on a single A100,
scoring-only (no KV cache/generation needed).

Usage:
python scripts/score_ultrarm.py \
    --completions_file ultrafeedback_eval/base_completions.json \
    --output_file ultrafeedback_eval/base_scored.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, PreTrainedModel


class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig
    # This class predates transformers' newer meta-device loading path, which expects
    # all_tied_weights_keys (a dict-like registry PreTrainedModel subclasses are normally expected to
    # populate) during _finalize_model_loading -- this model has no tied weights (no LM head sharing the
    # embedding matrix), so an empty dict is the correct value, not a workaround masking a real one.
    all_tied_weights_keys = {}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None):
        hidden_states = self.model(input_ids, attention_mask=attention_mask)[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)
        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1, 1)
        return torch.gather(rewards, 1, ends)


ULTRARM_TEMPLATE = "Human: {prompt}\n\nAssistant: {completion}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    data = json.loads(Path(args.completions_file).read_text())
    rows = data["rows"]

    print("loading tokenizer...", flush=True)
    tokenizer = LlamaTokenizer.from_pretrained("openbmb/UltraRM-13b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("loading model...", flush=True)
    # low_cpu_mem_usage=True: without it, from_pretrained fully randomly-initializes all 13B params (slow,
    # CPU-bound nn.Linear/nn.Embedding init) before overwriting them with the checkpoint's real weights --
    # skips straight to loading real values via the meta-device path instead.
    model = LlamaRewardModel.from_pretrained(
        "openbmb/UltraRM-13b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    print("moving model to cuda...", flush=True)
    model = model.to("cuda").eval()
    print("ready, scoring...", flush=True)

    scores = []
    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]
            texts = []
            for row in batch:
                prompt_text = row["prompt"] if isinstance(row["prompt"], str) else row["prompt"][-1]["content"]
                texts.append(ULTRARM_TEMPLATE.format(prompt=prompt_text, completion=row["completion"]))
            encoded = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length
            ).to("cuda")
            batch_scores = model(encoded["input_ids"], attention_mask=encoded["attention_mask"])
            scores.extend(batch_scores.squeeze(-1).float().cpu().tolist())

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
