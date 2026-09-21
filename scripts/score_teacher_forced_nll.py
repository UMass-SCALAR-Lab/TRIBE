# /// script
# dependencies = [
#     "transformers",
#     "torch",
# ]
# ///

"""
Teacher-forced NLL of a fixed set of (prompt, completion) pairs under a given model -- no generation, no
sampling. Used to check whether RAFT/SFT training actually moved the model's own likelihood toward its
training targets (scripts/convert_ultrafeedback_binarized_offpolicy.py's `chosen` completions): if the
trained checkpoint's NLL on those targets isn't clearly below the base model's, training didn't do what
naive gradient-descent-on-that-data intuition suggests.

Reuses _OffPolicyCollator and OffPolicyTrainer._get_per_token_logps directly (same tokenization -- chat
template + EOS-append convention -- and the same left-padding position_ids fix) so this is exactly the
quantity RAFT's own compute_loss would compute per example, just without gradients.

Usage:
python scripts/score_teacher_forced_nll.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --completions_file ultrafeedback_eval/ufb_chosen_completions.json \
    --output_file ultrafeedback_eval/base_nll_on_ufb_chosen.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from offpolicy_trainer import _OffPolicyCollator  # noqa: E402


def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, temperature):
    from trl.trainer.utils import selective_log_softmax

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.clamp_(min=0)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
        )
    logits = outputs.logits[:, :-1, :]
    logits = logits[:, -logits_to_keep:, :]
    logits = logits / temperature
    completion_ids = input_ids[:, -logits_to_keep:]
    return selective_log_softmax(logits, completion_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--completions_file", required=True, help="Provides (prompt, completion) pairs; 'completion' is teacher-forced, not generated.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    data = json.loads(Path(args.completions_file).read_text())
    rows = data["rows"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16).to("cuda").eval()

    collator = _OffPolicyCollator(tokenizer, max_completion_length=args.max_completion_length, max_prompt_length=None)

    examples = [{"prompt": r["prompt"], "completion": r["completion"], "reward": 0.0} for r in rows]

    seq_nlls = []
    for start in range(0, len(examples), args.batch_size):
        batch = collator(examples[start : start + args.batch_size])
        input_ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1).to("cuda")
        attention_mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1).to("cuda")
        completion_mask = batch["completion_mask"].to("cuda")
        logits_to_keep = batch["completion_ids"].size(1)

        per_token_logps = get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, temperature=1.0)
        seq_logps = (per_token_logps * completion_mask).sum(-1)
        token_counts = completion_mask.sum(-1).clamp(min=1)
        per_token_nll = (-seq_logps / token_counts).float().cpu().tolist()
        seq_nlls.extend(per_token_nll)

    result = {
        "name": data["name"],
        "model_path": args.model_path,
        "num_prompts": len(rows),
        "mean_nll": sum(seq_nlls) / len(seq_nlls),
        "per_example_nll": seq_nlls,
    }
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_example_nll"}, indent=2))
