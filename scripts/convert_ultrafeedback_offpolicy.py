# /// script
# dependencies = [
#     "datasets",
# ]
# ///

"""
Off-policy data CONVERSION (not generation) for UltraFeedback (https://huggingface.co/papers/2310.01377,
`openbmb/UltraFeedback`): unlike scripts/generate_offpolicy_{gsm8k,math}.py, which sample completions from
an untrained base model, UltraFeedback already ships 4 completions per prompt (from 4 different models,
e.g. alpaca-7b, gpt-3.5-turbo, ...) each scored by GPT-4 (`overall_score`, roughly 1-10). This script just
reshapes those into the same flat `(prompt, completion, reward)` schema every off-policy trainer in this
suite expects (see generate_offpolicy_gsm8k.py's own docstring), laid out as a contiguous group_size=4
block per prompt (required by scripts/offpolicy_split.py) — no model, no vLLM, no generation step.

`reward` is the average of the completion's 4 fine-grained per-aspect ratings (helpfulness, honesty,
instruction_following, truthfulness; each 1-5), NOT `completion["overall_score"]`. `overall_score` is a
separate holistic GPT-4 pass that disagrees with the fine-grained-average argmax on ~45% of prompts (5.98
mean vs 3.86 mean on their native scales, corr=0.758) — the same issue documented by the HuggingFace H4 team
when they built `ultrafeedback_binarized` from this same raw dataset, which is why that dataset also scores
via the fine-grained average rather than `overall_score`. `--reward_scale` (default 5.0, the fine-grained
scale's max) matches this new default; every other off-policy dataset in this suite uses a 0/1 correctness
reward, and TRIBE's Stage 1 trust region (tribe/offpolicy_stage1.py) computes the group-centered advantage
`A = reward - group_mean(reward)` directly from this value with no other normalization, so keeping the
reward roughly 0-1 scaled matters for `--trust_region_eps`/`--beta` to interact the way they were tuned for
on GSM8K/MATH.

`completions` within a prompt come in the order UltraFeedback stores them (by `model` field, not
shuffled) — flag `--shuffle_within_group` if that induces some order-dependent artifact.

Usage:
python scripts/convert_ultrafeedback_offpolicy.py \
    --output_dir /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/ultrafeedback

Debug mode: add `--limit 8` to convert only the first 8 prompts.
"""

import argparse

from datasets import Dataset, load_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--reward_scale", type=float, default=5.0, help="reward = fine_grained_avg_score / reward_scale."
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug mode: only the first N prompts.")
    args = parser.parse_args()

    split = "train" if args.limit is None else f"train[:{args.limit}]"
    dataset = load_dataset("openbmb/UltraFeedback", split=split)

    ASPECTS = ["helpfulness", "honesty", "instruction_following", "truthfulness"]

    rows = []
    num_skipped = 0
    for example in dataset:
        if example["source"] == "truthful_qa":
            # HuggingFace's ultrafeedback_binarized also drops this subset (811/63967 prompts, 1.3%): its
            # annotations reward models for restating well-known misconceptions as true, making its scores
            # unreliable independent of the overall_score-vs-fine-grained-average issue above.
            num_skipped += 1
            continue
        completions = example["completions"]
        if len(completions) != 4:
            # A handful of UltraFeedback rows have fewer than 4 completions (a generation from one of the
            # 4 source models failed upstream) — skip rather than break the fixed group_size=4 layout every
            # downstream off-policy trainer assumes.
            num_skipped += 1
            continue
        messages = [{"role": "user", "content": example["instruction"]}]
        for completion in completions:
            ratings = []
            for aspect in ASPECTS:
                annotation = completion["annotations"].get(aspect)
                if annotation is None or annotation.get("Rating") in (None, "N/A"):
                    continue
                ratings.append(float(annotation["Rating"]))
            reward = (sum(ratings) / len(ratings)) / args.reward_scale
            rows.append({"prompt": messages, "completion": completion["response"], "reward": reward})

    if num_skipped:
        print(f"Skipped {num_skipped}/{len(dataset)} prompts (truthful_qa source, or != 4 completions)")

    dataset_out = Dataset.from_list(rows)
    dataset_out.save_to_disk(args.output_dir)
    print(f"Saved {len(rows)} (prompt, completion, reward) rows to {args.output_dir}")
    print(f"Mean reward: {sum(r['reward'] for r in rows) / len(rows):.4f}")
