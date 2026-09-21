# /// script
# dependencies = [
#     "datasets",
# ]
# ///

"""
Build a DPO/SimPO preference dataset out of scripts/convert_ultrafeedback_offpolicy.py's output --
scripts/build_offpolicy_dpo_pairs.py's own pairing rule hardcodes a binary reward==1.0/0.0 split (GSM8K/
MATH's correctness reward), which UltraFeedback's continuous fine-grained-average reward never satisfies
(see convert_ultrafeedback_offpolicy.py's docstring) and would silently produce zero pairs. Pairing rule
here instead: within each group_size=4 block, chosen = highest-reward completion, rejected = lowest-reward
completion -- one pair per group (matches HuggingFaceH4/ultrafeedback_binarized's own chosen/rejected
construction, the established convention for this dataset).

The last --val_fraction of groups is held out BEFORE pairing (scripts/offpolicy_split.py) -- same held-out
prompts every other off-policy UltraFeedback driver script in this suite excludes from training.

Output columns (`prompt`, `chosen`, `rejected`) match trl.DPOTrainer's conversational format, same as
scripts/build_offpolicy_dpo_pairs.py.

Usage:
python scripts/build_ultrafeedback_dpo_pairs.py \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/ultrafeedback-v2 \
    --output_dir /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/ultrafeedback-v2-dpo-pairs \
    --group_size 4
"""

import argparse

from datasets import Dataset, load_from_disk

from offpolicy_split import split_off_policy_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)
    assert len(dataset) % args.group_size == 0, (
        f"Dataset length {len(dataset)} isn't a multiple of --group_size {args.group_size}."
    )
    dataset = split_off_policy_dataset(dataset, args.group_size, args.val_fraction)

    rows = []
    for start in range(0, len(dataset), args.group_size):
        group = dataset[start : start + args.group_size]
        prompt = group["prompt"][0]
        best_i = max(range(args.group_size), key=lambda i: group["reward"][i])
        worst_i = min(range(args.group_size), key=lambda i: group["reward"][i])
        if best_i == worst_i:
            continue
        rows.append(
            {
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": group["completion"][best_i]}],
                "rejected": [{"role": "assistant", "content": group["completion"][worst_i]}],
            }
        )

    dataset_out = Dataset.from_list(rows)
    dataset_out.save_to_disk(args.output_dir)
    print(f"Saved {len(rows)} (prompt, chosen, rejected) pairs to {args.output_dir}")
