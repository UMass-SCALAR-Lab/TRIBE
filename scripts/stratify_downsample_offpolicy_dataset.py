# /// script
# dependencies = [
#     "datasets",
# ]
# ///

"""
Downsample an off-policy (prompt, completion, reward) dataset (scripts/generate_offpolicy_gsm8k.py /
scripts/generate_offpolicy_math.py's output) from its original group_size to a smaller one, PER QUESTION
stratified by reward — each question's own positive:negative ratio in the downsampled group matches its
ratio in the original group_size-sized group as closely as integer rounding allows, rather than just
matching the ratio in aggregate across the whole dataset. Preserves the same "one contiguous block per
question" layout scripts/offpolicy_split.py and every off-policy trainer depend on, just with fewer rows
per block.

Rounding: target_pos = round(pos_count * new_group_size / group_size), clamped to the number actually
available; target_neg = new_group_size - target_pos, backfilled from the other stratum if that count isn't
actually available (guaranteed possible since pos_count + neg_count == group_size >= new_group_size).
"reward > 0" is treated as positive — matches every off-policy trainer's own weight<0 / weight>=0 or
signed_reward<0 / >=0 convention (this repo's rewards are always 0.0/1.0, so this is just reward==1.0 in
practice, but written as > 0 to not assume a strictly binary reward if that ever changes).

Pure dataset manipulation (no model, no vLLM) — runs in any env with `datasets` installed, no need for the
`llm_gen` env the generation scripts require.

Usage:
python scripts/stratify_downsample_offpolicy_dataset.py \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/math-llama-boxed \
    --output_dir /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/math-llama-boxed-n16 \
    --group_size 32 \
    --new_group_size 16
"""

import argparse
import random
from pathlib import Path

from datasets import Dataset, load_from_disk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--group_size", type=int, required=True, help="Original completions per question.")
    parser.add_argument("--new_group_size", type=int, required=True, help="Target completions per question.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert 0 < args.new_group_size <= args.group_size, "new_group_size must be in (0, group_size]"
    # save_to_disk on the same path as dataset_path would corrupt/lose the source dataset it's still
    # reading from (load_from_disk keeps the arrow files open) — refuse outright rather than let a typo'd
    # --output_dir destroy the original.
    assert Path(args.output_dir).resolve() != Path(args.dataset_path).resolve(), (
        "--output_dir must not be the same path as --dataset_path — this would overwrite the source dataset."
    )

    dataset = load_from_disk(args.dataset_path)
    assert len(dataset) % args.group_size == 0, (
        f"Dataset length {len(dataset)} isn't a multiple of group_size {args.group_size} — wrong group "
        "size, or the dataset isn't in the contiguous per-question blocks the generation scripts produce."
    )
    num_groups = len(dataset) // args.group_size
    rewards = dataset["reward"]

    rng = random.Random(args.seed)
    keep_indices = []
    for g in range(num_groups):
        start = g * args.group_size
        group_indices = list(range(start, start + args.group_size))
        pos_indices = [i for i in group_indices if rewards[i] > 0]
        neg_indices = [i for i in group_indices if rewards[i] <= 0]

        target_pos = round(len(pos_indices) * args.new_group_size / args.group_size)
        target_pos = min(target_pos, len(pos_indices))
        target_neg = args.new_group_size - target_pos
        if target_neg > len(neg_indices):
            # Not enough negatives to hit the target — take all of them, backfill the rest from positives.
            target_neg = len(neg_indices)
            target_pos = min(args.new_group_size - target_neg, len(pos_indices))

        chosen_pos = rng.sample(pos_indices, target_pos)
        chosen_neg = rng.sample(neg_indices, target_neg)
        chosen = sorted(chosen_pos + chosen_neg)
        assert len(chosen) == args.new_group_size, (
            f"Group {g} produced {len(chosen)} rows, expected {args.new_group_size} — "
            f"pos={len(pos_indices)}, neg={len(neg_indices)}, group_size={args.group_size}"
        )
        keep_indices.extend(chosen)

    downsampled = dataset.select(keep_indices)
    downsampled.save_to_disk(args.output_dir)

    orig_mean_reward = sum(r > 0 for r in rewards) / len(rewards)
    new_mean_reward = sum(r > 0 for r in downsampled["reward"]) / len(downsampled)
    print(f"Downsampled {len(dataset)} rows ({num_groups} groups of {args.group_size}) to "
          f"{len(downsampled)} rows ({num_groups} groups of {args.new_group_size})")
    print(f"Mean reward: {orig_mean_reward:.4f} (original) vs {new_mean_reward:.4f} (downsampled)")
