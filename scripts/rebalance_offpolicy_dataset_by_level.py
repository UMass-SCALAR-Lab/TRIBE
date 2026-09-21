# /// script
# dependencies = [
#     "datasets",
# ]
# ///

"""
Rebalance an off-policy MATH (prompt, completion, reward) dataset (scripts/generate_offpolicy_math.py's
output) by MATH difficulty LEVEL, to counteract the training-composition skew RAFT-style positive-only
filtering introduces (verified directly on both Qwen and Llama: easy-level questions solve correctly far
more often across a fixed per-question completion budget, so their correct completions dominate a
positive-only training set even though the raw question distribution isn't itself skewed that way — see
this session's RAFT level-skew finding, e.g. Level 1 solve rate ~76% vs Level 5's ~34% on Llama-MATH).

Operates at the GROUP level (each question's whole group_size-completion block, positive AND negative
completions together) — not just the filtered positives — so the rebalanced dataset is usable by every
off-policy trainer in this suite (RAFT, TRIBE, GRPO, TOPR, DPO/SimPO's pair-builder, ...), not only
RAFT's own positive-only filter.

Weighting: each MATH train QUESTION's level is recovered by re-pairing group index with the ORIGINAL
DigitalLearningGmbH/MATH-lighteval train split in order (generate_offpolicy_math.py preserves source
order 1:1, one contiguous group_size block per kept question — verified via this script's own assertion
that group count matches train-split length, which also catches an unparseable-solution-skip mismatch).
Per-level SOLVE RATE (mean reward) is computed directly from the input dataset's own labels, and each
level's groups are duplicated/downsampled with weight proportional to 1/solve_rate, normalized so the
output has approximately --target_total_groups groups — this targets roughly EQUAL representation of
CORRECT (reward>0) completions across levels after any downstream positive-only filtering, the precise
mechanism the skew comes from, while leaving every group's own completions (positive and negative) intact
so non-filtering methods keep the full original signal, just reweighted by how often each level's group
appears.

Held-out val groups (the SAME last val_fraction of groups every off-policy driver script's own
OffPolicyConfig.val_fraction / scripts/offpolicy_split.py hold out) are EXCLUDED from consideration before
weighting — rebalancing must never touch validation/eval questions, only the train-eligible portion.

Usage:
python scripts/rebalance_offpolicy_dataset_by_level.py \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/math-llama-boxed \
    --output_dir /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/math-llama-boxed-levelbalanced \
    --group_size 32 --val_fraction 0.1
"""

import argparse
import random
from collections import Counter
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--group_size", type=int, required=True)
    parser.add_argument(
        "--val_fraction", type=float, default=0.1,
        help="Same held-out fraction every off-policy driver script uses — excluded from rebalancing "
        "entirely, matching scripts/offpolicy_split.py's own last-val_fraction-of-groups convention.",
    )
    parser.add_argument(
        "--target_total_groups", type=int, default=None,
        help="Total groups (question blocks, with duplication) in the output. Defaults to the number of "
        "train-eligible groups (dataset's own group count minus the held-out val groups) — i.e. same "
        "overall size as the original train-eligible set, just reweighted.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min_level_groups", type=int, default=50,
        help="Levels with fewer than this many original train-eligible groups are EXCLUDED from the "
        "reweighting (and from the output) rather than reweighted like the rest — giving a tiny bucket "
        "'equal representation' alongside real difficulty levels forces massive duplication of a handful "
        "of questions (e.g. MATH-lighteval's malformed/missing-level 'Level ?' bucket has only 2 groups; "
        "reweighting it the same way as Level 1-5 tried to give it ~1100 output groups built from "
        "duplicating those same 2 questions ~560x each — a data-quality artifact, not a real difficulty "
        "tier worth preserving at all, let alone amplifying).",
    )
    args = parser.parse_args()

    assert Path(args.output_dir).resolve() != Path(args.dataset_path).resolve(), (
        "--output_dir must not be the same path as --dataset_path — this would overwrite the source dataset."
    )

    dataset = load_from_disk(args.dataset_path)
    assert len(dataset) % args.group_size == 0, (
        f"Dataset length {len(dataset)} isn't a multiple of group_size {args.group_size} — wrong group "
        "size, or the dataset isn't in the contiguous per-question blocks the generation scripts produce."
    )
    num_groups = len(dataset) // args.group_size
    num_val_groups = round(args.val_fraction * num_groups)
    num_train_groups = num_groups - num_val_groups

    train_split = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")
    assert len(train_split) == num_groups, (
        f"MATH train split has {len(train_split)} questions but the dataset has {num_groups} groups — "
        "generate_offpolicy_math.py's own order/skip assumption this script relies on doesn't hold here "
        "(e.g. it was run with --limit, or some gold solutions were unparseable and skipped)."
    )

    rewards = dataset["reward"]
    levels = [train_split[g]["level"] for g in range(num_train_groups)]  # train-eligible groups only
    level_group_count = Counter(levels)

    excluded_levels = {lvl for lvl, n in level_group_count.items() if n < args.min_level_groups}
    if excluded_levels:
        print(f"Excluding levels with < {args.min_level_groups} original groups (data-quality artifacts, "
              f"not real difficulty tiers — dropped from the output entirely, not reweighted):")
        for lvl in sorted(excluded_levels):
            print(f"  {lvl}: {level_group_count[lvl]} groups excluded")

    level_solve_sum = Counter()
    level_solve_count = Counter()
    for g in range(num_train_groups):
        if levels[g] in excluded_levels:
            continue
        group_rewards = rewards[g * args.group_size : (g + 1) * args.group_size]
        level_solve_sum[levels[g]] += sum(group_rewards)
        level_solve_count[levels[g]] += len(group_rewards)
    solve_rate = {lvl: level_solve_sum[lvl] / level_solve_count[lvl] for lvl in level_solve_count}
    print("\nPer-level solve rate (train-eligible groups only, excluded levels omitted):")
    for lvl in sorted(solve_rate):
        print(f"  {lvl}: {solve_rate[lvl]:.4f}")

    # Number of groups taken PER LEVEL is proportional to 1/solve_rate ALONE (deliberately NOT scaled by
    # that level's original group count) — this is what actually makes expected positive-completion
    # representation equal across levels regardless of how many original questions each level had:
    # expected_positive_level = n_take_level * group_size * solve_rate_level
    #                          = (inv_solve[lvl] * scale) * group_size * solve_rate_level
    #                          = scale * group_size   (solve_rate cancels — constant across levels)
    # An earlier version of this script scaled n_take by len(level_group_idxs[lvl]) too, which made the
    # result proportional to each level's ORIGINAL group count instead — verified via unit test to
    # silently reproduce close to the original (unbalanced) distribution when level group counts are
    # unequal (as MATH's are: Level 1 has 499 train-eligible groups vs Level 5's 2126) — a synthetic test
    # with EQUAL group counts per level didn't catch this, since the bug is invisible when counts match.
    target_total = args.target_total_groups or num_train_groups
    level_group_idxs = {lvl: [g for g in range(num_train_groups) if levels[g] == lvl] for lvl in solve_rate}
    inv_solve = {lvl: 1.0 / solve_rate[lvl] for lvl in solve_rate if solve_rate[lvl] > 0}
    raw_weight_total = sum(inv_solve.values())
    scale = target_total / raw_weight_total

    rng = random.Random(args.seed)
    selected_group_idxs = []
    print("\nRebalancing (target ~{} groups total):".format(target_total))
    for lvl in sorted(inv_solve):
        n_take = round(inv_solve[lvl] * scale)
        chosen = [rng.choice(level_group_idxs[lvl]) for _ in range(n_take)]  # sample WITH replacement
        selected_group_idxs.extend(chosen)
        orig_n = len(level_group_idxs[lvl])
        print(f"  {lvl}: {orig_n} original groups -> {n_take} groups in output "
              f"({n_take / orig_n:.2f}x, {'oversampled' if n_take > orig_n else 'downsampled'})")

    rng.shuffle(selected_group_idxs)  # avoid any residual level-blocking in the output's own order
    pieces = [dataset.select(range(g * args.group_size, (g + 1) * args.group_size)) for g in selected_group_idxs]
    rebalanced = concatenate_datasets(pieces)
    rebalanced.save_to_disk(args.output_dir)

    new_level_solve_sum = Counter()
    new_level_solve_count = Counter()
    new_rewards = rebalanced["reward"]
    for i, g in enumerate(selected_group_idxs):
        lvl = levels[g]
        chunk = new_rewards[i * args.group_size : (i + 1) * args.group_size]
        new_level_solve_sum[lvl] += sum(1 for r in chunk if r > 0)
        new_level_solve_count[lvl] += len(chunk)
    total_pos = sum(new_level_solve_sum.values())
    print(f"\nSaved {len(rebalanced)} rows ({len(selected_group_idxs)} groups) to {args.output_dir}")
    print("Resulting share of POSITIVE (correct) completions by level (the thing this script targets "
          "making roughly equal across levels):")
    for lvl in sorted(new_level_solve_sum):
        print(f"  {lvl}: {new_level_solve_sum[lvl] / total_pos:.4f}")
