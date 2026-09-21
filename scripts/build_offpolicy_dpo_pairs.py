# /// script
# dependencies = [
#     "datasets",
# ]
# ///

"""
Build a DPO preference dataset out of an existing off-policy (prompt, completion, reward) dataset
(scripts/generate_offpolicy_gsm8k.py / scripts/generate_offpolicy_math.py's output) — TOPR's own DPO
baseline construction (https://huggingface.co/papers/2503.14286, Section 4.3): "pairs of candidate
solutions are formed from these so as to obtain up to 16 contrastive pairs" (n=16 for GSM8K). The paper
gives no more detail than that one sentence — no stated pairing rule, no hyperparameters — so this
reconstructs the natural reading of it: for each question's group of completions, randomly pair a
correct (reward=1) completion with an incorrect (reward=0) one, one completion used at most once per
pair, capped at `--max_pairs_per_question` (defaults to `--group_size`, generalizing their "up to
n_completions pairs" rule to MATH's n=32 the same way).

Groups with no correct completion or no incorrect completion (all-right or all-wrong for that question)
produce zero pairs — there is no valid contrastive pair to form — and are skipped, not padded/faked.

The last `--val_fraction` of questions are held out BEFORE pairing (scripts/offpolicy_split.py) — same
split every other off-policy driver script applies to the raw dataset, so DPO/SimPO hold out exactly the
same questions as every other method for hyperparameter-selection validation, never the actual test set.

Output columns (`prompt`, `chosen`, `rejected`) match trl.DPOTrainer's conversational format directly:
`prompt` is the same structured message list carried over unchanged from the input dataset; `chosen`/
`rejected` are each wrapped as a single-message assistant turn.

Pure dataset manipulation (no model, no vLLM) — runs in any env with `datasets` installed, no need for the
`llm_gen` env this suite's generation scripts require.

Usage:
python scripts/build_offpolicy_dpo_pairs.py \
    --dataset_path /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/gsm8k \
    --output_dir /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/gsm8k-dpo-pairs \
    --group_size 16
"""

import argparse
import random

from datasets import Dataset, load_from_disk

from offpolicy_split import split_off_policy_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--group_size", type=int, default=16, help="Completions per question (16 for GSM8K, 32 for MATH).")
    parser.add_argument(
        "--max_pairs_per_question",
        type=int,
        default=None,
        help="Defaults to --group_size, generalizing TOPR's '16 completions -> up to 16 pairs' rule.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of questions held out (from the end) before pairing. See scripts/offpolicy_split.py.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    max_pairs = args.max_pairs_per_question or args.group_size

    rng = random.Random(args.seed)
    dataset = load_from_disk(args.dataset_path)
    assert len(dataset) % args.group_size == 0, (
        f"Dataset length {len(dataset)} isn't a multiple of --group_size {args.group_size} — wrong group "
        "size, or the dataset isn't in the contiguous per-question blocks the generation scripts produce."
    )
    dataset = split_off_policy_dataset(dataset, args.group_size, args.val_fraction)

    rows = []
    num_groups_skipped = 0
    for start in range(0, len(dataset), args.group_size):
        group = dataset[start : start + args.group_size]
        prompt = group["prompt"][0]
        correct = [c for c, r in zip(group["completion"], group["reward"]) if r == 1.0]
        incorrect = [c for c, r in zip(group["completion"], group["reward"]) if r == 0.0]
        if not correct or not incorrect:
            num_groups_skipped += 1
            continue
        rng.shuffle(correct)
        rng.shuffle(incorrect)
        for chosen, rejected in zip(correct[:max_pairs], incorrect[:max_pairs]):
            rows.append(
                {
                    "prompt": prompt,
                    "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": rejected}],
                }
            )

    print(f"Skipped {num_groups_skipped}/{len(dataset) // args.group_size} groups with no valid contrastive pair")
    dataset_out = Dataset.from_list(rows)
    dataset_out.save_to_disk(args.output_dir)
    print(f"Saved {len(rows)} (prompt, chosen, rejected) pairs to {args.output_dir}")
