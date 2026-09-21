# /// script
# dependencies = [
#     "datasets",
#     "math-verify",
# ]
# ///

"""
Convert PRM800K's raw GPT-4 rollouts (https://github.com/openai/prm800k, phase2_{train,test}.jsonl) into
this project's off-policy (prompt, completion, reward) schema (matching scripts/generate_offpolicy_math.py's
output), and separately save the held-out test split's unique problems for eval.

PRM800K's own completions end in a "# Answer\\n\\n<answer>" tail, not this project's `\\boxed{}` convention
(every other MATH pipeline here, and eval_math.py's own grading, expects `\\boxed{}`) — that tail is
stripped and replaced with an explicit boxed final line so a model trained on this data produces
completions in the same format it (and every other MATH-trained checkpoint in this project) gets
evaluated in. Reward is computed directly from PRM800K's own `ground_truth_answer` vs.
`pre_generated_answer` fields (both already-extracted final answers) via math_verify, rather than
re-extracting from the (inconsistently-formatted) raw completion text.

No grouping by problem: RAFT (scripts/train_gsm8k_offpolicy_raft.py, despite the filename, is dataset-
agnostic — plain reward==1-only NLL, no group-relative advantage) doesn't need groups, so this is a flat
(prompt, completion, reward) list regardless of PRM800K's per-problem sample count (median 5, mean 9, up
to 470 for some heavily-resampled problems).

Usage:
python scripts/convert_prm800k_offpolicy.py \
    --prm800k_dir /path/to/cloned/openai/prm800k/prm800k \
    --output_dir /scratch4/.../offpolicy-data/prm800k \
    --test_output_file /scratch4/.../offpolicy-data/prm800k_test_problems.json
"""

import argparse
import json
import re
import threading

from datasets import Dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem step by step, then put your final answer in "
    "\\boxed{}."
)

ANSWER_TAIL_RE = re.compile(r"\n*#\s*Answer\s*\n+.*$", re.DOTALL)


def build_prompt(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def build_completion(steps: list[str], final_answer: str) -> str:
    text = "\n\n".join(steps)
    text = ANSWER_TAIL_RE.sub("", text).rstrip()
    return f"{text}\nThe final answer is $\\boxed{{{final_answer}}}$."


def is_correct(gold_parsed, pred_answer: str) -> bool:
    is_main_thread = threading.current_thread() is threading.main_thread()
    parsing_timeout = 10 if is_main_thread else None
    verify_timeout = 5 if is_main_thread else None
    try:
        pred_parsed = parse(
            f"${pred_answer}$",
            extraction_config=[LatexExtractionConfig(normalization_config=NormalizationConfig(units=True))],
            parsing_timeout=parsing_timeout,
        )
        return bool(verify(gold_parsed, pred_parsed, timeout_seconds=verify_timeout))
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prm800k_dir", required=True, help="Path to the cloned repo's prm800k/ (data/*.jsonl inside).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_output_file", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Debug: only process the first N train records.")
    parser.add_argument(
        "--group_size",
        type=int,
        default=None,
        help="If set, lay the dataset out in uniform contiguous group_size blocks per problem (required by "
        "TRIBE/Stage-1-based trainers, unlike RAFT — see scripts/offpolicy_split.py's layout assumption). "
        "PRM800K's per-problem sample count varies (1 to 470, median 5) rather than being fixed like the "
        "generate_offpolicy_*.py scripts' own output, so problems with FEWER than group_size samples are "
        "dropped entirely (can't pad without duplicating rollouts, which would corrupt the trust-region "
        "math) and problems with MORE are randomly downsampled to exactly group_size. None (default, RAFT's "
        "own usage) keeps every row flat, no grouping.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the --group_size downsample.")
    parser.add_argument(
        "--ragged",
        action="store_true",
        help="Keep EVERY rollout for every problem (no downsampling, no dropping) — writes a 'group_id' "
        "column instead of laying groups out at a fixed --group_size, for use with "
        "scripts/offpolicy_trainer_ragged.py (tribe/stage1_ragged.py's variable-group-size Stage 1 solve). "
        "Mutually exclusive with --group_size.",
    )
    parser.add_argument(
        "--balance_negatives",
        action="store_true",
        help="Requires --ragged. PRM800K's raw pos:neg ratio is ~1:10 (vs. ~1:1 for this project's other "
        "self-generated off-policy datasets, GSM8K/MATH, where --num_generations and the base model's "
        "correctness rate happen to land near 50%%) — negative_fraction alone can't tilt a 1:10 set back "
        "toward balanced within its {0,...,1} range. This caps each problem's negatives to at most its own "
        "positive count (random subsample of the negatives if it has more), still keeping every positive, "
        "so the pre-negative_fraction base is ~1:1 like the other datasets and negative_fraction tilts "
        "further from there, same as everywhere else. Problems with zero positives are dropped entirely "
        "(no group-relative signal for Stage 1 regardless of how many negatives they have).",
    )
    args = parser.parse_args()
    if args.ragged and args.group_size is not None:
        parser.error("--ragged and --group_size are mutually exclusive.")
    if args.balance_negatives and not args.ragged:
        parser.error("--balance_negatives requires --ragged.")

    import random

    from collections import defaultdict

    rng = random.Random(args.seed)

    groups = defaultdict(list)
    n_skipped_unparseable_gold = 0
    n_skipped_no_pred = 0
    with open(f"{args.prm800k_dir}/data/phase2_train.jsonl") as f:
        lines = f.readlines()
        if args.limit:
            lines = lines[: args.limit]
        for line in lines:
            d = json.loads(line)
            q = d["question"]
            pred_answer = q.get("pre_generated_answer")
            if pred_answer is None:
                n_skipped_no_pred += 1
                continue
            gold_parsed = parse(q["ground_truth_answer"])
            if len(gold_parsed) == 0:
                n_skipped_unparseable_gold += 1
                continue
            reward = 1.0 if is_correct(gold_parsed, pred_answer) else 0.0
            completion = build_completion(q["pre_generated_steps"], pred_answer)
            row = {"prompt": build_prompt(q["problem"]), "completion": completion, "reward": reward}
            groups[q["problem"]].append(row)

    if args.group_size is not None:
        rows = []
        n_dropped_too_few = 0
        n_all_positive_groups = 0
        for problem, recs in groups.items():
            if len(recs) < args.group_size:
                n_dropped_too_few += 1
                continue
            # Positives are rare (~9% overall) — a plain random draw would often zero out the only correct
            # rollout(s) a problem has. Keep every positive first (up to group_size, since Stage 1 needs
            # groups laid out as exactly group_size), then fill remaining slots with random negatives, so
            # no correct rollout is discarded as long as a problem has <= group_size positives (true for
            # ~all of them at this overall rate).
            positives = [r for r in recs if r["reward"] == 1.0]
            negatives = [r for r in recs if r["reward"] == 0.0]
            if len(positives) >= args.group_size:
                selected = rng.sample(positives, args.group_size)
                n_all_positive_groups += 1
            else:
                n_neg_needed = args.group_size - len(positives)
                selected = positives + rng.sample(negatives, n_neg_needed)
            rows.extend(selected)
        print(
            f"group_size={args.group_size}: kept {len(groups) - n_dropped_too_few}/{len(groups)} problems "
            f"({n_dropped_too_few} had fewer than {args.group_size} samples, dropped; "
            f"{n_all_positive_groups} problems had >= {args.group_size} positives, capped to an all-positive group)"
        )
    elif args.ragged and args.balance_negatives:
        rows = []
        n_dropped_no_positive = 0
        n_capped = 0
        group_id = 0
        for recs in groups.values():
            positives = [r for r in recs if r["reward"] == 1.0]
            negatives = [r for r in recs if r["reward"] == 0.0]
            if len(positives) == 0:
                n_dropped_no_positive += 1
                continue
            if len(negatives) > len(positives):
                negatives = rng.sample(negatives, len(positives))
                n_capped += 1
            for row in positives + negatives:
                row["group_id"] = group_id
                rows.append(row)
            group_id += 1
        print(
            f"--ragged --balance_negatives: kept {len(rows)} rows across {group_id}/{len(groups)} problems "
            f"({n_dropped_no_positive} had zero positives, dropped; {n_capped} had more negatives than "
            "positives, capped)"
        )
    elif args.ragged:
        rows = []
        for group_id, recs in enumerate(groups.values()):
            for row in recs:
                row["group_id"] = group_id
                rows.append(row)
        print(f"--ragged: kept all {len(rows)} rows across {len(groups)} problems (no downsampling)")
    else:
        rows = [row for recs in groups.values() for row in recs]

    dataset_out = Dataset.from_list(rows)
    dataset_out.save_to_disk(args.output_dir)
    mean_reward = sum(r["reward"] for r in rows) / len(rows)
    print(f"Saved {len(rows)} (prompt, completion, reward) rows to {args.output_dir}")
    print(f"Mean reward: {mean_reward:.4f}")
    print(f"Skipped (no pre_generated_answer): {n_skipped_no_pred}, (unparseable gold): {n_skipped_unparseable_gold}")

    # Held-out test split: unique problems only (dedupe across possibly-multiple rollouts per problem).
    seen = {}
    with open(f"{args.prm800k_dir}/data/phase2_test.jsonl") as f:
        for line in f:
            d = json.loads(line)
            q = d["question"]
            seen[q["problem"]] = q["ground_truth_answer"]
    test_problems = [{"problem": p, "answer": a} for p, a in seen.items()]
    with open(args.test_output_file, "w") as f:
        json.dump(test_problems, f)
    print(f"Saved {len(test_problems)} unique test problems to {args.test_output_file}")


if __name__ == "__main__":
    main()
