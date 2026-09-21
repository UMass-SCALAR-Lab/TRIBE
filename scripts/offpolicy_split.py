"""
Shared, POST-HOC train/validation split for the off-policy suite — never rewrites anything on disk, just
filters an already-loaded `datasets.Dataset` in memory. Held out is a contiguous SEGMENT at the END of the
dataset (not a random scatter): the pre-generated datasets are laid out as one contiguous group_size block
per question, in the same order scripts/generate_offpolicy_{gsm8k,math}.py iterated the source HF dataset,
so "last val_fraction of groups" is a fixed, reusable set of held-out QUESTIONS (every one of that
question's group_size completions goes with it — never split a question's own completions across train and
validation) that lines up with the same suffix of the source dataset's question order.

Used by every off-policy driver script (via OffPolicyConfig.val_fraction) and by
scripts/build_offpolicy_dpo_pairs.py, so every method — including DPO/SimPO's paired data, built from this
same raw dataset — holds out exactly the same questions. Never touch the actual GSM8K/MATH TEST split for
hyperparameter selection; this validation segment (carved from the TRAINING questions) is what sweeps
(SimPO's beta/gamma, TRIBE's trust_region_eps, etc.) should be scored against instead — see
scripts/eval_gsm8k.py's --val_fraction flag for the matching eval-side split.
"""

from datasets import Dataset


def split_off_policy_dataset(dataset: Dataset, group_size: int, val_fraction: float) -> Dataset:
    """Returns the TRAIN portion only — the last val_fraction of question-groups are held out and dropped."""
    assert len(dataset) % group_size == 0, (
        f"Dataset length {len(dataset)} isn't a multiple of group_size {group_size} — wrong group size, or "
        "the dataset isn't in the contiguous per-question blocks the generation scripts produce."
    )
    num_groups = len(dataset) // group_size
    num_val_groups = round(val_fraction * num_groups)
    split_at = (num_groups - num_val_groups) * group_size
    return dataset.select(range(split_at))
