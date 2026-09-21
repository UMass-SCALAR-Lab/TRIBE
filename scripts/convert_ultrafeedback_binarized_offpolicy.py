# /// script
# dependencies = [
#     "datasets",
# ]
# ///

"""
Converts HuggingFaceH4/ultrafeedback_binarized's `train_sft`/`test_sft` splits (already deduplicated to one
`chosen` completion per prompt, no group structure) into this suite's flat `(prompt, completion, reward)`
schema, reward=1.0 for every row (OffPolicyRaftTrainer's own `reward == 1.0` filter is then a no-op, kept
only so the same trainer class works unmodified) -- a diagnostic control for
scripts/convert_ultrafeedback_offpolicy.py: isolates whether RAFT underperforming base on that dataset is a
data-selection artifact (our `overall_score`/fine-grained-average best-of-4 pick) or a bug in this suite's
own trainer/collator, by training on the literal dataset already independently verified (outside this repo)
to make plain SFT beat base.

No val split carved out here -- `test_sft` is HuggingFace's own official held-out split for this dataset,
used directly for eval instead.

Usage:
python scripts/convert_ultrafeedback_binarized_offpolicy.py \
    --output_dir /scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/ultrafeedback-binarized
"""

import argparse

from datasets import Dataset, load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    for split, subdir in [("train_sft", "train"), ("test_sft", "test")]:
        dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
        rows = []
        for example in dataset:
            messages = [{"role": "user", "content": example["prompt"]}]
            completion = example["chosen"][-1]["content"]
            rows.append({"prompt": messages, "completion": completion, "reward": 1.0})
        dataset_out = Dataset.from_list(rows)
        out_path = f"{args.output_dir}/{subdir}"
        dataset_out.save_to_disk(out_path)
        print(f"Saved {len(rows)} rows to {out_path}")
