"""
Prints a comparison table from scripts/eval_gsm8k.py's per-checkpoint JSON outputs. Plain stdlib, no
vllm/trl/torch needed — run this in any env once the eval jobs (interactive or SLURM) have finished.

Usage: python scripts/summarize_gsm8k_eval.py [gsm8k_eval]
"""

import json
import sys
from pathlib import Path


NAMES = ["base", "tribe", "grpo", "dapo", "rloo", "raft", "onlinedpo"]

if __name__ == "__main__":
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "gsm8k_eval")
    for name in NAMES:
        result_file = out_dir / f"{name}.json"
        if not result_file.exists():
            print(f"{name:12s}  (missing)")
            continue
        result = json.loads(result_file.read_text())
        print(f"{name:12s}  {result['accuracy']:.4f}  ({result['num_correct']}/{result['num_total']})")
