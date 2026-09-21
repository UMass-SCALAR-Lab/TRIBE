#!/bin/bash
# Evaluates the base model + all 6 finished GSM8K checkpoints (final save, not intermediate checkpoint-N
# subdirs) on the GSM8K test set via scripts/eval_gsm8k.py, one process per checkpoint (see that script's
# docstring for why: vLLM doesn't reliably free GPU memory between successive LLM() instances in one
# process). Prints a final comparison table once all 7 finish.
#
# Requires a GPU and the `llm_gen` conda env (has vllm; `tribe` doesn't — see eval_gsm8k.py's docstring).
# Runs all 7 sequentially in one process/GPU allocation; for the SLURM equivalent that runs them in
# parallel across 7 GPUs instead, see slurm/submit_eval_gsm8k.sh.
#
# Usage: conda activate llm_gen && bash scripts/eval_gsm8k_all.sh [--debug]
#   --debug: adds --limit 8 to every eval (first 8 test examples only), to confirm the script/env works
#            end-to-end in under a minute before committing to the full 1319-example x 7-checkpoint run.

set -euo pipefail

LIMIT_ARG=""
if [ "${1:-}" == "--debug" ]; then
    LIMIT_ARG="--limit 8"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_BASE="/scratch4/workspace/ychittepu_umass_edu-tribe/tribe-on-policy-2gpu"
OUT_DIR="$REPO_ROOT/gsm8k_eval"
mkdir -p "$OUT_DIR"

declare -A CHECKPOINTS=(
    [base]="Qwen/Qwen2.5-3B-Instruct"
    [tribe]="$RUNS_BASE/tribe/gsm8k"
    [grpo]="$RUNS_BASE/grpo/gsm8k"
    [dapo]="$RUNS_BASE/dapo/gsm8k"
    [rloo]="$RUNS_BASE/rloo/gsm8k"
    [raft]="$RUNS_BASE/raft/gsm8k"
    [onlinedpo]="$RUNS_BASE/onlinedpo/gsm8k"
)

for name in base tribe grpo dapo rloo raft onlinedpo; do
    echo "=== Evaluating $name (${CHECKPOINTS[$name]}) ==="
    python "$REPO_ROOT/scripts/eval_gsm8k.py" \
        --model_path "${CHECKPOINTS[$name]}" \
        --name "$name" \
        --output_file "$OUT_DIR/$name.json" \
        $LIMIT_ARG
done

echo
echo "===== GSM8K eval summary ====="
python3 "$REPO_ROOT/scripts/summarize_gsm8k_eval.py" "$OUT_DIR"
