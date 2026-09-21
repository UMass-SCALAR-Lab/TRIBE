#!/bin/bash
# Submits GSM8K test-set eval (scripts/eval_gsm8k.py) for the base model + all 6 finished checkpoints as
# 7 independent SLURM jobs (same one-job-per-unit-of-work pattern as slurm/submit_all.sh), so they run in
# parallel across 7 GPUs instead of one after another.
#
# Usage: bash slurm/submit_eval_gsm8k.sh [--debug] [name ...]
#   --debug: adds --limit 8 to every job (first 8 GSM8K test examples only) — use this first to confirm
#            the eval script/env/SLURM wiring actually works end-to-end before spending a full GPU-hour
#            per checkpoint on the real 1319-example run.
#   No names: submits all 7 (base, tribe, grpo, dapo, rloo, raft, onlinedpo).
#   With names: submits only those (e.g. `bash slurm/submit_eval_gsm8k.sh --debug tribe grpo`).
#
# Run from the Tribe/ repo root. Once all requested jobs finish, print the comparison table with:
#   python scripts/summarize_gsm8k_eval.py gsm8k_eval

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_BASE="/scratch4/workspace/ychittepu_umass_edu-tribe/tribe-on-policy-2gpu"
LOG_DIR="$REPO_ROOT/slurm/logs"
OUT_DIR="$REPO_ROOT/gsm8k_eval"
mkdir -p "$LOG_DIR" "$OUT_DIR"

LIMIT_ARG=""
if [ "${1:-}" == "--debug" ]; then
    LIMIT_ARG="--limit 8"
    shift
fi

declare -A CHECKPOINTS=(
    [base]="Qwen/Qwen2.5-3B-Instruct"
    [tribe]="$RUNS_BASE/tribe/gsm8k"
    [grpo]="$RUNS_BASE/grpo/gsm8k"
    [dapo]="$RUNS_BASE/dapo/gsm8k"
    [rloo]="$RUNS_BASE/rloo/gsm8k"
    [raft]="$RUNS_BASE/raft/gsm8k"
    [onlinedpo]="$RUNS_BASE/onlinedpo/gsm8k"
)
ALL_NAMES=(base tribe grpo dapo rloo raft onlinedpo)

WANTED=("$@")
if [ "${#WANTED[@]}" -eq 0 ]; then
    WANTED=("${ALL_NAMES[@]}")
fi

for name in "${WANTED[@]}"; do
    if [ -z "${CHECKPOINTS[$name]:-}" ]; then
        echo "Unknown checkpoint name '$name' (expected one of: ${ALL_NAMES[*]})" >&2
        exit 1
    fi
    job_name="eval-gsm8k-$name"
    cmd="python $REPO_ROOT/scripts/eval_gsm8k.py --model_path ${CHECKPOINTS[$name]} --name $name --output_file $OUT_DIR/$name.json $LIMIT_ARG"
    echo "Submitting $job_name ..."
    sbatch \
        --job-name="$job_name" \
        --output="$LOG_DIR/$job_name-%j.out" \
        --error="$LOG_DIR/$job_name-%j.err" \
        --export=ALL,EVAL_CMD="$cmd" \
        "$REPO_ROOT/slurm/run_eval_gsm8k.sbatch"
done
