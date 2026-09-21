#!/bin/bash
# Quick end-to-end smoke test for all baseline driver scripts (GSM8K + MATH), before committing to full
# runs. Uses a tiny model and a handful of steps so this finishes in a few minutes on one GPU — this is
# NOT checking whether any method learns anything, only that each script boots, generates, computes
# reward, and takes an optimizer step without crashing.
#
# Usage: bash scripts/smoke_test.sh [model_name_or_path]
# Requires: conda env `tribe` active, run from the Tribe/ repo root.

set -u
MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
# Written under the repo itself (not /tmp) so the logs land on shared storage, readable from any host
# that has this checkout mounted — /tmp is node-local on most cluster setups.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO_ROOT/smoke_test_logs"
rm -rf "$OUT"
mkdir -p "$OUT"

# Force offline mode so no script needs network/credentials during a smoke test.
export WANDB_MODE=offline

PASS=()
FAIL=()

run() {
    local name="$1"
    shift
    echo "=== $name ==="
    if "$@" > "$OUT/$name.log" 2>&1; then
        PASS+=("$name")
    else
        FAIL+=("$name")
        echo "FAILED — see $OUT/$name.log"
        tail -n 30 "$OUT/$name.log"
    fi
}

COMMON_GEN="--model_name_or_path $MODEL --num_generations 4 --per_device_train_batch_size 4 --max_steps 3 --gradient_checkpointing --report_to none"

run gsm8k_tribe python scripts/train_gsm8k.py $COMMON_GEN --output_dir $OUT/gsm8k_tribe --max_completion_length 64 --beta 0.04 --trust_region_eps 0.05
run gsm8k_grpo  python scripts/train_gsm8k_grpo.py $COMMON_GEN --output_dir $OUT/gsm8k_grpo --max_completion_length 64 --beta 0.1
run gsm8k_dapo  python scripts/train_gsm8k_grpo.py $COMMON_GEN --output_dir $OUT/gsm8k_dapo --max_completion_length 64 --beta 0.0 --loss_type dapo --epsilon 0.2 --epsilon_high 0.28 --mask_truncated_completions
run gsm8k_rloo  python scripts/train_gsm8k_rloo.py $COMMON_GEN --output_dir $OUT/gsm8k_rloo --max_completion_length 64 --beta 0.1
run gsm8k_raft  python scripts/train_gsm8k_raft.py $COMMON_GEN --output_dir $OUT/gsm8k_raft --max_completion_length 64
run gsm8k_onlinedpo python scripts/train_gsm8k_online_dpo.py --model_name_or_path "$MODEL" --output_dir $OUT/gsm8k_onlinedpo --per_device_train_batch_size 4 --max_steps 3 --max_new_tokens 64 --max_length 256 --beta 0.1 --gradient_checkpointing --report_to none

run math_tribe python scripts/train_math.py $COMMON_GEN --output_dir $OUT/math_tribe --max_completion_length 64 --beta 0.04 --trust_region_eps 0.05
run math_grpo  python scripts/train_math_grpo.py $COMMON_GEN --output_dir $OUT/math_grpo --max_completion_length 64 --beta 0.1
run math_rloo  python scripts/train_math_rloo.py $COMMON_GEN --output_dir $OUT/math_rloo --max_completion_length 64 --beta 0.1
run math_raft  python scripts/train_math_raft.py $COMMON_GEN --output_dir $OUT/math_raft --max_completion_length 64
run math_onlinedpo python scripts/train_math_online_dpo.py --model_name_or_path "$MODEL" --output_dir $OUT/math_onlinedpo --per_device_train_batch_size 4 --max_steps 3 --max_new_tokens 64 --max_length 256 --beta 0.1 --gradient_checkpointing --report_to none

echo
echo "===== SUMMARY ====="
echo "PASS (${#PASS[@]}): ${PASS[*]}"
echo "FAIL (${#FAIL[@]}): ${FAIL[*]}"
[ "${#FAIL[@]}" -eq 0 ]
