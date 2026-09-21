#!/bin/bash
# Submits all 12 baseline-suite experiments (GRPO, DAPO, RLOO, RAFT, Online DPO, TRIBE x {GSM8K, MATH})
# as independent SLURM jobs, each via slurm/run_experiment.sbatch. One job per experiment: failures/slow
# runs don't block the rest, and each gets its own job ID / log file for tracking.
#
# Usage: bash slurm/submit_all.sh [experiment_name ...]
#   No args: submits all 12.
#   With args: submits only the named ones (e.g. `bash slurm/submit_all.sh gsm8k-raft math-raft`), for
#   resubmitting a failed run without resubmitting everything else.
#
# Run from the Tribe/ repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# <algo>/<dset> structure, e.g. .../tribe-on-policy/grpo/gsm8k, .../tribe-on-policy/raft/math.
RUNS_BASE="/scratch4/workspace/ychittepu_umass_edu-tribe/tribe-on-policy"
LOG_DIR="$REPO_ROOT/slurm/logs"
mkdir -p "$LOG_DIR"

MODEL="Qwen/Qwen2.5-3B-Instruct"
# Shared across every RL baseline (GRPO/DAPO/RLOO/TRIBE/OnlineDPO) and RAFT's SFT phase — see each
# script's own docstring for why these specific values (beta=0.1 uniform, scale_rewards="group" for every
# method including TRIBE, etc.).
# --report_to wandb is required, not optional: TrainingArguments' own default is "none" (all reporting
# disabled), so without this every run would only ever print metrics to stdout, never to the dashboard.
# learning_rate=1e-6 (not the original 5e-6): a GSM8K ablation found TRIBE at 5e-6 had a large realized
# per-cycle policy shift (tribe/f_div_old_end_of_cycle growing, tribe/lambda and tribe/kl_ref both
# climbing) that wasn't reward-directed — Stage 1's rho* itself stayed correctly bounded, so the update
# step size (not the trust region) was the cause. Dropping to 1e-6 recovered most of TRIBE's gap to GRPO
# (0.6694 -> 0.7119 GSM8K test accuracy) and made f_div bounded again; applied to every method uniformly
# here for a fair comparison, not just TRIBE.
COMMON="--model_name_or_path $MODEL --learning_rate 1e-6 --num_train_epochs 1 --gradient_checkpointing --bf16 True --report_to wandb"
COMMON_RL="$COMMON --num_generations 8 --per_device_train_batch_size 16 --max_completion_length 512 --beta 0.1 --log_completions"
# MATH's harder, competition-style problems need more reasoning tokens than GSM8K before reaching
# \boxed{...} — 512 was truncating completions first (high completions/clipped_ratio, frequent
# answer_parsed="[unparseable]"). MATH_EXTRA overrides max_completion_length via the parser's own
# last-flag-wins behavior; appended after $COMMON_RL in each MATH add_job call below, GSM8K untouched.
MATH_EXTRA="--max_completion_length 1024"

NAMES=()
CMDS=()

add_job() {
    NAMES+=("$1")
    # --run_name required, not optional: TrainingArguments never auto-fills it from output_dir, and
    # WandbCallback only names the run explicitly if run_name is set — without this every run shows up
    # on the dashboard with wandb's own random auto-generated name instead of e.g. "gsm8k-tribe".
    CMDS+=("$2 --run_name $1")
}

add_job "gsm8k-tribe" \
    "python scripts/train_gsm8k.py $COMMON_RL --output_dir $RUNS_BASE/tribe/gsm8k --trust_region_eps 0.05 --stage2_loss_type grpo --scale_rewards group"
add_job "math-tribe" \
    "python scripts/train_math.py $COMMON_RL $MATH_EXTRA --output_dir $RUNS_BASE/tribe/math --trust_region_eps 0.05 --stage2_loss_type grpo --scale_rewards group"

add_job "gsm8k-grpo" \
    "python scripts/train_gsm8k_grpo.py $COMMON_RL --output_dir $RUNS_BASE/grpo/gsm8k"
add_job "math-grpo" \
    "python scripts/train_math_grpo.py $COMMON_RL $MATH_EXTRA --output_dir $RUNS_BASE/grpo/math"

add_job "gsm8k-dapo" \
    "python scripts/train_gsm8k_grpo.py $COMMON_RL --output_dir $RUNS_BASE/dapo/gsm8k --loss_type dapo --epsilon 0.2 --epsilon_high 0.28 --mask_truncated_completions"
add_job "math-dapo" \
    "python scripts/train_math_grpo.py $COMMON_RL $MATH_EXTRA --output_dir $RUNS_BASE/dapo/math --loss_type dapo --epsilon 0.2 --epsilon_high 0.28 --mask_truncated_completions"

add_job "gsm8k-rloo" \
    "python scripts/train_gsm8k_rloo.py $COMMON_RL --output_dir $RUNS_BASE/rloo/gsm8k"
add_job "math-rloo" \
    "python scripts/train_math_rloo.py $COMMON_RL $MATH_EXTRA --output_dir $RUNS_BASE/rloo/math"

add_job "gsm8k-onlinedpo" \
    "python scripts/train_gsm8k_online_dpo.py $COMMON --per_device_train_batch_size 16 --max_new_tokens 512 --max_length 1024 --beta 0.1 --missing_eos_penalty 1.0 --output_dir $RUNS_BASE/onlinedpo/gsm8k"
add_job "math-onlinedpo" \
    "python scripts/train_math_online_dpo.py $COMMON --per_device_train_batch_size 16 --max_new_tokens 1024 --max_length 2048 --beta 0.1 --missing_eos_penalty 1.0 --output_dir $RUNS_BASE/onlinedpo/math"

# RAFT now reuses GRPOTrainer's own generation/reward machinery (scripts/raft_trainer.py) rather than a
# hand-rolled loop, so it takes the same $COMMON_RL flags as every other method — beta is simply unused
# by RAFTTrainer's own loss (no KL term by design), harmless to pass for consistency.
add_job "gsm8k-raft" \
    "python scripts/train_gsm8k_raft.py $COMMON_RL --output_dir $RUNS_BASE/raft/gsm8k"
add_job "math-raft" \
    "python scripts/train_math_raft.py $COMMON_RL $MATH_EXTRA --output_dir $RUNS_BASE/raft/math"

# Optional filter: only submit the experiments named on the command line.
WANTED=()
if [ "$#" -gt 0 ]; then
    WANTED=("$@")
fi

submit_job() {
    local name="$1"
    local cmd="$2"
    echo "Submitting $name ..."
    sbatch \
        --job-name="tribe-$name" \
        --output="$LOG_DIR/$name-%j.out" \
        --error="$LOG_DIR/$name-%j.err" \
        --export=ALL,TRAIN_CMD="$cmd" \
        "$REPO_ROOT/slurm/run_experiment.sbatch"
}

# TRICKLE=1 (set by slurm/trickle_submit.sh when it sources this file instead of running it directly)
# skips submitting everything up front — the trickle script only wants NAMES/CMDS/WANTED/submit_job
# defined, so it can throttle submission itself instead of firing off all 12 at once.
if [ -z "${TRICKLE:-}" ]; then
    for i in "${!NAMES[@]}"; do
        name="${NAMES[$i]}"
        if [ "${#WANTED[@]}" -gt 0 ]; then
            skip=true
            for w in "${WANTED[@]}"; do
                [ "$w" == "$name" ] && skip=false
            done
            [ "$skip" == true ] && continue
        fi
        submit_job "$name" "${CMDS[$i]}"
    done
fi
