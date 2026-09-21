#!/bin/bash
# 2-GPU variant of submit_all.sh, for all 6 methods — every trainer here is a GRPOTrainer subclass
# (TribeTrainer, RAFTTrainer included) and already uses self.accelerator.gather()/reduce() internally, so
# multi-GPU support was already there, just never exercised since every script has only ever been launched
# as plain `python`, a single process. RAFT specifically used to be excluded here (an earlier version was
# a hand-written model+optimizer loop with zero Accelerate wrapping and no multi-GPU support at all); it's
# now scripts/raft_trainer.py's RAFTTrainer(GRPOTrainer), which only overrides _compute_loss and gets
# multi-GPU for free from the parent class, the same as every other method.
#
# Two differences from submit_all.sh, both required, not optional tuning:
#   1. `accelerate launch --num_processes=2` instead of plain `python` — actually starts 2 processes.
#   2. per_device_train_batch_size halved (16 -> 8): GRPOConfig's own num_generations docstring is
#      explicit that the effective batch is num_processes * per_device_train_batch_size *
#      gradient_accumulation_steps. Keeping per_device_train_batch_size at 16 with num_processes=2 would
#      silently DOUBLE the global batch (16 -> 32), confounding every hyperparameter already locked in
#      for the 1-GPU suite. Halving it keeps the global batch at 16, unchanged from submit_all.sh — and
#      is also exactly what reduces the per-GPU generation/activation memory that caused the OOM crashes
#      in the 1-GPU runs, since each GPU now only holds half the batch.
#
# Output goes to a SEPARATE path (tribe-on-policy-2gpu, not tribe-on-policy) so this can never collide
# with the 1-GPU suite's checkpoint files. Job names, SLURM log filenames, and wandb run_names are NOT
# distinguished by GPU count (by design — nothing else about these runs tracks GPU count as a meaningful
# axis), so running both suites for the same experiment at once will show ambiguous-looking duplicate
# names in squeue/the wandb dashboard; only the output_dir path tells them apart.
#
# Usage: bash slurm/submit_all_2gpu.sh [experiment_name ...]
#   No args: submits all 12.
#   With args: submits only the named ones (e.g. `bash slurm/submit_all_2gpu.sh gsm8k-tribe math-tribe`).
#
# Run from the Tribe/ repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_BASE="/scratch4/workspace/ychittepu_umass_edu-tribe/tribe-on-policy-2gpu"
LOG_DIR="$REPO_ROOT/slurm/logs"
mkdir -p "$LOG_DIR"

MODEL="Qwen/Qwen2.5-3B-Instruct"
# $MASTER_PORT is deliberately unexpanded here (escaped) — it's computed by run_experiment.sbatch at job
# runtime (a free port, so co-located 2-GPU jobs on the same physical node can't collide on accelerate
# launch's fixed default rendezvous port), not at submission time here, where it isn't set yet.
LAUNCH="accelerate launch --num_processes=2 --mixed_precision=bf16 --main_process_port=\$MASTER_PORT"
# --report_to wandb is required, not optional: TrainingArguments' own default is "none" (all reporting
# disabled), so without this every run would only ever print metrics to stdout, never to the dashboard.
# learning_rate=1e-6 (not the original 5e-6): a GSM8K ablation found TRIBE at 5e-6 had a large realized
# per-cycle policy shift (tribe/f_div_old_end_of_cycle growing, tribe/lambda and tribe/kl_ref both
# climbing) that wasn't reward-directed — Stage 1's rho* itself stayed correctly bounded, so the update
# step size (not the trust region) was the cause. Dropping to 1e-6 recovered most of TRIBE's gap to GRPO
# (0.6694 -> 0.7119 GSM8K test accuracy) and made f_div bounded again; applied to every method uniformly
# here for a fair comparison, not just TRIBE.
COMMON="--model_name_or_path $MODEL --learning_rate 1e-6 --num_train_epochs 1 --gradient_checkpointing --bf16 True --report_to wandb"
COMMON_RL="$COMMON --num_generations 8 --per_device_train_batch_size 8 --max_completion_length 512 --beta 0.1 --log_completions"
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
    # on the dashboard with wandb's own random auto-generated name.
    CMDS+=("$2 --run_name $1")
}

add_job "gsm8k-tribe" \
    "$LAUNCH scripts/train_gsm8k.py $COMMON_RL --output_dir $RUNS_BASE/tribe/gsm8k --trust_region_eps 0.05 --stage2_loss_type grpo --scale_rewards group"
add_job "math-tribe" \
    "$LAUNCH scripts/train_math.py $COMMON_RL $MATH_EXTRA --output_dir $RUNS_BASE/tribe/math --trust_region_eps 0.05 --stage2_loss_type grpo --scale_rewards group"

add_job "gsm8k-grpo" \
    "$LAUNCH scripts/train_gsm8k_grpo.py $COMMON_RL --output_dir $RUNS_BASE/grpo/gsm8k"
add_job "math-grpo" \
    "$LAUNCH scripts/train_math_grpo.py $COMMON_RL $MATH_EXTRA --output_dir $RUNS_BASE/grpo/math"

add_job "gsm8k-dapo" \
    "$LAUNCH scripts/train_gsm8k_grpo.py $COMMON_RL --output_dir $RUNS_BASE/dapo/gsm8k --loss_type dapo --epsilon 0.2 --epsilon_high 0.28 --mask_truncated_completions"
add_job "math-dapo" \
    "$LAUNCH scripts/train_math_grpo.py $COMMON_RL $MATH_EXTRA --output_dir $RUNS_BASE/dapo/math --loss_type dapo --epsilon 0.2 --epsilon_high 0.28 --mask_truncated_completions"

add_job "gsm8k-rloo" \
    "$LAUNCH scripts/train_gsm8k_rloo.py $COMMON_RL --output_dir $RUNS_BASE/rloo/gsm8k"
add_job "math-rloo" \
    "$LAUNCH scripts/train_math_rloo.py $COMMON_RL $MATH_EXTRA --output_dir $RUNS_BASE/rloo/math"

# beta is unused by RAFTTrainer's own loss (no KL term by design), harmless to pass via $COMMON_RL.
add_job "gsm8k-raft" \
    "$LAUNCH scripts/train_gsm8k_raft.py $COMMON_RL --output_dir $RUNS_BASE/raft/gsm8k"
add_job "math-raft" \
    "$LAUNCH scripts/train_math_raft.py $COMMON_RL $MATH_EXTRA --output_dir $RUNS_BASE/raft/math"

# per_device_train_batch_size halved twice now (8->4->2) vs the other 4 methods, with
# gradient_accumulation_steps quadrupled (1->4) to keep the same effective batch size. OnlineDPOTrainer's
# own _forward() does a single unchunked forward pass and materializes a full (batch, seq_len,
# vocab_size) log_softmax tensor with no internal micro-batching (unlike GRPOTrainer's
# _get_per_token_logps_and_entropies, which every other method here uses) — genuinely more
# activation-memory-hungry per sample than the others at this vocab size (~152k for Qwen2.5). batch_size=8
# OOM'd outright; batch_size=4 still showed recoverable near-OOM allocator warnings throughout.
add_job "gsm8k-onlinedpo" \
    "$LAUNCH scripts/train_gsm8k_online_dpo.py $COMMON --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --max_new_tokens 512 --max_length 1024 --beta 0.1 --missing_eos_penalty 1.0 --output_dir $RUNS_BASE/onlinedpo/gsm8k"
add_job "math-onlinedpo" \
    "$LAUNCH scripts/train_math_online_dpo.py $COMMON --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --max_new_tokens 1024 --max_length 2048 --beta 0.1 --missing_eos_penalty 1.0 --output_dir $RUNS_BASE/onlinedpo/math"

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
        --gpus=2 \
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
