#!/bin/bash
# Final, bug-fixed 3B MATH suite: 8 baselines (GRPO/TOPR/TIS/RAFT/RLOO/NaiveReinforce/DPO/SimPO) at their
# original lr=5e-7, plus TRIBE's negative_fraction sweep (0.0-1.0) at the NEWLY tuned lr=1e-7/eps=0.5/
# chi_squared (found this session's LR/eps sweep to match GRPO/TOPR/DPO at eps=0.5,lr=1e-7,negfrac=1.0:
# 53.07% val vs GRPO 53.87%/TOPR 52.8%/DPO 55.2%). divergence=chi_squared is mandatory, not a default --
# kl_new_old was confirmed dead (6/6 configs collapse to exactly 0% accuracy, lambda pinned at
# solve_lambda's -50 safety clamp) this same session.
#
# Same self-generated dataset as the original (pre-bug-fix) 3B suite (job IDs 63887125-63887138) --
# math-llama-boxed / math-llama-boxed-dpo-pairs, no new generation needed, no --ref_model_name_or_path
# (this model generated its own training data). Per-method batch size/deepspeed settings copied exactly
# from those same original jobs.
#
# What's different from the original run, beyond TRIBE's hyperparameters: every checkpoint gets
# scripts/overlay_base_config.py applied before eval (fixes the rope_theta/config-schema-drift bug that
# was silently corrupting every checkpoint's generation quality this whole investigation was about), and
# eval is pass@1 (greedy) + pass@4 (num_generations=4, temperature=1.0, top_k=50) on the actual MATH TEST
# split (not val -- hyperparameters are already chosen, this is the final-numbers pass). Both are handled
# by a Monitor loop outside this script (overlay needs to run in the `tribe` env between training and
# eval, not chainable via plain sbatch --dependency), not submitted here.
#
# Usage: bash slurm/run_final_3b_math_suite.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$REPO_ROOT/slurm/logs"
MODEL="meta-llama/Llama-3.2-3B-Instruct"
DATASET=/scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/math-llama-boxed
DPO_DATASET=/scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/math-llama-boxed-dpo-pairs
OUTBASE=/scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-suite-final/final-3b
LAUNCH="accelerate launch --num_processes=2 --main_process_port=\$MASTER_PORT"

JOBLIST_FILE="$REPO_ROOT/slurm/final_3b_jobs.tsv"
: > "$JOBLIST_FILE"

submit_train() {
  local name="$1" script="$2" extra="$3" bs="${4:-16}" ds="${5:-1}" mp="${6:-0}"
  local outdir="${OUTBASE}/${name}/math"
  local ds_flag="" mp_flag=""
  [ "$ds" = "1" ] && ds_flag="--deepspeed configs/deepspeed_zero3.json"
  [ "$mp" = "1" ] && mp_flag="--mixed_precision=bf16"
  local cmd="accelerate launch --num_processes=2 ${mp_flag} --main_process_port=\$MASTER_PORT scripts/${script} --model_name_or_path ${MODEL} --dataset_path ${DATASET} \
    --output_dir ${outdir} --group_size 32 --per_device_train_batch_size ${bs} \
    --max_completion_length 1024 --lr_scheduler_type constant \
    ${extra} ${ds_flag} --save_total_limit 1 --num_train_epochs 1 --gradient_checkpointing \
    --bf16 True --report_to wandb --run_name offpolicy-${name}-final-3b"
  local jid=$(sbatch --parsable --gpus=2 --mem=240G --time=1-00:00:00 \
    --job-name="offpolicy-${name}-final-3b" \
    --output="${LOGDIR}/offpolicy-${name}-final-3b-%j.out" \
    --error="${LOGDIR}/offpolicy-${name}-final-3b-%j.err" \
    --export=ALL,TRAIN_CMD="$cmd" \
    "$REPO_ROOT/slurm/run_experiment.sbatch")
  echo -e "${jid}\t${name}\t${outdir}\t${MODEL}" >> "$JOBLIST_FILE"
  echo "submitted $name: train=$jid"
}

submit_train "grpo"            "train_gsm8k_offpolicy_grpo.py"            "--learning_rate 5e-7"
submit_train "topr"            "train_gsm8k_offpolicy_topr.py"            "--learning_rate 5e-7 --weight_decay 0.0"
submit_train "tis"             "train_gsm8k_offpolicy_tis.py"             "--learning_rate 5e-7"
submit_train "raft"            "train_gsm8k_offpolicy_raft.py"            "--learning_rate 5e-7" 4 0 1
submit_train "rloo"            "train_gsm8k_offpolicy_reinforce.py"       "--learning_rate 5e-7"
submit_train "naive-reinforce" "train_gsm8k_offpolicy_naive_reinforce.py" "--learning_rate 5e-7"

for NF in 0.0 0.2 0.4 0.6 0.8 1.0; do
  submit_train "tribe-negfrac${NF}" "train_gsm8k_offpolicy_tribe.py" \
    "--learning_rate 1e-7 --trust_region_eps 0.5 --divergence chi_squared --negative_fraction ${NF}"
done

# DPO/SimPO use the paired dataset and their own bs/grad_accum recipe -- can't go through submit_train's
# --dataset_path DATASET default.
dpo_jid=$(sbatch --parsable --gpus=2 --mem=240G --time=1-00:00:00 \
  --job-name="offpolicy-dpo-final-3b" \
  --output="${LOGDIR}/offpolicy-dpo-final-3b-%j.out" \
  --error="${LOGDIR}/offpolicy-dpo-final-3b-%j.err" \
  --export=ALL,TRAIN_CMD="$LAUNCH scripts/train_gsm8k_offpolicy_dpo.py --model_name_or_path ${MODEL} \
    --dataset_path ${DPO_DATASET} --output_dir ${OUTBASE}/dpo/math \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 16 --learning_rate 5e-7 --beta 0.1 \
    --deepspeed configs/deepspeed_zero3.json --save_total_limit 1 --num_train_epochs 1 \
    --gradient_checkpointing --bf16 True --report_to wandb --run_name offpolicy-dpo-final-3b" \
  "$REPO_ROOT/slurm/run_experiment.sbatch")
echo -e "${dpo_jid}\tdpo\t${OUTBASE}/dpo/math\t${MODEL}" >> "$JOBLIST_FILE"
echo "submitted dpo: train=$dpo_jid"

simpo_jid=$(sbatch --parsable --gpus=2 --mem=240G --time=1-00:00:00 \
  --job-name="offpolicy-simpo-final-3b" \
  --output="${LOGDIR}/offpolicy-simpo-final-3b-%j.out" \
  --error="${LOGDIR}/offpolicy-simpo-final-3b-%j.err" \
  --export=ALL,TRAIN_CMD="$LAUNCH scripts/train_gsm8k_offpolicy_simpo.py --model_name_or_path ${MODEL} \
    --dataset_path ${DPO_DATASET} --output_dir ${OUTBASE}/simpo/math \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 16 --learning_rate 5e-7 --beta 2.5 \
    --simpo_gamma 1.4 --lr_scheduler_type cosine \
    --deepspeed configs/deepspeed_zero3.json --save_total_limit 1 --num_train_epochs 1 \
    --gradient_checkpointing --bf16 True --report_to wandb --run_name offpolicy-simpo-final-3b" \
  "$REPO_ROOT/slurm/run_experiment.sbatch")
echo -e "${simpo_jid}\tsimpo\t${OUTBASE}/simpo/math\t${MODEL}" >> "$JOBLIST_FILE"
echo "submitted simpo: train=$simpo_jid"

echo "All 3B jobs submitted. Job list written to $JOBLIST_FILE"
