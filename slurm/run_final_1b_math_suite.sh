#!/bin/bash
# Final, bug-fixed 1B weak-learner MATH suite: same structure as the original weak-learner run
# (slurm/run_llama1b_weaklearner_math_suite.sh, job IDs 63887125-... era) -- meta-llama/Llama-3.2-1B-Instruct
# trained on the 3B-generated offline data, lr=1e-7 flat for 8 baselines (unchanged, already-agreed choice,
# not each method's own optimum), TRIBE at the newly tuned eps=0.5/chi_squared (lr already matched 1e-7).
# divergence=chi_squared is mandatory -- kl_new_old confirmed dead this session (6/6 configs collapse to
# 0% accuracy, lambda pinned at solve_lambda's -50 safety clamp).
#
# GRPO/TOPR/TIS/DPO/TRIBE pass --ref_model_name_or_path meta-llama/Llama-3.2-3B-Instruct: the 1B model
# being trained did NOT generate this dataset (the 3B model did), so their frozen behavioral-policy/
# reference copy must be the actual 3B generator. RAFT/RLOO/NaiveReinforce/SimPO use no reference model
# (RAFT: positive-only SFT; SimPO: reference-free by design) so they omit the flag.
#
# Per-method batch size / deepspeed settings copied exactly from the original 3B commands (RAFT:
# per_device_bs=4, no deepspeed; DPO/SimPO: per_device_bs=4, grad_accum=16; everything else: per_device_bs=16,
# deepspeed zero3) -- not re-tuned for the smaller 1B model, same as the original weak-learner run.
#
# Every checkpoint gets scripts/overlay_base_config.py applied before eval, and eval is pass@1 (greedy) +
# pass@4 (num_generations=4, temperature=1.0, top_k=50) on the actual MATH TEST split -- both handled by a
# Monitor loop outside this script, not submitted here (overlay must run in the `tribe` env between
# training and eval). Base-1B reference numbers already exist (jobs 63962903/63962904), not resubmitted.
#
# Usage: bash slurm/run_final_1b_math_suite.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$REPO_ROOT/slurm/logs"
MODEL="meta-llama/Llama-3.2-1B-Instruct"
REF_MODEL="meta-llama/Llama-3.2-3B-Instruct"
DATASET=/scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/math-llama-boxed
DPO_DATASET=/scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/math-llama-boxed-dpo-pairs
OUTBASE=/scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-suite-final/final-1b
LAUNCH="accelerate launch --num_processes=2 --main_process_port=\$MASTER_PORT"

JOBLIST_FILE="$REPO_ROOT/slurm/final_1b_jobs.tsv"
: > "$JOBLIST_FILE"

submit_train() {
  local name="$1" script="$2" extra="$3" bs="${4:-16}" ds="${5:-1}"
  local outdir="${OUTBASE}/${name}/math"
  local ds_flag=""
  [ "$ds" = "1" ] && ds_flag="--deepspeed configs/deepspeed_zero3.json"
  local cmd="$LAUNCH scripts/${script} --model_name_or_path ${MODEL} --dataset_path ${DATASET} \
    --output_dir ${outdir} --group_size 32 --per_device_train_batch_size ${bs} \
    --max_completion_length 1024 --learning_rate 1e-7 --lr_scheduler_type constant \
    ${extra} ${ds_flag} --save_total_limit 1 --num_train_epochs 1 --gradient_checkpointing \
    --bf16 True --report_to wandb --run_name offpolicy-${name}-final-1b"
  local jid=$(sbatch --parsable --gpus=2 --mem=240G --time=1-00:00:00 \
    --job-name="offpolicy-${name}-final-1b" \
    --output="${LOGDIR}/offpolicy-${name}-final-1b-%j.out" \
    --error="${LOGDIR}/offpolicy-${name}-final-1b-%j.err" \
    --export=ALL,TRAIN_CMD="$cmd" \
    "$REPO_ROOT/slurm/run_experiment.sbatch")
  echo -e "${jid}\t${name}\t${outdir}\t${MODEL}" >> "$JOBLIST_FILE"
  echo "submitted $name: train=$jid"
}

submit_train "grpo"            "train_gsm8k_offpolicy_grpo.py"            "--ref_model_name_or_path ${REF_MODEL}"
submit_train "topr"            "train_gsm8k_offpolicy_topr.py"            "--weight_decay 0.0 --ref_model_name_or_path ${REF_MODEL}"
submit_train "tis"             "train_gsm8k_offpolicy_tis.py"             "--ref_model_name_or_path ${REF_MODEL}"
submit_train "raft"            "train_gsm8k_offpolicy_raft.py"            "" 4 0
submit_train "rloo"            "train_gsm8k_offpolicy_reinforce.py"       ""
submit_train "naive-reinforce" "train_gsm8k_offpolicy_naive_reinforce.py" ""

for NF in 0.0 0.2 0.4 0.6 0.8 1.0; do
  submit_train "tribe-negfrac${NF}" "train_gsm8k_offpolicy_tribe.py" \
    "--trust_region_eps 0.5 --divergence chi_squared --negative_fraction ${NF} --ref_model_name_or_path ${REF_MODEL}"
done

dpo_jid=$(sbatch --parsable --gpus=2 --mem=240G --time=1-00:00:00 \
  --job-name="offpolicy-dpo-final-1b" \
  --output="${LOGDIR}/offpolicy-dpo-final-1b-%j.out" \
  --error="${LOGDIR}/offpolicy-dpo-final-1b-%j.err" \
  --export=ALL,TRAIN_CMD="$LAUNCH scripts/train_gsm8k_offpolicy_dpo.py --model_name_or_path ${MODEL} \
    --ref_model_name_or_path ${REF_MODEL} \
    --dataset_path ${DPO_DATASET} --output_dir ${OUTBASE}/dpo/math \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 16 --learning_rate 1e-7 --beta 0.1 \
    --deepspeed configs/deepspeed_zero3.json --save_total_limit 1 --num_train_epochs 1 \
    --gradient_checkpointing --bf16 True --report_to wandb --run_name offpolicy-dpo-final-1b" \
  "$REPO_ROOT/slurm/run_experiment.sbatch")
echo -e "${dpo_jid}\tdpo\t${OUTBASE}/dpo/math\t${MODEL}" >> "$JOBLIST_FILE"
echo "submitted dpo: train=$dpo_jid"

simpo_jid=$(sbatch --parsable --gpus=2 --mem=240G --time=1-00:00:00 \
  --job-name="offpolicy-simpo-final-1b" \
  --output="${LOGDIR}/offpolicy-simpo-final-1b-%j.out" \
  --error="${LOGDIR}/offpolicy-simpo-final-1b-%j.err" \
  --export=ALL,TRAIN_CMD="$LAUNCH scripts/train_gsm8k_offpolicy_simpo.py --model_name_or_path ${MODEL} \
    --dataset_path ${DPO_DATASET} --output_dir ${OUTBASE}/simpo/math \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 16 --learning_rate 1e-7 --beta 2.5 \
    --simpo_gamma 1.4 --lr_scheduler_type cosine \
    --deepspeed configs/deepspeed_zero3.json --save_total_limit 1 --num_train_epochs 1 \
    --gradient_checkpointing --bf16 True --report_to wandb --run_name offpolicy-simpo-final-1b" \
  "$REPO_ROOT/slurm/run_experiment.sbatch")
echo -e "${simpo_jid}\tsimpo\t${OUTBASE}/simpo/math\t${MODEL}" >> "$JOBLIST_FILE"
echo "submitted simpo: train=$simpo_jid"

echo "All 1B jobs submitted. Job list written to $JOBLIST_FILE"
