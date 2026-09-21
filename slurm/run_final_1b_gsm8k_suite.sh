#!/bin/bash
# GSM8K counterpart of run_final_1b_math_suite.sh -- meta-llama/Llama-3.2-1B-Instruct trained on the
# 3B-generated gsm8k-llama-boxed / gsm8k-llama-boxed-dpo-pairs data (no new generation needed), same
# hyperparameters validated on the MATH weak-learner suite (lr=1e-7 flat for 8 baselines, TRIBE at the
# newly tuned eps=0.5/chi_squared). Batch size / deepspeed settings copied from this project's own
# GSM8K-Llama convention (see run_final_3b_gsm8k_suite.sh's header), not the MATH suite's settings.
#
# GRPO/TOPR/TIS/DPO/TRIBE pass --ref_model_name_or_path meta-llama/Llama-3.2-3B-Instruct (the actual
# generator of this dataset); RAFT/RLOO/NaiveReinforce/SimPO omit it (no reference model needed).
#
# Also submits the 1B base GSM8K eval (pass@1 + pass@4) since it doesn't exist yet, unlike MATH where it
# was already available.
#
# Usage: bash slurm/run_final_1b_gsm8k_suite.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$REPO_ROOT/slurm/logs"
MODEL="meta-llama/Llama-3.2-1B-Instruct"
REF_MODEL="meta-llama/Llama-3.2-3B-Instruct"
DATASET=/scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/gsm8k-llama-boxed
DPO_DATASET=/scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/gsm8k-llama-boxed-dpo-pairs
OUTBASE=/scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-suite-final/final-1b-gsm8k
LAUNCH="accelerate launch --num_processes=2 --mixed_precision=bf16 --main_process_port=\$MASTER_PORT"

JOBLIST_FILE="$REPO_ROOT/slurm/final_1b_gsm8k_jobs.tsv"
: > "$JOBLIST_FILE"

submit_train() {
  local name="$1" script="$2" extra="$3" bs="${4:-8}"
  local outdir="${OUTBASE}/${name}/gsm8k"
  local cmd="$LAUNCH scripts/${script} --model_name_or_path ${MODEL} --dataset_path ${DATASET} \
    --output_dir ${outdir} --group_size 16 --per_device_train_batch_size ${bs} \
    --learning_rate 1e-7 --lr_scheduler_type constant \
    ${extra} --save_total_limit 1 --num_train_epochs 1 --gradient_checkpointing \
    --bf16 True --report_to wandb --run_name offpolicy-${name}-final-1b-gsm8k"
  local jid=$(sbatch --parsable --gpus=2 --mem=240G --time=1-00:00:00 \
    --job-name="offpolicy-${name}-final-1b-gsm8k" \
    --output="${LOGDIR}/offpolicy-${name}-final-1b-gsm8k-%j.out" \
    --error="${LOGDIR}/offpolicy-${name}-final-1b-gsm8k-%j.err" \
    --export=ALL,TRAIN_CMD="$cmd" \
    "$REPO_ROOT/slurm/run_experiment.sbatch")
  echo -e "${jid}\t${name}\t${outdir}\t${MODEL}" >> "$JOBLIST_FILE"
  echo "submitted $name: train=$jid"
}

submit_train "grpo"            "train_gsm8k_offpolicy_grpo.py"            "--ref_model_name_or_path ${REF_MODEL}"
submit_train "topr"            "train_gsm8k_offpolicy_topr.py"            "--weight_decay 0.0 --ref_model_name_or_path ${REF_MODEL}"
submit_train "tis"             "train_gsm8k_offpolicy_tis.py"             "--ref_model_name_or_path ${REF_MODEL}"
submit_train "raft"            "train_gsm8k_offpolicy_raft.py"            "" 16
submit_train "rloo"            "train_gsm8k_offpolicy_reinforce.py"       ""
submit_train "naive-reinforce" "train_gsm8k_offpolicy_naive_reinforce.py" ""

for NF in 0.0 0.2 0.4 0.6 0.8 1.0; do
  submit_train "tribe-negfrac${NF}" "train_gsm8k_offpolicy_tribe.py" \
    "--trust_region_eps 0.5 --divergence chi_squared --negative_fraction ${NF} --ref_model_name_or_path ${REF_MODEL}"
done

dpo_jid=$(sbatch --parsable --gpus=2 --mem=240G --time=1-00:00:00 \
  --job-name="offpolicy-dpo-final-1b-gsm8k" \
  --output="${LOGDIR}/offpolicy-dpo-final-1b-gsm8k-%j.out" \
  --error="${LOGDIR}/offpolicy-dpo-final-1b-gsm8k-%j.err" \
  --export=ALL,TRAIN_CMD="accelerate launch --num_processes=2 --main_process_port=\$MASTER_PORT scripts/train_gsm8k_offpolicy_dpo.py --model_name_or_path ${MODEL} \
    --ref_model_name_or_path ${REF_MODEL} \
    --dataset_path ${DPO_DATASET} --output_dir ${OUTBASE}/dpo/gsm8k \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 16 --learning_rate 1e-7 --beta 0.1 \
    --deepspeed configs/deepspeed_zero3.json --save_total_limit 1 --num_train_epochs 1 \
    --gradient_checkpointing --bf16 True --report_to wandb --run_name offpolicy-dpo-final-1b-gsm8k" \
  "$REPO_ROOT/slurm/run_experiment.sbatch")
echo -e "${dpo_jid}\tdpo\t${OUTBASE}/dpo/gsm8k\t${MODEL}" >> "$JOBLIST_FILE"
echo "submitted dpo: train=$dpo_jid"

simpo_jid=$(sbatch --parsable --gpus=2 --mem=240G --time=1-00:00:00 \
  --job-name="offpolicy-simpo-final-1b-gsm8k" \
  --output="${LOGDIR}/offpolicy-simpo-final-1b-gsm8k-%j.out" \
  --error="${LOGDIR}/offpolicy-simpo-final-1b-gsm8k-%j.err" \
  --export=ALL,TRAIN_CMD="$LAUNCH scripts/train_gsm8k_offpolicy_simpo.py --model_name_or_path ${MODEL} \
    --dataset_path ${DPO_DATASET} --output_dir ${OUTBASE}/simpo/gsm8k \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 16 --learning_rate 1e-7 --beta 2.5 \
    --simpo_gamma 1.4 --lr_scheduler_type cosine \
    --save_total_limit 1 --num_train_epochs 1 \
    --gradient_checkpointing --bf16 True --report_to wandb --run_name offpolicy-simpo-final-1b-gsm8k" \
  "$REPO_ROOT/slurm/run_experiment.sbatch")
echo -e "${simpo_jid}\tsimpo\t${OUTBASE}/simpo/gsm8k\t${MODEL}" >> "$JOBLIST_FILE"
echo "submitted simpo: train=$simpo_jid"

# 1B base reference (doesn't exist yet, unlike MATH) -- no overlay needed, this is the untouched hub model.
mkdir -p "$REPO_ROOT/gsm8k_eval"
sbatch --gpus=1 --time=02:00:00 \
  --job-name="eval-base-llama1b-gsm8k-pass1" \
  --output="${LOGDIR}/eval-base-llama1b-gsm8k-pass1-%j.out" \
  --error="${LOGDIR}/eval-base-llama1b-gsm8k-pass1-%j.err" \
  --export=ALL,EVAL_CMD="python scripts/eval_gsm8k.py --model_path ${MODEL} --name base-llama1b-gsm8k-pass1 --answer_format boxed --output_file gsm8k_eval/base-llama1b-gsm8k-pass1.json" \
  "$REPO_ROOT/slurm/run_eval_gsm8k.sbatch"
sbatch --gpus=1 --time=02:00:00 \
  --job-name="eval-base-llama1b-gsm8k-pass4" \
  --output="${LOGDIR}/eval-base-llama1b-gsm8k-pass4-%j.out" \
  --error="${LOGDIR}/eval-base-llama1b-gsm8k-pass4-%j.err" \
  --export=ALL,EVAL_CMD="python scripts/eval_gsm8k.py --model_path ${MODEL} --name base-llama1b-gsm8k-pass4 --answer_format boxed --num_generations 4 --temperature 1.0 --top_k 50 --output_file gsm8k_eval/base-llama1b-gsm8k-pass4.json" \
  "$REPO_ROOT/slurm/run_eval_gsm8k.sbatch"
echo "submitted base-llama1b-gsm8k pass1+pass4 eval"

echo "All 1B GSM8K jobs submitted. Job list written to $JOBLIST_FILE"
