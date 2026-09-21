#!/bin/bash
# TRIBE val sweep: learning_rate x trust_region_eps x negative_fraction, MATH-Llama, 500-group subset.
# Reuses every other production hyperparameter unchanged from the negfrac sweep that produced the results
# table (offpolicy-tribe-floored-negfrac*-math-llama-boxed, e.g. job 63887138), per-batch OffPolicyTribeTrainer
# (scripts/train_gsm8k_offpolicy_tribe.py) -- matches what was actually run, not the 08-22 global-only decision.
#
# Uses the repo's existing run_experiment.sbatch / run_eval_gsm8k.sbatch launchers (see slurm/submit_all_2gpu.sh
# and slurm/submit_eval_gsm8k.sh) instead of hand-rolled sbatch --wrap: these already handle conda env
# activation (tribe for training, llm_gen for eval -- eval needs vllm, tribe deliberately doesn't), the
# cuda/gcc module loads DeepSpeed's JIT op builder needs (CUDA_HOME), and a real free-port scan for
# accelerate launch's rendezvous (avoids the port-collision bug hand-rolled ports hit earlier).
#
# Grid: 5 LRs x 3 trust_region_eps x 3 negative_fraction = 45 training runs, each followed by a val eval
# (pass@1 greedy, val_fraction=0.1, chained via --dependency=afterok so it only runs if training succeeds).
#
# Usage: bash slurm/run_tribe_val_sweep.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET=/scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/math-llama-boxed-lrsweep500
OUTBASE=/scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-suite-final/lrsweep-tribe
LOGDIR="$REPO_ROOT/slurm/logs"

# $MASTER_PORT deliberately unexpanded (escaped) -- computed by run_experiment.sbatch at job runtime, same
# pattern as slurm/submit_all_2gpu.sh.
LAUNCH="accelerate launch --num_processes=2 --main_process_port=\$MASTER_PORT"

LRS=(1e-7 5e-8 1e-8 5e-7 1e-6)
EPSS=(2.0 1.0 0.5)
NEGFRACS=(1.0 0.0 0.2)

for LR in "${LRS[@]}"; do
  for EPS in "${EPSS[@]}"; do
    for NF in "${NEGFRACS[@]}"; do
      TAG="lr${LR}-eps${EPS}-negfrac${NF}"
      OUTDIR="${OUTBASE}/tribe-${TAG}/math"
      RESULT_JSON="$REPO_ROOT/math_eval/tribe-lrsweep-${TAG}-val.json"

      EVAL_CMD="sed -i 's/\"tokenizer_class\": \"TokenizersBackend\"/\"tokenizer_class\": \"PreTrainedTokenizerFast\"/' ${OUTDIR}/tokenizer_config.json && \
        python scripts/eval_math.py --model_path ${OUTDIR} \
        --name tribe-lrsweep-${TAG}-val --val_fraction 0.1 \
        --output_file math_eval/tribe-lrsweep-${TAG}-val.json"

      # Resumable: skip configs whose val result already landed; re-eval-only configs whose checkpoint
      # exists but the eval failed (e.g. the quota-exceeded wave); full train+eval otherwise.
      if [ -f "$RESULT_JSON" ]; then
        echo "skipping ${TAG}: result already exists"
        continue
      fi

      if [ -f "${OUTDIR}/model.safetensors" ]; then
        sbatch \
          --job-name="eval-tribe-lrsweep-${TAG}-val" \
          --output="${LOGDIR}/eval-tribe-lrsweep-${TAG}-val-%j.out" \
          --error="${LOGDIR}/eval-tribe-lrsweep-${TAG}-val-%j.err" \
          --export=ALL,EVAL_CMD="$EVAL_CMD" \
          "$REPO_ROOT/slurm/run_eval_gsm8k.sbatch"
        echo "submitted ${TAG}: eval-only (checkpoint already exists)"
        continue
      fi

      TRAIN_CMD="$LAUNCH scripts/train_gsm8k_offpolicy_tribe.py \
        --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
        --dataset_path ${DATASET} \
        --output_dir ${OUTDIR} \
        --group_size 32 --per_device_train_batch_size 16 --max_completion_length 1024 \
        --learning_rate ${LR} --lr_scheduler_type constant \
        --trust_region_eps ${EPS} --divergence chi_squared --negative_fraction ${NF} \
        --deepspeed configs/deepspeed_zero3.json --save_total_limit 1 --num_train_epochs 1 \
        --gradient_checkpointing --bf16 True --report_to wandb \
        --run_name offpolicy-tribe-lrsweep-${TAG}"

      TRAIN_JOBID=$(sbatch --parsable \
        --gpus=2 \
        --job-name="offpolicy-tribe-lrsweep-${TAG}" \
        --output="${LOGDIR}/offpolicy-tribe-lrsweep-${TAG}-%j.out" \
        --error="${LOGDIR}/offpolicy-tribe-lrsweep-${TAG}-%j.err" \
        --export=ALL,TRAIN_CMD="$TRAIN_CMD" \
        "$REPO_ROOT/slurm/run_experiment.sbatch")

      sbatch \
        --job-name="eval-tribe-lrsweep-${TAG}-val" \
        --output="${LOGDIR}/eval-tribe-lrsweep-${TAG}-val-%j.out" \
        --error="${LOGDIR}/eval-tribe-lrsweep-${TAG}-val-%j.err" \
        --dependency=afterok:${TRAIN_JOBID} \
        --export=ALL,EVAL_CMD="$EVAL_CMD" \
        "$REPO_ROOT/slurm/run_eval_gsm8k.sbatch"

      echo "submitted ${TAG}: train=${TRAIN_JOBID}"
    done
  done
done
