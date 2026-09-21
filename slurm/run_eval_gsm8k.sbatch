#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=superpod-a100
#SBATCH --mem=80G

# Generic launcher shared by every checkpoint eval in slurm/submit_eval_gsm8k.sh — same one-file,
# many-independent-jobs pattern as slurm/run_experiment.sbatch, but a separate template because eval needs
# the `llm_gen` conda env (has vllm; `tribe`, used for training, deliberately doesn't — see
# scripts/eval_gsm8k.py's docstring) and far less time/memory than a training run.
set -euo pipefail

if [ -z "${EVAL_CMD:-}" ]; then
    echo "EVAL_CMD is not set — submit via slurm/submit_eval_gsm8k.sh, not sbatch run_eval_gsm8k.sbatch directly." >&2
    exit 1
fi

source ~/.bashrc
module load cuda/12.6
module load gcc/9.4.0
conda activate llm_gen

cd /work/pi_sniekum_umass_edu/ychittepu/Codes/Tribe
export HF_HOME=/scratch4/workspace/ychittepu_umass_edu-tribe/cache
# Same fix as run_experiment.sbatch: base model is already cached, so skip the network ETag-revalidation
# path entirely rather than risk it racing when multiple eval jobs start at once.
export HF_HUB_OFFLINE=1

echo "Running: $EVAL_CMD"
eval "$EVAL_CMD"
