#!/bin/bash
#SBATCH --job-name=gen-math-offline
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=superpod-a100
#SBATCH --mem=80G
#SBATCH --output=/scratch4/workspace/ychittepu_umass_edu-tribe/logs/gen-math-offline.out
#SBATCH --error=/scratch4/workspace/ychittepu_umass_edu-tribe/logs/gen-math-offline.err

# One-off launcher for the full MATH off-policy data generation (scripts/generate_offpolicy_math.py),
# data-parallel across 2 GPUs (2 independent vLLM processes, each tensor_parallel_size=1, each generating
# half the training set — see this session's discussion for why data parallelism beats tensor parallelism
# for a model this small) rather than the shared run_eval_gsm8k.sbatch template, since this needs two
# concurrent background processes + a merge step, not a single EVAL_CMD.
set -euo pipefail

source ~/.bashrc
module load cuda/12.6
module load gcc/9.4.0
conda activate llm_gen

cd /work/pi_sniekum_umass_edu/ychittepu/Codes/Tribe
export HF_HOME=/scratch4/workspace/ychittepu_umass_edu-tribe/cache
export HF_HUB_OFFLINE=1

SHARD0=/scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/math-shard0
SHARD1=/scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/math-shard1
FINAL=/scratch4/workspace/ychittepu_umass_edu-tribe/offpolicy-data/math

CUDA_VISIBLE_DEVICES=0 python scripts/generate_offpolicy_math.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --num_generations 32 \
    --num_shards 2 --shard_index 0 \
    --output_dir "$SHARD0" &

CUDA_VISIBLE_DEVICES=1 python scripts/generate_offpolicy_math.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --num_generations 32 \
    --num_shards 2 --shard_index 1 \
    --output_dir "$SHARD1" &

wait

python -c "
from datasets import load_from_disk, concatenate_datasets
d = concatenate_datasets([load_from_disk('$SHARD0'), load_from_disk('$SHARD1')])
d.save_to_disk('$FINAL')
print('Merged:', len(d), 'rows')
"
