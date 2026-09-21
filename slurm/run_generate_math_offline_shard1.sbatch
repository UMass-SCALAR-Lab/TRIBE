#!/bin/bash
#SBATCH --job-name=gen-math-offline-shard1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=superpod-a100
#SBATCH --mem=150G
#SBATCH --output=/scratch4/workspace/ychittepu_umass_edu-tribe/logs/gen-math-offline-shard1.out
#SBATCH --error=/scratch4/workspace/ychittepu_umass_edu-tribe/logs/gen-math-offline-shard1.err

# Recovery job: shard1 of the MATH off-policy generation (slurm/run_generate_math_offline.sbatch) was
# silently OOM-killed under the original job's shared --mem=80G (a single vLLM instance alone used ~44GB
# RSS; two concurrent instances exceeded the job's cgroup memory limit). Shard0 finished/is finishing
# unaffected. This regenerates shard1 alone with real headroom (150G, single GPU, no sibling process to
# share memory with), then merges it with shard0 into the final dataset.
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

python scripts/generate_offpolicy_math.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --num_generations 32 \
    --num_shards 2 --shard_index 1 \
    --output_dir "$SHARD1"

# Wait for shard0 to exist (it may still be finishing/saving from the original job) before merging.
while [ ! -f "$SHARD0/dataset_info.json" ]; do
    echo "Waiting for $SHARD0 to be saved by the original job..."
    sleep 60
done

python -c "
from datasets import load_from_disk, concatenate_datasets
d = concatenate_datasets([load_from_disk('$SHARD0'), load_from_disk('$SHARD1')])
d.save_to_disk('$FINAL')
print('Merged:', len(d), 'rows')
"
