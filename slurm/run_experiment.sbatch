#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=superpod-a100
#SBATCH --mem=240G

# Generic launcher shared by every experiment in scripts/submit_all.sh. Per-experiment identity
# (--job-name, --output, --error) and the actual training invocation (TRAIN_CMD) are supplied by the
# caller via `sbatch --job-name=... --output=... --error=... --export=ALL,TRAIN_CMD="..." run_experiment.sbatch`
# rather than hardcoded here — one file, twelve independent jobs, cluster-wide settings (partition/gpus/
# mem/time) adjustable in exactly one place. --gpus=1: every script in this project is launched as plain
# `python scripts/train_*.py`, not `accelerate launch`/`torchrun`, so it only ever uses a single GPU
# regardless of how many are allocated.
set -euo pipefail

if [ -z "${TRAIN_CMD:-}" ]; then
    echo "TRAIN_CMD is not set — submit via scripts/submit_all.sh, not sbatch run_experiment.sbatch directly." >&2
    exit 1
fi

source ~/.bashrc
module load cuda/12.6
module load gcc/9.4.0
conda activate tribe

cd /work/pi_sniekum_umass_edu/ychittepu/Codes/Tribe
export HF_HOME=/scratch4/workspace/ychittepu_umass_edu-tribe/cache
export WANDB_PROJECT=TRIBE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Model is already fully cached from many prior runs — without this, from_pretrained still hits the Hub
# over the network for ETag/revalidation checks even on a cache hit, and concurrent jobs racing on that
# network path against the same shared NFS-backed HF_HOME is exactly what's caused repeated
# "does not appear to have a file named ..." failures when >1 job starts around the same time. Offline
# mode reads only the local cache, no network involved, so there's nothing left to race on.
export HF_HUB_OFFLINE=1
# No WANDB_API_KEY here: wandb auth is expected to already be configured (`wandb login` / ~/.bashrc), not
# hardcoded in a git-tracked script file the way the reference script this was adapted from did.

# Free port for accelerate launch's rendezvous (multi-GPU jobs only — harmless to compute unconditionally
# even for single-GPU/RAFT jobs that never reference $MASTER_PORT). Needed because SLURM's --nodes=1
# --gpus=2 doesn't guarantee exclusive use of a physical node: if the cluster's nodes have more than 2
# GPUs, two of our 2-GPU jobs can land on the same node at once, and accelerate launch's fixed default
# port would then collide between them.
export MASTER_PORT="$(
    comm -23 \
        <(seq 10000 65535 | sort) \
        <(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
        shuf | head -n 1
)"

echo "Running: $TRAIN_CMD"
eval "$TRAIN_CMD"
