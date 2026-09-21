#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=superpod-a100
#SBATCH --mem=80G

# Same as run_eval_gsm8k.sbatch, but uses the `llm_gen_new` conda env (transformers 5.17.0 / vllm 0.29.0)
# instead of the pinned `llm_gen` env (transformers 4.45.2 / vllm 0.6.4.post1), for checkpoints whose
# architecture the pinned env's transformers is too old to recognize (e.g. Gemma-3's `gemma3_text`). Do
# NOT upgrade `llm_gen` itself — see slurm/run_experiment.sbatch's own comments on the RoPE-config
# writer/reader version-mismatch precedent from mixing transformers versions across this project's envs.
set -euo pipefail

if [ -z "${EVAL_CMD:-}" ]; then
    echo "EVAL_CMD is not set — submit via slurm/submit_eval_gsm8k.sh, not sbatch run_eval_gsm8k_newenv.sbatch directly." >&2
    exit 1
fi

source ~/.bashrc
module load cuda/12.6
module load gcc/9.4.0
conda activate llm_gen_new

cd /work/pi_sniekum_umass_edu/ychittepu/Codes/Tribe
export HF_HOME=/scratch4/workspace/ychittepu_umass_edu-tribe/cache
export HF_HUB_OFFLINE=1
# torch in this env is built against CUDA 13 (no bundled libcudart.so.12), but flashinfer's JIT-compiled
# kernels are built against the module-loaded cuda/12.6 toolkit and need libcudart.so.12 at runtime — the
# cuda/12.6 module doesn't add its lib dir to LD_LIBRARY_PATH here, so do it explicitly.
export LD_LIBRARY_PATH=/modules/opt/linux-ubuntu24.04-x86_64/nvhpc/Linux_x86_64/24.9/cuda/12.6/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

echo "Running: $EVAL_CMD"
eval "$EVAL_CMD"
