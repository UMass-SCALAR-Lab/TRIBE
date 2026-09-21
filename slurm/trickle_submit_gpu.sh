#!/bin/bash
# GPU-COUNT-aware throttled submitter: keeps total REQUESTED GPUs (across every one of your own
# pending+running jobs) under MAX_GPUS, submitting the next job from a job-list file as soon as there's
# enough headroom. Unlike slurm/trickle_submit.sh (which caps JOB COUNT, assuming every job is the same
# size and wired to the on-policy suite's own NAMES/CMDS contract), this handles a mixed fleet of
# differently-sized jobs (e.g. 1-GPU RAFT/DPO alongside 2-GPU accelerate-launch runs) against a fixed
# personal/lab GPU budget.
#
# GPU accounting uses `squeue -O tres-per-job` (the GPUs a job REQUESTED at submission, e.g. "gres/gpu:2")
# rather than AllocTRES (actually-allocated GPUs) — AllocTRES is empty for jobs still PENDING, which would
# undercount and let this over-submit past the real budget once those pending jobs start.
#
# Job list format (one per line, '|'-delimited; blank lines and lines starting with # are skipped):
#   <num_gpus>|<job_name>|<shell command to run as TRAIN_CMD inside run_experiment.sbatch>
#
# Usage: bash slurm/trickle_submit_gpu.sh <jobs_file> [MAX_GPUS]
#   Intended to be left running in the background/tmux — it blocks until every job is submitted, which can
#   take a long time:
#   nohup bash slurm/trickle_submit_gpu.sh slurm/jobs_offpolicy_gsm8k.txt 12 > slurm/trickle_gpu.log 2>&1 &
#
# Run from the Tribe/ repo root.

set -euo pipefail

JOBS_FILE="${1:?Usage: bash slurm/trickle_submit_gpu.sh <jobs_file> [MAX_GPUS]}"
MAX_GPUS="${2:-12}"
POLL_INTERVAL=60
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="/scratch4/workspace/ychittepu_umass_edu-tribe/logs"

used_gpus() {
    squeue -u "$USER" -h -O "tres-per-job:60" | grep -oP 'gres/gpu:\K[0-9]+' | awk '{s+=$1} END {print s+0}'
}

submit_job() {
    local gpus="$1" name="$2" cmd="$3"
    sbatch --job-name="$name" --gpus="$gpus" \
        --output="$LOG_DIR/${name}.out" \
        --error="$LOG_DIR/${name}.err" \
        --export=ALL,TRAIN_CMD="$cmd" \
        "$REPO_ROOT/slurm/run_experiment.sbatch"
}

echo "Trickling jobs from $JOBS_FILE, max $MAX_GPUS GPUs in-flight (pending+running) at once."
while IFS= read -r line || [ -n "$line" ]; do
    line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [ -z "$line" ] && continue
    case "$line" in \#*) continue ;; esac

    IFS='|' read -r gpus name cmd <<< "$line"
    gpus="$(echo "$gpus" | xargs)"
    name="$(echo "$name" | xargs)"
    cmd="$(echo "$cmd" | sed 's/^[[:space:]]*//')"

    while [ "$(( $(used_gpus) + gpus ))" -gt "$MAX_GPUS" ]; do
        sleep "$POLL_INTERVAL"
    done
    submit_job "$gpus" "$name" "$cmd"
    echo "Submitted $name ($gpus GPUs) at $(date)"
done < "$JOBS_FILE"

echo "All jobs from $JOBS_FILE submitted."
