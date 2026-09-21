#!/bin/bash
# Throttled submitter: keeps at most MAX_INFLIGHT of your own SLURM jobs pending/running at once, and
# submits the next one from the chosen suite as soon as a vacancy opens up, instead of firing off all 12
# at once like submit_all.sh/submit_all_2gpu.sh do.
#
# Doesn't redefine the 12 job specs — sources the existing suite script with TRICKLE=1 set, which makes
# that script define NAMES/CMDS/WANTED/submit_job without eagerly submitting everything (see the
# `if [ -z "${TRICKLE:-}" ]` guard at the bottom of each), then drives submit_job itself under the cap.
#
# "In-flight" = every one of your jobs squeue still shows (PENDING or RUNNING) — not just RUNNING. This
# intentionally throttles submission itself (so you never have more than MAX_INFLIGHT sitting in the
# queue at all), not just concurrent GPU usage. If you'd rather let extra jobs queue in SLURM and only cap
# actual GPU usage, change running_count() below to add `-t RUNNING`.
#
# Usage: bash slurm/trickle_submit.sh {1gpu|2gpu} [experiment_name ...]
#   No experiment names: trickles all 12 from that suite.
#   With names: trickles only those (e.g. resubmitting a handful of failed runs under the same cap).
#
# Intended to be left running in the background or inside tmux/screen — it blocks until every named job
# has been submitted, which can take hours:
#   nohup bash slurm/trickle_submit.sh 2gpu > slurm/trickle.log 2>&1 &
#
# Run from the Tribe/ repo root.

set -euo pipefail

MAX_INFLIGHT=6
POLL_INTERVAL=60

if [ "$#" -lt 1 ]; then
    echo "Usage: bash slurm/trickle_submit.sh (1gpu|2gpu) [experiment_name ...]" >&2
    exit 1
fi
SUITE="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$SUITE" in
    1gpu) SUITE_SCRIPT="$REPO_ROOT/slurm/submit_all.sh" ;;
    2gpu) SUITE_SCRIPT="$REPO_ROOT/slurm/submit_all_2gpu.sh" ;;
    *)
        echo "Unknown suite '$SUITE' (expected 1gpu or 2gpu)" >&2
        exit 1
        ;;
esac

export TRICKLE=1
source "$SUITE_SCRIPT" "$@"

running_count() {
    squeue -u "$USER" -h -o "%A" | wc -l
}

echo "Trickling ${#NAMES[@]} job(s) from the $SUITE suite, max $MAX_INFLIGHT of your jobs pending/running at once."
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    if [ "${#WANTED[@]}" -gt 0 ]; then
        skip=true
        for w in "${WANTED[@]}"; do
            [ "$w" == "$name" ] && skip=false
        done
        [ "$skip" == true ] && continue
    fi

    while [ "$(running_count)" -ge "$MAX_INFLIGHT" ]; do
        sleep "$POLL_INTERVAL"
    done
    submit_job "$name" "${CMDS[$i]}"
done

echo "All requested jobs submitted."
