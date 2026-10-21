#!/usr/bin/env bash
# New run only; launch.sh owns locks, snapshots, phases and checkpoint resume.
set -euo pipefail
umask 077
MC_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MC_WORKSPACE="${GENIAC_WORKSPACE:-$HOME/geniac_math}"
MC_ID="${1:-math_curriculum_20261021_3000_v1}"
MC_SOURCE="${2:-math_curriculum_20261021_v1}"
[[ "$MC_ID" =~ ^math_curriculum_[a-zA-Z0-9_-]+$ && "$MC_SOURCE" =~ ^math_curriculum_[a-zA-Z0-9_-]+$ && "$MC_ID" != "$MC_SOURCE" ]] || { echo 'Use distinct safe math_curriculum_ run IDs'; exit 2; }
test -f "$MC_WORKSPACE/math_curriculum/$MC_SOURCE/stage2/training/rft/complete.json" || { echo "Completed source run missing: $MC_SOURCE"; exit 1; }
# Stop only this run's old coordinator, preserving checkpoints/completed phases.
exec bash "$MC_HERE/start_submission_first.sh" "$MC_ID" "$MC_SOURCE"
