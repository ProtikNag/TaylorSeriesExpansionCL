#!/bin/bash
# Generic resumable HTCL array task: one array index runs exactly one job.
#
#   One array index == one job == one (dataset, method, task order), selected by
#   `--job $SLURM_ARRAY_TASK_ID` into the deterministic plan. Jobs checkpoint after
#   every task and write DONE when finished, so a requeued/resubmitted task resumes
#   from its checkpoint instead of restarting.
#
# No cluster-, account- or user-specific names live in this file. Partition, account,
# time limit and GPU request are passed at submit time via Slurm's own env vars, e.g.
#   export SBATCH_PARTITION=<partition>            # required
#   export SBATCH_ACCOUNT=<account>                # only if the partition needs one
#   export SBATCH_TIMELIMIT=<d-hh:mm:ss>           # only if the partition enforces a limit
#   export HTCL_DEVICE=cpu|cuda                     # engine device (default cpu)
#   sbatch --array=0-59%32 [--gres=gpu:1] scripts/slurm/array.sbatch configs/storyline.yaml [overrides...]
#
# The %N in --array caps simultaneously running tasks. Logs land under the git-ignored
# runs/slurm_logs/ (create it before submitting: mkdir -p runs/slurm_logs).
#SBATCH --job-name=htcl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/slurm_logs/%x-%A_%a.out
#SBATCH --error=runs/slurm_logs/%x-%A_%a.out
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
CONFIG=${1:?usage: sbatch array.sbatch CONFIG [overrides...]}; shift || true
PY="$REPO/.venv/bin/python"
export HTCL_DEVICE=${HTCL_DEVICE:-cpu}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
IDX=${SLURM_ARRAY_TASK_ID:?this script must be run as a Slurm array job}
echo "host=$(hostname) array=${SLURM_ARRAY_JOB_ID:-NA} task=$IDX config=$CONFIG device=$HTCL_DEVICE overrides=$* commit=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" || true
exec "$PY" -m htcl.run run "$CONFIG" --job "$IDX" --device "$HTCL_DEVICE" "$@"
