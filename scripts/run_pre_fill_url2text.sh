#!/bin/bash
# Submit with: sbatch scripts/run_pre_fill_url2text.sh

#SBATCH --job-name=villain-fill-url2text
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

echo "==========================================="
echo "SLURM Job: Villain Fill URL to Text Preprocessing"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Workdir: ${PROJECT_ROOT}"
echo "Started: $(date)"
echo "==========================================="

# Activate conda environment directly
set +u
source /home3/qvlw18/miniconda3/etc/profile.d/conda.sh
conda activate villain
set -u

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "Using Python: $(which python)"
python --version
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# Run your script
echo "Running fill URL to text preprocessing..."
bash scripts/pre_fill_url2text.sh

echo "Finished: $(date)"