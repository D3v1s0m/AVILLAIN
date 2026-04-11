#!/bin/bash
# Simple shell script to run the Multi-Agent Fact-Checking Pipeline
# Usage: sbatch ./scripts/run_multi_agent_pipeline.sh [config_file] [start_idx] [end_idx]

#SBATCH --job-name=villain-ma-pipe-qwen
#SBATCH --partition=gpu-bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:hopper:1
#SBATCH --mem=25G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

# Default config file
CONFIG_FILE="${1:-scripts/cfg/default.yaml}"

# Optional start/end indices for processing a subset
START_IDX="${2:-}"
END_IDX="${3:-}"

echo "==========================================="
echo "SLURM Job: Villain Multi-Agent Pipeline"
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

# nvcc --version | grep "release"
# Cuda compilation tools, release 12.4, V12.4.131
echo "CUDA Version: $(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)"

echo "=========================================="
echo "Multi-Agent Fact-Checking Pipeline"
echo "=========================================="
echo "Config: $CONFIG_FILE"

## Build command
#CMD="python src/run_multi_agent.py --config $CONFIG_FILE"
#
## Add optional index arguments
#if [ -n "$START_IDX" ]; then
#    CMD="$CMD --start_idx $START_IDX"
#    echo "Start index: $START_IDX"
#fi
#
#if [ -n "$END_IDX" ]; then
#    CMD="$CMD --end_idx $END_IDX"
#    echo "End index: $END_IDX"
#fi
#
#echo ""
#echo "Running: $CMD"
#echo ""
#
#eval $CMD

CMD_ARGS="--config $CONFIG_FILE"
if [ -n "$START_IDX" ]; then
    CMD_ARGS="$CMD_ARGS --start_idx $START_IDX"
fi
if [ -n "$END_IDX" ]; then
    CMD_ARGS="$CMD_ARGS --end_idx $END_IDX"
fi

python -u src/run_multi_agent.py $CMD_ARGS

echo "Finished: $(date)"