#!/bin/bash
#SBATCH --job-name=smc_bilstm_reg
#SBATCH --output=smc_bilstm_reg_%j.out
#SBATCH --error=smc_bilstm_reg_%j.err
#SBATCH --partition=a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

module purge
module load python

cd ~/SMC_pub

source .venv/bin/activate

mkdir -p outputs

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working dir: $(pwd)"

python --version
which python
nvidia-smi || true

python BiLSTM_regressor.py