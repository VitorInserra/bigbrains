#!/bin/bash
#SBATCH --job-name=smc_bilstm_reg
#SBATCH --output=smc_bilstm_reg_%j.out
#SBATCH --error=smc_bilstm_reg_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

module purge
module load python
module load cuda/12.9
module load cudnn/9.11.0

cd ~/bigbrains
source venv/bin/activate
python -m pip install --no-cache-dir -r requirements.txt
cd SMC_pub


mkdir -p outputs

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working dir: $(pwd)"

export PYTHONUNBUFFERED=1

echo "==== CUDA CHECK ===="
nvidia-smi
python - <<'PY'
import tensorflow as tf
print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs:", tf.config.list_physical_devices('GPU'))
PY


python -u BiLSTM_regressor.py