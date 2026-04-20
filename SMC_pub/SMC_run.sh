#!/bin/bash
#SBATCH --job-name=smc_bilstm_reg
#SBATCH --output=logs/smc_bilstm_reg_%j.out
#SBATCH --error=logs/smc_bilstm_reg_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

module purge
module load python

# Load a real CUDA toolkit module and cuDNN module on Longleaf.
# Pick the versions you actually have available / that match your environment.
module load cuda/12.9
module load cudnn/9.11.0

cd ~/SMC_pub
source venv/bin/activate

mkdir -p logs outputs
RUN_DIR="outputs/run_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

# Try to infer the CUDA toolkit root from nvcc first.
if command -v nvcc >/dev/null 2>&1; then
    CUDA_TOOLKIT_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
else
    CUDA_TOOLKIT_ROOT=""
fi

# Fallback: search for libdevice on the cluster if nvcc path is not enough.
if [ -z "${CUDA_TOOLKIT_ROOT}" ] || [ ! -d "${CUDA_TOOLKIT_ROOT}/nvvm/libdevice" ]; then
    LIBDEVICE_PATH="$(find /nas/longleaf/apps -path '*/nvvm/libdevice/libdevice*.bc' 2>/dev/null | head -n 1 || true)"
    if [ -n "${LIBDEVICE_PATH}" ]; then
        CUDA_TOOLKIT_ROOT="$(dirname "$(dirname "$(dirname "${LIBDEVICE_PATH}")")")"
    fi
fi

if [ -z "${CUDA_TOOLKIT_ROOT}" ] || [ ! -d "${CUDA_TOOLKIT_ROOT}/nvvm/libdevice" ]; then
    echo "ERROR: Could not find CUDA toolkit root with nvvm/libdevice"
    exit 1
fi

export CUDA_HOME="${CUDA_TOOLKIT_ROOT}"
export CUDA_DIR="${CUDA_TOOLKIT_ROOT}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_TOOLKIT_ROOT}"
export PATH="${CUDA_TOOLKIT_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_TOOLKIT_ROOT}/lib64:${LD_LIBRARY_PATH:-}"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo "nvcc: $(which nvcc || echo 'not found')"
echo "CUDA_TOOLKIT_ROOT: ${CUDA_TOOLKIT_ROOT}"
echo "Checking libdevice:"
ls -l "${CUDA_TOOLKIT_ROOT}/nvvm/libdevice" | head

python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY

python BiLSTM_regressor.py