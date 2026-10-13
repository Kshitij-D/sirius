#!/bin/bash
#SBATCH --job-name=sirius-hold
#SBATCH --account=zhaof
#SBATCH --partition=granite-gpu
#SBATCH --qos=granite-gpu-freecycle
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/general/vast/u1652955/jobs/hold_%j.out
echo "node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES start=$(date)"
nvidia-smi --query-gpu=pci.bus_id,name,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader
python3 - <<'PY'
import ctypes
cu = ctypes.CDLL('libcuda.so.1'); r = cu.cuInit(0)
n = ctypes.c_int(); cu.cuDeviceGetCount(ctypes.byref(n))
print('CUDA_CHECK cuInit=%d devices=%d -> %s' % (r, n.value, 'OK' if r == 0 and n.value > 0 else 'BROKEN'), flush=True)
PY
sleep infinity
