#!/usr/bin/env bash
# usage: run2.sh <out_tag> <data> <clients> [run_conc args...]   clean env (no spack gcc module), no nsys
TAG=$1; DATA=$2; C=$3; shift 3
cd /scratch/general/vast/u1652955/conc_bench
env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin PIXI_CACHE_DIR=/scratch/general/vast/u1652955/pixi-cache \
  OMP_NUM_THREADS=16 $(env | grep -E "^(SLURM_|CUDA_|GPU_DEVICE_ORDINAL|TMPDIR=)" | tr "\n" " ") \
  /scratch/general/vast/u1652955/pixi/bin/pixi run --frozen --manifest-path ../sirius/pixi.toml \
  python run_conc.py --data $DATA --clients $C --out results2/${TAG}_c${C} "$@" 2>&1 | grep -E "^clients=|serial sum|rror" | grep -v "^  #"
