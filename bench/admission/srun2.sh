#!/usr/bin/env bash
# usage: srun2.sh <jobid> <tag> <data> <clients> [run_conc args]   runs run2.sh inside the GPU job
J=$1; shift
srun -M granite --jobid=$J --overlap -c 16 /scratch/general/vast/u1652955/conc_bench/run2.sh "$@"
