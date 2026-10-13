#!/usr/bin/env bash
cd /scratch/general/vast/u1652955/conc_bench
J=2168195; D=/scratch/general/vast/u1652955/tpch_sf100
run() { echo -n "$1: "; shift; ./srun2.sh $J "$@" | grep -E "clients=|rror" | tr '\n' ' ' | cut -c1-230; echo; }
for r in 1 2; do
  for c in 4 6 10 12; do run "r=$r cap$c" adm_cap${c}_r${r} $D 16 --pin gpu --memlog --max-conc $c; done
done
# smaller GPU budget: 75% of 95.6 GB = 71.7 GB; admission threshold = budget - 2 GB (derived, not tuned)
for r in 1 2; do
  run "r=$r f75 cap8"     adm_f75_cap8_r${r}     $D 16 --pin gpu --memlog --gpu-frac 0.75 --max-conc 8
  run "r=$r f75 measured" adm_f75_measured_r${r} $D 16 --pin gpu --memlog --gpu-frac 0.75 --admit-gb 69.7 --admit-footprints results2/sf100gpu_rtx_c1.fp.tsv --admit-default-gb 4
  run "r=$r f75 plan"     adm_f75_plan_r${r}     $D 16 --pin gpu --memlog --gpu-frac 0.75 --admit-gb 69.7 --admit-plan-coefs 1.56533e+09,0,0,0,0,0,0.0171946,0.0560658,0 --admit-default-gb 4
done
echo ADMIT2DONE
