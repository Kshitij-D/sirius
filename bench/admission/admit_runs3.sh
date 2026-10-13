#!/usr/bin/env bash
cd /scratch/general/vast/u1652955/conc_bench
J=2168195; D=/scratch/general/vast/u1652955/tpch_sf100
FP=results2/sf100gpu_rtx_c1.fp.tsv
run() { echo -n "$1: "; shift; ./srun2.sh $J "$@" --pin gpu --memlog --querylog | grep -E "clients=|rror" | tr '\n' ' ' | cut -c1-230; echo; }
for r in 1 2; do
  run "r=$r full none"        a3_full_none_r$r      $D 16
  run "r=$r full cap4"        a3_full_cap4_r$r      $D 16 --max-conc 4
  run "r=$r full measured"    a3_full_measured_r$r  $D 16 --admit-gb 84 --admit-footprints $FP --admit-default-gb 4
  run "r=$r full learned1.0"  a3_full_learn10_r$r   $D 16 --admit-gb 84 --admit-learned 1.0 --admit-default-gb 4
  run "r=$r full learned1.15" a3_full_learn115_r$r  $D 16 --admit-gb 84 --admit-learned 1.15 --admit-default-gb 4
  run "r=$r f75 cap4"         a3_f75_cap4_r$r       $D 16 --gpu-frac 0.75 --max-conc 4
  run "r=$r f75 measured"     a3_f75_measured_r$r   $D 16 --gpu-frac 0.75 --admit-gb 69.7 --admit-footprints $FP --admit-default-gb 4
  run "r=$r f75 learned1.0"   a3_f75_learn10_r$r    $D 16 --gpu-frac 0.75 --admit-gb 69.7 --admit-learned 1.0 --admit-default-gb 4
  run "r=$r f75 learned1.15"  a3_f75_learn115_r$r   $D 16 --gpu-frac 0.75 --admit-gb 69.7 --admit-learned 1.15 --admit-default-gb 4
done
echo ADMIT3DONE
