#!/usr/bin/env bash
cd /scratch/general/vast/u1652955/conc_bench
J=2168195; D=/scratch/general/vast/u1652955/tpch_sf100
for r in 1 2 3; do
  for P in none cap8 reactive measured plan; do
    case $P in
      none)     A="" ;;
      cap8)     A="--max-conc 8" ;;
      reactive) A="--admit-gb 80 --admit-gap-ms 50" ;;
      measured) A="--admit-gb 84 --admit-footprints results2/sf100gpu_rtx_c1.fp.tsv --admit-default-gb 4" ;;
      plan)     A="--admit-gb 84 --admit-plan-coefs 1.56533e+09,0,0,0,0,0,0.0171946,0.0560658,0 --admit-default-gb 4" ;;
    esac
    echo -n "r=$r $P: "
    ./srun2.sh $J adm_${P}_r${r} $D 16 --pin gpu --memlog $A | grep -E "clients=|rror" | tr '\n' ' ' | cut -c1-230; echo
  done
done
echo ADMITDONE
