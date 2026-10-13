#!/usr/bin/env bash
# Resumable admission campaign: skips any run whose JSON already exists.
# usage: campaign.sh <jobid>
cd /scratch/general/vast/u1652955/conc_bench
J=${1:-2168195}; D=/scratch/general/vast/u1652955/tpch_sf100; O=results3; mkdir -p $O
HEAVY="1,9,13,18,21"; LIGHT="2,3,4,5,6,7,8,10,11,12,14,15,16,17,19,20,22"
declare -A POL=(
  [none]=""
  [cap4]="--max-conc 4"
  [learned]="--admit-learned 1.15 --admit-default-gb 4"
  [learned_rt]="--admit-learned 1.15 --admit-default-gb 4 --admit-runtime"
  [ours]="--admit-learned 1.15 --admit-default-gb 4 --admit-runtime --admit-aging-ms 2000 --admit-device-margin-gb 2"
)
ORDER="none cap4 learned learned_rt ours"
declare -A SCN=(
  [full]="16 --admit-gb-auto 0.90"
  [f75]="16 --gpu-frac 0.75 --admit-gb-auto 0.75"
  [mixed]="16 --groups $HEAVY:4;$LIGHT:12 --rounds 2 --admit-gb-auto 0.90"
  [cotenant]="16 --hog 5,20,15 --admit-gb-auto 0.90"
)
for S in full f75 mixed cotenant; do
  for r in 1 2; do
    for P in $ORDER; do
      tag=${S}_${P}_r${r}
      ls $O/${tag}_c*.json >/dev/null 2>&1 && continue
      set -- ${SCN[$S]}; C=$1; shift
      echo -n "$tag: "
      srun -M granite --jobid=$J --overlap -c 16 ./run2.sh "../$O/$tag" $D $C --pin gpu --memlog --querylog "$@" ${POL[$P]} \
        | grep -E "clients=|rror" | tr '\n' ' ' | cut -c1-230; echo
    done
  done
  echo "SCENARIO_DONE $S"
done
echo CAMPAIGNDONE
