#!/usr/bin/env bash
cd /scratch/general/vast/u1652955/conc_bench
J=2168195
./srun2.sh $J fp_sf10_gpu /scratch/general/vast/u1652955/tpch_sf10 1 --pin gpu --footprint-log
./srun2.sh $J fp_sf10_none /scratch/general/vast/u1652955/tpch_sf10 1 --pin none --footprint-log
./srun2.sh $J fp_sf100_none /scratch/general/vast/u1652955/tpch_sf100 1 --pin none --footprint-log
echo DONE
