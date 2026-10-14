#!/usr/bin/env bash
cd /scratch/general/vast/u1652955/conc_bench
python3 run_tests.py 2168195 default
python3 run_tests.py 2168195 manager
echo TESTSDONE
