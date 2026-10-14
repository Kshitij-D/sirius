#!/usr/bin/env bash
# When the replacement GPU job is up (CUDA OK): run the clean sweep (campaign6), then the final
# sweeps (campaign7); finished runs are skipped. Emails at start and at the end of each campaign.
NEW=2170725
cd /scratch/general/vast/u1652955/conc_bench
while true; do
  st=$(squeue -M granite -h -j $NEW -o %T 2>/dev/null | tail -1)
  if [ "$st" = "RUNNING" ] && grep -q "CUDA_CHECK.*OK" /scratch/general/vast/u1652955/jobs/hold_$NEW.out 2>/dev/null; then break; fi
  if [ -z "$st" ]; then echo "job $NEW gone" | mail -s "[sirius] replacement GPU job $NEW disappeared" k.dubey@utah.edu; exit 1; fi
  sleep 60
done
node=$(squeue -M granite -h -j $NEW -o %N | tail -1)
printf "Replacement RTX job %s is running on %s (CUDA OK). Starting campaign6 (manager vs baselines, 3 repeats) then campaign7 (budget/concurrency sweeps, oracle cap, open loop, TPC-DS).\n" $NEW "$node" | mail -s "[sirius] new GPU up ($node), sweeps started" k.dubey@utah.edu
python3 campaign6.py $NEW 1 2 3 > campaign6.out 2>&1
echo "campaign6 finished; $(ls results11/*.json 2>/dev/null | wc -l) results. Starting campaign7." | mail -s "[sirius] campaign6 done" k.dubey@utah.edu
python3 campaign7.py $NEW > campaign7.out 2>&1
echo "campaign7 finished; $(ls results12/*.json 2>/dev/null | wc -l) results." | mail -s "[sirius] campaign7 done" k.dubey@utah.edu
