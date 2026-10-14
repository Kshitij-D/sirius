# Sirius admission study — handoff 2026-10-13 21:55

Doc: https://claude.ai/artifact/4ouVLqH6wbM3ZLjx4dxXBJ (results + roadmap)
GPU: RTX PRO 6000 job 2168195 grn077 (ends ~01:10 Oct 14); H200 job 2158657 queued.

Confirmed (results7, 3 repeats, fresh process, randomized; bench/admission/results7/summary.csv):
- warm_f75: plan_chain3 1.37x [1.35-1.39] 0 OOM vs cap4 1.11x [0.75-1.48] 136 OOM 1 fb
- shift_f75: plan_chain3 1.36x 0 OOM vs cap4 0.71x 1838 OOM 3 fb
- warm_full: ours2_age500 1.78x [1.76-1.82] p99 2.3s vs cap4 1.69x [1.37-1.89] p99 4.6s
- mixed tie (2.13 vs 2.10); host_sf10 +5% (2.43 vs 2.31)
- cotenant+drain: OOM 2242 -> 428, fb 7 -> 2 (partial)
- broker: host 18.2 vs 5.6 q/s (3.3x), gpu 24.0 vs 65.7 (fusion 2.7x worse)

Layers (env-gated, see sirius_context.hpp comments): plan walk v2 (SIRIUS_ADMIT_STRUCTURAL[_FIELD=min]),
engine ledger (SIRIUS_ADMIT_LEARNED, _RUNTIME, _AGING_MS, _BESTFIT, _DEVICE_MARGIN_GB), chain history
(SIRIUS_TASK_CHAIN_HISTORY 1-3), op priors (SIRIUS_TASK_OP_PRIORS), stagger (SIRIUS_STAGGER_HEAVY),
runtime budget (SIRIUS_BUDGET_FILE, SIRIUS_BUDGET_DRAIN_MS; cuCascade patch cucascade_runtime_budget.patch).
Drivers: run_conc.py, run2.sh/srun2.sh, campaign3/4/5.py, summarize.py, fuse_broker.py, hog.cu.

Next: A) rescale chain history on budget shrink (co-tenant OOMs); C) results section + figures;
then replication on other GPUs (same campaign5 configs, no retuning); stage 2 deferred items.
Gotchas: never rebuild during a campaign; killing a driver leaves its srun step running (kill it);
build in env -i after sccache --stop-server; scratch purge (refresh timestamps).
