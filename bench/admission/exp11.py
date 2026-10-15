"""Overcommit-ratio sweep: plan-peak admission scaled by s (overcommit 1/s), no count limit. Is one ratio best across workloads?: grant manager vs baselines (paper numbers). Fresh process per run, randomized, resumable.
usage: python3 campaign8.py <slurm_jobid> [sections MS] [plan_scale]
M main comparison (3 repeats) | S sweeps (budget, clients, open loop, oracle cap; 1 repeat)"""
import glob, os, random, subprocess, sys
J = sys.argv[1]; SECTIONS = sys.argv[2] if len(sys.argv) > 2 else "MS"; PSCALE = sys.argv[3] if len(sys.argv) > 3 else "1.0"
O = "results15"
D100, D10, DDS = ("/scratch/general/vast/u1652955/tpch_sf100", "/scratch/general/vast/u1652955/tpch_sf10",
                  "/scratch/general/vast/u1652955/tpcds_sf10")
REF100 = "/scratch/general/vast/u1652955/conc_bench/results2/sf100gpu_rtx_c1.json"
LIGHT, HEAVY = "2,3,4,5,6,7,8,10,11,12,14,15,16,17,19,20,22", "1,9,13,18,21"
def plan(frac): return ["--admit-gb-auto", str(frac), "--admit-structural", PSCALE, "--admit-structural-field", "min",
                        "--task-chain-history", "3"]
GRANT = ["--env", "SIRIUS_GRANT_MANAGER=1", "--env", "SIRIUS_MPL_FIXED=4", "--env", "SIRIUS_GRANT_LEARNED_MIN=0"]
def planx(frac, sc): return ["--admit-gb-auto", str(frac), "--admit-structural", sc, "--admit-structural-field", "min",
                             "--task-chain-history", "3"]
POL = lambda frac: {"cap4": ["--max-conc", "4"], "plan1.0": planx(frac, "1.0"), "plan0.5": planx(frac, "0.5"),
                    "manager": ["--memory-manager"], "grant": GRANT}
G = ["--pin", "gpu", "--host-gb", "24"]
jobs = []
SC = {
  "warm_full": (D100, G + ["--serial-ref", REF100], 0.90),
  "warm_f75":  (D100, G + ["--serial-ref", REF100, "--gpu-frac", "0.75"], 0.75),
  "shift_f75": (D100, G + ["--serial-ref", REF100, "--phases", f"{LIGHT};{HEAVY}", "--gpu-frac", "0.75"], 0.75),
  "pt8_warm_f75": (D100, G + ["--serial-ref", REF100, "--gpu-frac", "0.75", "--pipe-threads", "8"], 0.75),
  "host_sf10": (D10, ["--pin", "host", "--host-gb", "16", "--rounds", "2"], 0.90),
  "tpcds_f75":  (DDS, G + ["--benchmark", "tpcds", "--rounds", "20", "--gpu-frac", "0.75"], 0.75),
}
for r in (1, 2):
    for scen, (data, args, frac) in SC.items():
        for sc in ("1.0", "0.75", "0.5", "0.35"):
            jobs.append((f"{scen}__s{sc}__r{r}", data, 16, args + planx(frac, sc)))
random.Random(1100).shuffle(jobs)
os.makedirs(O, exist_ok=True)
print(f"{len(jobs)} runs", flush=True)
for tag, data, c, args in jobs:
    if glob.glob(f"{O}/{tag}_c*.json"): print(f"{tag}: exists", flush=True); continue
    out = subprocess.run(["srun", "-M", "granite", f"--jobid={J}", "--overlap", "-t", "20", "-n", "1", "-c", "16", "./run2.sh", f"../{O}/{tag}",
                          data, str(c), "--memlog", "--querylog"] + args,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).stdout
    open(f"{O}/{tag}.log", "w").write(out)
    res = [l for l in out.splitlines() if "clients=" in l]
    print(f"{tag}: " + (res[0][:200] if res else "NO RESULT"), flush=True)
print("EXP11DONE", flush=True)
