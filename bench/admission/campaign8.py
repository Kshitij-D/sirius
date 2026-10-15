"""Final campaign: grant manager vs baselines (paper numbers). Fresh process per run, randomized, resumable.
usage: python3 campaign8.py <slurm_jobid> [sections MS] [plan_scale]
M main comparison (3 repeats) | S sweeps (budget, clients, open loop, oracle cap; 1 repeat)"""
import glob, os, random, subprocess, sys
J = sys.argv[1]; SECTIONS = sys.argv[2] if len(sys.argv) > 2 else "MS"; PSCALE = sys.argv[3] if len(sys.argv) > 3 else "1.0"
O = "results13"
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
if "M" in SECTIONS:
    SC = {  # name -> (data, args, gpu budget fraction for the plan baseline)
      "warm_full": (D100, G + ["--serial-ref", REF100], 0.90),
      "warm_f75":  (D100, G + ["--serial-ref", REF100, "--gpu-frac", "0.75"], 0.75),
      "shift_f75": (D100, G + ["--serial-ref", REF100, "--phases", f"{LIGHT};{HEAVY}", "--gpu-frac", "0.75"], 0.75),
      "mixed":     (D100, G + ["--serial-ref", REF100, "--groups", f"{HEAVY}:4;{LIGHT}:12", "--rounds", "2"], 0.90),
      "host_sf10": (D10, ["--pin", "host", "--host-gb", "16", "--rounds", "2"], 0.90),
      "cotenant":  (D100, G + ["--serial-ref", REF100, "--gpu-frac", "0.90", "--cotenant", "3,20,10", "--rounds", "2"], 0.90),
      "tpcds_full": (DDS, G + ["--benchmark", "tpcds", "--rounds", "20"], 0.90),
      "tpcds_f75":  (DDS, G + ["--benchmark", "tpcds", "--rounds", "20", "--gpu-frac", "0.75"], 0.75),
    }
    for r in (1, 2, 3):
        for scen, (data, args, frac) in SC.items():
            for p, a in POL(frac).items():
                jobs.append((f"{scen}__{p}__r{r}", data, 16, args + a))
if "S" in SECTIONS:
    for f in (0.95, 0.85, 0.75, 0.70, 0.65):
        for p in ("cap4", "grant"):
            jobs.append((f"budget{int(f*100)}__{p}__r1", D100, 16, G + ["--serial-ref", REF100, "--gpu-frac", str(f)] + POL(f)[p]))
    for c in (4, 8, 16, 32, 64):
        for p in ("cap4", "grant"):
            jobs.append((f"clients{c}__{p}__r1", D100, c, G + ["--serial-ref", REF100, "--gpu-frac", "0.75"] + POL(0.75)[p]))
    for rate in (4, 8, 12, 16):
        for p in ("cap4", "plan0.5", "grant"):
            jobs.append((f"openloop{rate}__{p}__r1", D100, 64, G + ["--serial-ref", REF100, "--gpu-frac", "0.75",
                         "--arrival", f"poisson:{rate}:240", "--workers", "64"] + POL(0.75)[p]))
    for scen, extra in (("warm_full", []), ("warm_f75", ["--gpu-frac", "0.75"]),
                        ("shift_f75", ["--phases", f"{LIGHT};{HEAVY}", "--gpu-frac", "0.75"])):
        for cap in (2, 3, 6, 8):
            jobs.append((f"oracle_{scen}__cap{cap}__r1", D100, 16, G + ["--serial-ref", REF100] + extra + ["--max-conc", str(cap)]))
main = [j for j in jobs if not j[0].startswith(("budget", "clients", "openloop", "oracle"))]
sweeps = [j for j in jobs if j not in main]
random.Random(800).shuffle(main); random.Random(801).shuffle(sweeps)
jobs = main + sweeps
os.makedirs(O, exist_ok=True)
print(f"{len(jobs)} runs (plan scale {PSCALE})", flush=True)
for tag, data, c, args in jobs:
    if glob.glob(f"{O}/{tag}_c*.json"): print(f"{tag}: exists", flush=True); continue
    out = subprocess.run(["srun", "-M", "granite", f"--jobid={J}", "--overlap", "-n", "1", "-c", "16", "./run2.sh", f"../{O}/{tag}",
                          data, str(c), "--memlog", "--querylog"] + args,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).stdout
    res = [l for l in out.splitlines() if "clients=" in l]
    nerr = sum(l.count("fallback to DuckDB") for l in out.splitlines())
    print(f"{tag}: " + (res[0][:200] if res else "NO RESULT") + (f" [{nerr} fallback msgs]" if nerr else ""), flush=True)
print("CAMPAIGN8DONE", flush=True)
