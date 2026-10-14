"""Final evaluation sweeps (resumable, randomized, fresh process per run).
usage: python3 campaign7.py <slurm_jobid> [sections, default ABCDE]
A budget sweep | B concurrency sweep | C oracle cap | D open loop | E TPC-DS"""
import glob, os, random, subprocess, sys
J = sys.argv[1]; SECTIONS = sys.argv[2] if len(sys.argv) > 2 else "ABCDE"; O = "results12"
D100, DDS = "/scratch/general/vast/u1652955/tpch_sf100", "/scratch/general/vast/u1652955/tpcds_sf10"
REF100 = "/scratch/general/vast/u1652955/conc_bench/results2/sf100gpu_rtx_c1.json"
LIGHT, HEAVY = "2,3,4,5,6,7,8,10,11,12,14,15,16,17,19,20,22", "1,9,13,18,21"
def pc3(frac): return ["--admit-gb-auto", str(frac), "--admit-structural", "1.0", "--admit-structural-field", "min", "--task-chain-history", "3"]
POL = lambda frac: {"cap4": ["--max-conc", "4"], "plan_chain3": pc3(frac), "manager": ["--memory-manager"]}
G = ["--pin", "gpu", "--host-gb", "24"]
jobs = []  # (tag, data, clients, args)
if "A" in SECTIONS:
    for f in (0.95, 0.85, 0.75, 0.70, 0.65):
        for p, a in POL(f).items():
            jobs.append((f"budget{int(f*100)}__{p}__r1", D100, 16, G + ["--serial-ref", REF100, "--gpu-frac", str(f)] + a))
if "B" in SECTIONS:
    for c in (4, 8, 16, 32, 64):
        for p in ("cap4", "manager"):
            jobs.append((f"clients{c}__{p}__r1", D100, c, G + ["--serial-ref", REF100, "--gpu-frac", "0.75"] + POL(0.75)[p]))
if "C" in SECTIONS:
    for scen, extra, f in (("warm_full", [], 0.90), ("warm_f75", ["--gpu-frac", "0.75"], 0.75),
                           ("shift_f75", ["--phases", f"{LIGHT};{HEAVY}", "--gpu-frac", "0.75"], 0.75)):
        for cap in (2, 3, 4, 6, 8):
            jobs.append((f"oracle_{scen}__cap{cap}__r1", D100, 16, G + ["--serial-ref", REF100] + extra + ["--max-conc", str(cap)]))
if "D" in SECTIONS:
    for rate in (4, 8, 12, 16):
        for p in ("cap4", "plan_chain3", "manager"):
            jobs.append((f"openloop{rate}__{p}__r1", D100, 64, G + ["--serial-ref", REF100, "--gpu-frac", "0.75",
                         "--arrival", f"poisson:{rate}:240", "--workers", "64"] + POL(0.75)[p]))
if "E" in SECTIONS:
    for r in (1, 2, 3):
        for scen, extra, f in (("tpcds_full", [], 0.90), ("tpcds_f75", ["--gpu-frac", "0.75"], 0.75)):
            for p, a in POL(f).items():
                jobs.append((f"{scen}__{p}__r{r}", DDS, 16, G + ["--benchmark", "tpcds", "--rounds", "2"] + extra + a))
random.Random(700).shuffle(jobs)
os.makedirs(O, exist_ok=True)
print(f"{len(jobs)} runs", flush=True)
for tag, data, c, args in jobs:
    if glob.glob(f"{O}/{tag}_c*.json"): print(f"{tag}: exists", flush=True); continue
    out = subprocess.run(["srun", "-M", "granite", f"--jobid={J}", "--overlap", "-n", "1", "-c", "16", "./run2.sh", f"../{O}/{tag}",
                          data, str(c), "--memlog", "--querylog"] + args,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).stdout
    res = [l for l in out.splitlines() if "clients=" in l]
    nerr = sum(l.count("fallback to DuckDB") for l in out.splitlines())
    print(f"{tag}: " + (res[0][:200] if res else "NO RESULT") + (f" [{nerr} fallback msgs]" if nerr else ""), flush=True)
print("CAMPAIGN7DONE", flush=True)
