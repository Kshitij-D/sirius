"""Plateau mechanism: does the throughput plateau follow the GPU executor's thread count?
Also: grant manager's concurrency limit vs shared scans on host-tier data. Fresh process per run, randomized, resumable.
usage: python3 exp9.py <slurm_jobid>"""
import glob, os, random, subprocess, sys
J = sys.argv[1]; O = "results14"
D100, D10 = "/scratch/general/vast/u1652955/tpch_sf100", "/scratch/general/vast/u1652955/tpch_sf10"
REF100 = "/scratch/general/vast/u1652955/conc_bench/results2/sf100gpu_rtx_c1.json"
G = ["--pin", "gpu", "--host-gb", "24", "--serial-ref", REF100]
GR = lambda p: ["--env", "SIRIUS_GRANT_MANAGER=1", "--env", f"SIRIUS_MPL_FIXED={p}", "--env", "SIRIUS_GRANT_LEARNED_MIN=0"]
jobs = []
for pt in (4, 8):
    for scen, extra in (("warm_full", []), ("warm_f75", ["--gpu-frac", "0.75"])):
        for cap in (2, 4, 6, 8):
            jobs.append((f"pt{pt}_{scen}__cap{cap}__r1", D100, 16, G + extra + ["--pipe-threads", str(pt), "--max-conc", str(cap)]))
for p in (4, 8, 16):
    jobs.append((f"host_sf10__grantP{p}__r1", D10, 16, ["--pin", "host", "--host-gb", "16", "--rounds", "2"] + GR(p)))
random.Random(900).shuffle(jobs)
os.makedirs(O, exist_ok=True)
print(f"{len(jobs)} runs", flush=True)
for tag, data, c, args in jobs:
    if glob.glob(f"{O}/{tag}_c*.json"): print(f"{tag}: exists", flush=True); continue
    out = subprocess.run(["srun", "-M", "granite", f"--jobid={J}", "--overlap", "-n", "1", "-c", "16", "./run2.sh", f"../{O}/{tag}",
                          data, str(c), "--memlog", "--querylog"] + args,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).stdout
    open(f"{O}/{tag}.log", "w").write(out)
    res = [l for l in out.splitlines() if "clients=" in l]
    print(f"{tag}: " + (res[0][:200] if res else "NO RESULT"), flush=True)
print("EXP9DONE", flush=True)
