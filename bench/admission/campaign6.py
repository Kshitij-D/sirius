"""Clean sweep: unified memory manager vs baselines. 3 repeats, fresh process per run, randomized, resumable.
usage: python3 campaign6.py <slurm_jobid> <repeat> [<repeat> ...]"""
import glob, os, random, subprocess, sys
J = sys.argv[1]; O = "results11"
D100, D10 = "/scratch/general/vast/u1652955/tpch_sf100", "/scratch/general/vast/u1652955/tpch_sf10"
REF100 = "/scratch/general/vast/u1652955/conc_bench/results2/sf100gpu_rtx_c1.json"
HEAVY, LIGHT = "1,9,13,18,21", "2,3,4,5,6,7,8,10,11,12,14,15,16,17,19,20,22"
PC3 = ["--admit-gb-auto", "0.90", "--admit-structural", "1.0", "--admit-structural-field", "min", "--task-chain-history", "3"]
P = {"cap4": ["--max-conc", "4"], "plan_chain3": PC3, "manager": ["--memory-manager"]}
G = ["--pin", "gpu", "--serial-ref", REF100, "--host-gb", "24"]
S = {
  "warm_full": (D100, G, list(P)),
  "warm_f75":  (D100, G + ["--gpu-frac", "0.75"], list(P)),
  "shift_f75": (D100, G + ["--phases", f"{LIGHT};{HEAVY}", "--gpu-frac", "0.75"], list(P)),
  "mixed":     (D100, G + ["--groups", f"{HEAVY}:4;{LIGHT}:12", "--rounds", "2"], list(P)),
  "host_sf10": (D10, ["--pin", "host", "--host-gb", "16", "--rounds", "2"], list(P)),
  "cotenant":  (D100, G + ["--gpu-frac", "0.90", "--cotenant", "3,20,10", "--rounds", "2"], list(P)),
}
os.makedirs(O, exist_ok=True)
def run(cmd): return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).stdout
for rep in sys.argv[2:]:
    jobs = [(s, p) for s, (_, _, ps) in S.items() for p in ps]
    random.Random(600 + int(rep)).shuffle(jobs)
    for s, p in jobs:
        tag = f"{s}__{p}__r{rep}"
        if glob.glob(f"{O}/{tag}_c*.json"): print(f"{tag}: exists", flush=True); continue
        data, args, _ = S[s]
        extra = list(P[p])
        if s == "warm_f75" or s == "shift_f75":
            extra = [x if x != "0.90" else "0.75" for x in extra]   # admission budget follows the GPU budget
        out = run(["srun", "-M", "granite", f"--jobid={J}", "--overlap", "-n", "1", "-c", "16", "./run2.sh", f"../{O}/{tag}", data, "16",
                   "--memlog", "--querylog"] + args + extra)
        res = [l for l in out.splitlines() if "clients=" in l]
        nerr = sum(l.count("fallback to DuckDB") for l in out.splitlines())
        print(f"{tag}: " + (res[0][:200] if res else "NO RESULT") + (f" [{nerr} fallback msgs]" if nerr else ""), flush=True)
    print(f"REPEAT_DONE {rep}", flush=True)
print("CAMPAIGN6DONE", flush=True)
