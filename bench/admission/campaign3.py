"""Stage-1 clean campaign: fresh process per run, randomized order per repeat, resumable.
usage: python3 campaign3.py <repeat> [...]"""
import glob, os, random, subprocess, sys
J = "2168195"; D = "/scratch/general/vast/u1652955/tpch_sf100"; O = "results5"
REF = "/scratch/general/vast/u1652955/conc_bench/results2/sf100gpu_rtx_c1.json"
HEAVY, LIGHT = "1,9,13,18,21", "2,3,4,5,6,7,8,10,11,12,14,15,16,17,19,20,22"
PLAN = ["--admit-structural", "1.0", "--admit-structural-field", "min"]
OURS2 = PLAN + ["--admit-learned", "1.15", "--admit-runtime", "--admit-aging-ms", "2000", "--task-chain-history", "3"]
P = {"none": [], "cap4": ["--max-conc", "4"], "plan": PLAN, "plan_chain3": PLAN + ["--task-chain-history", "3"],
     "ours2": OURS2, "ours2_bf": OURS2 + ["--admit-bestfit"]}
for ms in (500, 1000, 4000):
    P[f"ours2_age{ms}"] = [x if x != "2000" else str(ms) for x in OURS2]
S = {
  "warm_full": (["--admit-gb-auto", "0.90"], ["none", "cap4", "plan", "plan_chain3", "ours2", "ours2_bf", "ours2_age500", "ours2_age1000", "ours2_age4000"]),
  "warm_f75":  (["--gpu-frac", "0.75", "--admit-gb-auto", "0.75"], ["none", "cap4", "plan", "plan_chain3", "ours2", "ours2_bf", "ours2_age500", "ours2_age1000", "ours2_age4000"]),
  "cold_f75":  (["--no-warm", "--gpu-frac", "0.75", "--admit-gb-auto", "0.75"], ["none", "cap4", "plan", "plan_chain3", "ours2", "ours2_bf"]),
  "mixed":     (["--groups", f"{HEAVY}:4;{LIGHT}:12", "--rounds", "2", "--admit-gb-auto", "0.90"], ["none", "cap4", "plan", "plan_chain3", "ours2", "ours2_bf"]),
  "shift_f75": (["--phases", f"{LIGHT};{HEAVY}", "--gpu-frac", "0.75", "--admit-gb-auto", "0.75"], ["none", "cap4", "plan", "plan_chain3", "ours2", "ours2_bf"]),
}
ENV = ["env", "-i", f"HOME={os.environ['HOME']}", "USER=" + os.environ.get("USER", ""), "PATH=/usr/local/bin:/usr/bin:/bin",
       "PIXI_CACHE_DIR=/scratch/general/vast/u1652955/pixi-cache"] + [f"{k}={v}" for k, v in os.environ.items() if k.startswith(("SLURM_", "CUDA_"))]
PIXI = ["/scratch/general/vast/u1652955/pixi/bin/pixi", "run", "--frozen", "--manifest-path", "../sirius/pixi.toml"]
os.makedirs(O, exist_ok=True)
def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).stdout
for rep in sys.argv[1:]:
    jobs = [("tpch", s, p) for s, (_, ps) in S.items() for p in ps]
    jobs += [("broker", pin, mode) for pin in ("host", "gpu") for mode in ("concurrent", "broker")]
    random.Random(100 + int(rep)).shuffle(jobs)
    for kind, s, p in jobs:
        tag = f"{s}__{p}__r{rep}" if kind == "tpch" else f"broker_{s}__{p}__r{rep}"
        if glob.glob(f"{O}/{tag}*.json"):
            print(f"{tag}: exists", flush=True); continue
        if kind == "tpch":
            cmd = ["srun", "-M", "granite", f"--jobid={J}", "--overlap", "-c", "16", "./run2.sh", f"../{O}/{tag}", D, "16",
                   "--pin", "gpu", "--memlog", "--querylog", "--serial-ref", REF] + S[s][0] + P[p]
            out = run(cmd); line = " ".join(l for l in out.splitlines() if "clients=" in l or "rror" in l)[:220]
        else:
            cmd = ["srun", "-M", "granite", f"--jobid={J}", "--overlap", "-c", "16"] + ENV + PIXI + [
                   "python", "fuse_broker.py", "--data", D, "--pin", s, "--host-gb", "24", "--clients", "16",
                   "--per-client", "4" if s == "host" else "8", "--kmax", "16", "--mode", p, "--out", f"{O}/{tag}"]
            out = run(cmd); line = " ".join(l for l in out.splitlines() if "mode=" in l or "rror" in l)[:220]
        print(f"{tag}: {line}", flush=True)
    print(f"REPEAT_DONE {rep}", flush=True)
print("CAMPAIGN3DONE", flush=True)
