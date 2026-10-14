"""Confirmation campaign: paper configurations only, 3 repeats, fresh process per run, randomized, resumable."""
import glob, os, random, subprocess, sys
J = "2168195"; O = "results7"
D100, D10 = "/scratch/general/vast/u1652955/tpch_sf100", "/scratch/general/vast/u1652955/tpch_sf10"
REF100 = "/scratch/general/vast/u1652955/conc_bench/results2/sf100gpu_rtx_c1.json"
HEAVY, LIGHT = "1,9,13,18,21", "2,3,4,5,6,7,8,10,11,12,14,15,16,17,19,20,22"
PC3 = ["--admit-structural", "1.0", "--admit-structural-field", "min", "--task-chain-history", "3"]
OURS = PC3 + ["--admit-learned", "1.15", "--admit-runtime", "--admit-aging-ms", "2000"]
OURS05 = PC3 + ["--admit-learned", "1.15", "--admit-runtime", "--admit-aging-ms", "500"]
P = {"cap4": ["--max-conc", "4"], "plan_chain3": PC3, "ours2": OURS, "ours2_age500": OURS05}
G = ["--pin", "gpu", "--serial-ref", REF100]
S = {
  "warm_full": (D100, G + ["--admit-gb-auto", "0.90"], ["cap4", "plan_chain3", "ours2_age500"]),
  "warm_f75":  (D100, G + ["--gpu-frac", "0.75", "--admit-gb-auto", "0.75"], ["cap4", "plan_chain3", "ours2"]),
  "shift_f75": (D100, G + ["--phases", f"{LIGHT};{HEAVY}", "--gpu-frac", "0.75", "--admit-gb-auto", "0.75"], ["cap4", "plan_chain3", "ours2"]),
  "mixed":     (D100, G + ["--groups", f"{HEAVY}:4;{LIGHT}:12", "--rounds", "2", "--admit-gb-auto", "0.90"], ["cap4", "plan_chain3"]),
  "host_sf10": (D10, ["--pin", "host", "--host-gb", "16", "--admit-gb-auto", "0.90"], ["cap4", "plan_chain3"]),
  "cotenant":  (D100, G + ["--admit-gb-auto", "0.90", "--gpu-frac", "0.90", "--cotenant", "3,20,10", "--budget-drain-ms", "10000"], ["cap4", "plan_chain3"]),
}
ENV = ["env", "-i", f"HOME={os.environ['HOME']}", "USER=" + os.environ.get("USER", ""), "PATH=/usr/local/bin:/usr/bin:/bin",
       "PIXI_CACHE_DIR=/scratch/general/vast/u1652955/pixi-cache"] + [f"{k}={v}" for k, v in os.environ.items() if k.startswith(("SLURM_", "CUDA_"))]
PIXI = ["/scratch/general/vast/u1652955/pixi/bin/pixi", "run", "--frozen", "--manifest-path", "../sirius/pixi.toml"]
os.makedirs(O, exist_ok=True)
def run(cmd): return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).stdout
for rep in sys.argv[1:]:
    jobs = [("tpch", s, p) for s, (_, _, ps) in S.items() for p in ps]
    jobs += [("broker", pin, m) for pin in ("host", "gpu") for m in ("concurrent", "broker")]
    random.Random(300 + int(rep)).shuffle(jobs)
    for kind, s, p in jobs:
        tag = f"{s}__{p}__r{rep}" if kind == "tpch" else f"broker_{s}__{p}__r{rep}"
        if glob.glob(f"{O}/{tag}*.json"): print(f"{tag}: exists", flush=True); continue
        if kind == "tpch":
            data, args, _ = S[s]
            out = run(["srun", "-M", "granite", f"--jobid={J}", "--overlap", "-c", "16", "./run2.sh", f"../{O}/{tag}", data, "16",
                       "--memlog", "--querylog"] + args + P[p])
            res = [l for l in out.splitlines() if "clients=" in l]
        else:
            out = run(["srun", "-M", "granite", f"--jobid={J}", "--overlap", "-c", "16"] + ENV + PIXI + ["python", "fuse_broker.py",
                       "--data", D100, "--pin", s, "--host-gb", "24", "--clients", "16", "--per-client", "4" if s == "host" else "8",
                       "--kmax", "16", "--mode", p, "--out", f"{O}/{tag}"])
            res = [l for l in out.splitlines() if "mode=" in l]
        nerr = sum(l.count("fallback to DuckDB") for l in out.splitlines())
        print(f"{tag}: " + (res[0][:200] if res else "NO RESULT") + (f" [{nerr} fallback msgs]" if nerr else ""), flush=True)
    print(f"REPEAT_DONE {rep}", flush=True)
print("CAMPAIGN5DONE", flush=True)
