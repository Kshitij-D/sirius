"""Clean ablation campaign: fresh process per run, randomized order per repeat, resumable.
usage: python3 campaign2.py <repeat> [<repeat> ...]"""
import os, random, subprocess, sys, glob
J = "2168195"; D = "/scratch/general/vast/u1652955/tpch_sf100"; O = "results4"
REF = "/scratch/general/vast/u1652955/conc_bench/results2/sf100gpu_rtx_c1.json"
HEAVY, LIGHT = "1,9,13,18,21", "2,3,4,5,6,7,8,10,11,12,14,15,16,17,19,20,22"
P = {  # layered policies
  "none":     [],
  "cap4":     ["--max-conc", "4"],
  "plan":     ["--admit-structural", "1.0", "--admit-structural-field", "min"],
  "observed": ["--admit-structural", "1.0", "--admit-structural-field", "min", "--admit-learned", "1.15"],
  "obs_rt":   ["--admit-structural", "1.0", "--admit-structural-field", "min", "--admit-learned", "1.15", "--admit-runtime"],
  "ours":     ["--admit-structural", "1.0", "--admit-structural-field", "min", "--admit-learned", "1.15", "--admit-runtime",
               "--admit-aging-ms", "2000"],
}
S = {  # scenario -> (args, policies)
  "warm_full": (["--admit-gb-auto", "0.90"], list(P)),
  "warm_f75":  (["--gpu-frac", "0.75", "--admit-gb-auto", "0.75"], list(P)),
  "cold_full": (["--no-warm", "--admit-gb-auto", "0.90"], ["none", "cap4", "plan", "obs_rt", "ours"]),
  "cold_f75":  (["--no-warm", "--gpu-frac", "0.75", "--admit-gb-auto", "0.75"], ["none", "cap4", "plan", "obs_rt", "ours"]),
  "mixed":     (["--groups", f"{HEAVY}:4;{LIGHT}:12", "--rounds", "2", "--admit-gb-auto", "0.90"], ["none", "cap4", "observed", "ours"]),
}
os.makedirs(O, exist_ok=True)
for rep in sys.argv[1:]:
    jobs = [(s, p) for s, (_, ps) in S.items() for p in ps]
    random.Random(int(rep)).shuffle(jobs)
    for s, p in jobs:
        tag = f"{s}__{p}__r{rep}"
        if glob.glob(f"{O}/{tag}_c*.json"):
            print(f"{tag}: exists", flush=True); continue
        args = ["srun", "-M", "granite", f"--jobid={J}", "--overlap", "-c", "16", "./run2.sh", f"../{O}/{tag}", D, "16",
                "--pin", "gpu", "--memlog", "--querylog", "--serial-ref", REF] + S[s][0] + P[p]
        out = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).stdout
        line = " ".join(l for l in out.splitlines() if "clients=" in l or "rror" in l)[:220]
        print(f"{tag}: {line}", flush=True)
    print(f"REPEAT_DONE {rep}", flush=True)
print("CAMPAIGN2DONE", flush=True)
