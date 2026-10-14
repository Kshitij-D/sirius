"""Run sirius_unittest, auto-excluding tests that abort on this node's missing io_uring.
usage: run_tests.py <jobid> <mode: default|manager>"""
import os, re, subprocess, sys, time
J, mode = sys.argv[1], sys.argv[2]
SIR = "/scratch/general/vast/u1652955/sirius"
base_excl = ["[s3]", "[s3mix]", "[rest]", "[uring_readv]"]
names = []
envs = [f"{k}={v}" for k, v in os.environ.items() if k.startswith(("SLURM_", "CUDA_"))]
extra = ["SIRIUS_FORCE_KVIKIO=1"] + (["SIRIUS_MEMORY_MANAGER=1"] if mode == "manager" else [])
for it in range(40):
    spec = "".join("~" + t for t in base_excl) + "".join('~"' + n + '"' for n in names)
    cmd = ["srun", "-M", "granite", f"--jobid={J}", "--overlap", "-n", "1", "-c", "16", "env", "-i", f"HOME={os.environ['HOME']}",
           "PATH=/usr/bin:/bin", "PIXI_CACHE_DIR=/scratch/general/vast/u1652955/pixi-cache"] + extra + envs + [
           "/scratch/general/vast/u1652955/pixi/bin/pixi", "run", "--frozen", "--manifest-path", f"{SIR}/pixi.toml",
           f"{SIR}/build/release/extension/sirius/test/cpp/sirius_unittest", "--reporter", "compact", spec]
    t0 = time.time()
    out = subprocess.run(cmd, cwd=SIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, errors="replace").stdout
    open(f"tests_{mode}.log", "w").write(out)
    if "uring_reactor: ring init" in out:
        started = re.findall(r"^\[\d+/\d+\] \(\d+%\): (.+)$", out, re.M)
        if not started: print("crash before any test", flush=True); break
        nm = started[-1].split("srun: error")[0].split("terminate called")[0].strip()
        if nm in names: print(f"[{mode}] could not isolate crashing test after '{nm}'", flush=True); break
        names.append(nm); print(f"[{mode}] iter {it}: excluded '{nm}' ({time.time()-t0:.0f}s)", flush=True)
        continue
    tail = [l for l in out.splitlines() if re.search(r"test cases|assertions|Passed|Failed", l)][-3:]
    print(f"[{mode}] finished in {time.time()-t0:.0f}s:", *tail, sep="\n  ", flush=True)
    break
open(f"tests_{mode}_excluded.txt", "w").write("\n".join(base_excl + names) + "\n")
print(f"[{mode}] excluded: {len(base_excl)} tags + {len(names)} tests", flush=True)
