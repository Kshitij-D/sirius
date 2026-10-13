"""Summarize admission runs on speed, safety and reliability.

usage: summarize.py <glob-prefix> [...]   e.g. summarize.py results2/a3_
Groups runs by name minus the trailing _r<k>_c<clients>, writes <prefix>summary.csv and prints a table.
  speed:       speedup vs serial (mean, min, max), queries/s
  safety:      OOM retries, CPU fallbacks (sum over repeats), least free GPU memory
  reliability: speedup spread across repeats, p50/p95/p99 end-to-end query latency (wait + run)
"""
import csv, glob, json, os, re, statistics as st, sys
import numpy as np


def run_stats(path):
    R = json.load(open(path))
    lat = np.array([e - s for _, _, s, e in R["log"]])
    minfree = None
    mem = path[:-5] + ".mem.csv"
    if os.path.exists(mem):
        t0, t1 = R["marks"]["tp_start"], R["marks"]["tp_end"]
        free = [int(r["gpu_avail"]) for r in csv.DictReader(open(mem)) if t0 <= float(r["t"]) <= t1]
        minfree = min(free) / 2**30 if free else None
    return dict(speedup=R["speedup_vs_serial"], qps=R["qps"], oom=R.get("oom_retries") or 0,
                fb=R.get("gpu_fallbacks") or 0, minfree=minfree,
                p50=np.percentile(lat, 50), p95=np.percentile(lat, 95), p99=np.percentile(lat, 99))


def main():
    groups = {}
    for prefix in sys.argv[1:]:
        for p in sorted(glob.glob(prefix + "*.json")):
            name = re.sub(r"_r\d+_c\d+\.json$", "", os.path.basename(p))
            if name.endswith(".json"):
                continue
            groups.setdefault(name, []).append(run_stats(p))
    rows = []
    for name, rs in groups.items():
        sp = [r["speedup"] for r in rs]
        mf = [r["minfree"] for r in rs if r["minfree"] is not None]
        rows.append(dict(policy=name, n=len(rs), speedup_mean=st.mean(sp), speedup_min=min(sp), speedup_max=max(sp),
                         qps_mean=st.mean(r["qps"] for r in rs), oom_total=sum(r["oom"] for r in rs),
                         fallbacks_total=sum(r["fb"] for r in rs), min_free_gb=min(mf) if mf else float("nan"),
                         p50_s=st.mean(r["p50"] for r in rs), p95_s=st.mean(r["p95"] for r in rs),
                         p99_s=st.mean(r["p99"] for r in rs)))
    out = sys.argv[1] + "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{'policy':28s} {'n':>2} {'speedup mean [min-max]':>24} {'OOM':>5} {'fallb':>5} {'minfree':>7} {'p50 s':>6} {'p95 s':>6} {'p99 s':>6}")
    for r in rows:
        print(f"{r['policy']:28s} {r['n']:2d} {r['speedup_mean']:8.2f}x [{r['speedup_min']:.2f}-{r['speedup_max']:.2f}]    "
              f"{r['oom_total']:5d} {r['fallbacks_total']:5d} {r['min_free_gb']:6.1f}G {r['p50_s']:6.2f} {r['p95_s']:6.2f} {r['p99_s']:6.2f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
