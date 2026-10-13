"""Open-loop arrival simulation of Q6 instances from the measured kill-experiment cost curves.

Policies (one GPU):
  conc   - #1583-style FIFO admission up to CAP=16 in flight (max_concurrent_queries), then
           processor sharing; with n in flight each query
           progresses at 1/conc_s(n) of its work per second (conc_s = measured time for n concurrent).
  fuse   - greedy batching, no artificial wait: when the engine is idle, all queued queries
           (up to KMAX) run as one fused query costing fused_s(K).
  adapt  - regime-aware: when idle, run the queued batch either fused or concurrently,
           whichever measured curve is cheaper for that batch size.
Warm (compiled) costs; cost curves are log-log interpolated, linearly extrapolated past max K.
"""
import json, math, random, sys
import numpy as np

RES = "/scratch/general/vast/u1652955/conc_bench/results"

def curves(name, dist="overlapping"):
    R = json.load(open(f"{RES}/{name}.json"))
    rows = sorted((r for r in R["rows"] if r["dist"] == dist), key=lambda r: r["K"])
    K = np.array([r["K"] for r in rows], float)
    def mk(key):
        y = np.array([r[key] for r in rows])
        def f(k):
            if k <= K[-1]: return float(np.exp(np.interp(math.log(k), np.log(K), np.log(y))))
            slope = (y[-1] - y[-2]) / (K[-1] - K[-2]); return float(y[-1] + slope * (k - K[-1]))
        return f
    return mk("concurrent_s"), mk("fused_s"), int(K[-1])

def sim(policy, lam, conc, fused, kmax, n=20000, seed=1):
    rng = random.Random(seed)
    arr = np.cumsum([rng.expovariate(lam) for _ in range(n)])
    lat = []
    if policy == "conc":                       # processor sharing, state-dependent rate
        CAP = 16
        t, i, active, wait = 0.0, 0, [], []    # active: [remaining_work, arrival]
        while i < n or active or wait:
            if active:
                rate = 1.0 / conc(len(active))          # work units/s per query
                tdone = t + min(a[0] for a in active) / rate
            else:
                tdone = math.inf
            tnext = arr[i] if i < n else math.inf
            tn = min(tdone, tnext); dt = tn - t
            if active:
                for a in active: a[0] -= dt * rate
            t = tn
            if tnext <= tdone:
                wait.append(arr[i]); i += 1
            if tdone < tnext:                  # finish the earliest job exactly (no float stall)
                j = min(range(len(active)), key=lambda x: active[x][0])
                lat.append(t - active[j][1]); active.pop(j)
            while wait and len(active) < CAP: active.append([1.0, wait.pop(0)])
    else:
        t, i, q = 0.0, 0, []
        while i < n or q:
            if not q:
                t = max(t, arr[i])
            while i < n and arr[i] <= t: q.append(arr[i]); i += 1
            batch, q = q[:kmax], q[kmax:]
            k = len(batch)
            cost = fused(k) if policy == "fuse" else min(fused(k), conc(k))
            t += cost
            lat += [t - a for a in batch]
    lat = np.array(lat[n // 10:])              # drop warm-up
    return float(lat.mean()), float(np.percentile(lat, 95))

if __name__ == "__main__":
    out = {}
    for regime in ["q6kill_sf100host", "q6kill_sf10host", "q6kill_sf100gpu"]:
        conc, fused, kmeas = curves(regime)
        kmax = 32
        cap_conc = max(k / conc(k) for k in range(1, kmeas + 1))
        cap_fuse = max(k / fused(k) for k in range(1, kmax + 1))
        rows = []
        for frac in [0.1, 0.25, 0.5, 0.75, 0.9, 1.5, 3.0, 6.0]:
            lam = frac * cap_conc
            r = dict(load_vs_conc_cap=frac, lam=lam)
            for p in ["conc", "fuse", "adapt"]:
                m, p95 = sim(p, lam, conc, fused, kmax, n=20000 if frac < 1 else 5000)
                r[p] = (m, p95)
            rows.append(r)
        out[regime] = dict(cap_conc_qps=cap_conc, cap_fuse_qps=cap_fuse, rows=rows)
        print(flush=True); print(f"\n{regime}: conc capacity {cap_conc:.1f} q/s, fused capacity {cap_fuse:.1f} q/s (KMAX {kmax})")
        print(f"{'load':>6} {'q/s':>6} | {'conc mean/p95 ms':>18} | {'fuse mean/p95 ms':>18} | {'adapt mean/p95 ms':>18}")
        for r in rows:
            f = lambda x: f"{x[0]*1e3:8.0f}/{x[1]*1e3:<8.0f}" if x[1] < 60 else "   unstable       "
            print(f"{r['load_vs_conc_cap']:6.2f} {r['lam']:6.1f} | {f(r['conc'])} | {f(r['fuse'])} | {f(r['adapt'])}")
    json.dump(out, open("/scratch/general/vast/u1652955/conc_bench/sim/arrival_sim.json", "w"), indent=1)
