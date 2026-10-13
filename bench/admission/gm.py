"""Average nsys GPU metrics over the throughput window recorded by run_conc.py."""
import json, sqlite3, subprocess, sys, os
WANT = ["GR Active [Throughput %]", "SMs Active [Throughput %]", "SM Issue [Throughput %]",
        "Compute Warps in Flight [Throughput %]", "DRAM Read Bandwidth [Throughput %]",
        "DRAM Write Bandwidth [Throughput %]", "PCIe RX Throughput [Throughput %]", "PCIe TX Throughput [Throughput %]"]

def window_metrics(rep, t0, t1):
    sq = rep.replace(".nsys-rep", ".sqlite")
    if not os.path.exists(sq):
        subprocess.run(["nsys", "export", "-t", "sqlite", "-o", sq, "-f", "true", rep], check=True, capture_output=True)
    c = sqlite3.connect(sq)
    start = c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    names = {mid: n for _, _, _, mid, n in c.execute("select * from TARGET_INFO_GPU_METRICS")}
    lo, hi = int(t0 * 1e9) - start, int(t1 * 1e9) - start
    out = {}
    for mid, n in names.items():
        if n in WANT:
            v = c.execute("select avg(value), count(*) from GPU_METRICS where metricId=? and timestamp between ? and ?", (mid, lo, hi)).fetchone()
            out[n] = round(v[0], 1) if v[0] is not None else None
            if n == "GR Active [Throughput %]":
                # distribution: share of samples where the GPU is near-idle / saturated
                tot = v[1] or 1
                out["GR<10% share"] = round(c.execute("select count(*) from GPU_METRICS where metricId=? and timestamp between ? and ? and value<10", (mid, lo, hi)).fetchone()[0] / tot, 3)
    return out

if __name__ == "__main__":
    rep, js = sys.argv[1], sys.argv[2]
    R = json.load(open(js))
    for w in [("warm_start", "warm_end"), ("tp_start", "tp_end")]:
        print(w[0].split("_")[0], window_metrics(rep, R["marks"][w[0]], R["marks"][w[1]]))
