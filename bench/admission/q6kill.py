"""Q6 kill experiment: K Q6 instances as serial / concurrent (#1583) / one fused CASE query."""
import argparse, json, os, random, statistics, sys, threading, time, datetime as dt
REPO = "/scratch/general/vast/u1652955/sirius"
sys.path.insert(0, os.path.join(REPO, "test/tpch_performance"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb
from tpch_pin_columns import detect_pin_glob
from performance_test import _build_views_sql
import run_conc

def params(dist, K, r):
    if dist == "identical":
        return [(dt.date(1994, 1, 1), dt.date(1995, 1, 1), 0.06, 24)] * K
    if dist == "overlapping":   # qgen ranges: 5 possible years, discount .02-.09, qty 24-25
        out = []
        for _ in range(K):
            y = r.randint(1993, 1997); d = r.randint(2, 9) / 100
            out.append((dt.date(y, 1, 1), dt.date(y + 1, 1, 1), d, r.randint(24, 25)))
        return out
    if dist == "disjoint":      # K non-overlapping shipdate windows covering 1992-01-02..1998-12-01
        lo, hi = dt.date(1992, 1, 2), dt.date(1998, 12, 2); span = (hi - lo).days
        return [(lo + dt.timedelta(days=span * i // K), lo + dt.timedelta(days=span * (i + 1) // K),
                 r.randint(2, 9) / 100, r.randint(24, 25)) for i in range(K)]

def pred(p):
    a, b, d, q = p
    return (f"l_shipdate >= date '{a}' and l_shipdate < date '{b}' "
            f"and l_discount between {d - 0.01:.2f} and {d + 0.01:.2f} and l_quantity < {q}")
def single(p): return f"select sum(l_extendedprice * l_discount) as revenue from lineitem where {pred(p)}"
def fused(ps):
    cols = ",\n  ".join(f"sum(case when {pred(p)} then l_extendedprice * l_discount else 0 end) as q{i}" for i, p in enumerate(ps))
    return f"select\n  {cols}\nfrom lineitem"

def logcount(off):
    f = os.path.join(os.getcwd(), "log", time.strftime("sirius_%Y-%m-%d.log"))
    with open(f, errors="replace") as fh:
        fh.seek(off); t = fh.read()
    return dict(fallback=t.count("Transparent execution fallback"), gpu_err=t.count("Transparent GPU execution error"),
                gpu_plans=t.count("physical plan replaced with GPU operator"))
def logoff():
    f = os.path.join(os.getcwd(), "log", time.strftime("sirius_%Y-%m-%d.log"))
    return os.path.getsize(f) if os.path.exists(f) else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--pin", default="gpu")
    ap.add_argument("--ks", default="1,2,4,8,16,32"); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--host-gb", type=int, default=12); ap.add_argument("--out", required=True)
    a = ap.parse_args(); KS = [int(k) for k in a.ks.split(",")]; KMAX = max(KS)
    cfg = os.path.abspath(a.out + ".yaml")
    open(cfg, "w").write(run_conc.CONFIG.format(host_bytes=a.host_gb << 30, host_pools=(a.host_gb << 30) // (8 * 67108864),
        disk_dir="/scratch/general/vast/u1652955/sirius_disk", scan_threads=4, max_conc=KMAX, pipe_threads=4, scan_batch="2GB"))
    os.environ["SIRIUS_CONFIG_FILE"] = cfg; os.environ["SIRIUS_PIN_TIER"] = a.pin
    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    for s in _build_views_sql(a.data).split(";"):
        if s.strip(): con.execute(s)
    con.execute(f"LOAD '{run_conc.EXT}'"); con.execute("SET expression_evaluator_strategy = 'ast_jit'")
    cols = "'l_shipdate','l_discount','l_quantity','l_extendedprice'"
    con.execute(f"CALL pin_table('{detect_pin_glob(a.data, 'lineitem')}', tier='{a.pin}', name='lineitem', cols=[{cols}])").fetchall()
    curs = [con.cursor() for _ in range(KMAX)]
    for c in curs: c.execute("SET expression_evaluator_strategy = 'ast_jit'")
    t = time.time(); con.execute("select count(*) from nation").fetchall()
    triv = []
    for _ in range(20):
        t = time.time(); con.execute("select count(*) from nation").fetchall(); triv.append(time.time() - t)
    r = random.Random(11); rows = []
    for dist in ["identical", "overlapping", "disjoint"]:
        for K in KS:
            ps = params(dist, K, r); qs = [single(p) for p in ps]; fq = fused(ps)
            off = logoff()
            ref = [con.execute(q).fetchone()[0] for q in qs]                       # also warms JIT for singles
            t = time.time(); got = con.execute(fq).fetchone(); cold_fused = time.time() - t
            ok = all((x is None and y is None) or abs(float(x or 0) - float(y or 0)) <= 1e-6 * max(1, abs(float(y or 0))) + 0.01 for x, y in zip(got, ref))
            ser, conc, fus = [], [], []
            for _ in range(a.reps):
                t = time.time()
                for q in qs: con.execute(q).fetchall()
                ser.append(time.time() - t)
                bar = threading.Barrier(K + 1)
                def run(i):
                    bar.wait(); curs[i].execute(qs[i]).fetchall()
                th = [threading.Thread(target=run, args=(i,)) for i in range(K)]
                for x in th: x.start()
                bar.wait(); t = time.time()
                for x in th: x.join()
                conc.append(time.time() - t)
                t = time.time(); con.execute(fq).fetchall(); fus.append(time.time() - t)
            lc = logcount(off)
            row = dict(dist=dist, K=K, serial_s=statistics.median(ser), concurrent_s=statistics.median(conc),
                       fused_s=statistics.median(fus), fused_cold_s=cold_fused, results_match=ok, **lc)
            row["fused_vs_best"] = min(row["serial_s"], row["concurrent_s"]) / row["fused_s"]
            rows.append(row)
            print(f"{dist:11s} K={K:2d} serial {row['serial_s']*1e3:8.1f}ms  conc {row['concurrent_s']*1e3:8.1f}ms  "
                  f"fused {row['fused_s']*1e3:8.1f}ms (cold {cold_fused*1e3:7.1f})  fused/best {row['fused_vs_best']:.2f}x  "
                  f"match={ok} {lc}", flush=True)
    json.dump(dict(args=vars(a), trivial_query_s=statistics.median(triv), rows=rows), open(a.out + ".json", "w"), indent=1)
    os._exit(0)

if __name__ == "__main__":
    main()
