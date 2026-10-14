"""Shared-scan broker vs concurrent execution for same-template queries (Q6, qgen parameters).

Closed loop: C clients, each submits N Q6 instances back to back.
  concurrent: every client runs its own query on its own cursor (#1583 concurrency).
  broker:     clients enqueue; one broker thread takes everything pending (up to --kmax), runs it as one
              fused query (one SUM(CASE ...) per instance, a single scan of lineitem), and returns each
              client its own column. Waiting for a batch is part of the measured latency.
Reports throughput, latency percentiles and a correctness check of the first fused batch.
"""
import argparse, json, os, queue, random, statistics, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q6kill, run_conc
import duckdb
from tpch_pin_columns import detect_pin_glob
from performance_test import _build_views_sql


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--pin", default="host")
    ap.add_argument("--host-gb", type=int, default=24); ap.add_argument("--clients", type=int, default=16)
    ap.add_argument("--per-client", type=int, default=8); ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--mode", choices=["concurrent", "broker"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = os.path.abspath(a.out + ".yaml")
    open(cfg, "w").write(run_conc.CONFIG.format(host_bytes=a.host_gb << 30, host_pools=(a.host_gb << 30) // (8 * 67108864),
        disk_dir="/scratch/general/vast/u1652955/sirius_disk", scan_threads=4, max_conc=a.clients, pipe_threads=4,
        scan_batch="2GB", gpu_frac=0.90))
    os.environ["SIRIUS_CONFIG_FILE"] = cfg; os.environ["SIRIUS_PIN_TIER"] = a.pin
    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    for s in _build_views_sql(a.data).split(";"):
        if s.strip(): con.execute(s)
    con.execute(f"LOAD '{run_conc.EXT}'"); con.execute("SET expression_evaluator_strategy = 'ast_jit'")
    cols = "'l_shipdate','l_discount','l_quantity','l_extendedprice'"
    con.execute(f"CALL pin_table('{detect_pin_glob(a.data, 'lineitem')}', tier='{a.pin}', name='lineitem', cols=[{cols}])").fetchall()
    r = random.Random(7)
    work = [[q6kill.params("overlapping", 1, r)[0] for _ in range(a.per_client)] for _ in range(a.clients)]
    # warm both single and fused shapes for every batch size up to kmax (JIT)
    for k in sorted({1, 2, 4, 8, a.kmax} | set(range(1, a.kmax + 1))):
        con.execute(q6kill.fused([work[0][0]] * k) if k > 1 else q6kill.single(work[0][0])).fetchall()
    # correctness: fused batch == singles
    probe = [w[0] for w in work][: min(a.kmax, a.clients)]
    ref = [con.execute(q6kill.single(p)).fetchone()[0] for p in probe]
    got = con.execute(q6kill.fused(probe)).fetchone()
    ok = all(abs(float(x or 0) - float(y or 0)) <= 1e-6 * max(1, abs(float(y or 0))) + 0.01 for x, y in zip(got, ref))
    curs = [con.cursor() for _ in range(a.clients + 1)]
    for c in curs: c.execute("SET expression_evaluator_strategy = 'ast_jit'")
    lat, batches = [], []
    lock = threading.Lock()
    bar = threading.Barrier(a.clients + 1)
    if a.mode == "concurrent":
        def client(i):
            bar.wait()
            for p in work[i]:
                t = time.time(); curs[i].execute(q6kill.single(p)).fetchall()
                with lock: lat.append(time.time() - t)
        th = [threading.Thread(target=client, args=(i,)) for i in range(a.clients)]
    else:
        pending = queue.Queue(); stop = threading.Event()
        def broker():
            while not stop.is_set():
                try: first = pending.get(timeout=0.05)
                except queue.Empty: continue
                batch = [first]
                while len(batch) < a.kmax:
                    try: batch.append(pending.get_nowait())
                    except queue.Empty: break
                ps = [b[0] for b in batch]
                row = curs[-1].execute(q6kill.fused(ps) if len(ps) > 1 else q6kill.single(ps[0])).fetchone()
                batches.append(len(ps))
                for (p, ev, slot), v in zip(batch, row):
                    slot.append(v); ev.set()
        def client(i):
            bar.wait()
            for p in work[i]:
                ev, slot = threading.Event(), []
                t = time.time(); pending.put((p, ev, slot)); ev.wait()
                with lock: lat.append(time.time() - t)
        bt = threading.Thread(target=broker, daemon=True); bt.start()
        th = [threading.Thread(target=client, args=(i,)) for i in range(a.clients)]
    for x in th: x.start()
    bar.wait(); t0 = time.time()
    for x in th: x.join()
    mk = time.time() - t0
    if a.mode == "broker": stop.set()
    L = sorted(lat)
    pct = lambda q: L[min(len(L) - 1, int(q * len(L)))]
    R = dict(args=vars(a), makespan_s=mk, n=len(L), qps=len(L) / mk, p50=pct(.5), p95=pct(.95), p99=pct(.99),
             mean_batch=statistics.mean(batches) if batches else 1, fused_correct=ok)
    json.dump(R, open(a.out + ".json", "w"), indent=1)
    print(f"mode={a.mode} pin={a.pin} qps={R['qps']:.2f} p50={R['p50']*1e3:.0f}ms p95={R['p95']*1e3:.0f}ms "
          f"p99={R['p99']*1e3:.0f}ms mean_batch={R['mean_batch']:.1f} fused_correct={ok}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
