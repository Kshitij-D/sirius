"""Closed-loop concurrency sweep for Sirius (PR #1583).

N client threads, each on its own DuckDB cursor, each runs one TPC-H stream (spec permutation)
back-to-back. Reports throughput, per-query latency, host CPU use and wall-clock markers that
align with an nsys GPU-metrics timeline.
"""
import argparse, json, os, resource, statistics, sys, threading, time
REPO = "/scratch/general/vast/u1652955/sirius"
sys.path.insert(0, os.path.join(REPO, "test/tpch_performance"))
import duckdb
from queries import QUERIES
from tpch_pin_columns import emit_pin_all
from tpch_stream_permutations import stream_order
from performance_test import _build_views_sql

EXT = os.path.join(REPO, "build/release/extension/sirius/sirius.duckdb_extension")

CONFIG = """sirius:
    topology: {{ num_gpus: 1 }}
    memory:
        gpu: {{ usage_limit_fraction: 0.90, reservation_limit_fraction: 1.0, downgrade_trigger_fraction: 0.9, downgrade_stop_fraction: 0.85 }}
        host: {{ capacity_bytes: {host_bytes}, initial_number_pools: {host_pools}, pool_size: 8, block_size: 67108864 }}
        disk: {{ disk_id: 0, capacity_bytes: 500000000000, downgrade_root_dirs: "{disk_dir}" }}
    executor:
        scan_manager: {{ num_threads: {scan_threads}, max_concurrent_queries: {max_conc}, use_sirius_datasource: false, enable_prefetch_cache: false }}
        pipeline: {{ num_threads: {pipe_threads} }}
        downgrade: {{ num_threads: 1 }}
        task_creator: {{ num_threads: 2 }}
    operator_params:
        scan_task_batch_size: {scan_batch}
    telemetry: {{ enable_quent: false }}
"""

def now(): return time.time()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--clients", type=int, required=True)
    p.add_argument("--max-conc", type=int, default=None)
    p.add_argument("--queries", default="1-22", help="subset, e.g. 1,6,14 (kept in stream order)")
    p.add_argument("--rounds", type=int, default=1, help="streams per client")
    p.add_argument("--pin", default="gpu", choices=["gpu", "host", "none"])
    p.add_argument("--host-gb", type=int, default=12)
    p.add_argument("--scan-batch", default="2GB")
    p.add_argument("--pipe-threads", type=int, default=4)
    p.add_argument("--scan-threads", type=int, default=4)
    p.add_argument("--pre-sql", default="SET expression_evaluator_strategy = 'ast_jit'")
    p.add_argument("--out", required=True)
    p.add_argument("--memlog", action="store_true", help="sample GPU/host tier usage to <out>.mem.csv")
    p.add_argument("--admit-gb", type=float, default=0, help="memory-gated admission threshold (GPU allocated GB)")
    p.add_argument("--admit-gap-ms", type=int, default=0)
    a = p.parse_args()
    max_conc = a.max_conc or a.clients
    qset = set()
    for part in a.queries.split(","):
        lo, _, hi = part.partition("-"); qset.update(range(int(lo), int(hi or lo) + 1))

    cfg = os.path.abspath(a.out + ".yaml")
    os.makedirs(os.path.dirname(cfg), exist_ok=True)
    disk_dir = "/scratch/general/vast/u1652955/sirius_disk"
    os.makedirs(disk_dir, exist_ok=True)
    open(cfg, "w").write(CONFIG.format(host_bytes=a.host_gb << 30, host_pools=(a.host_gb << 30) // (8 * 67108864),
                                       disk_dir=disk_dir, scan_threads=a.scan_threads, max_conc=max_conc,
                                       pipe_threads=a.pipe_threads, scan_batch=a.scan_batch))
    os.environ["SIRIUS_CONFIG_FILE"] = cfg
    if a.memlog: os.environ["SIRIUS_MEMLOG"] = os.path.abspath(a.out + ".mem.csv")
    if a.admit_gb: os.environ["SIRIUS_ADMIT_GPU_BYTES"] = str(int(a.admit_gb * (1 << 30)))
    if a.admit_gap_ms: os.environ["SIRIUS_ADMIT_GAP_MS"] = str(a.admit_gap_ms)
    os.environ["SIRIUS_PIN_TIER"] = a.pin if a.pin != "none" else "gpu"

    R = dict(args=vars(a), max_conc=max_conc, marks={})
    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    for s in _build_views_sql(a.data).split(";"):
        if s.strip(): con.execute(s)
    t = now(); con.execute(f"LOAD '{EXT}'"); R["load_s"] = now() - t
    for s in a.pre_sql.split(";"):
        if s.strip(): con.execute(s).fetchall()
    if a.pin != "none":
        t = now()
        for s in emit_pin_all(a.data).split(";"):
            if s.strip(): con.execute(s).fetchall()
        R["pin_s"] = now() - t
    print(f"loaded+pinned in {now()-t:.1f}s", flush=True)

    # warm-up / serial single-query latencies (JIT compile happens on the first pass)
    R["marks"]["warm_start"] = now()
    serial = {}
    for rep in range(3):
        for q in sorted(qset):
            t = now(); con.execute(QUERIES[f"q{q}"]).fetchall(); dt = now() - t
            if rep > 0: serial.setdefault(q, []).append(dt)
    R["serial_latency_s"] = {q: min(v) for q, v in serial.items()}
    R["serial_sum_s"] = sum(R["serial_latency_s"].values())
    # fixed per-query overhead: trivial GPU query
    lat = []
    for _ in range(20):
        t = now(); con.execute("select count(*) from nation").fetchall(); lat.append(now() - t)
    R["trivial_query_s"] = statistics.median(lat)
    R["marks"]["warm_end"] = now()
    print(f"serial sum {R['serial_sum_s']:.3f}s trivial {R['trivial_query_s']*1e3:.1f}ms", flush=True)

    # closed-loop throughput phase
    curs = [con.cursor() for _ in range(a.clients)]
    for c in curs:
        for s in a.pre_sql.split(";"):
            if s.strip(): c.execute(s).fetchall()
    barrier = threading.Barrier(a.clients + 1)
    log, errors = [], []
    def client(i):
        c = curs[i]; barrier.wait()
        for r in range(a.rounds):
            for q in stream_order(1 + i + r * a.clients):
                if q not in qset: continue
                t0 = now()
                try: c.execute(QUERIES[f"q{q}"]).fetchall()
                except Exception as e: errors.append((i, q, repr(e)[:300]))
                log.append((i, q, t0, now()))
    th = [threading.Thread(target=client, args=(i,)) for i in range(a.clients)]
    for x in th: x.start()
    ru0 = resource.getrusage(resource.RUSAGE_SELF); barrier.wait(); t0 = now()
    R["marks"]["tp_start"] = t0
    LOGF = os.path.join(os.getcwd(), "log", time.strftime("sirius_%Y-%m-%d.log"))
    log_off = os.path.getsize(LOGF) if os.path.exists(LOGF) else 0
    for x in th: x.join()
    t1 = now(); ru1 = resource.getrusage(resource.RUSAGE_SELF)
    R["marks"]["tp_end"] = t1
    n = len(log)
    R.update(makespan_s=t1 - t0, n_queries=n, qps=n / (t1 - t0),
             cpu_cores_busy=((ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)) / (t1 - t0),
             log=log, errors=errors,
             serial_equiv_s=sum(R["serial_latency_s"][q] for _, q, _, _ in log))
    R["speedup_vs_serial"] = R["serial_equiv_s"] / R["makespan_s"]
    # engine-side failures during the throughput phase (a GPU failure silently falls back to DuckDB CPU)
    try:
        with open(LOGF, errors="replace") as f:
            f.seek(log_off); txt = f.read()
        R["gpu_fallbacks"] = txt.count("Transparent GPU execution error")
        R["oom_retries"] = txt.count("OOM at operator")
        R["plan_fallbacks"] = txt.count("Transparent execution fallback")
    except Exception as e:
        R["gpu_fallbacks"] = R["oom_retries"] = None
    # average number of queries in flight during the phase
    R["avg_inflight"] = sum(e - s for _, _, s, e in log) / (t1 - t0)
    json.dump(R, open(a.out + ".json", "w"), indent=1)
    print(f"clients={a.clients} makespan={R['makespan_s']:.2f}s qps={R['qps']:.2f} "
          f"speedup_vs_serial={R['speedup_vs_serial']:.2f} inflight={R['avg_inflight']:.2f} "
          f"cpu_cores={R['cpu_cores_busy']:.2f} errors={len(errors)} fallbacks={R['gpu_fallbacks']} oom_retries={R['oom_retries']} plan_fallbacks={R.get('plan_fallbacks')}", flush=True)
    os._exit(0)  # skip slow teardown

if __name__ == "__main__":
    main()
