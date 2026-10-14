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
        gpu: {{ usage_limit_fraction: {gpu_frac}, reservation_limit_fraction: 1.0, downgrade_trigger_fraction: 0.9, downgrade_stop_fraction: 0.85 }}
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
    p.add_argument("--footprint-log", action="store_true", help="per-query solo footprints + plan features to <out>.fp.tsv")
    p.add_argument("--admit-footprints", default=None, help="measured footprint TSV -> pre-admission")
    p.add_argument("--admit-plan-coefs", default=None, help="c0,c1..c6 -> query-aware plan pre-admission")
    p.add_argument("--admit-default-gb", type=float, default=0)
    p.add_argument("--admit-structural", type=float, default=0, help="scale * plan need_fast -> structural pre-admission")
    p.add_argument("--admit-learned", type=float, default=0, help="scale * engine-learned per-template peak")
    p.add_argument("--admit-runtime", action="store_true", help="online correction from engine ledger")
    p.add_argument("--admit-aging-ms", type=int, default=0, help="starvation bound for waiting queries")
    p.add_argument("--admit-device-margin-gb", type=float, default=None, help="device-aware budget (margin GB)")
    p.add_argument("--phases", default=None, help='shifting workload: "light-set;heavy-set": every client runs phase 1 then phase 2')
    p.add_argument("--groups", default=None, help='mixed workload: "1,9,13:4;6,14,19:12" = query subset:clients ...')
    p.add_argument("--admit-gb-auto", type=float, default=None,
                   help="for admission policies: budget = frac * device memory - 2 GB")
    p.add_argument("--no-warm", action="store_true", help="cold: skip serial warm-up (no template seen before)")
    p.add_argument("--serial-ref", default=None, help="JSON of a warm run whose serial latencies define speedup")
    p.add_argument("--hog", default=None, help="co-tenant: delay_s,gib,hold_s after throughput phase starts")
    p.add_argument("--admit-bestfit", action="store_true", help="largest fitting waiter first")
    p.add_argument("--stagger-heavy", type=int, default=0, help="max concurrent heavy merge/sort tasks")
    p.add_argument("--task-chain-history", type=int, default=0, help="cross-query chain history: 1 cold, 2 floor")
    p.add_argument("--task-op-priors", action="store_true", help="per-operator priors for never-seen chains")
    p.add_argument("--cotenant", default=None,
                   help="delay_s,gib,hold_s: shrink Sirius budget at runtime, run a co-tenant hog, restore")
    p.add_argument("--budget-drain-ms", type=int, default=0, help="drain-then-shrink bound for runtime budget cuts")
    p.add_argument("--demote", default=None,
                   help="with --cotenant: tables to unpin (GPU -> storage) before the budget cut, re-pinned after restore")
    p.add_argument("--demote-mode", default="unpin", choices=["unpin", "engine"],
                   help="unpin = drop to storage (driver prototype); engine = in-engine GPU->host demotion")
    p.add_argument("--demote-need-gb", type=float, default=11.0,
                   help="--demote auto: largest single-query GPU need the shrunk budget must still cover")
    p.add_argument("--shared-scan-mb", type=float, default=0, help="cooperative shared scan buffer (MB) for host-tier tables")
    p.add_argument("--memory-manager", action="store_true", help="SIRIUS_MEMORY_MANAGER=1: the unified manager, all layers")
    p.add_argument("--arrival", default=None,
                   help="open loop: poisson:RATE_QPS:N or burst:RATE_QPS:N:BURST (N queries, random templates)")
    p.add_argument("--workers", type=int, default=64, help="open loop: client connections available")
    p.add_argument("--benchmark", default="tpch", choices=["tpch", "tpcds"])
    p.add_argument("--tpcds-queries-dir", default="/scratch/general/vast/u1652955/tpcds_queries")
    p.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the engine (repeatable)")
    p.add_argument("--querylog", action="store_true", help="per-query wait/run/projection/engine peak -> <out>.q.tsv")
    p.add_argument("--tasklog", action="store_true", help="per-task Sirius estimate vs actual to <out>.task.tsv")
    p.add_argument("--admit-structural-field", default=None, choices=["min", "fast"])
    p.add_argument("--gpu-frac", type=float, default=0.90, help="Sirius GPU usage_limit_fraction")
    a = p.parse_args()
    global QUERIES
    if a.benchmark == "tpcds":
        # Sirius's GPU-supported TPC-DS subset (test/tpcds_performance/run_tpcds_super_gpu.sh).
        if a.queries == "1-22": a.queries = "3,7,22,26,32,37,42,52,55,62,82,85,92,93,97"
        QUERIES = {}
        for f in os.listdir(a.tpcds_queries_dir):
            if f.startswith("q") and f.endswith(".sql"):
                QUERIES[f[:-4]] = open(os.path.join(a.tpcds_queries_dir, f)).read()
    def parse(spec):
        out = set()
        for part in spec.split(","):
            lo, _, hi = part.partition("-"); out.update(range(int(lo), int(hi or lo) + 1))
        return out
    client_sets = None
    phase_sets = None
    if a.groups:
        client_sets = []
        for g in a.groups.split(";"):
            spec, _, n = g.rpartition(":")
            client_sets += [parse(spec)] * int(n)
        a.clients = len(client_sets)
        qset = set().union(*client_sets)
    elif a.phases:
        phase_sets = [parse(x) for x in a.phases.split(";")]
        qset = set().union(*phase_sets)
    else:
        qset = parse(a.queries)
    max_conc = a.max_conc or a.clients

    cfg = os.path.abspath(a.out + ".yaml")
    os.makedirs(os.path.dirname(cfg), exist_ok=True)
    disk_dir = "/scratch/general/vast/u1652955/sirius_disk"
    os.makedirs(disk_dir, exist_ok=True)
    open(cfg, "w").write(CONFIG.format(host_bytes=a.host_gb << 30, host_pools=(a.host_gb << 30) // (8 * 67108864),
                                       disk_dir=disk_dir, scan_threads=a.scan_threads, max_conc=max_conc,
                                       pipe_threads=a.pipe_threads, scan_batch=a.scan_batch, gpu_frac=a.gpu_frac))
    os.environ["SIRIUS_CONFIG_FILE"] = cfg
    if a.memlog: os.environ["SIRIUS_MEMLOG"] = os.path.abspath(a.out + ".mem.csv")
    if a.admit_gb_auto and (a.admit_learned or a.admit_footprints or a.admit_plan_coefs or a.admit_structural):
        import subprocess
        tot_mib = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]).split()[0])
        a.admit_gb = a.admit_gb_auto * tot_mib / 1024 - 2
    if a.admit_gb: os.environ["SIRIUS_ADMIT_GPU_BYTES"] = str(int(a.admit_gb * (1 << 30)))
    if a.admit_gap_ms: os.environ["SIRIUS_ADMIT_GAP_MS"] = str(a.admit_gap_ms)
    if a.footprint_log: os.environ["SIRIUS_FOOTPRINT_LOG"] = os.path.abspath(a.out + ".fp.tsv")
    if a.admit_footprints: os.environ["SIRIUS_ADMIT_FOOTPRINTS"] = os.path.abspath(a.admit_footprints)
    if a.admit_plan_coefs: os.environ["SIRIUS_ADMIT_PLAN_COEFS"] = a.admit_plan_coefs
    if a.admit_learned: os.environ["SIRIUS_ADMIT_LEARNED"] = str(a.admit_learned)
    if a.admit_runtime: os.environ["SIRIUS_ADMIT_RUNTIME"] = "1"
    if a.admit_aging_ms: os.environ["SIRIUS_ADMIT_AGING_MS"] = str(a.admit_aging_ms)
    if a.admit_device_margin_gb is not None: os.environ["SIRIUS_ADMIT_DEVICE_MARGIN_GB"] = str(a.admit_device_margin_gb)
    if a.admit_bestfit: os.environ["SIRIUS_ADMIT_BESTFIT"] = "1"
    if a.stagger_heavy: os.environ["SIRIUS_STAGGER_HEAVY"] = str(a.stagger_heavy)
    if a.task_chain_history: os.environ["SIRIUS_TASK_CHAIN_HISTORY"] = str(a.task_chain_history)
    if a.task_op_priors: os.environ["SIRIUS_TASK_OP_PRIORS"] = "1"
    if a.budget_drain_ms: os.environ["SIRIUS_BUDGET_DRAIN_MS"] = str(a.budget_drain_ms)
    if a.shared_scan_mb: os.environ["SIRIUS_SHARED_SCAN_MB"] = str(a.shared_scan_mb)
    if a.memory_manager: os.environ["SIRIUS_MEMORY_MANAGER"] = "1"
    for kv in a.env:
        k, _, v = kv.partition("="); os.environ[k] = v
    if a.querylog: os.environ["SIRIUS_QUERYLOG"] = os.path.abspath(a.out + ".q.tsv")
    if a.tasklog: os.environ["SIRIUS_TASKLOG"] = os.path.abspath(a.out + ".task.tsv")
    if a.admit_structural_field: os.environ["SIRIUS_ADMIT_STRUCTURAL_FIELD"] = a.admit_structural_field
    if a.admit_structural: os.environ["SIRIUS_ADMIT_STRUCTURAL"] = str(a.admit_structural)
    if a.admit_default_gb: os.environ["SIRIUS_ADMIT_DEFAULT_BYTES"] = str(int(a.admit_default_gb * (1 << 30)))
    budget_file = None
    if a.cotenant:
        import subprocess as _sp
        tot_gib = int(_sp.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]).split()[0]) / 1024
        budget_file = os.path.abspath(a.out + ".budget")
        R0_budget = a.gpu_frac * tot_gib
        open(budget_file, "w").write(f"{R0_budget:.2f}\n")
        os.environ["SIRIUS_BUDGET_FILE"] = budget_file
        placement_file = os.path.abspath(a.out + ".placement"); open(placement_file, "w").close()
        os.environ["SIRIUS_PLACEMENT_FILE"] = placement_file
    os.environ["SIRIUS_PIN_TIER"] = a.pin if a.pin != "none" else "gpu"

    R = dict(args=vars(a), max_conc=max_conc, marks={})
    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    if a.benchmark == "tpcds":
        import glob as _glob
        for f in sorted(_glob.glob(os.path.join(os.path.abspath(a.data), "*.parquet"))):
            t = os.path.basename(f)[:-8]
            con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{f}')")
    else:
        for s in _build_views_sql(a.data).split(";"):
            if s.strip(): con.execute(s)
    t = now(); con.execute(f"LOAD '{EXT}'"); R["load_s"] = now() - t
    for s in a.pre_sql.split(";"):
        if s.strip(): con.execute(s).fetchall()
    if a.pin != "none":
        t = now()
        from tpch_pin_columns import union_columns_by_table, _pin_call
        pin_marks = []
        if a.benchmark == "tpcds":
            import re as _re
            views = [r[0] for r in con.execute("SELECT table_name FROM duckdb_views() WHERE NOT internal").fetchall()]
            used = [t for t in views if any(_re.search(rf"\b{t}\b", QUERIES.get(f"q{q}", "")) for q in qset)]
            pin_tables = {t: [r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()] for t in used}
        else:
            pin_tables = union_columns_by_table()
        for tbl, cols in pin_tables.items():
            tb = now()
            for st in _pin_call(tbl, cols, a.data, "parquet").split(";"):
                if st.strip(): con.execute(st).fetchall()
            pin_marks.append((tbl, tb, now()))
        R["pin_marks"] = pin_marks
        R["pin_s"] = now() - t
    print(f"loaded+pinned in {now()-t:.1f}s", flush=True)

    # warm-up / serial single-query latencies (JIT compile happens on the first pass)
    R["marks"]["warm_start"] = now()
    serial = {}
    for rep in range(0 if a.no_warm else 3):
        for q in sorted(qset):
            t = now(); con.execute(QUERIES[f"q{q}"]).fetchall(); dt = now() - t
            if rep > 0: serial.setdefault(q, []).append(dt)
    if a.serial_ref:
        ref = json.load(open(a.serial_ref))["serial_latency_s"]
        R["serial_latency_s"] = {int(q): v for q, v in ref.items() if int(q) in qset}
    else:
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
    def client_order(seed):
        if a.benchmark == "tpcds":
            import random as _r
            o = sorted(qset); _r.Random(seed).shuffle(o); return o
        return stream_order(seed)
    def client(i):
        c = curs[i]; barrier.wait()
        order = []
        for r in range(a.rounds):
            so = client_order(1 + i + r * a.clients)
            if phase_sets:
                continue
            mine = client_sets[i] if client_sets else qset
            order += [q for q in so if q in mine]
        if phase_sets:
            for ps in phase_sets:
                for r in range(a.rounds):
                    order += [q for q in client_order(1 + i + r * a.clients) if q in ps]
        for q in order:
            if True:
                t0 = now()
                try: c.execute(QUERIES[f"q{q}"]).fetchall()
                except Exception as e: errors.append((i, q, repr(e)[:300]))
                log.append((i, q, t0, now()))
    if a.arrival:
        # Open loop: arrivals follow a schedule independent of completions; latency is
        # completion minus ARRIVAL, so any queueing (client pool or engine admission) counts.
        import queue, random as _rnd
        kind, *ps = a.arrival.split(":")
        rate, nq = float(ps[0]), int(ps[1])
        rng = _rnd.Random(42)
        templates = sorted(qset)
        sched, t = [], 0.0
        if kind == "poisson":
            for _ in range(nq):
                t += rng.expovariate(rate); sched.append((t, rng.choice(templates)))
        else:  # burst: groups of BURST simultaneous arrivals at the same mean rate
            burst = int(ps[2])
            while len(sched) < nq:
                t += rng.expovariate(rate / burst)
                for _ in range(min(burst, nq - len(sched))): sched.append((t, rng.choice(templates)))
        R["arrival"] = dict(kind=kind, rate=rate, n=nq, workers=a.workers)
        work = queue.Queue()
        extra = [con.cursor() for _ in range(max(0, a.workers - len(curs)))]
        for c in extra:
            for st in a.pre_sql.split(";"):
                if st.strip(): c.execute(st).fetchall()
        pool = curs + extra
        def worker(i):
            c = pool[i]; barrier.wait()
            while True:
                item = work.get()
                if item is None: return
                q, t_arr = item
                try: c.execute(QUERIES[f"q{q}"]).fetchall()
                except Exception as e: errors.append((i, q, repr(e)[:300]))
                log.append((i, q, t_arr, now()))
        def dispatcher():
            barrier.wait()
            t_base = now()
            for t_rel, q in sched:
                d = t_base + t_rel - now()
                if d > 0: time.sleep(d)
                work.put((q, t_base + t_rel))
            for _ in range(a.workers): work.put(None)
        th = [threading.Thread(target=worker, args=(i,)) for i in range(a.workers)]
        barrier = threading.Barrier(a.workers + 2)  # workers + dispatcher + main
        th.append(threading.Thread(target=dispatcher))
        a.clients = a.workers
    else:
        th = [threading.Thread(target=client, args=(i,)) for i in range(a.clients)]
    for x in th: x.start()
    ru0 = resource.getrusage(resource.RUSAGE_SELF); barrier.wait(); t0 = now()
    if a.cotenant:
        def cotenant():
            import subprocess as _sp
            d, g, h = (float(x) for x in a.cotenant.split(","))
            time.sleep(d)
            if a.demote == "auto":
                # Choose victims: least query use per resident GB first, until the shrunk
                # budget covers resident data + the largest single-query need.
                import csv as _csv
                from tpch_pin_columns import QUERY_COLUMNS
                rows = list(_csv.DictReader(open(a.out + ".mem.csv")))
                def alloc_at(t):
                    best = min(rows, key=lambda r: abs(float(r["t"]) - t))
                    return int(best["gpu_alloc"])
                tbytes = {tbl: max(0, alloc_at(t1) - alloc_at(t0)) for tbl, t0, t1 in R["pin_marks"]}
                uses = {tbl: sum(1 for q, ct in QUERY_COLUMNS.items() if q in qset and tbl in ct) for tbl in tbytes}
                resident = sum(tbytes.values())
                deficit = resident + a.demote_need_gb * 2**30 - (R0_budget - g - 1) * 2**30
                # Cost of demoting a set = data the workload must now read from storage,
                # sum(uses x bytes); pick the cheapest set that covers the deficit (8 tables:
                # exhaustive over subsets).
                import itertools
                cand = [t for t in tbytes if tbytes[t] > 0]
                best = None
                for k in range(1, len(cand) + 1):
                    for combo in itertools.combinations(cand, k):
                        fb = sum(tbytes[t] for t in combo)
                        if fb < deficit: continue
                        cost = sum(uses[t] * tbytes[t] for t in combo)
                        if best is None or cost < best[0]: best = (cost, combo, fb)
                chosen, freed = (list(best[1]), best[2]) if best else ([], 0)
                R["demote_plan"] = dict(table_gb={k: round(v / 2**30, 2) for k, v in tbytes.items()}, uses=uses,
                                        deficit_gb=round(deficit / 2**30, 2), chosen=chosen, freed_gb=round(freed / 2**30, 2))
                a.demote = ",".join(chosen)
            if a.demote:
                # Placement decision: give up residency of the named tables so the shrunk
                # budget still covers the working set, instead of an OOM storm.
                dc = con.cursor()
                R["marks"]["demote_start"] = now()
                for t in a.demote.split(","):
                    if a.demote_mode == "engine":
                        open(placement_file, "a").write(f"demote {t}\n")
                    else:
                        dc.execute(f"CALL unpin_table('{t}')").fetchall()
                if a.demote_mode == "engine": time.sleep(0.5)  # watcher polls every 100 ms; copies take ~0.2 s/12 GB
                R["marks"]["demote_done"] = now()
            R["marks"]["cotenant_shrink"] = now()
            open(budget_file, "w").write(f"{R0_budget - g - 1:.2f}\n")
            hog_out = _sp.run([os.path.join(os.path.dirname(os.path.abspath(__file__)), "hog"), "0", str(g), str(h), "30"],
                              stdout=_sp.PIPE, stderr=_sp.STDOUT, universal_newlines=True).stdout
            R["cotenant_hog"] = hog_out.strip()
            R["marks"]["cotenant_restore"] = now()
            open(budget_file, "w").write(f"{R0_budget:.2f}\n")
            if a.demote and a.demote_mode == "engine":
                for t in a.demote.split(","):
                    open(placement_file, "a").write(f"promote {t}\n")
                R["marks"]["repin_done"] = now()
            elif a.demote:
                from tpch_pin_columns import union_columns_by_table, _pin_call
                cols = union_columns_by_table()
                rc = con.cursor()
                for t in a.demote.split(","):
                    for st in _pin_call(t, cols[t], a.data, "parquet").split(";"):
                        if st.strip(): rc.execute(st).fetchall()
                R["marks"]["repin_done"] = now()
        threading.Thread(target=cotenant, daemon=True).start()
    hog = None
    if a.hog:
        import subprocess
        d, g, h = a.hog.split(",")
        hog = subprocess.Popen([os.path.join(os.path.dirname(os.path.abspath(__file__)), "hog"), d, g, h],
                               stdout=open(a.out + ".hog.txt", "w"), stderr=subprocess.STDOUT)
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
