"""Shared-scan correctness: all TPC-H queries on host-pinned tables with sharing on vs DuckDB CPU."""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_conc
sys.path.insert(0, os.path.join(run_conc.REPO, "test/tpch_performance"))
import duckdb
from queries import QUERIES
from tpch_pin_columns import emit_pin_all
from performance_test import _build_views_sql
data = sys.argv[1]
out = os.path.abspath("results2/test_shared")
open(out + ".yaml", "w").write(run_conc.CONFIG.format(host_bytes=16 << 30, host_pools=(16 << 30) // (8 * 67108864),
     disk_dir="/scratch/general/vast/u1652955/sirius_disk", scan_threads=4, max_conc=16, pipe_threads=4, scan_batch="2GB", gpu_frac=0.9))
os.environ.update(SIRIUS_CONFIG_FILE=out + ".yaml", SIRIUS_PIN_TIER="host", SIRIUS_SHARED_SCAN_MB="2048")
cpu = duckdb.connect(":memory:")
for s in _build_views_sql(data).split(";"):
    if s.strip(): cpu.execute(s)
con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
for s in _build_views_sql(data).split(";"):
    if s.strip(): con.execute(s)
con.execute(f"LOAD '{run_conc.EXT}'")
for s in emit_pin_all(data).split(";"):
    if s.strip(): con.execute(s).fetchall()
def close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-2)
    return a == b
bad = []
for q in range(1, 23):
    g = con.execute(QUERIES[f"q{q}"]).fetchall(); c = cpu.execute(QUERIES[f"q{q}"]).fetchall()
    ok = len(g) == len(c) and all(len(x) == len(y) and all(close(u, v) for u, v in zip(x, y)) for x, y in zip(sorted(g, key=str), sorted(c, key=str)))
    if not ok: bad.append(q)
print(f"shared-scan correctness: {22 - len(bad)}/22 match DuckDB CPU; mismatches: {bad}", flush=True)
os._exit(0)
