"""Correctness + timing test for in-engine demote/promote of a pinned table."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_conc
REPO = run_conc.REPO
sys.path.insert(0, os.path.join(REPO, "test/tpch_performance"))
import duckdb
from queries import QUERIES
from tpch_pin_columns import emit_pin_all, QUERY_COLUMNS
from performance_test import _build_views_sql
data, table = sys.argv[1], sys.argv[2]
out = os.path.abspath(f"results2/test_placement_{table}")
cfg = out + ".yaml"
open(cfg, "w").write(run_conc.CONFIG.format(host_bytes=16 << 30, host_pools=(16 << 30) // (8 * 67108864),
     disk_dir="/scratch/general/vast/u1652955/sirius_disk", scan_threads=4, max_conc=4, pipe_threads=4,
     scan_batch="2GB", gpu_frac=0.9))
pf = out + ".placement"; open(pf, "w").close()
os.environ.update(SIRIUS_CONFIG_FILE=cfg, SIRIUS_PIN_TIER="gpu", SIRIUS_PLACEMENT_FILE=pf, SIRIUS_MEMLOG=out + ".mem.csv")
con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
for s in _build_views_sql(data).split(";"):
    if s.strip(): con.execute(s)
con.execute(f"LOAD '{run_conc.EXT}'")
for s in emit_pin_all(data).split(";"):
    if s.strip(): con.execute(s).fetchall()
qs = [q for q, ct in QUERY_COLUMNS.items() if table in ct]
def run_all():
    res, t = {}, time.time()
    for q in qs: res[q] = con.execute(QUERIES[f"q{q}"]).fetchall()
    return res, time.time() - t
def norm(rows): return [tuple(round(x, 6) if isinstance(x, float) else x for x in r) for r in rows]
def alloc_gb():
    import csv
    rows = list(csv.DictReader(open(out + ".mem.csv")))
    return int(rows[-1]["gpu_alloc"]) / 2**30
base, tb = run_all(); time.sleep(0.3); a0 = alloc_gb()
open(pf, "a").write(f"demote {table}\n"); time.sleep(2.5)
dem, td = run_all(); time.sleep(0.3); a1 = alloc_gb()
open(pf, "a").write(f"promote {table}\n"); time.sleep(2.5)
pro, tp = run_all(); time.sleep(0.3); a2 = alloc_gb()
ok1 = all(norm(base[q]) == norm(dem[q]) for q in qs); ok2 = all(norm(base[q]) == norm(pro[q]) for q in qs)
print(f"queries using {table}: {qs}")
print(f"GPU alloc GB: pinned {a0:.2f} -> demoted {a1:.2f} -> promoted {a2:.2f}")
print(f"run time s: gpu {tb:.2f} | host {td:.2f} | gpu again {tp:.2f}")
print(f"results match after demote: {ok1}, after promote: {ok2}")
sys.stdout.flush(); os._exit(0)
