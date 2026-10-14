import duckdb, sys, os
sf, out, qdir = sys.argv[1], sys.argv[2], sys.argv[3]
con = duckdb.connect()
con.execute("INSTALL tpcds; LOAD tpcds")
for nr, q in con.execute("SELECT query_nr, query FROM tpcds_queries()").fetchall():
    open(os.path.join(qdir, f"q{nr}.sql"), "w").write(q)
con.execute(f"CALL dsdgen(sf={sf})")
con.execute(f"EXPORT DATABASE '{out}' (FORMAT PARQUET)")
print("done", flush=True)
