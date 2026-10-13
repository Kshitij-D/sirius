"""Summarize SIRIUS_TASKLOG: actual task peak vs input and vs Sirius's own estimate, by operator chain.
cols: query, pipeline, task, input, est_peak, had_history, reservation, actual_peak, output, ops"""
import sys, collections, statistics as st
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
by = collections.defaultdict(list)
for q, p, t, inp, est, hh, res, peak, out, ops in rows:
    by[ops].append((int(inp), int(est), int(hh), int(res), int(peak), int(out)))
print(f"{'operator chain':60s} {'tasks':>6} {'in MB':>7} {'peak/in':>8} {'est/in':>7} {'est/peak':>8} {'out/in':>7}")
for ops, v in sorted(by.items(), key=lambda kv: -sum(x[4] for x in kv[1]))[:25]:
    v = [x for x in v if x[0] > 0]
    if not v: continue
    med = lambda f: st.median(f(x) for x in v)
    print(f"{ops[:60]:60s} {len(v):6d} {med(lambda x:x[0])/2**20:7.0f} {med(lambda x:x[4]/x[0]):8.2f} "
          f"{med(lambda x:x[1]/x[0]):7.2f} {st.median(x[1]/x[4] for x in v if x[4]>0) if any(x[4] for x in v) else float('nan'):8.2f} {med(lambda x:x[5]/x[0]):7.2f}")
# Sirius's estimator quality overall: history vs no-history
for hh in (0, 1):
    r = [int(x[4]) / int(x[7]) for x in rows if x[5] == str(hh) and int(x[7]) > 0]
    if r: print(f"had_history={hh}: tasks {len(r)}, Sirius est/actual median {st.median(r):.2f}, under {sum(a<1 for a in r)/len(r)*100:.0f}%")
