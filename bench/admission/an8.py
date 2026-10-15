import json,glob,re,statistics as st,collections,csv,sys
D=collections.defaultdict(list)
for f in sorted(glob.glob('results13/*.json')):
    m=re.match(r'results13/(.+?)__(.+?)__r(\d)_c\d+\.json',f)
    if not m: continue
    R=json.load(open(f)); sc,p,r=m.groups()
    lat=sorted(e-s for _,_,s,e in R['log']); p99=lat[int(.99*(len(lat)-1))]
    D[(sc,p)].append((R['speedup_vs_serial'],R.get('oom_retries') or 0,R.get('gpu_fallbacks') or 0,p99,R.get('avg_inflight')))
pre=sys.argv[1] if len(sys.argv)>1 else ''
for k in sorted(D):
    if not k[0].startswith(pre): continue
    v=D[k]; s=[x[0] for x in v]
    print(f"{k[0]:22s} {k[1]:9s} n={len(v)} speedup {st.mean(s):.2f} [{min(s):.2f}-{max(s):.2f}] oom={sum(x[1] for x in v)} fb={sum(x[2] for x in v)} p99={st.mean(x[3] for x in v):.2f}s")
