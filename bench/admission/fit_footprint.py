"""Fit and evaluate the query-aware footprint model.

Input: <run>.fp.tsv files written by SIRIUS_FOOTPRINT_LOG
  fingerprint, peak_bytes, projected, valid, f0..f7, query_head
Model: peak ~= c0 + sum_i ci * fi   (ci >= 0), fit by NNLS on relative error.
Reports prediction error per evaluation split and the coefficient string for
SIRIUS_ADMIT_PLAN_COEFS.
usage: fit_footprint.py --train a.fp.tsv [...] --test b.fp.tsv [...] [--out coefs.json]
"""
import argparse, collections, json
import numpy as np

NAMES = ["scan", "scan_capped", "join_build", "join_out", "agg", "sort", "base", "base_capped"]


def load(paths):
    by = collections.defaultdict(list)
    head = {}
    for p in paths:
        for line in open(p):
            r = line.rstrip("\n").split("\t")
            if len(r) < 11 or r[3] != "1":
                continue
            nf = min(8, len(r) - 5)  # 6 (old logs) or 8 features; later logs add need_fast/need_min
            f = np.zeros(8); f[:nf] = [float(x) for x in r[4:4 + nf]]
            if not f.any():
                continue  # trivial queries (no scan) carry no signal
            by[r[0]].append((int(r[1]), f))
            head[r[0]] = r[-1][:40]
    X, y, keys = [], [], []
    for k, v in by.items():
        X.append(v[0][1]); y.append(float(np.median([p for p, _ in v]))); keys.append(k)
    return np.array(X), np.array(y), keys, head


def nnls(A, b, iters=5000):
    """Lawson-Hanson active-set NNLS."""
    n = A.shape[1]; x = np.zeros(n); P = np.zeros(n, bool)
    w = A.T @ (b - A @ x)
    for _ in range(iters):
        if P.all() or w[~P].max() <= 1e-12:
            break
        j = np.argmax(np.where(P, -np.inf, w)); P[j] = True
        while True:
            z = np.zeros(n); z[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if (z[P] > 0).all():
                x = z; break
            neg = P & (z <= 0)
            alpha = np.min(x[neg] / (x[neg] - z[neg]))
            x = x + alpha * (z - x); P &= x > 1e-12
        w = A.T @ (b - A @ x)
    return x


def fit(X, y, cols):
    # relative-error weighting: divide each row by its target
    A = np.hstack([np.ones((len(y), 1)), X[:, cols]]) / y[:, None]
    c = nnls(A, np.ones(len(y)))
    full = np.zeros(1 + X.shape[1]); full[0] = c[0]; full[1 + np.array(cols)] = c[1:]
    return full


def predict(c, X):
    return c[0] + X @ c[1:]


def report(name, yhat, y):
    ape = np.abs(yhat - y) / y
    under = (yhat < y).mean()
    print(f"  {name:34s} median APE {np.median(ape)*100:6.1f}%  p90 {np.percentile(ape,90)*100:6.1f}%  "
          f"underestimates {under*100:4.0f}%  worst under {np.min(yhat/y):5.2f}x")
    return dict(median_ape=float(np.median(ape)), p90_ape=float(np.percentile(ape, 90)),
                under_frac=float(under), worst_ratio=float(np.min(yhat / y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    Xtr, ytr, ktr, _ = load(a.train)
    Xte, yte, kte, head = load(a.test)
    print(f"train {len(ytr)} queries, test {len(yte)} queries")
    res = {}
    allc = list(range(8))
    models = {
        "plan model (all features)": allc,
        "scan bytes only": [0],
        "scan + join build": [0, 2],
        "base capped + join build + agg": [7, 2, 4],
        "capped scans + breakers": [1, 7, 2, 4, 5],
    }
    print("held-out test set (fit on train only):")
    coefs = {}
    for name, cols in models.items():
        c = fit(Xtr, ytr, cols); coefs[name] = c
        res[name] = report(name, predict(c, Xte), yte)
    const = np.median(ytr) * yte.max() / max(ytr.max(), 1)  # scale-aware constant baseline
    res["constant (train median, scaled)"] = report("constant (train median, scaled)", np.full_like(yte, const), yte)
    # leave-one-query-out inside the test set: how well the features explain footprints at one scale
    loo = []
    for i in range(len(yte)):
        m = np.arange(len(yte)) != i
        loo.append(predict(fit(Xte[m], yte[m], allc), Xte[i:i + 1])[0])
    print("leave-one-query-out within test set:")
    res["plan model LOQO"] = report("plan model (all features), LOQO", np.array(loo), yte)
    c = coefs["plan model (all features)"]
    print("coefficients (c0 bytes, then per feature):", {**{"c0": c[0]}, **dict(zip(NAMES, c[1:]))})
    coef_str = ",".join(f"{v:.6g}" for v in c)
    print("SIRIUS_ADMIT_PLAN_COEFS=" + coef_str)
    print("per-query (test):")
    for k, yy, p in sorted(zip(kte, yte, predict(c, Xte)), key=lambda t: -t[1]):
        print(f"  {head[k]!r:44s} measured {yy/2**30:6.2f} GB  predicted {p/2**30:6.2f} GB")
    if a.out:
        json.dump(dict(coefs=coef_str, results=res, n_train=len(ytr), n_test=len(yte)), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
