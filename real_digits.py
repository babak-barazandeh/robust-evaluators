"""Real-data group-shift experiment (supplement).

Real inputs and labels: sklearn digits (1797 8x8 images, offline), task =
digit parity (even/odd), which varies within every source group. Source
groups = K=5 k-means clusters of the raw pixels: data-driven but real modes
of the input distribution (writing/shape styles). Training mix is skewed
over sources; test is per-source. A second variant holds one source out of
training entirely. No planted shortcut: this measures what robust training
buys on real heterogeneity.

Writes results_digits.json.
"""
import json
import numpy as np
from sklearn.datasets import load_digits

from experiments import train_pointwise, group_errors

K = 5
LAM = 0.05


def kmeans(X, k, rng, iters=60):
    # k-means++ init
    centers = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(centers)
    for _ in range(iters):
        g = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if (g == j).any():
                C[j] = X[g == j].mean(0)
    return np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)


def main():
    d = load_digits()
    X = d.data / 16.0
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    y = (d.target % 2).astype(float)

    res = {"full": {}, "heldout": {}}
    methods = ["erm", "balanced", "gdro", "minmax"]
    acc = {v: {m: {"worst": [], "macro": []} for m in methods}
           for v in ["full"]}
    ho = {m: [] for m in methods}
    train_counts_log = []
    for seed in range(10):
        rng = np.random.default_rng(42 + seed)
        for attempt in range(8):
            g = kmeans(X, K, rng)
            if min(np.sum(g == j) for j in range(K)) >= 60:
                break
        # order clusters by size (largest first) for a stable skew assignment
        order = np.argsort([-np.sum(g == j) for j in range(K)])
        relabel = np.empty(K, dtype=int)
        relabel[order] = np.arange(K)
        g = relabel[g]
        props = np.array([0.45, 0.30, 0.15, 0.07, 0.03])
        want = np.round(props * 800).astype(int)
        tr_idx, te_idx = [], []
        counts = []
        for j in range(K):
            idx = np.where(g == j)[0]
            rng.shuffle(idx)
            n_tr = max(12, min(want[j], int(0.8 * len(idx))))
            counts.append(int(n_tr))
            tr_idx.append(idx[:n_tr])
            te_idx.append(idx[n_tr:])
        train_counts_log.append(counts)
        tr = np.concatenate(tr_idx); te = np.concatenate(te_idx)
        Xtr, ytr, gtr = X[tr], y[tr], g[tr]
        Xte, yte, gte = X[te], y[te], g[te]

        def run(mode, Xt, yt, gt, k):
            kw = dict(steps=4000, batch_per_group=8, seed=seed)
            if mode == "erm":
                return train_pointwise(Xt, yt, gt, k, lam=0, erm=True, **kw)[0]
            if mode == "balanced":
                return train_pointwise(Xt, yt, gt, k, lam=0, balanced=True,
                                       **kw)[0]
            if mode == "gdro":
                return train_pointwise(Xt, yt, gt, k, lam=0.0, **kw)[0]
            return train_pointwise(Xt, yt, gt, k, lam=LAM, **kw)[0]

        # (a) all sources in training
        for m in methods:
            p = run(m, Xtr, ytr, gtr, K)
            errs = group_errors(p, Xte, yte, gte, K)
            acc["full"][m]["worst"].append(float(errs.max()))
            acc["full"][m]["macro"].append(float(errs.mean()))
        # (b) hold the smallest source (4) out of training entirely
        keep = gtr != 4
        g4 = np.where(g == 4)[0]
        for m in methods:
            p = run(m, Xtr[keep], ytr[keep], gtr[keep], K - 1)
            from experiments import forward
            s, _ = forward(p, X[g4])
            ho[m].append(float(((s > 0).astype(float) != y[g4]).mean()))

    res["train_counts"] = train_counts_log
    res["full"] = {m: {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                       for k, v in acc["full"][m].items()} for m in methods}
    res["heldout_source"] = {m: {"mean": float(np.mean(v)),
                                 "std": float(np.std(v))} for m, v in ho.items()}
    with open("results_digits.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res["full"], indent=1))
    print("held-out source:", json.dumps(res["heldout_source"], indent=1))
    print("train counts:", train_counts_log[0])


if __name__ == "__main__":
    main()
