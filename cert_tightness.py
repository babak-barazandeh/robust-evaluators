"""E11: certificate tightness vs. validation size n_k (companion Fig. S1c).
Reuses the pointwise min--max evaluator; sweeps n_k at radius rho=log K.
Run: python3 cert_tightness.py  (appends E11 to results.json)."""
import json
import numpy as np
from experiments import (make_e1_params, make_e1_data, train_pointwise,
                         group_errors, K1)

delta = 0.05

def worst_mix(L, rho):
    L = np.asarray(L); K = len(L); u = np.ones(K)/K; best = float(u@L)
    for tau in np.logspace(-3, 3, 400):
        z = L/tau; z -= z.max(); q = u*np.exp(z); q /= q.sum()
        if np.sum(q*np.log(q/u)) <= rho + 1e-12:
            best = max(best, float(q@L))
    return best

def cert(Lc, rho):
    L = np.asarray(Lc)
    return min(min(lam*(np.log(np.mean(np.exp((L/lam)-(L/lam).max())))
                        + (L/lam).max()) + lam*rho
                   for lam in np.logspace(-3, 2, 300)), float(L.max()))

def main():
    rho = float(np.log(K1))
    nvals = [50, 100, 250, 500, 1500, 5000]
    emp, crt = {n: [] for n in nvals}, {n: [] for n in nvals}
    for seed in range(10):
        params = make_e1_params(seed)
        Xtr, ytr, gtr = make_e1_data(8000, np.random.default_rng(100+seed), params)
        Xte, yte, gte = make_e1_data(10000, np.random.default_rng(900+seed),
                                     params, props=np.ones(K1)/K1)
        p, _ = train_pointwise(Xtr, ytr, gtr, K1, lam=0.05, seed=seed)
        e_wm = worst_mix(group_errors(p, Xte, yte, gte, K1), rho)
        for n in nvals:
            Xv, yv, gv = make_e1_data(n*K1, np.random.default_rng(700+seed+n),
                                      params, props=np.ones(K1)/K1)
            ck = np.sqrt(np.log(K1/delta)/(2*n))
            emp[n].append(e_wm)
            crt[n].append(cert(group_errors(p, Xv, yv, gv, K1)+ck, rho))
    r = json.load(open("results.json"))
    r["E11"] = {"rho": rho, "n": nvals,
                "emp_mean": [float(np.mean(emp[n])) for n in nvals],
                "cert_mean": [float(np.mean(crt[n])) for n in nvals],
                "cert_std": [float(np.std(crt[n])) for n in nvals]}
    json.dump(r, open("results.json", "w"), indent=2)
    for n in nvals:
        print(f"n_k={n:5d}  emp {np.mean(emp[n]):.3f}  cert {np.mean(crt[n]):.3f}")

if __name__ == "__main__":
    main()
