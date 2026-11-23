"""Numerical unit tests for Propositions 1-4 of the paper. Run: python3 test_props.py"""
import numpy as np

from experiments import (klW_of_v, klv_of_v, pol_pair_values)

rng = np.random.default_rng(0)
PASS = []


def check(name, ok):
    PASS.append((name, bool(ok)))
    print(("PASS" if ok else "FAIL"), name)


# ---------- Proposition 1 ----------
K = 5
L = rng.uniform(0, 1, K)
u = np.ones(K) / K
lam = 0.3
phi = lam * np.log(np.mean(np.exp(L / lam)))
# (i) closed form >= Psi(w) for sampled w, equality at Gibbs w*
ws = rng.dirichlet(np.ones(K), 2000)
psis = ws @ L - lam * np.sum(ws * np.log(ws * K + 1e-300), axis=1)
wstar = np.exp(L / lam); wstar /= wstar.sum()
psi_star = wstar @ L - lam * np.sum(wstar * np.log(wstar * K))
check("P1(i) closed form = max Psi", psis.max() <= phi + 1e-9
      and abs(psi_star - phi) < 1e-9)
# (ii) certificate for random q
qs = rng.dirichlet(np.ones(K), 2000)
lhs = qs @ L
rhs = phi + lam * np.sum(qs * np.log(qs * K + 1e-300), axis=1)
check("P1(ii) certificate", np.all(lhs <= rhs + 1e-9))
# (iii) duality: tilted worst-case equals inf over lambda grid
rho = 0.4
taus = np.logspace(-3, 3, 2000)
best = -np.inf
for tau in taus:
    q = np.exp(L / tau); q /= q.sum()
    if np.sum(q * np.log(q * K)) <= rho:
        best = max(best, q @ L)
lams = np.logspace(-3, 3, 4000)
dual = min(l * np.log(np.mean(np.exp(L / l))) + l * rho for l in lams)
check("P1(iii) strong duality (grid)", abs(best - dual) < 2e-3)

# ---------- Proposition 3 ----------
for K in [4, 5]:
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    G = len(pairs)
    # formula for KL(W(v)||u_G)
    v = rng.dirichlet(np.ones(K)); v = np.clip(v, 1e-9, None); v /= v.sum()
    Wv = np.array([v[i] * v[j] for (i, j) in pairs]); W = Wv / Wv.sum()
    direct = np.sum(W * np.log(W * G))
    check(f"P4(i) KL(W(v)||uG) formula K={K}",
          abs(direct - klW_of_v(v, G)) < 1e-9)
    # (ii) identity: F(W) = Phi_pair - lam*KL(W||W*)
    Lp = rng.standard_normal(G)
    lam = 0.25
    z = Lp / lam; z -= z.max()
    Wstar = np.exp(z) / np.exp(z).sum()
    phi_pair = lam * (np.log(np.mean(np.exp(z))) + (Lp / lam).max())
    Wr = rng.dirichlet(np.ones(G))
    F = Wr @ Lp - lam * np.sum(Wr * np.log(Wr * G + 1e-300))
    ident = phi_pair - lam * np.sum(Wr * np.log((Wr + 1e-300) / Wstar))
    check(f"P3(ii) Gibbs identity K={K}", abs(F - ident) < 1e-9)
# (iii) K=3 always exact
pairs3 = [(0, 1), (0, 2), (1, 2)]
L3 = rng.standard_normal(3)
pp = pol_pair_values(L3, pairs3, 0.25, restarts=10, iters=500)
check("P3(iii) K=3 exact", abs(pp[2]) < 1e-8)
# (iii) K=4 cycle violation -> positive observed gap (lam large enough that
# the Gibbs distribution is not concentrated on a single pair)
rng4 = np.random.default_rng(7)
pairs4 = [(i, j) for i in range(4) for j in range(i + 1, 4)]
a = rng4.standard_normal(4) * 0.3
b = rng4.standard_normal(6); b = (b - b.mean()) / b.std()
L4 = np.array([a[i] + a[j] for (i, j) in pairs4]) + 0.5 * b
cyc = abs((L4[0] + L4[5]) - (L4[1] + L4[4]))  # (01)+(23) vs (02)+(13)
pp = pol_pair_values(L4, pairs4, 0.5, restarts=20, iters=800)
check("P3(iii) K=4 cycle violated & gap>0", cyc > 1e-3 and pp[2] > 1e-4)
# eta=0 exact for K=4
L40 = np.array([a[i] + a[j] for (i, j) in pairs4])
pp0 = pol_pair_values(L40, pairs4, 0.3, restarts=5, iters=400)
check("P3(iii) K=4 additive exact", abs(pp0[2]) < 1e-8)

# ---------- Proposition 4 (induced radius) ----------
for K in [3, 5, 8]:
    G = K * (K - 1) // 2
    d = rng.standard_normal(K); d -= d.mean(); d /= np.linalg.norm(d)
    v = np.ones(K) / K + 1e-4 * d
    ratio = klW_of_v(v, G) / klv_of_v(v, K)
    check(f"P4(iii) local slope K={K}",
          abs(ratio - 2 * (K - 2) / (K - 1)) < 1e-3)
# cap: point mass on a pair
K = 5; G = 10
v = np.array([0.5, 0.5, 1e-12, 1e-12, 1e-12]); v /= v.sum()
check("P4(ii) cap at rho=ln(K/2)",
      abs(klW_of_v(v, G) - np.log(G)) < 1e-6
      and abs(klv_of_v(v, K) - np.log(K / 2)) < 1e-6)

# ---------- Proposition 2 (Hoeffding coverage) ----------
Kk, n, delta = 5, 200, 0.05
c = np.sqrt(np.log(Kk / delta) / (2 * n))
true_p = rng.uniform(0.1, 0.6, Kk)
viol = 0
T = 4000
for t in range(T):
    Lhat = rng.binomial(n, true_p) / n
    if np.any(true_p > Lhat + c):
        viol += 1
check("P2 coverage >= 1-delta", viol / T <= delta)

print("\n%d/%d tests passed" % (sum(p for _, p in PASS), len(PASS)))
assert all(p for _, p in PASS)
