"""Experiments for "Training learned evaluators against worst-case policy shift" (ESANN 2027).

Pure numpy. Three experiments:
  E1: pointwise evaluator, K=5 policies, spurious style correlation, ERM vs min-max.
  E2: pairwise (Bradley-Terry) judge, pair-level adversary vs rank-one policy-level adversary.
  E3: lambda sweep -> average/worst-group trade-off frontier.

Outputs: results.json, fig_groups.pdf, fig_frontier.pdf
"""
import json
import numpy as np

rng_global = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# Two-layer MLP scorer with manual gradients
# ----------------------------------------------------------------------------

def init_params(d, h, rng):
    return {
        "W1": rng.normal(0, 1.0 / np.sqrt(d), (h, d)),
        "b1": np.zeros(h),
        "w2": rng.normal(0, 1.0 / np.sqrt(h), h),
        "b2": 0.0,
    }


def forward(p, X):
    Z = X @ p["W1"].T + p["b1"]          # (n, h)
    A = np.tanh(Z)
    s = A @ p["w2"] + p["b2"]            # (n,)
    return s, (Z, A)


def backward(p, X, cache, ds):
    """ds: (n,) dLoss/dscore. Returns gradient dict (summed over samples)."""
    Z, A = cache
    gw2 = A.T @ ds
    gb2 = ds.sum()
    dA = np.outer(ds, p["w2"])
    dZ = dA * (1 - A ** 2)
    gW1 = dZ.T @ X
    gb1 = dZ.sum(axis=0)
    return {"W1": gW1, "b1": gb1, "w2": gw2, "b2": gb2}


def sgd_update(p, g, lr, wd=1e-4):
    for k in p:
        p[k] = p[k] - lr * (g[k] + wd * p[k])
    return p


def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def logistic_loss_grad(s, y):
    """y in {0,1}. Returns per-sample loss and dLoss/ds."""
    yy = 2 * y - 1
    m = -yy * s
    loss = np.where(m > 0, m + np.log1p(np.exp(-m)), np.log1p(np.exp(m)))
    ds = -yy * sigmoid(-yy * s)
    return loss, ds


# ----------------------------------------------------------------------------
# E1 data: K policies, content + style features, minority policy anti-correlated
# ----------------------------------------------------------------------------

D_CONTENT, D_STYLE = 10, 10
K1 = 5
TRAIN_PROPS = np.array([0.30, 0.30, 0.20, 0.15, 0.05])
STYLE_ALIGN = np.array([+1, +1, +1, +1, -1])  # policy 4 (minority): anti-correlated style


def make_e1_params(seed):
    rng = np.random.default_rng(10_000 + seed)
    beta = np.ones(D_CONTENT) / np.sqrt(D_CONTENT)
    gamma = rng.standard_normal(D_STYLE)
    gamma = gamma / np.linalg.norm(gamma)   # single shared style direction
    return {"beta": beta, "gamma": gamma}


def make_e1_data(n_total, rng, params, props=TRAIN_PROPS):
    beta, gamma = params["beta"], params["gamma"]
    Xs, ys, gs = [], [], []
    for k in range(K1):
        n_k = int(round(n_total * props[k]))
        xc = rng.standard_normal((n_k, D_CONTENT))
        margin = xc @ beta + 0.5 * rng.standard_normal(n_k)
        y = (margin > 0).astype(float)
        yy = 2 * y - 1
        # style: label-correlated shortcut along a direction shared by all
        # policies, with the sign flipped for the minority policy. Helpful on
        # average, worst-case misleading.
        xs = (STYLE_ALIGN[k] * yy)[:, None] * gamma[None, :] * 1.2 \
            + 0.5 * rng.standard_normal((n_k, D_STYLE))
        Xs.append(np.hstack([xc, xs]))
        ys.append(y)
        gs.append(np.full(n_k, k))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)


# ----------------------------------------------------------------------------
# Training: ERM and min-max (mirror descent-ascent over group weights)
# ----------------------------------------------------------------------------

def group_losses(p, X, y, groups, K):
    s, _ = forward(p, X)
    loss, _ = logistic_loss_grad(s, y)
    return np.array([loss[groups == k].mean() for k in range(K)])


def group_errors(p, X, y, groups, K):
    s, _ = forward(p, X)
    pred = (s > 0).astype(float)
    return np.array([(pred[groups == k] != y[groups == k]).mean() for k in range(K)])


def proj_floor_simplex(w, floor):
    """Euclidean projection onto {w in simplex : w_k >= floor}."""
    K = len(w)
    z = w - floor
    mass = 1.0 - K * floor
    # project z onto the simplex scaled to total mass `mass`
    u = np.sort(z)[::-1]
    css = np.cumsum(u) - mass
    rho = np.nonzero(u - css / (np.arange(K) + 1) > 0)[0][-1]
    tau = css[rho] / (rho + 1.0)
    return np.maximum(z - tau, 0) + floor


def train_pointwise(X, y, groups, K, lam, steps=6000, lr=0.08, eta_w=0.4,
                    batch_per_group=16, h=32, seed=1, erm=False,
                    balanced=False, w_floor=1e-8, ucb=None, ascent="eg"):
    """erm: pooled sampling, average loss. balanced: group-balanced batches,
    fixed uniform w (uniform policy-weighted ERM). Otherwise min-max with
    mirror ascent on w (lam=0 gives unregularized group DRO).
    All modes use batch_per_group*K examples and gradients per step."""
    rng = np.random.default_rng(seed)
    p = init_params(X.shape[1], h, rng)
    w = np.ones(K) / K
    idx_by_g = [np.where(groups == k)[0] for k in range(K)]
    n = len(y)
    for t in range(steps):
        if erm:
            idx = rng.integers(0, n, batch_per_group * K)
            Xb, yb = X[idx], y[idx]
            s, cache = forward(p, Xb)
            _, ds = logistic_loss_grad(s, yb)
            g = backward(p, Xb, cache, ds / len(idx))
            p = sgd_update(p, g, lr)
            continue
        # group-balanced batch
        batch_idx = [rng.choice(idx_by_g[k], batch_per_group) for k in range(K)]
        Lhat = np.zeros(K)
        grads = None
        for k in range(K):
            Xb, yb = X[batch_idx[k]], y[batch_idx[k]]
            s, cache = forward(p, Xb)
            loss, ds = logistic_loss_grad(s, yb)
            Lhat[k] = loss.mean()
            gk = backward(p, Xb, cache, ds / len(yb))
            if grads is None:
                grads = {kk: w[k] * vv for kk, vv in gk.items()}
            else:
                for kk in gk:
                    grads[kk] += w[k] * gk[kk]
        p = sgd_update(p, grads, lr)
        if balanced:
            continue  # w stays uniform
        # ascent on w:  grad_w = Lhat (+ucb bonus) - lam*(log(K w) + 1)
        L_asc = Lhat if ucb is None else Lhat + ucb
        logw = np.log(np.clip(w, 1e-12, None))
        if ascent == "euclid":
            g_w = L_asc - lam * (logw + np.log(K) + 1.0)
            w = proj_floor_simplex(w + eta_w * g_w, w_floor / K)
        else:
            w = w * np.exp(eta_w * (L_asc - lam * (logw + np.log(K))))
            w = np.maximum(w, w_floor)
            w = w / w.sum()
    return p, w


# ----------------------------------------------------------------------------
# E2: pairwise Bradley-Terry judge over policy pairs
# ----------------------------------------------------------------------------

K2 = 5


def make_e2_params(seed):
    rng = np.random.default_rng(20_000 + seed)
    beta = np.ones(D_CONTENT) / np.sqrt(D_CONTENT)
    gammas = [rng.standard_normal(D_STYLE) for _ in range(K2)]
    gammas = [g / np.linalg.norm(g) for g in gammas]
    # policy 3 mimics the style of the top policy (0) but has the worst
    # true quality: style is uninformative on pairs involving 3 and 0.
    gammas[3] = gammas[0]
    offsets = np.array([1.0, 0.5, 0.0, -2.0, -0.75])    # true mean quality
    style_sig = np.array([1.0, 0.5, 0.0, 1.0, -0.75])   # what style advertises
    return {"beta": beta, "gammas": gammas, "offsets": offsets,
            "style_sig": style_sig}


def make_e2_data(n_pairs_total, rng, params, skew=True):
    """Responses from K2 policies; comparisons grouped by unordered pair (i,j)."""
    beta, gammas, offsets = params["beta"], params["gammas"], params["offsets"]
    style_sig = params["style_sig"]
    pairs = [(i, j) for i in range(K2) for j in range(i + 1, K2)]
    if skew:
        freq = np.array([0.05 if 3 in pr else (0.4 if 4 in pr else 1.0)
                         for pr in pairs])
    else:
        freq = np.ones(len(pairs))
    freq = freq / freq.sum()

    def sample_response(k, n, alpha=0.25):
        # quality is partly visible in content (per-sample, fraction alpha of
        # the policy's mean quality plus sample variation) and partly latent at
        # the policy level. Honest styles advertise the policy; the mimic
        # (policy 3) advertises policy 0's style.
        xc = rng.standard_normal((n, D_CONTENT)) + alpha * offsets[k] * beta[None, :]
        q = xc @ beta + (1 - alpha) * offsets[k] + 0.3 * rng.standard_normal(n)
        xs = style_sig[k] * gammas[k][None, :] * np.ones((n, 1)) * 1.2 \
            + 0.6 * rng.standard_normal((n, D_STYLE))
        return np.hstack([xc, xs]), q

    Xi, Xj, lab, grp = [], [], [], []
    for g_id, (i, j) in enumerate(pairs):
        n_g = max(30, int(round(n_pairs_total * freq[g_id])))
        xi, qi = sample_response(i, n_g)
        xj, qj = sample_response(j, n_g)
        Xi.append(xi); Xj.append(xj)
        lab.append((qi > qj).astype(float))
        grp.append(np.full(n_g, g_id))
    return (np.vstack(Xi), np.vstack(Xj), np.concatenate(lab),
            np.concatenate(grp), pairs)


def pair_loss_grad(p, Xi, Xj, y):
    si, ci = forward(p, Xi)
    sj, cj = forward(p, Xj)
    loss, ds = logistic_loss_grad(si - sj, y)
    return loss, ds, ci, cj


def pair_group_metrics(p, Xi, Xj, y, grp, G):
    si, _ = forward(p, Xi)
    sj, _ = forward(p, Xj)
    loss, _ = logistic_loss_grad(si - sj, y)
    pred = (si - sj > 0).astype(float)
    errs = np.array([(pred[grp == g] != y[grp == g]).mean() for g in range(G)])
    losses = np.array([loss[grp == g].mean() for g in range(G)])
    return errs, losses


def train_pairwise(Xi, Xj, y, grp, pairs, lam, steps=6000, lr=0.08, eta_w=0.4,
                   batch_per_group=12, h=32, seed=1, mode="pair"):
    """mode in {erm, pair, rank1}. rank1: adversary weights W_ij = v_i v_j / Z."""
    rng = np.random.default_rng(seed)
    G = len(pairs)
    p = init_params(Xi.shape[1], h, rng)
    w = np.ones(G) / G                     # pair-level weights
    v = np.ones(K2) / K2                   # policy-level weights (rank1)
    idx_by_g = [np.where(grp == g)[0] for g in range(G)]
    n = len(y)
    for t in range(steps):
        if mode == "erm":
            idx = rng.integers(0, n, batch_per_group * G)
            loss, ds, ci, cj = pair_loss_grad(p, Xi[idx], Xj[idx], y[idx])
            gi = backward(p, Xi[idx], ci, ds / len(idx))
            gj = backward(p, Xj[idx], cj, -ds / len(idx))
            p = sgd_update(p, {k: gi[k] + gj[k] for k in gi}, lr)
            continue
        batch_idx = [rng.choice(idx_by_g[g], batch_per_group) for g in range(G)]
        Lhat = np.zeros(G)
        grads = None
        if mode == "rank1":
            Wv = np.array([v[i] * v[j] for (i, j) in pairs])
            w_eff = Wv / Wv.sum()
        else:
            w_eff = w
        for g in range(G):
            idx = batch_idx[g]
            loss, ds, ci, cj = pair_loss_grad(p, Xi[idx], Xj[idx], y[idx])
            Lhat[g] = loss.mean()
            gi = backward(p, Xi[idx], ci, ds / len(idx))
            gj = backward(p, Xj[idx], cj, -ds / len(idx))
            gk = {k: gi[k] + gj[k] for k in gi}
            if grads is None:
                grads = {kk: w_eff[g] * vv for kk, vv in gk.items()}
            else:
                for kk in gk:
                    grads[kk] += w_eff[g] * gk[kk]
        p = sgd_update(p, grads, lr)
        if mode == "pair":
            logw = np.log(np.clip(w, 1e-12, None))
            w = w * np.exp(eta_w * (Lhat - lam * (logw + np.log(G))))
            w = np.maximum(w, 1e-8)
            w = w / w.sum()
        elif mode == "rank1":
            v = rank1_ascent_step(v, Lhat, pairs, lam, eta_w)
    return p, (w if mode == "pair" else v)


def rank1_ascent_step(v, L, pairs, lam, eta, v_floor=1e-8):
    """One EG ascent step on F(v) = W(v).L - lam*KL(W(v)||u_G), where
    W(v)_ij = v_i v_j / Z(v), Z(v) = sum_{i<j} v_i v_j (comparisons drawn
    between two distinct policies sampled from v)."""
    K = len(v)
    G = len(pairs)
    Wv = np.array([v[i] * v[j] for (i, j) in pairs])
    Z = Wv.sum()
    W = Wv / Z
    # dF/dW_g for the entropically regularized pair objective
    M = L - lam * (np.log(G * np.clip(W, 1e-12, None)) + 1.0)
    S = W @ M
    gv = np.zeros(K)
    for g_id, (i, j) in enumerate(pairs):
        gv[i] += v[j] * M[g_id]
        gv[j] += v[i] * M[g_id]
    dZ = v.sum() - v
    gv = gv / Z - S * dZ / Z
    gv = np.clip(gv, -50.0, 50.0)   # guard against overflow near the boundary
    v = v * np.exp(eta * gv)
    v = np.maximum(v, v_floor)
    return v / v.sum()


def pol_pair_values(L, pairs, lam, restarts=100, iters=2000, eta=0.5, seed=0):
    """Phi_pair (closed form), Phi_pol (EG ascent with random restarts), the
    gap, and the identity check gap == lam*KL(W(v*)||W_gibbs)."""
    L = np.asarray(L)
    G = len(L)
    K = max(max(pr) for pr in pairs) + 1
    z = L / lam
    z = z - z.max()
    phi_pair = lam * (np.log(np.mean(np.exp(z))) + (L / lam).max())
    Wstar = np.exp(z) / np.exp(z).sum()
    rng = np.random.default_rng(seed)
    # least-squares additive fit L_ij ~ a_i + a_j + c -> smart init v ~ e^{a/lam}
    A = np.zeros((G, K + 1))
    for g_id, (i, j) in enumerate(pairs):
        A[g_id, i] = 1.0; A[g_id, j] = 1.0; A[g_id, K] = 1.0
    a_fit = np.linalg.lstsq(A, L, rcond=None)[0][:K]
    v_fit = np.exp((a_fit - a_fit.max()) / lam)
    v_fit = np.maximum(v_fit / v_fit.sum(), 1e-8); v_fit = v_fit / v_fit.sum()
    def F_of(v):
        Wv = np.array([v[i] * v[j] for (i, j) in pairs])
        W = Wv / Wv.sum()
        return float(W @ L
                     - lam * np.sum(W * np.log(G * np.clip(W, 1e-12, None))))

    def F_of_W(W):
        W = np.clip(W, 1e-300, None); W = W / W.sum()
        return float(W @ L - lam * np.sum(W * np.log(G * W)))

    best_F, best_v = -np.inf, None
    # candidate: LS-additive-fit pair distribution computed in log space
    # (W(v) with v_i ~ e^{a_i/lam}; avoids the simplex floor entirely)
    logWfit = np.array([(a_fit[i] + a_fit[j]) for (i, j) in pairs]) / lam
    logWfit -= logWfit.max()
    Wfit = np.exp(logWfit); Wfit = Wfit / Wfit.sum()
    vfit_exact = np.exp((a_fit - a_fit.max()) / lam)
    vfit_exact = vfit_exact / vfit_exact.sum()
    F = F_of_W(Wfit)
    if np.isfinite(F):
        best_F, best_v = F, np.maximum(vfit_exact, 1e-300)
        best_v = best_v / best_v.sum()
    for r in range(restarts):
        if r == 0:
            v = np.ones(K) / K
        elif r == 1:
            v = v_fit.copy()
        else:
            v = rng.dirichlet(np.ones(K))
        for cand in [v.copy()]:
            F = F_of(cand)
            if np.isfinite(F) and F > best_F:
                best_F, best_v = F, cand
        for t in range(iters):
            v = rank1_ascent_step(v, L, pairs, lam, eta)
        F = F_of(v)
        if np.isfinite(F) and F > best_F:
            best_F, best_v = F, v
    if best_v is None:
        best_v = np.ones(K) / K
        Wv = np.array([best_v[i] * best_v[j] for (i, j) in pairs])
        W = Wv / Wv.sum()
        best_F = float(W @ L - lam * np.sum(W * np.log(G * np.clip(W, 1e-12, None))))
    Wv = np.array([best_v[i] * best_v[j] for (i, j) in pairs])
    W = Wv / Wv.sum()
    kl_proj = float(np.sum(W * np.log(np.clip(W, 1e-12, None) / Wstar)))
    return float(phi_pair), float(best_F), float(phi_pair - best_F), \
        lam * kl_proj, best_v.tolist()


# ----------------------------------------------------------------------------
# v4 additions: style-randomization, polKL adversary, metrics, rho-bar
# ----------------------------------------------------------------------------

def train_pointwise_stylerand(X, y, groups, K, steps=6000, lr=0.08,
                              batch_per_group=16, h=32, seed=1):
    """ERM with style features shuffled within each minibatch (oracle-knowledge
    counterfactual augmentation: breaks the style-label association)."""
    rng = np.random.default_rng(seed)
    p = init_params(X.shape[1], h, rng)
    n = len(y)
    for t in range(steps):
        idx = rng.integers(0, n, batch_per_group * K)
        Xb, yb = X[idx].copy(), y[idx]
        perm = rng.permutation(len(idx))
        Xb[:, D_CONTENT:] = Xb[perm, D_CONTENT:]
        s, cache = forward(p, Xb)
        _, ds = logistic_loss_grad(s, yb)
        g = backward(p, Xb, cache, ds / len(idx))
        p = sgd_update(p, g, lr)
    return p


def polkl_ascent_step(v, L, pairs, tau, eta, v_floor=1e-8):
    """EG ascent step on W(v).L - tau*KL(v||u_K): the DIRECT policy-space
    adversary (penalty form)."""
    K = len(v)
    Wv = np.array([v[i] * v[j] for (i, j) in pairs])
    Z = Wv.sum()
    W = Wv / Z
    S = W @ L
    gv = np.zeros(K)
    for g_id, (i, j) in enumerate(pairs):
        gv[i] += v[j] * L[g_id]
        gv[j] += v[i] * L[g_id]
    dZ = v.sum() - v
    gv = gv / Z - S * dZ / Z
    gv = gv - tau * (np.log(np.clip(v, 1e-12, None) * K) + 1.0)
    gv = np.clip(gv, -50.0, 50.0)
    v = v * np.exp(eta * gv)
    v = np.maximum(v, v_floor)
    return v / v.sum()


def train_pairwise_v4(Xi, Xj, y, grp, pairs, lam, steps=6000, lr=0.08,
                      eta_w=0.4, batch_per_group=12, h=32, seed=1,
                      mode="pair"):
    """mode in {erm, balanced, pair, rank1, polKL}."""
    rng = np.random.default_rng(seed)
    G = len(pairs)
    p = init_params(Xi.shape[1], h, rng)
    w = np.ones(G) / G
    v = np.ones(K2) / K2
    idx_by_g = [np.where(grp == g)[0] for g in range(G)]
    n = len(y)
    for t in range(steps):
        if mode == "erm":
            idx = rng.integers(0, n, batch_per_group * G)
            loss, ds, ci, cj = pair_loss_grad(p, Xi[idx], Xj[idx], y[idx])
            gi = backward(p, Xi[idx], ci, ds / len(idx))
            gj = backward(p, Xj[idx], cj, -ds / len(idx))
            p = sgd_update(p, {k: gi[k] + gj[k] for k in gi}, lr)
            continue
        batch_idx = [rng.choice(idx_by_g[g], batch_per_group) for g in range(G)]
        Lhat = np.zeros(G)
        grads = None
        if mode in ("rank1", "polKL"):
            Wv = np.array([v[i] * v[j] for (i, j) in pairs])
            w_eff = Wv / Wv.sum()
        else:
            w_eff = w
        for g in range(G):
            idx = batch_idx[g]
            loss, ds, ci, cj = pair_loss_grad(p, Xi[idx], Xj[idx], y[idx])
            Lhat[g] = loss.mean()
            gi = backward(p, Xi[idx], ci, ds / len(idx))
            gj = backward(p, Xj[idx], cj, -ds / len(idx))
            gk = {k: gi[k] + gj[k] for k in gi}
            if grads is None:
                grads = {kk: w_eff[g] * vv for kk, vv in gk.items()}
            else:
                for kk in gk:
                    grads[kk] += w_eff[g] * gk[kk]
        p = sgd_update(p, grads, lr)
        if mode == "pair":
            logw = np.log(np.clip(w, 1e-12, None))
            w = w * np.exp(eta_w * (Lhat - lam * (logw + np.log(G))))
            w = np.maximum(w, 1e-8)
            w = w / w.sum()
        elif mode == "rank1":
            v = rank1_ascent_step(v, Lhat, pairs, lam, eta_w)
        elif mode == "polKL":
            v = polkl_ascent_step(v, Lhat, pairs, lam, eta_w)
        # mode == balanced: keep w uniform
    return p


def auroc(scores, labels):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(np.argsort(allv))  # ranks 0..n-1
    r_pos = order[:len(pos)].sum() + len(pos)
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) /
                 (len(pos) * len(neg)))


def ece(scores, labels, bins=10):
    pconf = sigmoid(scores)
    e = 0.0
    n = len(labels)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (pconf >= lo) & (pconf < hi) if b < bins - 1 else \
            (pconf >= lo) & (pconf <= hi)
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(labels[m].mean() - pconf[m].mean())
    return float(e)


def group_metric(fn, p, X, y, groups, K):
    s, _ = forward(p, X)
    return np.array([fn(s[groups == k], y[groups == k]) for k in range(K)])


# ---- induced radius rho_bar ------------------------------------------------

def klW_of_v(v, G):
    Z = (1 - np.sum(v ** 2)) / 2
    return float((1 / Z) * np.sum(v * (1 - v) * np.log(v))
                 - np.log(Z) + np.log(G))


def klv_of_v(v, K):
    return float(np.sum(v * np.log(v * K)))


def rho_bar_curve(K, rho_grid, n_samp=120000, seed=0):
    """Numerically computed rho_bar(rho) = sup KL(W(v)||u_G) over
    KL(v||u)<=rho. Dense Dirichlet sampling at several concentrations plus
    penalized EG-ascent refinement; exact cap log G for rho >= log(K/2)."""
    rng = np.random.default_rng(seed)
    G = K * (K - 1) // 2
    pts = []
    for alpha in [0.03, 0.1, 0.3, 1.0, 3.0]:
        V = rng.dirichlet(np.ones(K) * alpha, size=n_samp // 5)
        V = np.clip(V, 1e-12, None)
        V = V / V.sum(axis=1, keepdims=True)
        Z = (1 - np.sum(V ** 2, axis=1)) / 2
        ok = Z > 1e-9
        V = V[ok]; Z = Z[ok]
        klw = (1 / Z) * np.sum(V * (1 - V) * np.log(V), axis=1) \
            - np.log(Z) + np.log(G)
        klv = np.sum(V * np.log(V * K), axis=1)
        pts.append(np.stack([klv, klw], axis=1))
    # penalized ascent refinement
    def grad_klW(v):
        Z = (1 - np.sum(v ** 2)) / 2
        S = np.sum(v * (1 - v) * np.log(v))
        Sp = (1 - 2 * v) * np.log(v) + (1 - v)
        return Sp / Z + S * v / Z ** 2 + v / Z
    for tau in np.logspace(-1.5, 1.5, 25):
        for r in range(6):
            v = rng.dirichlet(np.ones(K)) if r else np.ones(K) / K
            for t in range(800):
                g = grad_klW(v) - tau * (np.log(v * K) + 1)
                v = v * np.exp(0.3 * np.clip(g, -30, 30))
                v = np.maximum(v, 1e-12); v = v / v.sum()
            pts.append(np.array([[klv_of_v(v, K), klW_of_v(v, G)]]))
    pts = np.vstack(pts)
    out = []
    cap_rho = np.log(K / 2)
    for rho in rho_grid:
        if rho >= cap_rho:
            out.append(float(np.log(G)))
        else:
            feas = pts[pts[:, 0] <= rho + 1e-12, 1]
            val = float(feas.max()) if len(feas) else 0.0
            out.append(min(val, float(np.log(G))))  # KL to u_G <= log G
    return out


# ----------------------------------------------------------------------------
# Staged experiment driver (v4). results.json is updated incrementally.
# ----------------------------------------------------------------------------

def load_results():
    import os
    if os.path.exists("results.json"):
        return json.load(open("results.json"))
    return {}


def save_results(results):
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


def ms_list(v):
    return {"mean": float(np.mean(v)), "std": float(np.std(v))}


def stage1():
    """Pointwise: E1 (7 methods + AUROC/ECE), E3 (+lambda selection), E4
    certificate, E9 multi-angle held-out, E8 rho-bar curves."""
    import time
    results = load_results()
    n_seeds = 10
    lam_main = 0.05
    delta = 0.05
    props = TRAIN_PROPS
    methods = ["erm", "balanced", "gdro", "minmax", "ucb", "stylerand",
               "content"]
    e1 = {m: {"worst": [], "macro": [], "pooled_trainmix": [],
              "worst_auroc": [], "worst_ece": []} for m in methods}
    per_policy = {m: [] for m in methods}
    timings = {}
    models = {}
    for seed in range(n_seeds):
        params = make_e1_params(seed)
        rng = np.random.default_rng(100 + seed)
        Xtr, ytr, gtr = make_e1_data(8000, rng, params)
        Xte, yte, gte = make_e1_data(10000, np.random.default_rng(900 + seed),
                                     params, props=np.ones(K1) / K1)
        n_tr = np.array([(gtr == k).sum() for k in range(K1)])
        c_tr = np.sqrt(np.log(K1 / delta) / (2 * n_tr))
        def timed(name, fn):
            t0 = time.time(); out = fn(); timings.setdefault(name, []).append(
                time.time() - t0); return out
        trained = {
            "erm": timed("erm", lambda: train_pointwise(
                Xtr, ytr, gtr, K1, lam=0, erm=True, seed=seed)[0]),
            "balanced": timed("balanced", lambda: train_pointwise(
                Xtr, ytr, gtr, K1, lam=0, balanced=True, seed=seed)[0]),
            "gdro": timed("gdro", lambda: train_pointwise(
                Xtr, ytr, gtr, K1, lam=0.0, seed=seed)[0]),
            "minmax": timed("minmax", lambda: train_pointwise(
                Xtr, ytr, gtr, K1, lam=lam_main, seed=seed)[0]),
            "ucb": timed("ucb", lambda: train_pointwise(
                Xtr, ytr, gtr, K1, lam=lam_main, ucb=c_tr, seed=seed)[0]),
            "stylerand": timed("stylerand", lambda: train_pointwise_stylerand(
                Xtr, ytr, gtr, K1, seed=seed)),
            "content": timed("content", lambda: train_pointwise(
                Xtr[:, :D_CONTENT], ytr, gtr, K1, lam=0, balanced=True,
                seed=seed)[0]),
        }
        models[seed] = (params, trained)
        for name, p in trained.items():
            Xev = Xte[:, :D_CONTENT] if name == "content" else Xte
            errs = group_errors(p, Xev, yte, gte, K1)
            aucs = group_metric(auroc, p, Xev, yte, gte, K1)
            eces = group_metric(ece, p, Xev, yte, gte, K1)
            e1[name]["worst"].append(float(errs.max()))
            e1[name]["macro"].append(float(errs.mean()))
            e1[name]["pooled_trainmix"].append(float(props @ errs))
            e1[name]["worst_auroc"].append(float(aucs.min()))
            e1[name]["worst_ece"].append(float(eces.max()))
            per_policy[name].append(errs.tolist())
    e1["per_policy_mean"] = {k: np.array(v).mean(axis=0).tolist()
                             for k, v in per_policy.items()}
    e1["per_policy_std"] = {k: np.array(v).std(axis=0).tolist()
                            for k, v in per_policy.items()}
    e1["c_train"] = c_tr.tolist()
    e1["lam"] = lam_main
    results["E1"] = e1
    results["timings_sec_per_run"] = {k: ms_list(v) for k, v in timings.items()}

    # E3 frontier + validation-based lambda selection (5 seeds)
    lams = [1.0, 0.3, 0.1, 0.05, 0.02, 0.01, 0.0]
    fr = {"lams": lams, "worst": [], "macro": [], "worst_std": [],
          "macro_std": [], "selected": {"lam": [], "worst": [], "macro": []}}
    for seed in range(5):
        params = make_e1_params(seed)
        rng = np.random.default_rng(100 + seed)
        Xtr, ytr, gtr = make_e1_data(8000, rng, params)
        Xva, yva, gva = make_e1_data(5000, np.random.default_rng(700 + seed),
                                     params, props=np.ones(K1) / K1)
        Xte, yte, gte = make_e1_data(10000, np.random.default_rng(900 + seed),
                                     params, props=np.ones(K1) / K1)
        row_w, row_m, row_vw = [], [], []
        for lam in lams:
            p_mm, _ = train_pointwise(Xtr, ytr, gtr, K1, lam=lam, seed=seed)
            errs = group_errors(p_mm, Xte, yte, gte, K1)
            verrs = group_errors(p_mm, Xva, yva, gva, K1)
            row_w.append(float(errs.max())); row_m.append(float(errs.mean()))
            row_vw.append(float(verrs.max()))
        fr.setdefault("per_seed_worst", []).append(row_w)
        fr.setdefault("per_seed_macro", []).append(row_m)
        i_sel = int(np.argmin(row_vw))
        fr["selected"]["lam"].append(lams[i_sel])
        fr["selected"]["worst"].append(row_w[i_sel])
        fr["selected"]["macro"].append(row_m[i_sel])
    pw = np.array(fr["per_seed_worst"]); pm = np.array(fr["per_seed_macro"])
    fr["worst"] = pw.mean(0).tolist(); fr["worst_std"] = pw.std(0).tolist()
    fr["macro"] = pm.mean(0).tolist(); fr["macro_std"] = pm.std(0).tolist()
    fr["erm_worst"] = float(np.mean(e1["erm"]["worst"]))
    fr["erm_macro"] = float(np.mean(e1["erm"]["macro"]))
    fr["balanced_worst"] = float(np.mean(e1["balanced"]["worst"]))
    fr["balanced_macro"] = float(np.mean(e1["balanced"]["macro"]))
    results["E3"] = fr

    # E4 certificate (unequal n_k), 10 seeds, reusing models
    def worst_mixture_curve(L, rhos):
        L = np.asarray(L); Kn = len(L); u = np.ones(Kn) / Kn
        taus = np.logspace(-3, 3, 400)
        kls, vals = [], []
        for tau in taus:
            z = L / tau; z = z - z.max()
            q = u * np.exp(z); q = q / q.sum()
            kls.append(float(np.sum(q * np.log(q / u))))
            vals.append(float(q @ L))
        kls = np.array(kls); vals = np.array(vals)
        base = float(u @ L)
        out = []
        for rho in rhos:
            feas = vals[kls <= rho + 1e-12]
            out.append(float(max(feas.max() if len(feas) else base, base)))
        return out

    def certificate_curve(Lc, rhos):
        L = np.asarray(Lc)
        lgrid = np.logspace(-3, 2, 300)
        out = []
        for rho in rhos:
            vals = []
            for lam in lgrid:
                z = L / lam; z = z - z.max()
                vals.append(lam * (np.log(np.mean(np.exp(z)))
                                   + (L / lam).max()) + lam * rho)
            out.append(float(min(min(vals), L.max())))
        return out

    n_val = np.array([1500, 1500, 1000, 750, 250])
    c_k = np.sqrt(np.log(K1 / delta) / (2 * n_val))
    rhos = np.linspace(0, np.log(K1), 25).tolist()
    e4 = {"rhos": rhos, "n_val": n_val.tolist(), "delta": delta,
          "c_k": c_k.tolist(), "erm": [], "minmax": [], "erm_cert": [],
          "minmax_cert": []}
    for seed in range(n_seeds):
        params, trained = models[seed]
        Xte, yte, gte = make_e1_data(10000, np.random.default_rng(900 + seed),
                                     params, props=np.ones(K1) / K1)
        Xva, yva, gva = make_e1_data(int(n_val.sum()),
                                     np.random.default_rng(700 + seed),
                                     params, props=n_val / n_val.sum())
        for name in ["erm", "minmax"]:
            p = trained[name]
            err_te = group_errors(p, Xte, yte, gte, K1)
            err_va = group_errors(p, Xva, yva, gva, K1)
            e4[name].append(worst_mixture_curve(err_te, rhos))
            e4[name + "_cert"].append(certificate_curve(err_va + c_k, rhos))
    for k in ["erm", "minmax", "erm_cert", "minmax_cert"]:
        e4[k + "_mean"] = np.array(e4[k]).mean(axis=0).tolist()
        e4[k + "_std"] = np.array(e4[k]).std(axis=0).tolist()
    for name in ["erm", "minmax"]:
        ok = (np.array(e4[name + "_cert"]) >= np.array(e4[name]) - 1e-9)
        e4[name + "_cert_valid_frac"] = float(ok.mean())
    results["E4"] = e4

    # E9: held-out policies at several angles (10 seeds, all methods)
    angles = [0, 30, 45, 60, 90]
    e9 = {m: {str(a): [] for a in angles} for m in methods}
    for seed in range(n_seeds):
        params, trained = models[seed]
        rng = np.random.default_rng(600 + seed)
        gperp = rng.standard_normal(D_STYLE)
        gperp -= (gperp @ params["gamma"]) * params["gamma"]
        gperp /= np.linalg.norm(gperp)
        for a in angles:
            th = np.deg2rad(a)
            gnew = np.cos(th) * params["gamma"] + np.sin(th) * gperp
            n5 = 4000
            xc = rng.standard_normal((n5, D_CONTENT))
            margin = xc @ params["beta"] + 0.5 * rng.standard_normal(n5)
            y5 = (margin > 0).astype(float)
            yy = 2 * y5 - 1
            xs = (-1.0 * yy)[:, None] * gnew[None, :] * 1.2 \
                + 0.5 * rng.standard_normal((n5, D_STYLE))
            X5 = np.hstack([xc, xs])
            for name, p in trained.items():
                Xev = X5[:, :D_CONTENT] if name == "content" else X5
                s, _ = forward(p, Xev)
                e9[name][str(a)].append(
                    float(((s > 0).astype(float) != y5).mean()))
    results["E9"] = {m: {a: ms_list(v) for a, v in d.items()}
                     for m, d in e9.items()}

    # E8: rho-bar curves
    e8 = {}
    for K in [3, 5, 8]:
        cap = np.log(K / 2)
        grid = np.linspace(0, np.log(K), 40).tolist()
        e8[str(K)] = {"rho": grid, "rho_bar": rho_bar_curve(K, grid),
                      "cap_rho": float(cap),
                      "logG": float(np.log(K * (K - 1) / 2)),
                      "slope0": float(2 * (K - 2) / (K - 1))}
    results["E8"] = e8
    save_results(results)
    print("stage1 done")
    for m in methods:
        print(f"E1 {m:10s} worst={np.mean(e1[m]['worst']):.3f}"
              f"±{np.std(e1[m]['worst']):.3f}"
              f" macro={np.mean(e1[m]['macro']):.3f}"
              f" wAUC={np.mean(e1[m]['worst_auroc']):.3f}"
              f" wECE={np.mean(e1[m]['worst_ece']):.3f}")
    print("selected lam:", fr["selected"])


def stage2():
    """Pairwise: E2 (5 modes + certified interval), E6 stress (+restart
    stability), E7 minority-size sweep."""
    results = load_results()
    n_seeds = 10
    lam2 = 0.05
    modes = ["erm", "balanced", "pair", "rank1", "polKL"]
    e2 = {m: {"worst": [], "avg": []} for m in modes}
    gaps = []
    for seed in range(n_seeds):
        params2 = make_e2_params(seed)
        rng = np.random.default_rng(200 + seed)
        Xi, Xj, y, grp, pairs = make_e2_data(6000, rng, params2)
        Xi_t, Xj_t, y_t, grp_t, _ = make_e2_data(
            8000, np.random.default_rng(800 + seed), params2, skew=False)
        G = len(pairs)
        trained2 = {}
        for mode in modes:
            p = train_pairwise_v4(Xi, Xj, y, grp, pairs,
                                  lam=(0 if mode in ("erm", "balanced")
                                       else lam2),
                                  mode=mode, seed=seed)
            trained2[mode] = p
            errs, _ = pair_group_metrics(p, Xi_t, Xj_t, y_t, grp_t, G)
            e2[mode]["worst"].append(float(errs.max()))
            e2[mode]["avg"].append(float(errs.mean()))
            e2[mode].setdefault("per_pair", []).append(errs.tolist())
        _, Lte = pair_group_metrics(trained2["pair"], Xi_t, Xj_t, y_t,
                                    grp_t, G)
        ppv = pol_pair_values(Lte, pairs, lam2, restarts=50, iters=1500,
                              seed=seed)
        gaps.append({"phi_pair": ppv[0], "phi_pol_tilde": ppv[1],
                     "gap_obs": ppv[2], "lam_klproj": ppv[3],
                     "v_star": ppv[4]})
    e2["lam"] = lam2
    e2["pairs"] = pairs
    e2["gaps"] = gaps
    e2["gap_obs_mean"] = float(np.mean([g["gap_obs"] for g in gaps]))
    e2["gap_obs_rel"] = float(np.mean([g["gap_obs"] / g["phi_pair"]
                                       for g in gaps]))
    results["E2"] = e2

    # E6 stress test: K x eta x lambda, 3 draws, 20 restarts
    etas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    lams_st = [0.1, 0.3]
    e6 = {"etas": etas, "lams": lams_st, "gap": {}}
    for K in [3, 4, 5, 8]:
        prs = [(i, j) for i in range(K) for j in range(i + 1, K)]
        for lam_st in lams_st:
            rows = []
            for eta in etas:
                gs = []
                for dr in range(3):
                    rng = np.random.default_rng(3000 + 17 * K + dr)
                    a = rng.standard_normal(K)
                    b = rng.standard_normal(len(prs))
                    b = (b - b.mean()) / b.std()
                    L = np.array([a[i] + a[j] for (i, j) in prs]) + eta * b
                    pp = pol_pair_values(L, prs, lam_st, restarts=20,
                                         iters=800, seed=dr)
                    gs.append(pp[2])
                rows.append(float(np.mean(gs)))
            e6["gap"][f"K{K}_lam{lam_st}"] = rows
    # restart stability at (K=5, eta=0.3, lam=0.3)
    prs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    rng = np.random.default_rng(3000 + 17 * 5)
    a = rng.standard_normal(5)
    b = rng.standard_normal(len(prs)); b = (b - b.mean()) / b.std()
    L = np.array([a[i] + a[j] for (i, j) in prs]) + 0.3 * b
    e6["restart_stability"] = {
        str(R): pol_pair_values(L, prs, 0.3, restarts=R, iters=800, seed=0)[2]
        for R in [1, 5, 20, 50]}
    results["E6"] = e6

    # E7: minority training-size sweep, minmax vs ucb vs erm (5 seeds)
    delta = 0.05
    n_mins = [40, 100, 200, 400, 800]
    e7 = {"n_min": n_mins,
          **{m: {str(n): [] for n in n_mins} for m in ["erm", "minmax",
                                                       "ucb"]}}
    for n_min in n_mins:
        counts = np.array([2400, 2400, 1600, 1200, n_min])
        props7 = counts / counts.sum()
        for seed in range(5):
            params = make_e1_params(seed)
            rng = np.random.default_rng(100 + seed)
            Xtr, ytr, gtr = make_e1_data(int(counts.sum()), rng, params,
                                         props=props7)
            Xte, yte, gte = make_e1_data(10000,
                                         np.random.default_rng(900 + seed),
                                         params, props=np.ones(K1) / K1)
            n_tr = np.array([(gtr == k).sum() for k in range(K1)])
            c_tr = np.sqrt(np.log(K1 / delta) / (2 * n_tr))
            bpg = min(16, max(8, n_min // 4))
            runs = {
                "erm": train_pointwise(Xtr, ytr, gtr, K1, lam=0, erm=True,
                                       seed=seed)[0],
                "minmax": train_pointwise(Xtr, ytr, gtr, K1, lam=0.05,
                                          batch_per_group=bpg, seed=seed)[0],
                "ucb": train_pointwise(Xtr, ytr, gtr, K1, lam=0.05, ucb=c_tr,
                                       batch_per_group=bpg, seed=seed)[0],
            }
            for name, p in runs.items():
                errs = group_errors(p, Xte, yte, gte, K1)
                e7[name][str(n_min)].append(float(errs.max()))
    results["E7"] = {"n_min": n_mins,
                     **{m: {n: ms_list(v) for n, v in e7[m].items()}
                        for m in ["erm", "minmax", "ucb"]}}
    save_results(results)
    print("stage2 done")
    for m in modes:
        print(f"E2 {m:9s} worst={np.mean(e2[m]['worst']):.3f}"
              f"±{np.std(e2[m]['worst']):.3f}"
              f" macro={np.mean(e2[m]['avg']):.3f}")
    print("gap_obs mean %.2e rel %.4f%%" % (e2["gap_obs_mean"],
                                            100 * e2["gap_obs_rel"]))
    print("restart stability:", e6["restart_stability"])
    print("E7:", json.dumps(results["E7"], indent=0)[:600])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "stage2":
        stage2()
    else:
        stage1()
