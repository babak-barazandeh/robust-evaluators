"""Supplement figure: minority-size sweep + held-out angle sweep."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
})
r = json.load(open("results.json"))
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.1))

ax = axes[0]
e7 = r["E7"]
nm = e7["n_min"]
for m, col, lab in [("erm", "#c0392b", "ERM"),
                    ("minmax", "#2471a3", "min--max"),
                    ("ucb", "#148f77", "confidence-aware (UCB)")]:
    mu = [e7[m][str(n)]["mean"] for n in nm]
    sd = [e7[m][str(n)]["std"] for n in nm]
    ax.errorbar(nm, mu, yerr=sd, fmt="o-", ms=3.5, lw=1.3, capsize=2,
                color=col, label=lab)
ax.set_xscale("log")
ax.set_xlabel("minority-policy training examples $n_5$")
ax.set_ylabel("worst-policy test error")
ax.set_title("(a) confidence-aware DRO under scarcity")
ax.legend(frameon=False)

ax = axes[1]
e9 = r["E9"]
angles = [0, 30, 45, 60, 90]
for m, col, lab in [("erm", "#c0392b", "ERM"),
                    ("balanced", "#b7950b", "balanced ERM"),
                    ("minmax", "#2471a3", "min--max"),
                    ("stylerand", "#6c3483", "style randomization"),
                    ("content", "#148f77", "content oracle")]:
    mu = [e9[m][str(a)]["mean"] for a in angles]
    sd = [e9[m][str(a)]["std"] for a in angles]
    ax.errorbar(angles, mu, yerr=sd, fmt="o-", ms=3.5, lw=1.3, capsize=2,
                color=col, label=lab)
ax.axhline(0.5, ls=":", c="gray", lw=0.8)
ax.set_xlabel(r"held-out style angle to $\gamma$ (deg)")
ax.set_ylabel("error on held-out policy")
ax.set_title("(b) held-out policies")
ax.legend(frameon=False, fontsize=6.5)

ax = axes[2]
e11 = r["E11"]
nk = e11["n"]
ax.plot(nk, e11["emp_mean"], "s--", c="gray", lw=1.1, ms=3.5,
        label="empirical worst-mix")
ax.errorbar(nk, e11["cert_mean"], yerr=e11["cert_std"], fmt="o-", ms=3.5,
            lw=1.3, capsize=2, c="#2471a3", label="certificate (Prop. 2)")
ax.set_xscale("log")
ax.set_xlabel(r"validation size $n_k$")
ax.set_ylabel(r"worst-mixture error at $\rho=\log K$")
ax.set_title("(c) certificate tightness")
ax.legend(frameon=False, fontsize=6.8)

fig.tight_layout(pad=0.4)
fig.savefig("fig_supp.pdf", bbox_inches="tight")
print("fig_supp written")
