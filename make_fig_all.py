"""Single 2x2 figure for the main paper."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 6.2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
})
r = json.load(open("results.json"))
C_ERM, C_BAL, C_MM = "#c0392b", "#b7950b", "#2471a3"
fig, axes = plt.subplots(2, 2, figsize=(5.9, 2.0))

ax = axes[0, 0]
K = 5
x = np.arange(K)
for off, name, col, lab in [(-0.27, "erm", C_ERM, "ERM"),
                            (0.0, "balanced", C_BAL, "balanced ERM"),
                            (0.27, "minmax", C_MM, "min--max")]:
    m = np.array(r["E1"]["per_policy_mean"][name])
    s = np.array(r["E1"]["per_policy_std"][name])
    ax.bar(x + off, m, 0.27, yerr=s, color=col, label=lab, capsize=1.2)
ax.axhline(0.5, ls=":", c="gray", lw=0.7)
ax.set_xticks(x); ax.set_xticklabels([f"$P_{k}$" for k in range(K)])
ax.set_ylabel("test error")
ax.set_title("(a) per-policy test error")
ax.legend(frameon=False, loc="upper left", labelspacing=0.3, ncol=2,
          columnspacing=1.0, handlelength=1.2, fontsize=6.5)

ax = axes[0, 1]
e4 = r["E4"]
rhos = np.array(e4["rhos"])
for name, col, lab in [("erm", C_ERM, "ERM"), ("minmax", C_MM, "min--max")]:
    m = np.array(e4[name + "_mean"]); s = np.array(e4[name + "_std"])
    ax.plot(rhos, m, c=col, lw=1.3, label=lab)
    ax.fill_between(rhos, m - s, m + s, color=col, alpha=0.15, lw=0)
    ax.plot(rhos, np.array(e4[name + "_cert_mean"]), c=col, lw=0.9, ls="--")
ax.set_xlabel(r"shift radius $\rho$")
ax.set_ylabel("worst-mix. error", fontsize=7, labelpad=1.5)
ax.set_title("(b) certified error under shift")
ax.set_xlim(right=2.05)
ax.text(1.66, e4["erm_mean"][-1], "ERM", color=C_ERM, fontsize=6.5,
        va="center", ha="left")
ax.text(1.66, e4["minmax_mean"][-1], "min--max", color=C_MM, fontsize=6.5,
        va="center", ha="left")
ax.text(0.03, 0.97, "solid: empirical\ndashed: certificate", fontsize=6,
        va="top", ha="left", transform=ax.transAxes, color="0.35")

ax = axes[1, 0]
cols = {"3": "#7d3c98", "5": "#2471a3", "8": "#148f77"}
for K_ in ["3", "5", "8"]:
    d = r["E8"][K_]
    rho = np.array(d["rho"]); rb = np.array(d["rho_bar"])
    ax.plot(rho, rb, c=cols[K_], lw=1.3)
    ax.plot([0, d["cap_rho"] * 0.45], [0, d["slope0"] * d["cap_rho"] * 0.45],
            c=cols[K_], lw=0.7, ls=":")
    ax.plot([d["cap_rho"]], [d["logG"]], "o", c=cols[K_], ms=3)
    ax.text(rho[-1] + 0.03, rb[-1], f"$K={K_}$", c=cols[K_], fontsize=6.5,
            va="center", ha="left")
ax.set_xlim(right=2.4)
ax.set_xlabel(r"policy radius $\rho$")
ax.set_ylabel(r"$\tilde\rho(\rho)$")
ax.set_title(r"(c) induced radius: estimate $\tilde\rho$")

ax = axes[1, 1]
e6 = r["E6"]
etas = e6["etas"]
cd = {"3": "#7d3c98", "4": "#b03a2e", "5": "#2471a3", "8": "#148f77"}
for K_ in ["3", "4", "5", "8"]:
    y = 100 * np.array(e6["gap"][f"K{K_}_lam0.3"])
    lab = f"$K={K_}$" if K_ in ("3", "4") else None
    ax.plot(etas, y, "o-", ms=2.5, lw=1.1, c=cd[K_], label=lab)
    if K_ in ("5", "8"):
        ax.text(1.02, y[-1], f"$K={K_}$", color=cd[K_], fontsize=6.5,
                va="center", ha="left")
ax.set_xlim(right=1.17)
ax.set_xlabel(r"non-separability $\eta$")
ax.set_ylabel(r"gap $\times10^{-2}$")
ax.set_title("(d) relaxation-gap stress test")
ax.legend(frameon=False, loc="upper left", handlelength=1.1,
          labelspacing=0.2)

fig.tight_layout(pad=0.3, w_pad=3.0, h_pad=0.6)
fig.savefig("fig_all.pdf", bbox_inches="tight")
print("fig_all written")
