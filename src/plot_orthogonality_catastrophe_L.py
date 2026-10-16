"""
Global vs reduced fidelity for the critical pair h = 0.9797 -> 1.0033 (TFIM = ANNNI k=0),
combining the chi = 64 runs of exec_orthogonality_catastrophe.py over L = 50 ... 800.

Usage (from src/):  python plot_orthogonality_catastrophe_L.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt

project = "C:/Users/HP/Desktop/projects/2_ANNNI"
data    = f"{project}/results/data/oc_ising"
tag     = "h_0.9797-1.0033"
chi     = "64"

# Pinned (eps = 1e-2) series, stitched over the three L ranges
pinned_files = [f"oc_ising_{tag}_chi_32-64.json",
                f"oc_ising_{tag}_chi_64_L_400-800.json",
                f"oc_ising_{tag}_chi_64_L_600-800.json"]
sym_file = f"oc_ising_{tag}_eps_0_init_upz_chi_64.json"

blocks = ["1", "4", "16"]


def load(files):
    L, F, Fr, meta = [], [], {b: [] for b in blocks}, None
    for name in files:
        with open(f"{data}/{name}") as f:
            r = json.load(f)
        meta = meta or r
        run = r["runs"][chi]
        n = len(run["F_global"])          # files are saved after each L: keep completed points only
        L += r["Ls"][:n]
        F += run["F_global"]
        for b in blocks:
            Fr[b] += run["F_red"][b][:n]
    order = np.argsort(L)
    L = np.array(L, dtype=float)[order]
    F = np.array(F)[order]
    Fr = {b: np.array(v, dtype=float)[order] for b, v in Fr.items()}
    return L, F, Fr, meta


L, F, Fr, meta = load(pinned_files)
Ls, Fs, Frs, _ = load([sym_file])
h1, h2 = meta["hs"]
alpha = meta["alpha_exact"]
xi1, xi2 = 1 / abs(np.log(h1)), 1 / abs(np.log(h2))

# ── Style (light surface, recessive axes; categorical slots 1-2, blue ramp for l_A) ──
ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
c_pin, c_sym = "#2a78d6", "#eb6834"
c_blocks = {"1": "#86b6ef", "4": "#2a78d6", "16": "#104281"}
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 4.5, "legend.frameon": False,
})


def xi_marks(ax, y_text):
    for xi, name in [(xi1, r"$\xi_1$"), (xi2, r"$\xi_2$")]:
        ax.axvline(xi, color=grid, lw=1, ls=":", zorder=0)
        ax.text(xi, y_text, f" {name}", color=ink2, fontsize=7, va="top", transform=ax.get_xaxis_transform())


fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
xx = np.linspace(0, L.max() * 1.02, 200)

# (a) -ln F vs L
ax[0].plot(xx, alpha * xx, ls="--", lw=1, color=ink2,
           label=rf"exact bulk $\alpha L$, $\alpha={alpha * 1e3:.2f}\times10^{{-3}}$")
ax[0].plot(L, -np.log(F), "o-", color=c_pin, label=r"pinned, $\varepsilon=10^{-2}$")
ax[0].plot(Ls, -np.log(Fs), "s", mfc="none", mew=1.2, color=c_sym, label=r"symmetric, $\varepsilon=0$")
ax[0].set_xlabel("$L$"); ax[0].set_ylabel(r"$-\ln \mathcal{F}$")
ax[0].set_title(rf"global fidelity, $h={h1:.4f}\to{h2:.4f}$", color=ink)
xi_marks(ax[0], 0.98)
ax[0].legend(loc="lower right")

# (b) rate per site
ax[1].axhline(alpha, ls="--", lw=1, color=ink2, label=r"exact bulk $\alpha$")
ax[1].plot(L, -np.log(F) / L, "o-", color=c_pin, label=r"pinned, $\varepsilon=10^{-2}$")
ax[1].plot(Ls, -np.log(Fs) / Ls, "s", mfc="none", mew=1.2, color=c_sym, label=r"symmetric, $\varepsilon=0$")
ax[1].set_xlabel("$L$"); ax[1].set_ylabel(r"$-\ln \mathcal{F}\,/\,L$")
ax[1].set_title("rate per site", color=ink)
ax[1].set_ylim(0, None)
xi_marks(ax[1], 0.98)
ax[1].legend(loc="lower right")

# (c) reduced vs global infidelity (pinned)
ax[2].plot(L, 1 - F, "s--", color=ink2, lw=1.2, label=r"global $1-\mathcal{F}$")
for b in blocks:
    ax[2].plot(L, 1 - Fr[b], "o-", color=c_blocks[b], label=rf"reduced, $\ell_A$={b}")
ax[2].set_yscale("log")
ax[2].set_xlabel("$L$"); ax[2].set_ylabel(r"$1-\mathcal{F}$")
ax[2].set_title(r"reduced vs global, pinned $\varepsilon=10^{-2}$", color=ink)
xi_marks(ax[2], 0.98)
ax[2].legend(loc="lower right")

for a in ax:
    a.grid(True, color=grid, lw=0.6)
    a.set_axisbelow(True)
    a.set_xlim(0, L.max() * 1.02)

plt.tight_layout()
for ext in ("png", "pdf"):
    out = f"{project}/figures/ISING_orthogonality_catastrophe_{tag}_chi_{chi}_L_50-800.{ext}"
    fig.savefig(out, dpi=300)
    print(f"Saved → {out}")
