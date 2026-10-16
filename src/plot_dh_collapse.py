"""
Collapse in x = N*dh of the global and reduced fidelity at h_bar = 1 (TFIM, symmetric states),
from the symmetric-pair runs h = 1 -+ dh/2 of exec_orthogonality_catastrophe.py.

(a) finite-difference susceptibilities normalized by their dh -> 0 value at the same N
(b) -ln F / (N dh / 4): crossover from chi_F dh^2 (small x) to the extensive N dh / 4 (large x)

Usage (from src/):  python plot_dh_collapse.py
"""
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

project = "C:/Users/HP/Desktop/projects/2_ANNNI"
data    = f"{project}/results/data/oc_ising"
chi     = "64"
dh_ref  = 1e-4      # deep in the quadratic regime and above the reduced-fidelity precision floor

rows = []
for f in glob.glob(f"{data}/oc_ising_h_0.9*_eps_0_init_upz_chi_*.json"):
    r = json.load(open(f))
    h1, h2 = r["hs"]
    if abs((h1 + h2) / 2 - 1) > 1e-9:
        continue                      # keep only symmetric pairs around h = 1
    dh, run = h2 - h1, r["runs"][chi]
    for i, F in enumerate(run["F_global"]):
        rows.append(dict(dh=dh, N=r["Ls"][i], chiF=-np.log(F) / dh ** 2,
                         chiA1=(1 - run["F_red"]["1"][i]) / dh ** 2,
                         chiA4=(1 - run["F_red"]["4"][i]) / dh ** 2))
ref = {r["N"]: r for r in rows if abs(r["dh"] - dh_ref) < 1e-12}
Ns = sorted(ref)

# ── Style (light surface, recessive axes) ─────────────────────────────────────
ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
c_qty = {"chiF": "#2a78d6", "chiA1": "#eb6834", "chiA4": "#1baf7a"}          # categorical slots 1-3
m_qty = {"chiF": "o", "chiA1": "s", "chiA4": "^"}
lab_qty = {"chiF": r"global $\chi_F$", "chiA1": r"RFS $\chi_A$, $\ell_A=1$", "chiA4": r"RFS $\chi_A$, $\ell_A=4$"}
c_N = dict(zip(Ns, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]))         # sequential blue ramp
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 4.5, "legend.frameon": False,
})

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))

# (a) normalized susceptibilities vs x (only x >= 0.03: below that every ratio is 1)
for q in ["chiF", "chiA1", "chiA4"]:
    pts = sorted((r["N"] * r["dh"], r[q] / ref[r["N"]][q]) for r in rows if r["N"] * r["dh"] >= 0.03)
    x, y = np.array(pts).T
    ax[0].plot(x, y, m_qty[q], color=c_qty[q], mfc="none", mew=1.2, label=lab_qty[q])
ax[0].axhline(1, color=ink2, lw=1, ls="--")
ax[0].axhline(0.99, color=grid, lw=1, ls=":")
ax[0].text(0.035, 0.985, "1% deviation", color=ink2, fontsize=7, va="top")
ax[0].set_xscale("log")
ax[0].set_xlabel(r"$x = N\,\delta h$")
ax[0].set_ylabel(r"$\chi(\delta h, N)\,/\,\chi(\delta h \to 0, N)$")
ax[0].set_title(r"end of the susceptibility regime, $\bar h = 1$", color=ink)
ax[0].legend(loc="lower left")

# (b) -ln F / (N dh / 4) vs x, coloured by N
amp = np.mean([ref[N]["chiF"] / N ** 2 for N in Ns[-2:]])                    # chi_F / N^2 (open chain)
xx = np.logspace(-4.5, 1.5, 200)
ax[1].plot(xx, 4 * amp * xx, ls="--", lw=1, color=ink2, label=rf"$\chi_F\,\delta h^2$, $\chi_F\simeq N^2/{1 / amp:.1f}$")
ax[1].axhline(1, color=ink2, lw=1, ls=":", label=r"$N\delta h/4$ (exact bulk)")
for N in Ns:
    pts = sorted((r["N"] * r["dh"], r["chiF"] * r["dh"] ** 2 / (r["N"] * r["dh"] / 4)) for r in rows if r["N"] == N)
    x, y = np.array(pts).T
    ax[1].plot(x, y, "o", color=c_N[N], ms=4, label=f"$N={N}$")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_ylim(1e-5, 2)
ax[1].set_xlabel(r"$x = N\,\delta h$")
ax[1].set_ylabel(r"$-\ln\mathcal{F}\,/\,(N\delta h/4)$")
ax[1].set_title("global fidelity: quadratic to extensive", color=ink)
ax[1].legend(loc="lower right")

for a in ax:
    a.grid(True, color=grid, lw=0.6)
    a.set_axisbelow(True)

plt.tight_layout()
for ext in ("png", "pdf"):
    out = f"{project}/figures/ISING_dh_collapse_hbar_1_eps_0_chi_{chi}.{ext}"
    fig.savefig(out, dpi=300)
    print(f"Saved → {out}")
