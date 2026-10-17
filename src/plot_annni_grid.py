"""
Global vs reduced fidelity susceptibility across the floating phase of the ANNNI chain (k = 0.75),
from the grid runs of exec_fidelity_grid.py (neighbouring-point finite differences, dh = grid step).

Pairs involving a symmetry-broken state (|<sx>_mid| > sx_cut) are dropped.
Reference transitions (Beccaria, Campostrini, Feo, PRB 76, 094410 (2007), Table I, k = 0.75):
  B1 = 0.2574(2)  commensurate-incommensurate (antiphase -> floating)
  B2 ~ 0.424(3) (gap), 0.44(1) (entropy), 0.425(10) (domain walls)   floating -> paramagnet (BKT)

Usage (from src/):  python plot_annni_grid.py
"""
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

project = "C:/Users/HP/Desktop/projects/2_ANNNI"
data    = f"{project}/results/data/annni_grid"
sx_cut  = 1e-3
B1, B2, B2_err = 0.2574, 0.435, 0.015     # B2: spread of the three determinations in Table I


def load(L):
    f = glob.glob(f"{data}/grid_k_0.75_eps_0_L_{L}_chi_64_h_0.27-0.47_dh_0.002.json")[0]
    r = json.load(open(f))
    broken = np.array([abs(p["m_mid"][0]) > sx_cut for p in r["points"]])
    ok = ~(broken[:-1] | broken[1:])                                  # pair j uses points j, j+1
    hb = np.array([q["h_bar"] for q in r["pairs"]])[ok]
    chiF = np.array([q["chi_F"] for q in r["pairs"]])[ok]
    chiA = {b: np.array([q["chi_A"][b] for q in r["pairs"]])[ok] for b in r["pairs"][0]["chi_A"]}
    return dict(L=L, h=hb, chiF=chiF, chiA=chiA, n_drop=int((~ok).sum()), dh=r["dh"])


Ls = [50, 100, 150, 200]
runs = {}
for L in Ls:
    try:
        runs[L] = load(L)
    except IndexError:
        print(f"L={L}: no file yet")

ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
c_L = dict(zip(Ls, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]))
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "lines.markersize": 2.5, "legend.frameon": False,
})
fig, ax = plt.subplots(2, 2, figsize=(10, 7))
panels = [(ax[0, 0], "chiF", r"$\chi_F\,/\,N$", "global fidelity susceptibility per site", True),
          (ax[0, 1], "1",    r"$\chi_A$ ($\ell_A=1$)", "reduced, single site", False),
          (ax[1, 0], "4",    r"$\chi_A$ ($\ell_A=4$)", "reduced, 4 sites", False)]
for a, key, ylab, title, per_site in panels:
    for L, r in runs.items():
        y = r["chiF"] / L if per_site else r["chiA"][key]
        a.plot(r["h"], y, "o-", color=c_L[L], label=f"$N={L}$")
    a.set_yscale("log"); a.set_ylabel(ylab); a.set_title(title, color=ink)

# contrast panel: chi_F / N over chi_A(1), a unit-free comparison of the two signals
for L, r in runs.items():
    ax[1, 1].plot(r["h"], (r["chiF"] / L) / r["chiA"]["1"], "o-", color=c_L[L], label=f"$N={L}$")
ax[1, 1].set_yscale("log"); ax[1, 1].set_ylabel(r"$(\chi_F/N)\,/\,\chi_A(\ell_A=1)$")
ax[1, 1].set_title("global per site over reduced", color=ink)

for a in ax.flat:
    a.axvline(B1, color=ink2, lw=1, ls="--")
    a.axvspan(B2 - B2_err, B2 + B2_err, color=grid, alpha=0.8, lw=0)
    a.set_xlabel(r"$\bar h$"); a.grid(True, color=grid, lw=0.6); a.set_axisbelow(True); a.set_xlim(0.27, 0.47)
    a.legend(loc="best")
ax[0, 0].text(B1, 1.02, " $B_1$", transform=ax[0, 0].get_xaxis_transform(), color=ink2, fontsize=7, va="bottom")
ax[0, 0].text(B2, 1.02, " $B_2$", transform=ax[0, 0].get_xaxis_transform(), color=ink2, fontsize=7, va="bottom")
plt.tight_layout()
for ext in ("png", "pdf"):
    out = f"{project}/figures/ANNNI_k0.75_fidelity_grid_eps_0_chi_64.{ext}"
    fig.savefig(out, dpi=300)
    print(f"Saved → {out}")
for L, r in runs.items():
    print(f"N={L}: {len(r['h'])} pairs kept, {r['n_drop']} dropped (broken); "
          f"chi_F/N range {np.min(r['chiF']/L):.3g}-{np.max(r['chiF']/L):.3g}; chi_A(1) range {r['chiA']['1'].min():.3g}-{r['chiA']['1'].max():.3g}")
