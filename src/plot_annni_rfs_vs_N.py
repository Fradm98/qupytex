"""
RFS of 2- and 4-site central blocks vs N at fixed h, ANNNI k = 0.75 (exec_fidelity_grid.py data,
N = 50, 100, 150, dh = 2e-3). Operators are not used: only reduced fidelities of block RDMs.

Usage (from src/):  python plot_annni_rfs_vs_N.py
"""
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

project = "C:/Users/HP/Desktop/projects/2_ANNNI"
data    = f"{project}/results/data/annni_grid"
Ls      = (50, 100, 150)
hs      = [0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.47]

D = {}
for L in Ls:
    r = json.load(open(glob.glob(f"{data}/grid_k_0.75_eps_0_L_{L}_chi_64*.json")[0]))
    broken = np.array([abs(p["m_mid"][0]) > 1e-3 for p in r["points"]])     # symmetry check only
    ok = ~(broken[:-1] | broken[1:])
    D[L] = dict(h=np.array([q["h_bar"] for q in r["pairs"]]), ok=ok,
                A={b: np.array([q["chi_A"][b] for q in r["pairs"]]) for b in ("2", "4")})

ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#6250d6", "#eda100", "#52514e", "#0b0b0b"]
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "lines.markersize": 4.5, "legend.frameon": False,
})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
print("chi_A at fixed h (N = 50 | 100 | 150)")
for a, b in zip(ax, ("2", "4")):
    print(f" l_A={b}")
    for c, h0 in zip(cols, hs):
        y = []
        for L in Ls:
            i = int(np.argmin(abs(D[L]["h"] - h0)))
            y.append(D[L]["A"][b][i] if D[L]["ok"][i] else np.nan)
        a.plot(Ls, y, "o-", color=c, label=f"$h={h0}$")
        print(f"   h={h0}: " + " | ".join(f"{v:8.3f}" for v in y))
    a.set_yscale("log"); a.set_xlabel("$N$"); a.set_ylabel(rf"$\chi_A$ ($\ell_A={b}$)")
    a.set_title(rf"RFS of a central block of {b} sites vs $N$ (ANNNI $\kappa=0.75$)", color=ink)
    a.legend(loc="best", ncol=2); a.grid(True, color=grid, lw=0.6); a.set_axisbelow(True)
plt.tight_layout()
for ext in ("png", "pdf"):
    out = f"{project}/figures/ANNNI_k0.75_rfs_vs_N_blocks_2_4.{ext}"
    fig.savefig(out, dpi=300)
    print(f"Saved → {out}")
