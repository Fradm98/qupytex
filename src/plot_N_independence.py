"""
N-dependence at fixed h of the single-site reduced fidelity susceptibility.

(left)  Ising (ANNNI k = 0, symmetric states, scan of exec_fidelity_scan.py, dh = 1e-4):
        chi_A(l_A = 1)(N) / chi_A^inf(h) vs N, with the exact infinite-chain value
        chi_A^inf = (d_h m_z)^2 / (4 (1 - m_z^2)),  m_z(h) = (1/pi) int_0^pi (h - cos k)/sqrt(1 + h^2 - 2 h cos k) dk
(right) ANNNI k = 0.75 (exec_fidelity_grid.py, N = 50, 100, 150): <sigma^z> on the central site vs N.

Usage (from src/):  python plot_N_independence.py
"""
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

project = "C:/Users/HP/Desktop/projects/2_ANNNI"
ising   = f"{project}/results/data/oc_ising/scan"
annni   = f"{project}/results/data/annni_grid"


def mz_inf(h):
    return quad(lambda k: (h - np.cos(k)) / np.sqrt(1 + h * h - 2 * h * np.cos(k)), 0, np.pi, limit=200)[0] / np.pi


def chiA1_inf(h, e=1e-5):
    dmz = (mz_inf(h + e) - mz_inf(h - e)) / (2 * e)
    return dmz ** 2 / (4 * (1 - mz_inf(h) ** 2))


# ── Ising: chi_A(1) vs N at fixed h (coarse grid is the same for all N) ───────
Ls = [50, 100, 150, 200]
coarse = {}
for L in Ls:
    f = glob.glob(f"{ising}/scan_eps_0_L_{L}_chi_32_dh_0.0001.json")[0]
    r = json.load(open(f))
    coarse[L] = {round(p["h_bar"], 5): p for p in r["coarse"]
                 if max(abs(p["m_mid"][0][0]), abs(p["m_mid"][1][0])) < 5e-6}
common = sorted(set.intersection(*[set(c) for c in coarse.values()]))
targets = [0.951, 0.978, 0.990, 1.011, 1.023, 1.050]
hs = [min(common, key=lambda x: abs(x - t)) for t in targets]

ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#6250d6", "#eda100"]
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.3, "lines.markersize": 4.5, "legend.frameon": False,
})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for c, h in zip(cols, hs):
    ratio = [coarse[L][h]["chi_A"]["1"] / chiA1_inf(h) for L in Ls]
    xi = 1 / abs(np.log(h))
    ax[0].plot(Ls, ratio, "o-", color=c, label=rf"$h={h:.3f}$, $\xi={xi:.0f}$")
ax[0].axhline(1, color=ink2, lw=1, ls="--")
ax[0].set_xlabel("$N$"); ax[0].set_ylabel(r"$\chi_A(N)\,/\,\chi_A^\infty(h)$  ($\ell_A=1$)")
ax[0].set_title("Ising: approach to the infinite chain", color=ink)
ax[0].legend(loc="best")

# ── ANNNI k = 0.75: central-site <sigma^z> vs N ───────────────────────────────
D = {}
for L in (50, 100, 150):
    r = json.load(open(glob.glob(f"{annni}/grid_k_0.75_eps_0_L_{L}_chi_64*.json")[0]))
    D[L] = r["points"]
for c, h0 in zip(cols, [0.34, 0.38, 0.40, 0.44, 0.46, 0.47]):
    y = []
    for L in D:
        p = min(D[L], key=lambda q: abs(q["h"] - h0))
        y.append(np.nan if abs(p["m_mid"][0]) > 1e-3 else p["m_mid"][1])
    ax[1].plot(list(D), y, "o-", color=c, label=f"$h={h0}$")
ax[1].set_xlabel("$N$"); ax[1].set_ylabel(r"$\langle\sigma^z\rangle$ (central site)")
ax[1].set_title(r"ANNNI $\kappa=0.75$: three sizes only", color=ink)
ax[1].legend(loc="best")
for a in ax:
    a.grid(True, color=grid, lw=0.6); a.set_axisbelow(True)
plt.tight_layout()
for ext in ("png", "pdf"):
    out = f"{project}/figures/N_independence_ising_vs_annni.{ext}"
    fig.savefig(out, dpi=300)
    print(f"Saved → {out}")
print("Ising rows (N=50,100,150,200) of chi_A/chi_A_inf:")
for h in hs:
    print(f"  h={h:.3f}  xi={1/abs(np.log(h)):.0f}: ", np.round([coarse[L][h]['chi_A']['1'] / chiA1_inf(h) for L in Ls], 3).tolist())
