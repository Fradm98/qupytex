"""
Off-critical vs critical regime of the global and reduced fidelity susceptibilities in the
symmetric-state scan of exec_fidelity_scan.py (TFIM = ANNNI k = 0, dh = 1e-4: x = N dh <= 0.02,
so -ln F = chi_F dh^2 and 1 - F_A = chi_A dh^2 throughout).

Thermodynamic-limit references (no fit parameters):
  chi_F / N = 1 / [16 (1 - h^2)] (h < 1),  1 / [16 h^2 (h^2 - 1)] (h > 1)      (Damski 2013)
  chi_A(l_A = 1) = (d_h m_z)^2 / [4 (1 - m_z^2)],
  m_z(h) = (1/pi) int_0^pi (h - cos k) / sqrt(1 + h^2 - 2 h cos k) dk          (Pfeuty 1970)
The second line is the exact single-site Uhlmann fidelity of a parity-symmetric state.

Usage (from src/):  python plot_scan_regimes.py
"""
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

project = "C:/Users/HP/Desktop/projects/2_ANNNI"
data    = f"{project}/results/data/oc_ising/scan"
dh      = 1e-4
sx_cut  = 5e-6     # drop (nearly) symmetry-broken states, see analyze_fidelity_scan.py


def load():
    out = {}
    for f in glob.glob(f"{data}/scan_eps_0_L_*_chi_32_dh_0.0001*.json"):
        r = json.load(open(f))
        pts = out.setdefault(r["L"], {})
        for p in r["coarse"] + r["refine"]:
            if max(abs(p["m_mid"][0][0]), abs(p["m_mid"][1][0])) > sx_cut:
                continue
            pts[round(p["h_bar"], 7)] = (p["chi_F"], p["chi_A"]["1"])
    return {L: (np.array(sorted(p)), *np.array([p[h] for h in sorted(p)]).T) for L, p in sorted(out.items())}


def chiF_density(h):
    h = np.asarray(h)
    return np.where(h < 1, 1 / (16 * (1 - h ** 2)), 1 / (16 * h ** 2 * (h ** 2 - 1)))


def mz_inf(h):
    return quad(lambda k: (h - np.cos(k)) / np.sqrt(1 + h * h - 2 * h * np.cos(k)), 0, np.pi, limit=200)[0] / np.pi


def chiA1_inf(h, e=1e-5):
    dmz = (mz_inf(h + e) - mz_inf(h - e)) / (2 * e)
    m = mz_inf(h)
    return dmz ** 2 / (4 * (1 - m ** 2))


scan = load()
hh = np.concatenate([np.linspace(0.895, 0.999, 300), np.linspace(1.001, 1.055, 200)])
cF = chiF_density(hh)
cA = np.array([chiA1_inf(h) for h in hh])

# Regime boundary: smallest |y| beyond which the data are within 5% of the infinite chain
print("ratio to the thermodynamic limit (5% boundary in y = N (h - 1)):")
for L, (h, chiF, chiA) in scan.items():
    y = L * (h - 1)
    RF = chiF / (L * chiF_density(h))
    RA = chiA / np.array([chiA1_inf(x) for x in h])
    for name, R in [("chi_F", RF), ("chi_A1", RA)]:
        bad = np.abs(R - 1) > 0.05
        yl = y[(y < 0) & bad].min() if np.any((y < 0) & bad) else np.nan
        yr = y[(y > 0) & bad].max() if np.any((y > 0) & bad) else np.nan
        print(f"  N={L:3d} {name:6s}: |R-1| > 5% for y in [{yl:+.1f}, {yr:+.1f}];"
              f" R at window edges = {R[0]:.3f} (h={h[0]:.3f}), {R[-1]:.3f} (h={h[-1]:.3f})")

# ── Figure ────────────────────────────────────────────────────────────────────
ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
c_L = dict(zip(scan, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]))
c_ref = "#eb6834"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 3, "legend.frameon": False,
})
fig, ax = plt.subplots(2, 2, figsize=(9.5, 7))

# (a) global: chi_F / N vs h, with -ln F / N on the right axis
for L, (h, chiF, chiA) in scan.items():
    ax[0, 0].plot(h, chiF / L, "o-", ms=2, lw=1, color=c_L[L], label=f"$N={L}$")
for part in (hh < 1, hh > 1):
    ax[0, 0].plot(hh[part], cF[part], "--", lw=1.2, color=c_ref, label=r"$N\to\infty$" if part[0] else None)
ax[0, 0].set_yscale("log"); ax[0, 0].set_ylim(0.2, 20)
ax[0, 0].set_xlabel(r"$\bar h$"); ax[0, 0].set_ylabel(r"$\chi_F / N$")
ax[0, 0].set_title("global: susceptibility per site", color=ink)
sec = ax[0, 0].secondary_yaxis("right", functions=(lambda v: v * dh ** 2, lambda v: v / dh ** 2))
sec.set_ylabel(r"$-\ln\mathcal{F}\,/\,N$")
ax[0, 0].legend(loc="upper left")

# (b) reduced: chi_A(1) vs h, with 1 - F_A on the right axis
for L, (h, chiF, chiA) in scan.items():
    ax[0, 1].plot(h, chiA, "o-", ms=2, lw=1, color=c_L[L], label=f"$N={L}$")
for part in (hh < 1, hh > 1):
    ax[0, 1].plot(hh[part], cA[part], "--", lw=1.2, color=c_ref, label=r"$N\to\infty$" if part[0] else None)
ax[0, 1].set_ylim(0.2, 1.8)
ax[0, 1].set_xlabel(r"$\bar h$"); ax[0, 1].set_ylabel(r"$\chi_A$ ($\ell_A=1$)")
ax[0, 1].set_title("reduced: single-site susceptibility", color=ink)
sec = ax[0, 1].secondary_yaxis("right", functions=(lambda v: v * dh ** 2, lambda v: v / dh ** 2))
sec.set_ylabel(r"$1-F_A$")
ax[0, 1].legend(loc="upper left")

# (c), (d) ratio to the thermodynamic limit vs y = N (h - 1)
for L, (h, chiF, chiA) in scan.items():
    y = L * (h - 1)
    ax[1, 0].plot(y, chiF / (L * chiF_density(h)), "o", ms=2.5, color=c_L[L], label=f"$N={L}$")
    ax[1, 1].plot(y, chiA / np.array([chiA1_inf(x) for x in h]), "o", ms=2.5, color=c_L[L], label=f"$N={L}$")
for a, name in [(ax[1, 0], r"\chi_F / (N\,\chi_F^\infty/N)"), (ax[1, 1], r"\chi_A / \chi_A^\infty")]:
    a.axhline(1, color=ink2, lw=1, ls="--")
    a.axhspan(0.95, 1.05, color=grid, alpha=0.7, lw=0)
    a.set_xlabel(r"$y = N(\bar h - 1)$")
    a.set_ylabel(f"${name}$")
    a.legend(loc="upper left")
ax[1, 0].set_yscale("log")
ax[1, 0].legend(loc="lower right")
ax[1, 0].set_title("global: ratio to the infinite chain", color=ink)
ax[1, 1].set_title("reduced: ratio to the infinite chain", color=ink)

for a in ax.flat:
    a.grid(True, color=grid, lw=0.6)
    a.set_axisbelow(True)
plt.tight_layout()
for ext in ("png", "pdf"):
    out = f"{project}/figures/ISING_fidelity_scan_regimes_eps_0_chi_32.{ext}"
    fig.savefig(out, dpi=300)
    print(f"Saved → {out}")
