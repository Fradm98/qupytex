"""
Finite-size scaling of the global (chi_F) and reduced (chi_A) fidelity susceptibilities of the
TFIM (ANNNI k = 0) across h_c = 1, from the symmetric-state scans of exec_fidelity_scan.py.

  * peaks: local parabolic fit of chi(h_bar) around the maximum (symmetry-broken points dropped)
  * h*(L) = h_c + a / L        -> pseudo-critical shift and h_c(L -> inf)
  * chi_F^max ~ L^{2/nu},  chi_A^max ~ (ln L)^2  (alpha = 0)
  * single-site identity in symmetric states: chi_A(1) = (d_h m_z)^2 / (4 (1 - m_z^2))
  * collapse chi_F / L^2 vs L (h_bar - 1)

Usage (from src/):  python analyze_fidelity_scan.py
"""
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

project = "C:/Users/HP/Desktop/projects/2_ANNNI"
data    = f"{project}/results/data/oc_ising/scan"
qtys    = ["chi_F", "chi_A1", "chi_A2", "chi_A4", "chi_A8"]


def load():
    out = {}
    for f in glob.glob(f"{data}/scan_eps_0_L_*_chi_32_dh_0.0001*.json"):
        r = json.load(open(f))
        pts = out.setdefault(r["L"], {})
        for p in r["coarse"] + r["refine"]:
            # Drop (nearly) symmetry-broken states. Even a leak <sx> ~ 1e-5 matters: divided by
            # dh = 1e-4 it adds an O(0.1) slope d<sx>/dh to the single-site RDM, i.e. to chi_A.
            if max(abs(p["m_mid"][0][0]), abs(p["m_mid"][1][0])) > 5e-6:
                continue
            mz1, mz2 = p["m_mid"][0][1], p["m_mid"][1][1]
            pts[round(p["h_bar"], 7)] = dict(
                chi_F=p["chi_F"], **{f"chi_A{b}": p["chi_A"][b] for b in p["chi_A"]},
                dmz=(mz2 - mz1) / (p["h2"] - p["h1"]), mz=(mz1 + mz2) / 2)
    res = {}
    for L, pts in out.items():
        hb = np.array(sorted(pts))
        res[L] = dict(h=hb, **{k: np.array([pts[h][k] for h in hb]) for k in pts[hb[0]]})
    return dict(sorted(res.items()))


def peak(h, y, half=3):
    """Vertex of a parabola through the 2*half+1 points around the maximum."""
    i = int(np.argmax(y))
    sl = slice(max(i - half, 0), min(i + half + 1, len(h)))
    c = np.polyfit(h[sl] - h[i], y[sl], 2)
    x0 = -c[1] / (2 * c[0])
    return h[i] + x0, np.polyval(c, x0)


scan = load()
Ls = np.array(list(scan), dtype=float)
pk = {q: np.array([peak(scan[L]["h"], scan[L][q]) for L in scan]) for q in qtys}

print("peaks (position, height):")
for q in qtys:
    print(f"  {q:7s}", "  ".join(f"L={int(L)}: {p[0]:.5f}, {p[1]:.5g}" for L, p in zip(Ls, pk[q])))

# Pseudo-critical points: h*(L) = h_c + a/L (+ b/L^2)
print("\nh*(L) = h_c + a/L + b/L^2:")
fits_pos = {}
for q in qtys:
    A = np.vstack([np.ones_like(Ls), 1 / Ls, 1 / Ls ** 2]).T
    hc, a, b = np.linalg.lstsq(A, pk[q][:, 0], rcond=None)[0]
    fits_pos[q] = (hc, a, b)
    print(f"  {q:7s} h_c = {hc:.5f}, a = {a:+.3f}, b = {b:+.2f}")

# Peak heights
lnF = np.log(pk["chi_F"][:, 1])
print("\nchi_F^max: local exponents d ln chi / d ln L =",
      np.round(np.diff(lnF) / np.diff(np.log(Ls)), 3), " (2/nu = 2)")
print("  chi_F^max / L^2 =", np.round(pk["chi_F"][:, 1] / Ls ** 2, 5))
print("chi_A^max vs (ln L)^2: slope between consecutive L")
for q in qtys[1:]:
    s = np.diff(pk[q][:, 1]) / np.diff(np.log(Ls) ** 2)
    print(f"  {q:7s}", np.round(s, 4))

# Single-site identity
print("\nsingle-site identity chi_A(1) / [ (d_h m_z)^2 / (4 (1 - m_z^2)) ]:")
for L in scan:
    s = scan[L]
    ratio = s["chi_A1"] / (s["dmz"] ** 2 / (4 * (1 - s["mz"] ** 2)))
    print(f"  L={L}: min {ratio.min():.6f}, max {ratio.max():.6f}")

# ── Figure ────────────────────────────────────────────────────────────────────
ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
c_L = dict(zip(scan, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]))
c_qty = {"chi_F": "#2a78d6", "chi_A1": "#eb6834", "chi_A4": "#1baf7a"}
lab = {"chi_F": r"$\chi_F$", "chi_A1": r"$\chi_A$, $\ell_A=1$", "chi_A4": r"$\chi_A$, $\ell_A=4$"}
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 4, "legend.frameon": False,
})
fig, ax = plt.subplots(2, 2, figsize=(9, 6.6))

# (a) collapse chi_F / L^2 vs L (h - 1)
for L in scan:
    ax[0, 0].plot(L * (scan[L]["h"] - 1), scan[L]["chi_F"] / L ** 2, "o-", ms=2.5, lw=1,
                  color=c_L[L], label=f"$N={L}$")
ax[0, 0].set_xlabel(r"$N(\bar h-1)$"); ax[0, 0].set_ylabel(r"$\chi_F / N^2$")
ax[0, 0].set_title("global FS collapse", color=ink)
ax[0, 0].legend(loc="upper right")

# (b) chi_A(1) vs h_bar
for L in scan:
    ax[0, 1].plot(scan[L]["h"], scan[L]["chi_A1"], "o-", ms=2.5, lw=1, color=c_L[L], label=f"$N={L}$")
ax[0, 1].set_xlabel(r"$\bar h$"); ax[0, 1].set_ylabel(r"$\chi_A$ ($\ell_A=1$)")
ax[0, 1].set_title("RFS, single site", color=ink)
ax[0, 1].legend(loc="upper left")

# (c) peak positions vs 1/L
xx = np.linspace(0, 1 / Ls.min() * 1.05, 50)
for q in ["chi_F", "chi_A1", "chi_A4"]:
    hc, a, b = fits_pos[q]
    ax[1, 0].plot(1 / Ls, pk[q][:, 0], "o", color=c_qty[q], label=lab[q] + rf", $h_c$={hc:.4f}")
    ax[1, 0].plot(xx, hc + a * xx + b * xx ** 2, "-", lw=1, color=c_qty[q])
ax[1, 0].axhline(1, color=ink2, lw=1, ls="--")
ax[1, 0].set_xlabel(r"$1/N$"); ax[1, 0].set_ylabel(r"peak position $h^*(N)$")
ax[1, 0].set_title("pseudo-critical points", color=ink)
ax[1, 0].legend(loc="lower left")

# (d) peak heights
ax2 = ax[1, 1]
ax2.loglog(Ls, pk["chi_F"][:, 1], "o", color=c_qty["chi_F"], label=lab["chi_F"])
LL = np.linspace(Ls.min() * 0.9, Ls.max() * 1.1, 50)
ax2.loglog(LL, pk["chi_F"][-1, 1] * (LL / Ls[-1]) ** 2, "--", lw=1, color=ink2, label=r"$\propto N^2$")
ax2.set_xlabel("$N$"); ax2.set_ylabel(r"$\chi_F^{\max}$")
ax2.set_title(r"peak heights: $\chi_F^{\max}$ (left), $\chi_A^{\max}$ vs $(\ln N)^2$ (inset)", color=ink)
ax2.legend(loc="upper left")
ins = ax2.inset_axes([0.58, 0.1, 0.38, 0.38])
for q in ["chi_A1", "chi_A4"]:
    y = pk[q][:, 1] / pk[q][0, 1]
    ins.plot(np.log(Ls) ** 2, y, "o-", ms=3, lw=1, color=c_qty[q], label=lab[q])
ins.set_xlabel(r"$(\ln N)^2$", fontsize=6); ins.set_ylabel(r"$\chi_A^{\max}/\chi_A^{\max}(50)$", fontsize=6)
ins.tick_params(labelsize=6)
ins.legend(fontsize=5.5)

for a in ax.flat:
    a.grid(True, color=grid, lw=0.6)
    a.set_axisbelow(True)
plt.tight_layout()
for ext in ("png", "pdf"):
    out = f"{project}/figures/ISING_fidelity_scan_fss_eps_0_chi_32.{ext}"
    fig.savefig(out, dpi=300)
    print(f"Saved → {out}")
