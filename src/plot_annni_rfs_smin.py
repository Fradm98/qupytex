"""Log map of the smallest kept Schmidt value (mid-chain) over the ANNNI RFS grid.

Shows where DMRG at fixed chi was hardest. Works on a running grid: every saved
point is drawn, unfinished points are hatched.

usage: python plot_annni_rfs_smin.py [--data-dir DIR] [--out PNG]
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from analyze_annni_rfs_opd import draw_lines, INK2

ROOT = "C:/Users/HP/Desktop/projects/2_ANNNI"
TAG = "rfs_grid_pinned_eps_0.01_L_50_chi_64_k_0.1-1_h_0.05-0.75_n_64x64_block_6"


def load_smin(data_dir):
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    kappas, hs = np.array(meta["kappas"]), np.array(meta["hs"])
    smin = np.full((len(hs), len(kappas)), np.nan)
    sweeps = np.full_like(smin, np.nan)
    for path in glob.glob(os.path.join(data_dir, "col_???.npz")):
        j = int(os.path.basename(path)[4:7])
        with np.load(path) as z:
            n = len(z["smin"])
            smin[:n, j], sweeps[:n, j] = z["smin"], z["sweeps"]
    return meta, kappas, hs, smin, sweeps


def edges(x):
    x = np.asarray(x, float)
    mid = 0.5 * (x[1:] + x[:-1])
    return np.concatenate([[x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=f"{ROOT}/results/data/annni_rfs_grid/{TAG}")
    p.add_argument("--out", default=f"{ROOT}/figures/annni_rfs_opd/{TAG}/smin_log_map.png")
    a = p.parse_args()

    meta, kappas, hs, smin, sweeps = load_smin(a.data_dir)
    n_done = int(np.isfinite(smin).sum())
    pos = smin[np.isfinite(smin) & (smin > 0)]
    vmin, vmax = max(pos.min(), 1e-14), pos.max()

    # single-hue sequential ramp: light = easy (tiny smin), dark = hard
    cmap = LinearSegmentedColormap.from_list("ink", ["#f4f3ef", "#a9a7a0", "#3d3c39", "#0b0b0b"])
    cmap.set_bad("none")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ke, he = edges(kappas), edges(hs)
    # unfinished points: hatched background
    ax.add_patch(plt.Rectangle((ke.min(), min(he)), np.ptp(ke), np.ptp(he), facecolor="white",
                               edgecolor="#c9c7c0", hatch="////", lw=0, zorder=0))
    m = ax.pcolormesh(ke, he, np.ma.masked_invalid(np.clip(smin, vmin, None)), cmap=cmap,
                      norm=LogNorm(vmin=vmin, vmax=vmax), shading="flat", zorder=1)
    ext = (ke.min(), ke.max(), min(he), max(he))
    draw_lines(ax, ext)
    cb = fig.colorbar(m, ax=ax, pad=0.02)
    cb.set_label(r"smallest kept Schmidt value $s_{\chi}$ (mid-chain)")
    cb.outline.set_edgecolor(INK2)

    j, i = np.unravel_index(np.nanargmax(smin), smin.shape)
    ax.plot(kappas[i], hs[j], marker="o", ms=9, mfc="none", mec="#d03b3b", mew=1.8, zorder=5)
    ax.annotate(f"max {smin[j, i]:.1e}", (kappas[i], hs[j]), xytext=(-60, 14),
                textcoords="offset points", fontsize=8, color="#0b0b0b",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$h$")
    ax.set_title(f"ANNNI DMRG, L={meta.get('L')}, $\\chi$={meta.get('chi')}, "
                 f"$\\epsilon$={meta.get('eps')}   ({n_done}/{smin.size} points; hatched = not yet run)",
                 fontsize=9)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out)
    print("saved", a.out)
    print(f"points {n_done}/{smin.size}; smin range {pos.min():.2e} .. {pos.max():.2e}; "
          f"argmax kappa={kappas[i]:.3f} h={hs[j]:.3f}; max sweeps {np.nanmax(sweeps):.0f}")
    for thr in (1e-6, 1e-7, 1e-8):
        print(f"  frac s_chi > {thr:.0e}: {(pos > thr).mean():.2f}")


if __name__ == "__main__":
    main()
