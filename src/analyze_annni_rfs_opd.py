"""
analyze_annni_rfs_opd.py
========================
Analysis of an exec_annni_rfs_grid.py run (central 6-site RDMs on a (kappa, h) grid):

1. Fig. 4 of Mariella et al. (arXiv:2408.01400): angle of the RFS vector field P = -grad g built
   from the two middle-site RDMs, and its stream plot, with the Ising / KT / PT lines of App. F.
2. Labels y_i = sin(theta_i + eta) from the field; eta scanned over [0, 2 pi).
3. OPD (Eq. 15) for k = 1..4 central sites: matrix A (Eq. 45), its spectrum and the
   indefiniteness condition (App. B), the optimal M, and for comparison the two-state closed
   form of Eq. (17) with averaged rho+ / rho- (what the earlier qupytex notebooks used).
4. Eigendecomposition M = sum alpha_i M^(i) (Eq. 20); for k = 2 the angles (theta_1, theta_2)
   of B(theta_1, theta_2) (Eq. 21) from the leading Schmidt term of each eigenvector, and the
   expectation maps of each projector (Figs. 5-6 analogues).
5. Question "is a 1-site RDM enough for AP / FP / PM?": for every k and phase pair (theory
   lines of App. F, a buffer band around them excluded)
     - indefiniteness of A built with the theory labels,
     - AUC of |<M>| for the pair-optimal M, and of the best linear functional (Fisher LDA),
     - collision ratio: smallest cross-phase Hilbert-Schmidt distance between RDMs divided by
       the median nearest-neighbour distance inside a phase. A ratio << 1 means two points in
       different phases have (numerically) the same k-site RDM, so no function of that RDM,
       linear or not, can tell the phases apart there.
     - effective dimension of the RDM cloud (how many Pauli directions vary at all).
0. Z2 parity (before everything else): the raw asymmetry ||rho - P rho P|| of every 6-site RDM
   is mapped (parity_asymmetry.png). For eps = 0 runs every RDM is then replaced by its
   parity-symmetric part, at every grid point regardless of phase, because H commutes with
   P = prod Z and so does the exact ground-state RDM. DMRG at eps = 0 collapses from the cat
   state to a broken state somewhere inside FM / AP; without this step that collapse is a
   fake RFS line. --parity auto (default) symmetrizes iff meta eps == 0; on / off force it.

Usage (from src/):
  python analyze_annni_rfs_opd.py --data-dir <folder written by exec_annni_rfs_grid.py>
  python analyze_annni_rfs_opd.py --eps 0.01            # rebuild the folder name from the run args
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qphaset.phases import phases_vfield
from qphaset import opd

# Categorical slots 1-4 of the validated default palette, in fixed order.
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
LINE_STYLE = {"Ising": C_AQUA, "KT": C_ORANGE, "PT": C_BLUE, "PE": C_YELLOW}
PHASE_COLOR = {"FM": C_BLUE, "PM": C_ORANGE, "FP": C_AQUA, "AP": C_YELLOW}
PHASE_CODE = {"FM": 0, "PM": 1, "FP": 2, "AP": 3}
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10,
                     "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})


# ── ANNNI theory lines (App. F, Eqs. 63-65) ───────────────────────────────────
def h_ising(k):
    k = np.asarray(k, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (1 - k) / k * (1 - np.sqrt((1 - 3 * k + 4 * k ** 2) / (1 - k)))


def h_kt(k):
    k = np.asarray(k, dtype=float)
    return 1.05 * np.sqrt(np.clip((k - 0.5) * (k - 0.1), 0, None))


def h_pt(k):
    return 1.05 * (np.asarray(k, dtype=float) - 0.5)


def h_pe(k):
    """Peschel-Emery disorder line for H = -sum XX + k sum X_i X_{i+2} - h sum Z.

    Not a phase transition (the PM ground state is exactly a product of tilted
    spins there, so entanglement nearly vanishes). It starts at the Lifshitz
    point (1/2, 0). Drawn as a reference only and kept out of `line_samples`,
    so it never enters the phase labels or the distance-to-transition buffer.
    """
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore"):
        return 1.0 / (4.0 * k) - k


def theory_phase(k, h):
    if k < 0.5:
        return "FM" if h < h_ising(k) else "PM"
    if h < h_pt(k):
        return "AP"
    return "FP" if h < h_kt(k) else "PM"


def line_samples(kmin, kmax, n=800):
    ka = np.linspace(max(kmin, 1e-3), min(kmax, 0.5), n)
    kb = np.linspace(max(kmin, 0.5), kmax, n)
    return {"Ising": (ka, h_ising(ka)), "KT": (kb, h_kt(kb)), "PT": (kb, h_pt(kb))}


def distance_to_lines(K, H, kmin, kmax):
    pts = np.concatenate([np.stack(v, 1) for v in line_samples(kmin, kmax, 2000).values()])
    pts = pts[np.isfinite(pts).all(1)]
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(np.stack([K.ravel(), H.ravel()], 1))
    return d.reshape(K.shape)


def draw_lines(ax, extent, legend=True, show_pe=True, legend_loc="lower left"):
    for name, (kk, hh) in line_samples(extent[0], extent[1]).items():
        m = np.isfinite(hh) & (hh >= extent[2]) & (hh <= extent[3])
        if m.sum() < 2:
            continue
        ax.plot(kk[m], hh[m], "--", lw=1.6, color=LINE_STYLE[name], label=name,
                path_effects=[pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
    if show_pe:  # disorder line: dotted, since it is not a transition
        kk = np.linspace(max(extent[0], 1e-3), min(extent[1], 0.5), 800)
        hh = h_pe(kk)
        m = (hh >= extent[2]) & (hh <= extent[3])
        if m.sum() >= 2:
            ax.plot(kk[m], hh[m], ":", lw=1.8, color=LINE_STYLE["PE"], label="PE (disorder)",
                    path_effects=[pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()])
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    if legend:
        ax.legend(loc=legend_loc, fontsize=8, frameon=True, framealpha=0.85)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_grid(data_dir, allow_partial=False):
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    kappas, hs = np.array(meta["kappas"]), np.array(meta["hs"])
    n1, n2, D = len(kappas), len(hs), 2 ** meta["block"]
    rdms = np.full((n2, n1, D, D), np.nan, dtype=complex)
    diag = {kk: np.full((n2, n1), np.nan) for kk in ("E", "dE", "sweeps", "S_mid", "smin", "time")}
    done = np.zeros(n1, bool)
    for path in glob.glob(os.path.join(data_dir, "col_*.npz")):
        if path.endswith("_last_state.npz") or ".tmp" in path:
            continue
        j = int(os.path.basename(path)[4:7])
        with np.load(path) as z:
            n = len(z["rdms"])
            rdms[:n, j] = z["rdms"]
            for kk in diag:
                diag[kk][:n, j] = z[kk]
            done[j] = bool(z["done"]) and n == n2
    if not done.all():
        if not allow_partial:
            raise RuntimeError(f"{(~done).sum()} / {n1} columns are incomplete; "
                               "rerun later or pass --allow-partial")
        runs, start = [], None
        for j, d_ in enumerate(np.append(done, False)):
            if d_ and start is None:
                start = j
            if not d_ and start is not None:
                runs.append((start, j))
                start = None
        if not runs or max(b - a for a, b in runs) < 5:
            raise RuntimeError("not enough contiguous complete columns for a 2D analysis")
        a, b = max(runs, key=lambda r: r[1] - r[0])
        print(f"[partial] using complete columns {a}..{b - 1} (kappa {kappas[a]:.3f}-{kappas[b - 1]:.3f})")
        rdms, kappas = rdms[:, a:b], kappas[a:b]
        diag = {kk: v[:, a:b] for kk, v in diag.items()}
    return meta, kappas, hs, rdms, diag


def central_positions(meta, k):
    """Block positions of the k central sites; k = 2 gives the paper's two middle sites
    (1-indexed L/2, L/2 + 1), k = 1 gives the site 1-indexed L/2."""
    sites = list(meta["block_sites_0idx"])
    p0 = sites.index(meta["L"] // 2 - 1)
    start = p0 - (k - 1) // 2
    pos = list(range(start, start + k))
    if pos[0] < 0 or pos[-1] >= len(sites):
        raise ValueError(f"k = {k} does not fit in the stored {len(sites)}-site block")
    return pos


# ── RFS field ─────────────────────────────────────────────────────────────────
def rfs_field(rdms, dk, dh, scale=1):
    """g (paper sign, >= 0) and P = -grad_lambda g as a complex number P_kappa + i P_h.

    Grid rows are h descending, columns kappa ascending. phases_vfield returns the Sobel
    gradient of the library's g_lib = sum (f - 1) = -g along (columns, rows); rows run
    towards lower h, hence P = conj(grad g_lib) in index units. We then rescale each
    component by its grid step so that angles are physical when dk != dh."""
    grad = phases_vfield(rdms, scale=scale)
    g = -phases_vfield(rdms, scale=scale, grad=False)
    P = np.conj(grad)
    P = P.real / dk + 1j * P.imag / dh
    return g, P


# ── Plots ─────────────────────────────────────────────────────────────────────
def imshow_map(ax, Z, extent, cmap="plasma", vmin=None, vmax=None, lines=True, legend=False):
    im = ax.imshow(Z, extent=extent, origin="upper", aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    if lines:
        draw_lines(ax, extent, legend=legend)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$h$")
    return im


def fig4(P_up, ext_inner, path):
    """Panel (a) angle of P (upsampled x2 as in App. A), panel (b) stream plot."""
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4))
    im = imshow_map(axs[0], np.angle(P_up), ext_inner, cmap="twilight", vmin=-np.pi, vmax=np.pi,
                    legend=True)
    cb = fig.colorbar(im, ax=axs[0], ticks=[-np.pi, 0, np.pi])
    cb.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    axs[0].set_title(r"(a) $\theta(\lambda)=\mathrm{Arg}\,P$")
    ny, nx = P_up.shape
    X, Y = np.meshgrid(np.linspace(ext_inner[0], ext_inner[1], nx),
                       np.linspace(ext_inner[2], ext_inner[3], ny))
    Pf = P_up[::-1]                                   # rows ascending in h for streamplot
    axs[1].streamplot(X, Y, Pf.real, Pf.imag, density=2.6, color=C_BLUE, linewidth=0.7,
                      arrowsize=0.7)
    axs[1].set_xlim(ext_inner[:2])
    axs[1].set_ylim(ext_inner[2:])
    axs[1].set_xlabel(r"$\kappa$")
    axs[1].set_ylabel(r"$h$")
    axs[1].set_title(r"(b) $P(\lambda)=-\nabla_\lambda g$")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


def plot_rfs_g(rhos, ks, dk, dh, ext_inner, fig_dir):
    """Bare RFS g for each block size k: one file per k and a side-by-side panel.
    Each panel has its own log colour scale, since g grows with the block size."""
    gs = {k: rfs_field(rhos[k], dk, dh, scale=1)[0] for k in ks}
    for k in ks:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        im = imshow_map(ax, np.log10(np.maximum(gs[k], 1e-12)), ext_inner, cmap="viridis", legend=True)
        fig.colorbar(im, ax=ax, label=r"$\log_{10} g$")
        ax.set_title(f"reduced fidelity susceptibility g ({k}-site)")
        fig.savefig(f"{fig_dir}/rfs_g_{k}site.png")
        plt.close(fig)
    fig, axs = plt.subplots(1, len(ks), figsize=(4.3 * len(ks), 3.9), sharey=True)
    for i, (ax, k) in enumerate(zip(np.atleast_1d(axs), ks)):
        im = imshow_map(ax, np.log10(np.maximum(gs[k], 1e-12)), ext_inner, cmap="viridis",
                        legend=i == 0)
        fig.colorbar(im, ax=ax, label=r"$\log_{10} g$" if i == len(ks) - 1 else None)
        ax.set_title(f"{k}-site")
        if i:
            ax.set_ylabel("")
    fig.suptitle("reduced fidelity susceptibility g of the central k-site RDM")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/rfs_g_all_k.png")
    plt.close(fig)
    return gs


def plot_diagnostics(diag, extent, path):
    keys = [("sweeps", "DMRG sweeps"), ("smin", r"smallest kept Schmidt value (log$_{10}$)"),
            ("S_mid", "mid-chain entropy"), ("time", "wall time per point [s]")]
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (kk, title) in zip(axs, keys):
        Z = np.log10(np.maximum(diag[kk], 1e-18)) if kk == "smin" else diag[kk]
        im = imshow_map(ax, Z, extent, cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{path}.png")
    plt.close(fig)


PARITY_BROKEN = 0.5          # raw asymmetry above this = DMRG returned a (nearly) broken state


def plot_parity(asym, extent, applied, path):
    """log10 ||rho - P rho P||_F of the raw 6-site RDMs; broken points outlined in red."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ink", ["#f4f3ef", "#a9a7a0", "#3d3c39", "#0b0b0b"])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = imshow_map(ax, np.log10(np.maximum(asym, 1e-16)), extent, cmap=cmap, vmin=-16, vmax=0.5,
                    lines=False)
    draw_lines(ax, extent, legend=False)
    broken = asym > PARITY_BROKEN
    if broken.any():
        ny, nx = asym.shape
        kk = np.linspace(extent[0], extent[1], nx)
        hh = np.linspace(extent[3], extent[2], ny)
        ax.contour(kk, hh, broken.astype(float), levels=[0.5], colors=[C_FAIL], linewidths=1.6)
        ax.plot([], [], color=C_FAIL, lw=1.6, label=f"broken (> {PARITY_BROKEN})")
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.85)   # FM/AP corners hold the breaks
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$\log_{10}\,\|\rho - P\rho P\|_F$  (raw 6-site RDM)")
    ax.set_title(f"Z2 parity of the DMRG RDMs: {broken.sum()} / {broken.size} points broken; "
                 f"{'symmetrized everywhere' if applied else 'NOT symmetrized'}", fontsize=9)
    fig.savefig(f"{path}.png")
    plt.close(fig)


def plot_component_maps(rhos_grid_flat, comps, shape, extent, path, title_prefix, n_max=4):
    comps = comps[:n_max]
    fig, axs = plt.subplots(1, len(comps), figsize=(4.2 * len(comps), 3.7), squeeze=False)
    for ax, c in zip(axs[0], comps):
        Z = opd.expectations(rhos_grid_flat, opd.projector(c["v"])).reshape(shape)
        im = imshow_map(ax, Z, extent, vmin=0)
        fig.colorbar(im, ax=ax)
        t = rf"$M^{{({c['index']})}}$, $\alpha$={c['alpha']:+.3f}"
        if "theta1" in c:
            t += "\n" + rf"$\theta_1$={c['theta1'] / np.pi:+.3f}$\pi$, $\theta_2$={c['theta2'] / np.pi:+.3f}$\pi$, prod={c['product_overlap']:.2f}"
        ax.set_title(t, fontsize=8.5)
    fig.suptitle(title_prefix, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{path}.png")
    plt.close(fig)


C_PASS, C_FAIL, C_ZERO = "#0ca30c", "#d03b3b", "#9a9893"      # status good / critical, muted


def plot_ladder(ladders, row_names, path, title):
    """Indefiniteness check of App. B for every k (columns) and label set (rows).

    Each panel: sorted eigenvalues of A on a symlog axis (A's spectrum lies in [-1, 1]), the
    +/- tolerance band shaded, the verdict written in the title. The figure is produced whether
    the checks pass or fail, so the report shows that every k below the one used was tested."""
    ks = ladders[0]["performed_for_k"]
    fig, axs = plt.subplots(len(ladders), len(ks), figsize=(3.3 * len(ks), 2.8 * len(ladders)),
                            squeeze=False)
    for r, (lad, rn) in enumerate(zip(ladders, row_names)):
        for c, k in enumerate(ks):
            ax = axs[r, c]
            chk = lad["per_k"][k]
            tol = chk["tol"]
            w = np.sort(np.asarray(chk["eigs"]))
            col = np.where(w < -tol, C_BLUE, np.where(w > tol, C_ORANGE, C_ZERO))
            ax.axhspan(-tol, tol, color=C_ZERO, alpha=0.3, lw=0)
            ax.scatter(np.arange(len(w)), w, c=col, s=14 if len(w) < 40 else 6, zorder=3)
            ax.axhline(0, color=INK2, lw=0.6)
            ax.set_yscale("symlog", linthresh=tol / 10)
            ax.set_ylim(-1.2, 1.2)
            ticks = [-1, -1e-3, -tol, 0, tol, 1e-3, 1]
            ax.set_yticks(ticks)
            ax.set_yticklabels(["-1", r"$-10^{-3}$", "-tol", "0", "tol", r"$10^{-3}$", "1"], fontsize=7.5)
            ax.minorticks_off()
            verdict = "PASS" if chk["indefinite"] else "FAIL"
            ax.set_title(f"{rn}, k = {k}: {verdict}\nneg {chk['n_neg']} / pos {chk['n_pos']} / ~0 {chk['n_zero']}",
                         fontsize=8.5, color=C_PASS if chk["indefinite"] else C_FAIL)
            if c == 0:
                ax.set_ylabel("eig(A)")
            if r == len(ladders) - 1:
                ax.set_xlabel("eigenvalue index")
    kmins = ", ".join(f"{rn}: {lad['k_min_pass']}" for lad, rn in zip(ladders, row_names))
    fig.suptitle(f"{title}\nchecked for k = {ks}; |eig| <= max({lad['abs_tol']:g}, {lad['rel_tol']:g} max|eig|) "
                 f"counts as 0 (grey band); blue < 0, orange > 0; smallest passing k: {kmins}", fontsize=9.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


def ladder_json(lad):
    return dict(performed_for_k=lad["performed_for_k"], k_min_pass=lad["k_min_pass"], rel_tol=lad["rel_tol"],
                abs_tol=lad["abs_tol"],
                per_k={str(k): {kk: (v.tolist() if isinstance(v, np.ndarray) else v) for kk, v in c.items()}
                       for k, c in lad["per_k"].items()})


def print_ladder(lad, label):
    print(f"   INDEFINITENESS CHECK (App. B), {label}: performed for k = {lad['performed_for_k']}, "
          f"tol = max({lad['abs_tol']:g}, {lad['rel_tol']:g} max|eig|)")
    for k, c in lad["per_k"].items():
        print(f"      k = {k}: {c['status']}  (neg {c['n_neg']}, pos {c['n_pos']}, ~0 {c['n_zero']}, "
              f"lam_min = {c['lam_min']:+.2e}, lam_max = {c['lam_max']:+.2e})")
        if not c["indefinite"]:
            print(f"      WARNING: k = {k} fails the indefiniteness condition for {label}; "
                  "the M found at this k is not a valid order parameter.")
    print(f"      smallest k passing: {lad['k_min_pass']}")


VARIANTS = {
    "eq17_hadamard": "Eq. 17 on averaged rho+/rho-, inner product trace(a * b) (old script / 'final' notebook)",
    "eq17": "Eq. 17 on averaged rho+/rho- (exec_order_parameter_discovery*.py, phases.py)",
    "eq16_exact": "exact minimizer of Eq. 16 on the same averaged rho+/rho-",
    "eq15": "full Eq. 15 over all labelled points (min eigenvector of A, Eq. 45)",
}


def opd_variants(rhos, R, ys, k, sol15):
    """The four observables, each normalized to ||M||_F = 1, from the old practice to Eq. 15."""
    rp, rm = rhos[ys > 0].mean(0), rhos[ys < 0].mean(0)
    a_, b_ = rp / np.linalg.norm(rp), rm / np.linalg.norm(rm)
    dot_h = np.real(np.trace(a_ * b_))                       # elementwise product: diagonal terms only
    out = {"eq17_hadamard": (a_ - dot_h * b_) / np.sqrt(abs(1 - dot_h ** 2)),
           "eq17": opd.closed_form_observable(rp, rm)}
    Rab = opd.pauli_coeffs(np.stack([rp, rm]), k)
    out["eq16_exact"] = opd.solve_opd(Rab, np.array([1.0, -1.0]), k)["M"]
    out["eq15"] = sol15["M"]
    for n, M in out.items():
        M = M / np.linalg.norm(M)
        if np.mean(opd.expectations(rhos[ys > 0], M)) < 0:
            M = -M
        out[n] = M
    return out


def plot_variants(rhos, Ms, shape, extent, path, title, n_comp=4):
    """Rows: the four variants. Columns: |<M>| and the expectation of its leading projectors."""
    n_comp = min(n_comp, rhos.shape[-1])
    fig, axs = plt.subplots(len(Ms), 1 + n_comp, figsize=(3.6 * (1 + n_comp), 3.0 * len(Ms)), squeeze=False)
    for r, (name, M) in enumerate(Ms.items()):
        im = imshow_map(axs[r, 0], np.abs(opd.expectations(rhos, M)).reshape(shape), extent, vmin=0)
        fig.colorbar(im, ax=axs[r, 0])
        axs[r, 0].set_title(f"{name}: |<M>|", fontsize=8.5)
        for c, comp in enumerate(opd.decompose_observable(M, int(np.log2(len(M))))[:n_comp]):
            ax = axs[r, 1 + c]
            Z = opd.expectations(rhos, opd.projector(comp["v"])).reshape(shape)
            im = imshow_map(ax, Z, extent, vmin=0)
            fig.colorbar(im, ax=ax)
            t = rf"$M^{{({comp['index']})}}$ $\alpha$={comp['alpha']:+.2f}"
            if "theta1" in comp:
                t += rf", $\theta$=({comp['theta1'] / np.pi:+.3f}, {comp['theta2'] / np.pi:+.3f})$\pi$"
            ax.set_title(t, fontsize=8)
    for ax in axs.ravel():
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{path}.png")
    plt.close(fig)


def objective(rhos, ys, M):
    """Value of the Eq. (15) objective, normalized as in Eq. (44): mean_{I+} -<M>^2/p + mean_{I-} <M>^2/p."""
    ev = opd.expectations(rhos, M)
    p = np.real(np.einsum("nij,nji->n", rhos, rhos))
    q = ev ** 2 / p
    return float(-q[ys > 0].mean() + q[ys < 0].mean())


# ── Analysis steps ────────────────────────────────────────────────────────────
def build_labels(P, eta):
    return np.sin(np.angle(P) + eta)


def eta_scan(R_by_k, theta_flat, etas):
    out = {}
    for k, R in R_by_k.items():
        lam = []
        for eta in etas:
            ys = np.sin(theta_flat + eta)
            if (ys > 0).sum() == 0 or (ys < 0).sum() == 0:
                lam.append(np.nan)
                continue
            lam.append(np.linalg.eigvalsh(opd.qcqp_matrix(R, ys))[0])
        out[k] = np.array(lam)
    return out


def pair_ladders(R_by_k, phase_flat, keep, pairs, rel_tol):
    """Indefiniteness check for the theory-label pairs, for every k."""
    out = {}
    for a_, b_ in pairs:
        ia = keep & (phase_flat == PHASE_CODE[a_])
        ib = keep & (phase_flat == PHASE_CODE[b_])
        if ia.sum() < 3 or ib.sum() < 3:
            continue
        sel = ia | ib
        ys = np.where(ia[sel], 1.0, -1.0)
        out[f"{a_}-{b_}"] = opd.indefiniteness_ladder({k: R[sel] for k, R in R_by_k.items()}, ys, rel_tol)
    return out


def separability(R_by_k, rhos_by_k, phase_flat, keep, ks, pairs, shape, rel_tol=opd.INDEF_REL_TOL):
    res = {}
    for k in ks:
        R, rhos = R_by_k[k], rhos_by_k[k]
        neff, sv = opd.effective_dimension(R[keep])
        scale = opd.grid_step_scale(R.reshape(shape + (-1,)), keep.reshape(shape))
        res[k] = dict(effective_dimension=neff, singular_values=sv[:8].tolist(), grid_step_hs=scale, pairs={})
        for a, b in pairs:
            ia = np.nonzero(keep & (phase_flat == PHASE_CODE[a]))[0]
            ib = np.nonzero(keep & (phase_flat == PHASE_CODE[b]))[0]
            if len(ia) < 3 or len(ib) < 3:
                res[k]["pairs"][f"{a}-{b}"] = dict(skipped=f"|{a}|={len(ia)}, |{b}|={len(ib)}")
                continue
            ys = np.zeros(len(R))
            ys[ia], ys[ib] = 1, -1
            sel = np.concatenate([ia, ib])
            sol = opd.solve_opd(R[sel], ys[sel], k, rel_tol=rel_tol)
            ev = np.abs(opd.expectations(rhos[sel], sol["M"]))
            auc_M = opd.auc(ev[: len(ia)], ev[len(ia):])
            w = opd.fisher_lda(R[ia], R[ib])
            auc_lda = opd.auc(R[ia] @ w, R[ib] @ w)
            dmin, i, j = opd.cross_phase_collision(R[ia], R[ib])
            res[k]["pairs"][f"{a}-{b}"] = dict(
                n=[len(ia), len(ib)], indefinite=sol["indefinite"], status=sol["status"], n_neg=sol["n_neg"],
                n_pos=sol["n_pos"], rel_neg=sol["rel_neg"], auc_absM=auc_M, auc_lda=auc_lda,
                hs_min=dmin, collision_ratio=dmin / scale if scale > 0 else np.nan,
                tr_dist_min_pair=opd.trace_distance(rhos[ia[i]], rhos[ib[j]]),
                min_pair_flat_idx=[int(ia[i]), int(ib[j])])
    return res


def summarize_obs(sol, k):
    comps = opd.decompose_observable(sol["M"], k)
    out = dict(valid=sol.get("valid"), status=sol.get("status"),
               indefinite=sol["indefinite"], n_neg=sol["n_neg"], n_pos=sol["n_pos"],
               n_zero=sol["n_zero"], lam_min=sol["lam_min"], lam_max=sol["lam_max"],
               rel_neg=sol["rel_neg"], min_eig_degeneracy=sol["min_degeneracy"],
               A_eigs=sol["A_eigs"].tolist(), top_pauli=opd.top_pauli_terms(sol["M"], k),
               components=[])
    for c in comps:
        d = dict(index=c["index"], alpha=c["alpha"], product_overlap=c["product_overlap"],
                 schmidt=c["schmidt"].tolist(), v=np.round(np.real_if_close(c["v"]), 5).tolist()
                 if np.isrealobj(np.real_if_close(c["v"])) else None)
        if "theta1" in c:
            d.update(theta1_over_pi=c["theta1"] / np.pi, theta2_over_pi=c["theta2"] / np.pi,
                     complex_residual=c["complex_residual"], bell_overlaps=c["bell_overlaps"])
        out["components"].append(d)
    return out, comps


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None)
    p.add_argument("--fig-dir", default=None)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--k-ref", type=int, default=2, help="k used to choose eta*")
    p.add_argument("--opd-kmin", type=float, default=0.5,
                   help="OPD region R: kappa >= this (0.5 = right of the Lifshitz point, as in Sec. 5.2)")
    p.add_argument("--buffer", type=float, default=0.03,
                   help="exclude points closer than this to a theory line in the phase-pair tests")
    p.add_argument("--indef-tol", type=float, default=opd.INDEF_REL_TOL,
                   help="eigenvalues of A with |lambda| <= tol * max|lambda| count as zero (App. B check)")
    p.add_argument("--n-eta", type=int, default=72)
    p.add_argument("--eta", type=float, default=None, help="fix eta instead of scanning")
    p.add_argument("--parity", choices=["auto", "on", "off"], default="auto",
                   help="replace every RDM by (rho + P rho P)/2, P = prod Z: auto = iff the run has eps = 0")
    # the same arguments as exec_annni_rfs_grid.py, to rebuild the folder name
    p.add_argument("--L", type=int, default=50)
    p.add_argument("--chi", type=int, default=64)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--kmin", type=float, default=0.1)
    p.add_argument("--kmax", type=float, default=1.0)
    p.add_argument("--hmin", type=float, default=0.05)
    p.add_argument("--hmax", type=float, default=0.75)
    p.add_argument("--n1", type=int, default=64)
    p.add_argument("--n2", type=int, default=64)
    p.add_argument("--block", type=int, default=6)
    p.add_argument("--tag-suffix", default="")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.data_dir is None:
        import exec_annni_rfs_grid as ex
        a.data_dir = ex.out_dir(ex.parse_args([
            "--L", str(a.L), "--chi", str(a.chi), "--eps", str(a.eps), "--kmin", str(a.kmin),
            "--kmax", str(a.kmax), "--hmin", str(a.hmin), "--hmax", str(a.hmax), "--n1", str(a.n1),
            "--n2", str(a.n2), "--block", str(a.block), "--tag-suffix", a.tag_suffix]))
    meta, kappas, hs, rdm6, diag = load_grid(a.data_dir, a.allow_partial)
    tag = os.path.basename(os.path.normpath(a.data_dir))
    fig_dir = a.fig_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.normpath(a.data_dir))))), "figures", "annni_rfs_opd", tag)
    os.makedirs(fig_dir, exist_ok=True)
    n2, n1 = rdm6.shape[:2]
    nb = meta["block"]
    dk, dh = kappas[1] - kappas[0], hs[0] - hs[1]
    extent = (kappas[0], kappas[-1], hs[-1], hs[0])
    ext_inner = (kappas[1], kappas[-2], hs[-2], hs[1])
    print(f"── {tag}: {n2}x{n1} grid, eps = {meta['eps']}, L = {meta['L']}, chi = {meta['chi']}")
    print(f"   figures → {fig_dir}")

    # DMRG diagnostics
    plot_diagnostics(diag, extent, f"{fig_dir}/dmrg_diagnostics")
    diag_sum = dict(max_sweeps=float(np.nanmax(diag["sweeps"])),
                    n_at_sweep_cap=int(np.sum(diag["sweeps"] >= meta.get("n_sweeps", 12))),
                    max_smin=float(np.nanmax(diag["smin"])), max_dE=float(np.nanmax(diag["dE"])),
                    total_hours=float(np.nansum(diag["time"]) / 3600))

    # 0. Z2 parity: diagnose on the raw RDMs, then symmetrize everywhere if H is parity-symmetric
    symmetric_H = float(meta["eps"]) == 0.0
    apply_parity = {"auto": symmetric_H, "on": True, "off": False}[a.parity]
    asym = opd.parity_asymmetry(rdm6, nb)
    broken = asym > PARITY_BROKEN
    plot_parity(asym, extent, apply_parity, f"{fig_dir}/parity_asymmetry")
    parity_sum = dict(mode=a.parity, hamiltonian_parity_symmetric=symmetric_H, applied=bool(apply_parity),
                      broken_threshold=PARITY_BROKEN, n_broken=int(broken.sum()), n_points=int(broken.size),
                      max_asymmetry=float(np.nanmax(asym)),
                      max_asymmetry_unbroken=float(np.nanmax(np.where(broken, np.nan, asym)))
                      if (~broken).any() else None)
    mu = parity_sum["max_asymmetry_unbroken"]
    print(f"PARITY: H {'is' if symmetric_H else 'is NOT'} Z2-symmetric (eps = {meta['eps']}); "
          f"raw RDMs: {broken.sum()} / {broken.size} broken, max asymmetry of the rest "
          f"{'n/a' if mu is None else f'{mu:.1e}'} -> "
          f"{'symmetrizing ALL points' if apply_parity else 'left as is'}")
    if apply_parity and not symmetric_H:
        print("WARNING: --parity on with eps != 0: H is not parity-symmetric, this removes real physics")
    if symmetric_H and not apply_parity and broken.any():
        print("WARNING: eps = 0 with broken RDMs left unsymmetrized: expect fake RFS lines at the collapse")
    if apply_parity:
        rdm6 = opd.parity_symmetrize(rdm6, nb)

    # k-site central RDMs
    ks = sorted(set(a.ks) | {1, 2})
    rhos = {k: opd.ptrace(rdm6, nb, central_positions(meta, k)) for k in ks}

    # 1. Fig. 4
    g_up, P_up = rfs_field(rhos[2], dk, dh, scale=2)
    fig4(P_up, ext_inner, f"{fig_dir}/fig4_rfs_2site")
    g1, P1 = rfs_field(rhos[2], dk, dh, scale=1)               # (n2-2, n1-2), labels live here
    plot_rfs_g(rhos, ks, dk, dh, ext_inner, fig_dir)

    # inner grid bookkeeping
    K, H = np.meshgrid(kappas[1:-1], hs[1:-1])
    shape_in = K.shape
    phase = np.vectorize(lambda k_, h_: PHASE_CODE[theory_phase(k_, h_)])(K, H)
    dist = distance_to_lines(K, H, kappas[0], kappas[-1])
    flat = lambda Z: Z.reshape(-1, *Z.shape[2:]) if Z.ndim > 2 else Z.ravel()
    rhos_in = {k: flat(rhos[k][1:-1, 1:-1]) for k in ks}
    R_in = {k: opd.pauli_coeffs(rhos_in[k], k) for k in ks}
    theta_flat = np.angle(P1).ravel()
    phase_flat, dist_flat = phase.ravel(), dist.ravel()

    summary = dict(tag=tag, meta={kk: meta[kk] for kk in ("L", "chi", "eps", "block")},
                   grid=dict(n_kappa=n1, n_h=n2, kappa=[kappas[0], kappas[-1]], h=[hs[-1], hs[0]]),
                   dmrg=diag_sum, parity=parity_sum, regions={})

    # 1-site Bloch components (sanity: <Y> must vanish for a real Hamiltonian)
    r1 = R_in[1] * np.sqrt(2)                                  # Tr(rho P) for P in I, X, Y, Z
    summary["one_site"] = dict(max_abs_Y=float(np.abs(r1[:, 2]).max()),
                               X_range=[float(r1[:, 1].min()), float(r1[:, 1].max())],
                               Z_range=[float(r1[:, 3].min()), float(r1[:, 3].max())])
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.9))
    for ax, (ci, lab) in zip(axs[:2], [(1, r"$\langle\sigma^x\rangle$"), (3, r"$\langle\sigma^z\rangle$")]):
        im = imshow_map(ax, r1[:, ci].reshape(shape_in), ext_inner, cmap="plasma", legend=ci == 1)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"1-site {lab} (site L/2)")
    m = dist_flat > a.buffer
    for name, code in PHASE_CODE.items():
        s = m & (phase_flat == code)
        if s.any():
            axs[2].scatter(r1[s, 1], r1[s, 3], s=9, color=PHASE_COLOR[name], label=name,
                           edgecolors="white", linewidths=0.3)
    axs[2].set_xlabel(r"$\langle\sigma^x\rangle$")
    axs[2].set_ylabel(r"$\langle\sigma^z\rangle$")
    axs[2].set_title(f"1-site RDMs by theory phase (buffer {a.buffer})")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/one_site_bloch.png")
    plt.close(fig)

    regions = {"right_of_LP": K.ravel() >= a.opd_kmin, "full": np.ones(K.size, bool)}
    etas = np.linspace(0, 2 * np.pi, a.n_eta, endpoint=False)
    for rname, rmask in regions.items():
        if rmask.sum() < 10:
            continue
        ext_r = ext_inner if rname == "full" else (max(a.opd_kmin, ext_inner[0]),) + ext_inner[1:]
        cols = np.nonzero(kappas[1:-1] >= (a.opd_kmin if rname != "full" else -np.inf))[0]
        shape_r = (shape_in[0], len(cols))
        Rr = {k: R_in[k][rmask] for k in ks}
        rr = {k: rhos_in[k][rmask] for k in ks}
        th = theta_flat[rmask]

        # 2. eta scan and labels
        if a.eta is None:
            scan = eta_scan(Rr, th, etas)
            eta_star = float(etas[np.nanargmin(scan[a.k_ref])])
        else:
            scan, eta_star = {}, a.eta
        ys = np.sin(th + eta_star)
        reg = dict(eta_star=eta_star, n_plus=int((ys > 0).sum()), n_minus=int((ys < 0).sum()),
                   label_composition={}, k={})
        for name, code in PHASE_CODE.items():
            s = phase_flat[rmask] == code
            if s.any():
                reg["label_composition"][name] = dict(n=int(s.sum()), frac_plus=float((ys[s] > 0).mean()))
        if scan:
            fig, ax = plt.subplots(figsize=(5.6, 3.6))
            for ci, k in enumerate(ks):
                ax.plot(etas / np.pi, scan[k] / np.nanmax(np.abs(scan[k])), lw=1.8,
                        color=[C_BLUE, C_ORANGE, C_AQUA, C_YELLOW][ci % 4], label=f"k = {k}")
            ax.axvline(eta_star / np.pi, color=INK2, lw=1, ls=":")
            ax.set_xlabel(r"$\eta/\pi$")
            ax.set_ylabel(r"$\lambda_{\min}(A_\eta)$ / max$|\lambda_{\min}|$")
            ax.set_title(f"label rotation scan ({rname})")
            ax.legend(fontsize=8)
            fig.savefig(f"{fig_dir}/eta_scan_{rname}.png")
            plt.close(fig)
            reg["eta_scan_lam_min"] = {str(k): scan[k].tolist() for k in ks}

        # explicit indefiniteness check for every k with these labels (App. B)
        lad = opd.indefiniteness_ladder(Rr, ys, a.indef_tol)
        reg["indefiniteness_check"] = ladder_json(lad)
        plot_ladder([lad], ["RFS labels"], f"{fig_dir}/indefiniteness_ladder_{rname}",
                    f"Indefiniteness of A (Eq. 45), RFS labels, region {rname}")

        # label map
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        lab = np.sign(ys).reshape(shape_r)
        imshow_map(ax, lab, ext_r, cmap="cividis", vmin=-1, vmax=1, legend=True)
        ax.set_title(rf"sign of labels $y=\sin(\theta+\eta^*)$, $\eta^*$={eta_star / np.pi:.2f}$\pi$")
        fig.savefig(f"{fig_dir}/labels_{rname}.png")
        plt.close(fig)

        # 3-4. OPD per k
        for k in ks:
            sol = opd.solve_opd(Rr[k], ys, k, rel_tol=a.indef_tol)
            info, comps = summarize_obs(sol, k)
            info["objective_eq15"] = objective(rr[k], ys, sol["M"])
            invalid = "" if sol["valid"] else "  [INVALID: A not indefinite]"
            evM = np.abs(opd.expectations(rr[k], sol["M"])).reshape(shape_r)
            fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
            im = imshow_map(axs[0], evM, ext_r, vmin=0, legend=True)
            fig.colorbar(im, ax=axs[0])
            axs[0].set_title(rf"$|\langle M\rangle|$, k = {k} ({rname}){invalid}",
                             color=INK if sol["valid"] else C_FAIL)
            w = sol["A_eigs"]
            axs[1].plot(np.arange(len(w)), w / np.max(np.abs(w)), "o", ms=4, color=C_BLUE)
            axs[1].axhline(0, color=INK2, lw=0.8)
            axs[1].set_xlabel("index")
            axs[1].set_ylabel(r"eig$(A)$ / max|eig|")
            axs[1].set_title(f"spectrum of A: {sol['n_neg']} neg, {sol['n_pos']} pos, {sol['n_zero']} ~0")
            fig.tight_layout()
            fig.savefig(f"{fig_dir}/opd_M_k{k}_{rname}.png")
            plt.close(fig)
            plot_component_maps(rr[k], comps, shape_r, ext_r, f"{fig_dir}/opd_components_k{k}_{rname}",
                                f"projectors of M, k = {k}, region {rname}{invalid}")
            # old practice -> Eq. 15, step by step (all normalized to ||M||_F = 1)
            Ms = opd_variants(rr[k], Rr[k], ys, k, sol)
            info["variants"] = {}
            for vname, Mv in Ms.items():
                comps_v = opd.decompose_observable(Mv, k)
                info["variants"][vname] = dict(
                    description=VARIANTS[vname], objective_eq15=objective(rr[k], ys, Mv),
                    overlap_with_eq15=float(abs(np.real(np.trace(Mv @ Ms["eq15"])))),
                    top_pauli=opd.top_pauli_terms(Mv, k),
                    components=[dict(index=c["index"], alpha=c["alpha"], product_overlap=c["product_overlap"],
                                     **({"theta1_over_pi": c["theta1"] / np.pi, "theta2_over_pi": c["theta2"] / np.pi,
                                         "bell_overlaps": c["bell_overlaps"]} if "theta1" in c else {}))
                                for c in comps_v])
            plot_variants(rr[k], Ms, shape_r, ext_r, f"{fig_dir}/opd_variants_k{k}_{rname}",
                          f"k = {k}, region {rname}: rows = Eq. 17 (diag. inner product) / Eq. 17 / exact Eq. 16 / Eq. 15")
            reg["k"][str(k)] = info
        summary["regions"][rname] = reg

    # 5. phase-pair separability on the right of the Lifshitz point
    keep = (K.ravel() >= a.opd_kmin) & (dist_flat > a.buffer)
    pairs = [("AP", "FP"), ("FP", "PM"), ("AP", "PM")]
    summary["separability"] = dict(buffer=a.buffer, kmin=a.opd_kmin, results={
        str(k): v for k, v in separability(R_in, rhos_in, phase_flat, keep, ks, pairs, shape_in,
                                           a.indef_tol).items()})
    plads = pair_ladders(R_in, phase_flat, keep, pairs, a.indef_tol)
    if plads:
        plot_ladder(list(plads.values()), list(plads.keys()), f"{fig_dir}/indefiniteness_ladder_phase_pairs",
                    "Indefiniteness of A with theory-line labels (one phase vs another)")
    summary["separability"]["indefiniteness_check"] = {n: ladder_json(l) for n, l in plads.items()}

    with open(f"{fig_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print_summary(summary)
    return summary


def print_summary(s):
    print(f"\nDMRG: max sweeps {s['dmrg']['max_sweeps']:.0f} ({s['dmrg']['n_at_sweep_cap']} at cap), "
          f"max smallest Schmidt value {s['dmrg']['max_smin']:.1e}, max dE {s['dmrg']['max_dE']:.1e}")
    o = s["one_site"]
    print(f"1-site: max|<Y>| = {o['max_abs_Y']:.1e}, <X> in {o['X_range']}, <Z> in {o['Z_range']}")
    for rname, reg in s["regions"].items():
        print(f"\n[{rname}] eta* = {reg['eta_star'] / np.pi:.3f} pi, |I+| = {reg['n_plus']}, |I-| = {reg['n_minus']}")
        print("   label composition (fraction labelled +): " + ", ".join(
            f"{n}: {v['frac_plus']:.2f} of {v['n']}" for n, v in reg["label_composition"].items()))
        print_ladder(reg["indefiniteness_check"], f"RFS labels, region {rname}")
        for k, info in reg["k"].items():
            print(f"   k = {k}: {info['status']}, lam_min/max|lam| = {-info['rel_neg']:.3f}, "
                  f"objective {info['objective_eq15']:+.4f}, "
                  f"top terms {[(p, round(c, 3)) for p, c in info['top_pauli'][:4]]}")
            for vname, v in info["variants"].items():
                line = (f"      {vname:14s} objective {v['objective_eq15']:+.4f}, "
                        f"|<M, M_eq15>| = {v['overlap_with_eq15']:.3f}")
                if k == "2":
                    line += "  angles/pi: " + "; ".join(
                        f"M{c['index']}({c['alpha']:+.2f}) {c['theta1_over_pi']:+.3f},{c['theta2_over_pi']:+.3f}"
                        f" Psi+ {c['bell_overlaps']['Psi+']:.2f}" for c in v["components"])
                print(line)
    print(f"\nPhase-pair separability (kappa >= {s['separability']['kmin']}, buffer {s['separability']['buffer']}):")
    for pname, lad in s["separability"]["indefiniteness_check"].items():
        print_ladder(lad, f"theory labels {pname}")
    for k, r in s["separability"]["results"].items():
        line = f"   k = {k} (eff. dim {r['effective_dimension']}): "
        parts = []
        for pname, v in r["pairs"].items():
            if "skipped" in v:
                parts.append(f"{pname} skipped ({v['skipped']})")
            else:
                parts.append(f"{pname} AUC|M| {v['auc_absM']:.2f} AUC_lda {v['auc_lda']:.2f} "
                             f"collision {v['collision_ratio']:.2f} indef {v['indefinite']}")
        print(line + " | ".join(parts))


if __name__ == "__main__":
    main()
