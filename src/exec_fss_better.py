"""
finite_size_scaling.py
======================
Two analysis functions for the ANNNI (and related) models:

  plot_chi(...)  – fix L, vary chi  → drfs curves + |drfs(chi_max) - drfs(chi_i)|
  plot_L(...)    – fix chi, vary L  → drfs curves + maxima + log fit  a·ln(L) + b

Both functions handle the fact that different (L, chi) runs may have been
computed on different coupling grids (different n_points and/or different
[lambda_min, lambda_max]).  After loading, every curve is restricted to a
common symmetric window  [1 - half_width, 1 + half_width]  centred on 1.0
so that boundary artefacts are excluded before any comparison or peak search.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from qs_mps.applications.ISING.utils import discrete_fidelity_susceptibility
from qs_mps.utils import create_sequential_colors

from qphaset.fidelity import uhlmann_fidelity
from qphaset.phases import (
    gstates_to_rdms_matrix_qs_mps,
    sanitize_state,
    extract_submatrix,
)
from qupytex_io import load_gstates, describe_manifest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_base_filename(model_name, l, lam1_min, lam1_max,
                         lam2_min, lam2_max, n1, n2, chi, c1):
    return (
        f"{model_name}_L_{l}"
        f"_lambda_1_{lam1_min}-{lam1_max}"
        f"_lambda_2_{lam2_min}-{lam2_max}"
        f"_npoints_{n1}x{n2}_chi_{chi}_eps_{c1}"
    )


def _load_and_compute_drfs(
    path_to_tensor,
    base_filename,
    direction,
    sites_fn=None,          # callable(l) -> list[int]; defaults to [l//2-1, l//2]
    lambda1_range=None,
    lambda2_range=None,
):
    """
    Load ground states, build RDMs, compute row-wise fidelities and drfs.

    Returns
    -------
    axis_vals : 1-D np.ndarray   – coupling values (length n2_sub - 1 midpoints)
    drfs_rows : list[np.ndarray] – one drfs array per row of the params grid
    l         : int              – system size as stored in the file
    n2_sub    : int
    """
    result       = load_gstates(
        path_to_tensor=path_to_tensor,
        base_filename=base_filename,
        lambda1_range=lambda1_range,
        lambda2_range=lambda2_range,
    )
    params       = result["params"]
    params_grid  = result["params_grid"]
    gstates_grid = result["gstates_grid"]
    n1_sub       = result["n1_sub"]
    n2_sub       = result["n2_sub"]
    l            = result["l"]

    sites = sites_fn(l) if sites_fn is not None else [l // 2 - 1, l // 2]

    gstates = [s for row in gstates_grid for s in row]
    gstates = [sanitize_state(s) for s in gstates]

    # Derive axis and step directly from params_grid — never trust the
    # passed-in lambda args, which may belong to a different (wider) grid.
    if direction == "h":
        axis_raw = params_grid[0, :, 0]        # λ₁ values along columns
    elif direction == "v":
        axis_raw = params_grid[:, 0, 1]        # λ₂ values along rows
    else:
        raise ValueError("direction must be 'h' or 'v'")

    a = abs(axis_raw[1] - axis_raw[0]) if len(axis_raw) > 1 else 1.0
    axis_mid = 0.5 * (axis_raw[:-1] + axis_raw[1:])   # midpoints

    rdms = gstates_to_rdms_matrix_qs_mps(
        gstates, sites=sites, shape=(n1_sub, n2_sub), generalized=True
    )


    drfs_rows = []
    for i in range(n1_sub):
        fids = [
            uhlmann_fidelity(rdms[i, j], rdms[i, j + 1])
            for j in range(n2_sub - 1)
        ]
        drfs_rows.append(
            np.array(discrete_fidelity_susceptibility(fid=fids, a=a))
        )

    return axis_mid, drfs_rows, l, n2_sub


def _restrict_to_window(axis, drfs_row, center=1.0, half_width=0.05):
    """
    Keep only the points in  [center - half_width, center + half_width].

    Returns (axis_cut, drfs_cut).
    """
    mask = (axis >= center - half_width) & (axis <= center + half_width)
    return axis[mask], drfs_row[mask]


def _interpolate_to_common_grid(curves):
    """
    Given a list of (axis, drfs) tuples with possibly different grids,
    build a common axis (union of all points, then fine linspace) and
    interpolate every curve onto it.

    Returns (common_axis, list_of_interpolated_drfs).
    """
    lo = max(ax[0]  for ax, _ in curves)
    hi = min(ax[-1] for ax, _ in curves)
    # use the finest resolution found
    n  = max(len(ax) for ax, _ in curves)
    common = np.linspace(lo, hi, n)
    interp = [np.interp(common, ax, dr) for ax, dr in curves]
    return common, interp


# ─────────────────────────────────────────────────────────────────────────────
# plot_chi  –  fix L, compare different chi values
# ─────────────────────────────────────────────────────────────────────────────

def plot_chi(
    *,
    model_name,
    l,
    chis,                   # list of chi values, e.g. [50, 100, 150]
    path_to_tensor,
    path_to_figures,
    direction="v",
    # --- per-chi grid specs: list of dicts with keys
    #     n1, n2, lam1_min, lam1_max, lam2_min, lam2_max, c1
    #     (one entry per chi; or a single dict reused for all)
    grid_specs=None,
    # --- fallback / default grid (used when grid_specs is None or missing entry)
    n1=1, n2=201,
    lam1_min=0.001, lam1_max=0.001,
    lam2_min=0.9,   lam2_max=1.1,
    c1=0,
    # --- window around coupling = 1.0
    center=1.0,
    half_width=0.05,        # restrict to [center-half_width, center+half_width]
    # ---
    row_index=0,            # which params-grid row to use (usually 0 for 1-D scan)
    sites_fn=None,
    ax=None,
    verbose=True,
):
    """
    For a fixed L, plot drfs vs coupling for every chi in `chis`.
    Also plots  |drfs(chi_max) - drfs(chi_i)|  for all non-maximum chi.

    Returns
    -------
    dict with keys:
        'chis'        : list[int]
        'axes'        : list[np.ndarray]   – restricted coupling axes
        'drfs'        : list[np.ndarray]   – restricted drfs arrays
        'differences' : dict {chi: np.ndarray}  – |drfs(chi_max) - drfs(chi_i)|
                        on the common axis; only for chi < chi_max
        'common_axis' : np.ndarray
    """
    if grid_specs is None:
        grid_specs = [{}] * len(chis)
    elif isinstance(grid_specs, dict):
        grid_specs = [grid_specs] * len(chis)

    colors = create_sequential_colors(len(chis))
    show   = ax is None
    if show:
        fig, axes_plot = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
        ax_main, ax_diff = axes_plot
    else:
        ax_main = ax
        ax_diff = None

    restricted = []   # (axis_cut, drfs_cut) after window restriction
    for idx, chi in enumerate(chis):
        spec = {**dict(n1=n1, n2=n2,
                       lam1_min=lam1_min, lam1_max=lam1_max,
                       lam2_min=lam2_min, lam2_max=lam2_max,
                       c1=c1),
                **grid_specs[idx]}

        bf = _build_base_filename(
            model_name, l,
            spec["lam1_min"], spec["lam1_max"],
            spec["lam2_min"], spec["lam2_max"],
            spec["n1"],       spec["n2"],
            chi,              spec["c1"],
        )
        if verbose:
            describe_manifest(path_to_tensor, bf)

        axis_mid, drfs_rows, l_loaded, _ = _load_and_compute_drfs(
            path_to_tensor=path_to_tensor,
            base_filename=bf,
            direction=direction,
            sites_fn=sites_fn,
        )
        drfs_row = drfs_rows[row_index]
        axis_cut, drfs_cut = _restrict_to_window(
            axis_mid, drfs_row, center=center, half_width=half_width
        )
        restricted.append((axis_cut, drfs_cut))
        ax_main.plot(axis_cut, drfs_cut, color=colors[idx], label=f"χ={chi}")

    ax_main.set_ylabel("drfs")
    ax_main.set_title(f"{model_name}  L={l}  –  drfs vs χ")
    ax_main.legend()
    ax_main.axvline(center, ls=":", color="gray", lw=0.8)

    # ── differences  |drfs(chi_max) - drfs(chi_i)| ───────────────────────────
    chi_max   = max(chis)
    idx_max   = chis.index(chi_max)
    common_ax, interp_all = _interpolate_to_common_grid(restricted)

    drfs_ref  = interp_all[idx_max]
    diff_dict = {}
    diff_colors = create_sequential_colors(len(chis) - 1)
    dc = 0
    for idx, chi in enumerate(chis):
        if chi == chi_max:
            continue
        diff = np.abs(drfs_ref - interp_all[idx])
        diff_dict[chi] = diff
        if ax_diff is not None:
            ax_diff.plot(common_ax, diff, color=diff_colors[dc],
                         label=f"|drfs({chi_max}) - drfs({chi})|")
            dc += 1

    if ax_diff is not None:
        ax_diff.set_xlabel(f"coupling")
        ax_diff.set_ylabel("|Δ drfs|")
        ax_diff.set_title("Differences w.r.t. largest χ")
        ax_diff.legend()
        ax_diff.axvline(center, ls=":", color="gray", lw=0.8)

    plt.tight_layout()
    if show:
        out = f"{path_to_figures}/{model_name}_L{l}_chi_comparison.png"
        plt.savefig(out, dpi=300)
        print(f"Saved → {out}")
        plt.show()

    return dict(
        chis=chis,
        axes=[r[0] for r in restricted],
        drfs=[r[1] for r in restricted],
        differences=diff_dict,
        common_axis=common_ax,
    )


# ─────────────────────────────────────────────────────────────────────────────
# plot_L  –  fix chi, compare different L values + log fit of maxima
# ─────────────────────────────────────────────────────────────────────────────

def _log_model(x, a, b):
    return a * np.log(x) + b


def plot_L(
    *,
    model_name,
    Ls,                     # list of system sizes
    chi,
    path_to_tensor,
    path_to_figures,
    direction="v",
    # --- per-L grid specs (same logic as plot_chi)
    grid_specs=None,
    n1=1, n2=201,
    lam1_min=0.001, lam1_max=0.001,
    lam2_min=0.9,   lam2_max=1.1,
    c1=0,
    # --- window
    center=1.0,
    half_width=0.05,
    # --- chi errors for the log fit (dict {chi: drfs_array on common axis})
    #     e.g. pass the 'differences' output of plot_chi
    chi_errors=None,        # dict {chi_i: array} – uncertainty from chi convergence
    # --- intrinsic axis error (half step size; set None to auto-compute)
    axis_error=None,        # e.g. 0.001
    row_index=0,
    sites_fn=None,
    ax=None,
    verbose=True,
):
    """
    For a fixed chi, plot drfs vs coupling for every L in `Ls`.
    Finds the maximum of drfs in the restricted window for each L.
    Fits  y = a·ln(L) + b  to the maxima.

    The y-errors on the maxima combine:
      • intrinsic axis uncertainty  (step/2 × |d(drfs)/d(coupling)|  at the peak)
      • chi-convergence error       (max |Δdrfs| in the window, from chi_errors)

    Returns
    -------
    dict with keys:
        'Ls'         : list[int]
        'axes'       : list[np.ndarray]
        'drfs'       : list[np.ndarray]
        'peak_couplings' : np.ndarray   – coupling at which drfs is maximum
        'peak_values'    : np.ndarray   – drfs value at the maximum
        'peak_errors'    : np.ndarray   – combined y-uncertainty on each peak
        'fit_a'      : float
        'fit_b'      : float
        'fit_a_err'  : float
        'fit_b_err'  : float
    """
    if grid_specs is None:
        grid_specs = [{}] * len(Ls)
    elif isinstance(grid_specs, dict):
        grid_specs = [grid_specs] * len(Ls)

    colors = create_sequential_colors(len(Ls))
    show   = ax is None
    if show:
        fig, (ax_main, ax_fit) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        ax_main = ax
        ax_fit  = None

    restricted  = []
    peak_coups  = []
    peak_vals   = []
    peak_errs   = []

    for idx, l in enumerate(Ls):
        spec = {**dict(n1=n1, n2=n2,
                       lam1_min=lam1_min, lam1_max=lam1_max,
                       lam2_min=lam2_min, lam2_max=lam2_max,
                       c1=c1),
                **grid_specs[idx]}

        bf = _build_base_filename(
            model_name, l,
            spec["lam1_min"], spec["lam1_max"],
            spec["lam2_min"], spec["lam2_max"],
            spec["n1"],       spec["n2"],
            chi,              spec["c1"],
        )
        if verbose:
            describe_manifest(path_to_tensor, bf)

        axis_mid, drfs_rows, l_loaded, n2_sub = _load_and_compute_drfs(
            path_to_tensor=path_to_tensor,
            base_filename=bf,
            direction=direction,
            sites_fn=sites_fn,
        )
        drfs_row = drfs_rows[row_index]
        axis_cut, drfs_cut = _restrict_to_window(
            axis_mid, drfs_row, center=center, half_width=half_width
        )
        restricted.append((axis_cut, drfs_cut))

        # ── peak ─────────────────────────────────────────────────────────────
        i_peak  = np.argmax(drfs_cut)
        pk_coup = axis_cut[i_peak]
        pk_val  = drfs_cut[i_peak]
        peak_coups.append(pk_coup)
        peak_vals.append(pk_val)

        # ── y-error: intrinsic axis uncertainty ──────────────────────────────
        step = axis_cut[1] - axis_cut[0] if len(axis_cut) > 1 else (
            axis_error if axis_error else 0.001
        )
        ax_err = axis_error if axis_error is not None else step / 2
        # propagate: δy ≈ |dy/dx| · δx  at the peak
        if 0 < i_peak < len(drfs_cut) - 1:
            slope = abs(drfs_cut[i_peak + 1] - drfs_cut[i_peak - 1]) / (2 * step)
        else:
            slope = 0.0
        y_err_axis = slope * ax_err

        # ── y-error: chi convergence ─────────────────────────────────────────
        y_err_chi = 0.0
        if chi_errors:
            # chi_errors is a dict {chi_val: diff_array on some axis}
            # We take the maximum |Δdrfs| within our window from each entry
            for chi_val, diff_arr_tuple in chi_errors.items():
                # diff_arr_tuple may be (common_axis, diff) or just diff_array
                if isinstance(diff_arr_tuple, tuple):
                    common_ax_err, diff_arr = diff_arr_tuple
                    # restrict to our window
                    mask = ((common_ax_err >= center - half_width) &
                            (common_ax_err <= center + half_width))
                    local_diff = diff_arr[mask]
                else:
                    local_diff = diff_arr_tuple   # assume already restricted
                if len(local_diff):
                    y_err_chi = max(y_err_chi, float(np.max(local_diff)))

        y_err_total = np.sqrt(y_err_axis**2 + y_err_chi**2)
        peak_errs.append(y_err_total)

        ax_main.plot(axis_cut, drfs_cut, color=colors[idx], label=f"L={l}")
        ax_main.axvline(pk_coup, color=colors[idx], ls="--", lw=0.7, alpha=0.6)

    peak_coups = np.array(peak_coups)
    peak_vals  = np.array(peak_vals)
    peak_errs  = np.array(peak_errs)

    ax_main.set_xlabel("coupling")
    ax_main.set_ylabel("drfs")
    ax_main.set_title(f"{model_name}  χ={chi}  –  drfs vs L")
    ax_main.legend()
    ax_main.axvline(center, ls=":", color="gray", lw=0.8)

    # ── log fit ───────────────────────────────────────────────────────────────
    Ls_arr = np.array(Ls, dtype=float)

    sigma = peak_errs if np.any(peak_errs > 0) else None
    try:
        popt, pcov = curve_fit(
            _log_model, Ls_arr, peak_vals,
            sigma=sigma, absolute_sigma=True, p0=[1.0, 0.0]
        )
        perr = np.sqrt(np.diag(pcov))
        a_fit, b_fit   = popt
        a_err, b_err   = perr
    except RuntimeError as e:
        print(f"[plot_L] curve_fit failed: {e}")
        a_fit = b_fit = a_err = b_err = np.nan

    if ax_fit is not None:
        L_fine  = np.linspace(Ls_arr.min() * 0.9, Ls_arr.max() * 1.1, 300)
        ax_fit.plot(L_fine, _log_model(L_fine, a_fit, b_fit),
                    "k-", lw=1.5,
                    label=f"fit: {a_fit:.3f}·ln(L) + {b_fit:.3f}")
        ax_fit.errorbar(
            Ls_arr, peak_vals, yerr=peak_errs if sigma is not None else None,
            fmt="o", color="tab:blue", capsize=4, label="peak drfs"
        )
        ax_fit.set_xlabel("L")
        ax_fit.set_ylabel("peak drfs")
        ax_fit.set_title(
            f"Log fit:  a={a_fit:.4f}±{a_err:.4f},  b={b_fit:.4f}±{b_err:.4f}"
        )
        ax_fit.legend()

    plt.tight_layout()
    if show:
        out = f"{path_to_figures}/{model_name}_chi{chi}_L_scaling.png"
        plt.savefig(out, dpi=300)
        print(f"Saved → {out}")
        plt.show()

    print(
        f"\n[plot_L] Log fit results  (χ={chi})\n"
        f"  a = {a_fit:.5f} ± {a_err:.5f}\n"
        f"  b = {b_fit:.5f} ± {b_err:.5f}\n"
        f"  Peak locations: {peak_coups}\n"
        f"  Peak values   : {peak_vals}\n"
        f"  Peak errors   : {peak_errs}"
    )

    return dict(
        Ls=Ls,
        axes=[r[0] for r in restricted],
        drfs=[r[1] for r in restricted],
        peak_couplings=peak_coups,
        peak_values=peak_vals,
        peak_errors=peak_errs,
        fit_a=a_fit, fit_b=b_fit,
        fit_a_err=a_err, fit_b_err=b_err,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Example usage  (edit to match your actual grids)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── common config ─────────────────────────────────────────────────────────
    model_name = "ANNNI"
    device     = "ngt"
    device_path = "/eos/user/f/fdimarca" if device == "ngt" else "D:/work"
    path_to_tensor  = f"{device_path}/projects/2_ANNNI/results/data"
    path_to_figures = f"{device_path}/projects/2_ANNNI/figures"

    # Each (L, chi) pair may have been run on a different grid.
    # Describe them here; omit a key to fall back to the defaults below.
    #
    # Example: chi=50 was run on 401 pts in (0.8, 1.2),
    #          chi=100 on 201 pts in (0.9, 1.1).
    chi_grid_specs = {
        50:  dict(n1=1, n2=401, lam1_min=0.001, lam1_max=0.001,
                  lam2_min=0.8, lam2_max=1.2, c1=0),
        100: dict(n1=1, n2=201, lam1_min=0.001, lam1_max=0.001,
                  lam2_min=0.9, lam2_max=1.1, c1=0),
        150: dict(n1=1, n2=201, lam1_min=0.001, lam1_max=0.001,
                  lam2_min=0.9, lam2_max=1.1, c1=0),
    }

    # ── 1) plot_chi: fix L=120, vary chi ─────────────────────────────────────
    L_fixed = 120
    chis    = [50, 100, 150]

    res_chi = plot_chi(
        model_name=model_name,
        l=L_fixed,
        chis=chis,
        path_to_tensor=path_to_tensor,
        path_to_figures=path_to_figures,
        direction="v",
        grid_specs=[chi_grid_specs[c] for c in chis],
        center=1.0,
        half_width=0.05,   # restrict to [0.95, 1.05]; adjust as needed
    )

    # ── 2) plot_L: fix chi=50, vary L ────────────────────────────────────────
    Ls   = [120, 140, 160, 180, 200]
    chi  = 50

    # pass differences from step 1 as chi_errors so the log-fit y-errors
    # include chi-convergence uncertainty
    # Format: {chi_val: diff_array_on_common_axis}
    chi_errors_for_fit = {
        c: res_chi["differences"][c]
        for c in res_chi["differences"]
    }

    res_L = plot_L(
        model_name=model_name,
        Ls=Ls,
        chi=chi,
        path_to_tensor=path_to_tensor,
        path_to_figures=path_to_figures,
        direction="v",
        # if all Ls share the same grid:
        n1=1, n2=201,
        lam1_min=0.001, lam1_max=0.001,
        lam2_min=0.9,   lam2_max=1.1,
        c1=0,
        # or pass per-L grid_specs list of dicts (same structure as chi_grid_specs)
        center=1.0,
        half_width=0.05,
        chi_errors=chi_errors_for_fit,
        axis_error=0.001,   # intrinsic step uncertainty
    )