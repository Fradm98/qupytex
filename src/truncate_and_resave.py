"""
truncate_and_resave.py
======================
Truncate one or more gstate files to a common coupling window and resave them
under a new base_filename that reflects the new grid specs.

The key insight: load_gstates already does the slicing (via lambda2_range /
lambda1_range) and returns exactly the fields that save_gstates expects.
So truncation is just load → repack → save, with no recomputation.

Assumption: the target grid [lam2_min_target, lam2_max_target] with
n2_target points must be an exact subset of the source grid, i.e. the
source points at those values exist (up to floating-point tolerance).
This is guaranteed when the coarser step divides the finer one evenly,
which you confirmed holds for your files.
"""

import numpy as np
from qupytex_io import load_gstates, save_gstates, describe_manifest


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────

def truncate_and_resave(
    *,
    path_to_tensor,
    base_filename_src,
    # target coupling window (λ₂ axis, i.e. the axis you scan along)
    lam2_min_target,
    lam2_max_target,
    # target λ₁ window — set to None to keep whatever is in the source file
    lam1_min_target=None,
    lam1_max_target=None,
    # new base_filename; if None it is built automatically from the source name
    # by replacing the λ₂ and npoints parts
    base_filename_dst=None,
    # model/grid identifiers needed to build the new filename automatically
    model_name=None,
    l=None,
    n1_new=None,        # if None, kept from the loaded sub-grid (n1_sub)
    n2_new=None,        # if None, counted from the loaded sub-grid (n2_sub)
    chi=None,
    c1=None,
    lam1_min_new=None,  # for the filename; defaults to lam1_min_target
    lam1_max_new=None,
    max_file_gb=45,
    verbose=True,
):
    """
    Load, slice, and resave a single file.

    Returns
    -------
    base_filename_dst : str   – the filename stem used for the new files
    result            : dict  – what load_gstates returned (the sliced data)
    """
    # ── load & slice ──────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'─'*60}")
        print(f"Source : {base_filename_src}")
        describe_manifest(path_to_tensor, base_filename_src)

    lambda2_range = (lam2_min_target, lam2_max_target)
    lambda1_range = (
        (lam1_min_target, lam1_max_target)
        if lam1_min_target is not None
        else None
    )

    result = load_gstates(
        path_to_tensor=path_to_tensor,
        base_filename=base_filename_src,
        lambda1_range=lambda1_range,
        lambda2_range=lambda2_range,
    )

    n1_sub = result["n1_sub"]
    n2_sub = result["n2_sub"]

    # ── verify the sub-grid really matches the target point count ─────────────
    if n2_new is not None and n2_sub != n2_new:
        raise ValueError(
            f"Expected n2_new={n2_new} points in λ₂ window "
            f"[{lam2_min_target}, {lam2_max_target}], "
            f"but got {n2_sub}.  "
            f"Check that the source grid contains exactly those points."
        )

    # ── build target base_filename if not supplied ─────────────────────────────
    if base_filename_dst is None:
        _model   = model_name  or result["model_name"]
        _l       = l           or result["l"]
        _chi     = chi         or result["chi"]
        _c1      = c1          if c1 is not None else 0
        _n1      = n1_new      or n1_sub
        _n2      = n2_new      or n2_sub
        _lam1_lo = lam1_min_new if lam1_min_new is not None else (
                       lam1_min_target if lam1_min_target is not None
                       else float(result["params_grid"][0, 0, 0])
                   )
        _lam1_hi = lam1_max_new if lam1_max_new is not None else (
                       lam1_max_target if lam1_max_target is not None
                       else float(result["params_grid"][0, -1, 0])
                   )
        base_filename_dst = (
            f"{_model}_L_{_l}"
            f"_lambda_1_{_lam1_lo}-{_lam1_hi}"
            f"_lambda_2_{lam2_min_target}-{lam2_max_target}"
            f"_npoints_{_n1}x{_n2}_chi_{_chi}_eps_{_c1}"
        )

    # ── repack into the dict expected by save_gstates ─────────────────────────
    # save_gstates wants a flat gstates list and params array (n1*n2, 2)
    data_to_save = dict(
        params      = result["params"],          # (n1_sub*n2_sub, 2)  already flat
        gstates     = result["gstates"],         # flat list, length n1_sub*n2_sub
        n1          = n1_sub,
        n2          = n2_sub,
        l           = result["l"],
        d           = result["d"],
        chi         = result["chi"],
        model_name  = result["model_name"],
        dmrg_params = result["dmrg_params"],
        stats       = result["stats"],
    )

    if verbose:
        print(f"\nTarget : {base_filename_dst}")
        print(f"  sub-grid : {n1_sub}×{n2_sub}")
        print(f"  λ₂ window: [{lam2_min_target}, {lam2_max_target}]")

    save_gstates(
        path_to_tensor=path_to_tensor,
        base_filename=base_filename_dst,
        data=data_to_save,
        max_file_gb=max_file_gb,
    )

    if verbose:
        print(f"Done ✓  →  {base_filename_dst}")

    return base_filename_dst, result


# ─────────────────────────────────────────────────────────────────────────────
# Batch helper: align a list of (L, chi) files to a common window
# ─────────────────────────────────────────────────────────────────────────────

def align_files_to_common_window(
    *,
    path_to_tensor,
    file_specs,             # list of dicts – see example below
    lam2_min_target,
    lam2_max_target,
    n2_target,              # expected point count in the target window
    lam1_min_target=None,
    lam1_max_target=None,
    max_file_gb=45,
    verbose=True,
):
    """
    Truncate and resave several files to the same coupling window.

    Parameters
    ----------
    file_specs : list of dicts, each with keys:
        base_filename_src  : str   – source file stem
        base_filename_dst  : str or None  – if None, auto-built
        # optional overrides for auto-naming:
        model_name, l, chi, c1, n1_new, lam1_min_new, lam1_max_new

    Returns
    -------
    list of base_filename_dst strings (in the same order as file_specs)
    """
    results = []
    for spec in file_specs:
        dst, _ = truncate_and_resave(
            path_to_tensor    = path_to_tensor,
            base_filename_src = spec["base_filename_src"],
            lam2_min_target   = lam2_min_target,
            lam2_max_target   = lam2_max_target,
            n2_new            = n2_target,
            lam1_min_target   = lam1_min_target,
            lam1_max_target   = lam1_max_target,
            base_filename_dst = spec.get("base_filename_dst"),
            model_name        = spec.get("model_name"),
            l                 = spec.get("l"),
            n1_new            = spec.get("n1_new"),
            chi               = spec.get("chi"),
            c1                = spec.get("c1"),
            lam1_min_new      = spec.get("lam1_min_new"),
            lam1_max_new      = spec.get("lam1_max_new"),
            max_file_gb       = max_file_gb,
            verbose           = verbose,
        )
        results.append(dst)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    device      = "ngt"
    device_path = "/eos/user/f/fdimarca" if device == "ngt" else "D:/work"
    path        = f"{device_path}/projects/2_ANNNI/results/data"

    # Common target window: 201 points in [0.9, 1.1]
    # (the narrowest grid you have; every wider grid contains these points)
    LAM2_LO  = 0.9
    LAM2_HI  = 1.1
    N2_TARGET = 201

    # ── Example 1: single file ────────────────────────────────────────────────
    # chi=50 was run on 401 pts in [0.8, 1.2] → truncate to 201 pts in [0.9, 1.1]
    truncate_and_resave(
        path_to_tensor    = path,
        base_filename_src = (
            "ANNNI_L_100"
            "_lambda_1_0.001-0.001"
            "_lambda_2_0.8-1.2"
            "_npoints_1x401_chi_50_eps_0"
        ),
        lam2_min_target   = LAM2_LO,
        lam2_max_target   = LAM2_HI,
        n2_new            = N2_TARGET,
        # auto-build destination filename from these fields:
        lam1_min_new      = 0.001,
        lam1_max_new      = 0.001,
        c1                = 0,
    )

    truncate_and_resave(
        path_to_tensor    = path,
        base_filename_src = (
            "ANNNI_L_120"
            "_lambda_1_0.001-0.001"
            "_lambda_2_0.8-1.2"
            "_npoints_1x401_chi_100_eps_0"
        ),
        lam2_min_target   = LAM2_LO,
        lam2_max_target   = LAM2_HI,
        n2_new            = N2_TARGET,
        # auto-build destination filename from these fields:
        lam1_min_new      = 0.001,
        lam1_max_new      = 0.001,
        c1                = 0,
    )