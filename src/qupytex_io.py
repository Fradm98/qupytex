"""
qupytex_io.py
=============
Chunked save/load for DMRG ground-state grids on filesystems with per-file
size limits (e.g. CERN EOS, 50 GB cap).

Layout on disk
--------------
Every simulation produces a *manifest* file plus one or more *chunk* files:

    <base>.manifest.pkl.gz          ← index: grid metadata + chunk registry
    <base>.chunk_000.pkl.gz         ← rows [0, R)   of the n×n grid
    <base>.chunk_001.pkl.gz         ← rows [R, 2R)
    ...

The manifest is always tiny (kilobytes) and is the only file you need to
open when you want to query which chunks overlap a parameter sub-region.

Public API
----------
    save_gstates(path_to_tensor, base_filename, data, max_file_gb=45)
    load_gstates(path_to_tensor, base_filename,
                 lambda1_range=None, lambda2_range=None) -> data_dict
"""

import os
import math
import pickle
import gzip
import numpy as np
import glob

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _manifest_path(directory, base):
    return os.path.join(directory, f"{base}.manifest.pkl.gz")

def _chunk_path(directory, base, idx):
    return os.path.join(directory, f"{base}.chunk_{idx:03d}.pkl.gz")

def _gz_dump(obj, path):
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def _gz_load(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def _estimate_bytes_per_state(l, d, chi):
    """
    Rough upper bound: each MPS site tensor has shape (chi, d, chi) or
    (1, d, chi) / (chi, d, 1) at the boundaries. We use the bulk size.
    complex128 = 16 bytes.
    """
    return 16 * l * d * chi ** 2

def _rows_per_chunk(n, l, d, chi, max_file_gb):
    bytes_per_row = n * _estimate_bytes_per_state(l, d, chi)   # n states per row
    max_bytes     = max_file_gb * 1024 ** 3
    rows          = max(1, int(max_bytes // bytes_per_row))
    return min(rows, n)                                          # never more than n


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_gstates(path_to_tensor, base_filename, data, max_file_gb=45):
    """
    Split and save a DMRG result dict into EOS-safe chunk files.

    Parameters
    ----------
    path_to_tensor : str
        Directory where files are written.
    base_filename : str
        Stem shared by all chunk/manifest files (no extension).
    data : dict
        Must contain at minimum:
            params      : np.ndarray, shape (n*n, 2)
            gstates     : list of length n*n
            l, n, d     : int
            chi         : int  (bond dimension)
            model_name  : str
            dmrg_params : dict
            stats       : anything pickle-able
    max_file_gb : float
        Hard ceiling per chunk file (default 45 to stay under EOS 50 GB cap).
    """
    os.makedirs(path_to_tensor, exist_ok=True)

    params      = np.asarray(data["params"])   # (n*n, 2)
    gstates     = data["gstates"]
    n           = data["n"]
    l           = data["l"]
    d           = data["d"]
    chi         = data["chi"]

    assert len(gstates) == n * n, \
        f"Expected {n*n} gstates, got {len(gstates)}"

    # ── reshape into n×n grid ──────────────────────────────────────────────
    # params[i*n + j] corresponds to grid point (row=i, col=j)
    params_grid = params.reshape(n, n, 2)
    gstates_grid = [gstates[i * n:(i + 1) * n] for i in range(n)]

    # ── decide chunk size ─────────────────────────────────────────────────
    R = _rows_per_chunk(n, l, d, chi, max_file_gb)
    n_chunks = math.ceil(n / R)

    print(f"[qupytex_io] grid={n}×{n}, rows_per_chunk={R}, n_chunks={n_chunks}")

    chunk_registry = []   # list of dicts, one per chunk

    for c in range(n_chunks):
        row_start = c * R
        row_end   = min(row_start + R, n)           # exclusive

        chunk_params  = params_grid[row_start:row_end]          # (rows, n, 2)
        chunk_gstates = gstates_grid[row_start:row_end]         # list of lists

        # ── parameter extent for this chunk ───────────────────────────────
        flat = chunk_params.reshape(-1, 2)
        lam1_min = float(chunk_params[:, :, 0].min())
        lam1_max = float(chunk_params[:, :, 0].max())
        lam2_min = float(chunk_params[:, :, 1].min())
        lam2_max = float(chunk_params[:, :, 1].max())

        chunk_meta = dict(
            chunk_idx   = c,
            row_start   = row_start,
            row_end     = row_end,               # exclusive
            lam1_range  = (float(lam1_min), float(lam1_max)),
            lam2_range  = (float(lam2_min), float(lam2_max)),
            filename    = _chunk_path(path_to_tensor, base_filename, c),
        )
        chunk_registry.append(chunk_meta)

        chunk_data = dict(
            params      = chunk_params,            # shape (rows, n, 2)
            gstates     = chunk_gstates,           # list[list[MPS tensors]]
            row_start   = row_start,
            row_end     = row_end,
            n           = n,
            l           = data["l"],
            d           = d,
            chi         = chi,
            model_name  = data["model_name"],
            dmrg_params = data["dmrg_params"],
            stats       = data.get("stats"),
        )

        path = _chunk_path(path_to_tensor, base_filename, c)
        print(f"  chunk {c:03d}: rows [{row_start}, {row_end}) "
              f"λ₁∈[{lam1_min:.3f},{lam1_max:.3f}] "
              f"λ₂∈[{lam2_min:.3f},{lam2_max:.3f}] → {os.path.basename(path)}")
        _gz_dump(chunk_data, path)

    # ── write manifest ─────────────────────────────────────────────────────
    manifest = dict(
        base_filename = base_filename,
        n             = n,
        l             = data["l"],
        d             = d,
        chi           = chi,
        model_name    = data["model_name"],
        dmrg_params   = data["dmrg_params"],
        params_grid   = params_grid,           # (n, n, 2) — tiny vs gstates
        chunks        = chunk_registry,
    )
    mpath = _manifest_path(path_to_tensor, base_filename)
    _gz_dump(manifest, mpath)
    print(f"[qupytex_io] manifest → {os.path.basename(mpath)}")
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_gstates(path_to_tensor, base_filename,
                 lambda1_range=None, lambda2_range=None):
    """
    Load ground states for a (optionally restricted) parameter sub-region.

    Parameters
    ----------
    path_to_tensor : str
        Directory containing the manifest and chunk files.
    base_filename : str
        Stem shared by all files.
    lambda1_range : (float, float) or None
        Inclusive [min, max] for λ₁. None = load all.
    lambda2_range : (float, float) or None
        Inclusive [min, max] for λ₂. None = load all.

    Returns
    -------
    dict with keys:
        params      : np.ndarray, shape (n'*n', 2)   — flat, matching gstates
        params_grid : np.ndarray, shape (n', n', 2)  — 2-D grid view
        gstates     : list of length n'*n'            — flat, row-major
        gstates_grid: list of n' lists, each of n' states
        row_indices : np.ndarray, shape (n',)         — original row indices
        col_indices : np.ndarray, shape (n',)         — original col indices
        n_sub       : int                             — n' (sub-grid size per axis)
        l, d, chi, model_name, dmrg_params, stats
    """
    mpath    = _manifest_path(path_to_tensor, base_filename)
    manifest = _gz_load(mpath)

    n            = manifest["n"]
    params_grid  = manifest["params_grid"]   # (n, n, 2)

    # ── find which rows/cols fall inside the requested ranges ─────────────
    lam1_vals = params_grid[0, :, 0]   # λ₁ varies along rows (axis 0)
    lam2_vals = params_grid[:, 0, 1]   # λ₂ varies along cols (axis 1)


    if lambda1_range is None:
        row_mask = np.ones(n, dtype=bool)
    else:
        lo, hi   = lambda1_range
        row_mask = (lam1_vals >= lo) & (lam1_vals <= hi)

    if lambda2_range is None:
        col_mask = np.ones(n, dtype=bool)
    else:
        lo, hi   = lambda2_range
        col_mask = (lam2_vals >= lo) & (lam2_vals <= hi)

    row_indices = np.where(row_mask)[0]
    col_indices = np.where(col_mask)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        raise ValueError("No grid points found in the requested parameter range.")

    n_sub = len(row_indices)   # n' rows
    m_sub = len(col_indices)   # n' cols (may differ from n_sub)

    print(f"[qupytex_io] sub-grid: {n_sub}×{m_sub} "
          f"(rows {row_indices[0]}–{row_indices[-1]}, "
          f"cols {col_indices[0]}–{col_indices[-1]})")

    # ── find which chunks overlap the needed rows ─────────────────────────
    needed_rows = set(row_indices.tolist())
    chunks_to_load = [
        c for c in manifest["chunks"]
        if set(range(c["row_start"], c["row_end"])) & needed_rows
    ]

    print(f"[qupytex_io] loading {len(chunks_to_load)} / "
          f"{len(manifest['chunks'])} chunk(s)...")

    # ── load and assemble ─────────────────────────────────────────────────
    # We build a temporary row→list-of-states dict
    row_cache = {}    # global_row_idx → list of n states (full row)

    for chunk_meta in chunks_to_load:
        cdata = _gz_load(chunk_meta["filename"])
        for local_row, global_row in enumerate(
                range(chunk_meta["row_start"], chunk_meta["row_end"])):
            if global_row in needed_rows:
                row_cache[global_row] = cdata["gstates"][local_row]

    # ── extract sub-grid ──────────────────────────────────────────────────
    gstates_grid = []
    for ri in row_indices:
        full_row = row_cache[ri]           # list of n states
        sub_row  = [full_row[ci] for ci in col_indices]
        gstates_grid.append(sub_row)

    # flat lists / arrays — row-major, matching original convention
    gstates_flat = [s for row in gstates_grid for s in row]
    params_sub   = params_grid[np.ix_(row_indices, col_indices)]  # (n', m', 2)
    params_flat  = params_sub.reshape(-1, 2)

    # recover stats from the last loaded chunk (or None)
    stats = chunks_to_load[-1].get("stats") if chunks_to_load else None
    # try to get stats from chunk data
    last_chunk_data = _gz_load(chunks_to_load[-1]["filename"]) if chunks_to_load else {}
    stats = last_chunk_data.get("stats")

    return dict(
        params       = params_flat,
        params_grid  = params_sub,
        gstates      = gstates_flat,
        gstates_grid = gstates_grid,
        row_indices  = row_indices,
        col_indices  = col_indices,
        n_sub        = n_sub,
        m_sub        = m_sub,
        l            = manifest["l"],
        d            = manifest["d"],
        chi          = manifest["chi"],
        model_name   = manifest["model_name"],
        dmrg_params  = manifest["dmrg_params"],
        stats        = stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: list available chunks for a manifest
# ─────────────────────────────────────────────────────────────────────────────

def find_manifest(path_to_tensor, model_name=None, l=None, n=None, chi=None):
    """
    Find manifest files in a directory, optionally filtering by model/params.
    Useful when you're not sure of the exact filename.
    """
    pattern = os.path.join(path_to_tensor, "*.manifest.pkl.gz")
    files   = glob.glob(pattern)

    if not files:
        print(f"No manifest files found in {path_to_tensor}")
        return []

    matches = []
    for f in files:
        name = os.path.basename(f)
        if model_name and model_name not in name: continue
        if l   and f"_L_{l}_"       not in name: continue
        if n   and f"_{n}x{n}_"     not in name: continue
        if chi and f"_chi_{chi}_"    not in name: continue
        matches.append(f)

    for f in matches:
        print(f"  {os.path.basename(f)}")
    return matches

def describe_manifest(path_to_tensor, base_filename):
    """Print a summary of what's stored in a manifest."""
    mpath    = _manifest_path(path_to_tensor, base_filename)
    manifest = _gz_load(mpath)
    print(f"Model      : {manifest['model_name']}")
    print(f"Grid       : {manifest['n']}×{manifest['n']}")
    print(f"L, d, chi  : {manifest['l']}, {manifest['d']}, {manifest['chi']}")
    print(f"Chunks     : {len(manifest['chunks'])}")
    for c in manifest["chunks"]:
        print(f"  [{c['chunk_idx']:03d}] rows [{c['row_start']:3d},{c['row_end']:3d}) "
              f"λ₁∈[{c['lam1_range'][0]:.3f},{c['lam1_range'][1]:.3f}] "
              f"λ₂∈[{c['lam2_range'][0]:.3f},{c['lam2_range'][1]:.3f}]")
