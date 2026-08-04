"""
qupytex_io.py
=============
Chunked save/load for DMRG ground-state grids on filesystems with per-file
size limits (e.g. CERN EOS, 50 GB cap).

Layout on disk
--------------
Every simulation produces a *manifest* file plus one or more *chunk* files:

    <base>.manifest.pkl.gz          ← index: grid metadata + chunk registry
    <base>.chunk_000.pkl.gz         ← rows [0, R)   of the n1×n2 grid
    <base>.chunk_001.pkl.gz         ← rows [R, 2R)
    ...

The manifest is always tiny (kilobytes) and is the only file you need to
open when you want to query which chunks overlap a parameter sub-region.

Public API
----------
    save_gstates(path_to_tensor, base_filename, data, max_file_gb=45)
    load_gstates(path_to_tensor, base_filename,
                 lambda1_range=None, lambda2_range=None) -> data_dict

    # Zenodo-friendly RDM archive (single .npz, no pickle)
    save_rdms(path_to_rdms, base_filename, rdms, params_grid,
              sites, model_name, l, chi, dmrg_params=None, extra=None) -> path
    load_rdms(path_to_rdms, base_filename,
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

def _rows_per_chunk(n1, n2, l, d, chi, max_file_gb):
    """Compute how many rows fit in one chunk file."""
    bytes_per_row = n2 * _estimate_bytes_per_state(l, d, chi)  # n2 states per row
    max_bytes     = max_file_gb * 1024 ** 3
    rows          = max(1, int(max_bytes // bytes_per_row))
    return min(rows, n1)                                         # never more than n1


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
            params      : np.ndarray, shape (n1*n2, 2)
            gstates     : list of length n1*n2
            l           : int
            n1          : int  (number of rows    / λ₂ grid points)
            n2          : int  (number of columns / λ₁ grid points)
            d           : int
            chi         : int  (bond dimension)
            model_name  : str
            dmrg_params : dict
            stats       : anything pickle-able
    max_file_gb : float
        Hard ceiling per chunk file (default 45 to stay under EOS 50 GB cap).
    """
    os.makedirs(path_to_tensor, exist_ok=True)

    params      = np.asarray(data["params"])   # (n1*n2, 2)
    gstates     = data["gstates"]
    n1          = data["n1"]
    n2          = data["n2"]
    l           = data["l"]
    d           = data["d"]
    chi         = data["chi"]

    assert len(gstates) == n1 * n2, \
        f"Expected {n1*n2} gstates, got {len(gstates)}"

    # ── reshape into n1×n2 grid ────────────────────────────────────────────
    # params[i*n2 + j] corresponds to grid point (row=i, col=j)
    params_grid  = params.reshape(n1, n2, 2)
    gstates_grid = [gstates[i * n2:(i + 1) * n2] for i in range(n1)]

    # ── decide chunk size ──────────────────────────────────────────────────
    R        = _rows_per_chunk(n1, n2, l, d, chi, max_file_gb)
    n_chunks = math.ceil(n1 / R)

    print(f"[qupytex_io] grid={n1}×{n2}, rows_per_chunk={R}, n_chunks={n_chunks}")

    chunk_registry = []   # list of dicts, one per chunk

    for c in range(n_chunks):
        row_start = c * R
        row_end   = min(row_start + R, n1)          # exclusive

        chunk_params  = params_grid[row_start:row_end]      # (rows, n2, 2)
        chunk_gstates = gstates_grid[row_start:row_end]     # list of lists

        # ── parameter extent for this chunk ───────────────────────────────
        lam1_min = float(chunk_params[:, :, 0].min())
        lam1_max = float(chunk_params[:, :, 0].max())
        lam2_min = float(chunk_params[:, :, 1].min())
        lam2_max = float(chunk_params[:, :, 1].max())

        chunk_meta = dict(
            chunk_idx   = c,
            row_start   = row_start,
            row_end     = row_end,               # exclusive
            lam1_range  = (lam1_min, lam1_max),
            lam2_range  = (lam2_min, lam2_max),
            filename    = _chunk_path(path_to_tensor, base_filename, c),
        )
        chunk_registry.append(chunk_meta)

        chunk_data = dict(
            params      = chunk_params,          # shape (rows, n2, 2)
            gstates     = chunk_gstates,         # list[list[MPS tensors]]
            row_start   = row_start,
            row_end     = row_end,
            n1          = n1,
            n2          = n2,
            l           = l,
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
        n1            = n1,
        n2            = n2,
        l             = l,
        d             = d,
        chi           = chi,
        model_name    = data["model_name"],
        dmrg_params   = data["dmrg_params"],
        params_grid   = params_grid,             # (n1, n2, 2) — tiny vs gstates
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
        Inclusive [min, max] for λ₁ (column axis). None = load all.
    lambda2_range : (float, float) or None
        Inclusive [min, max] for λ₂ (row axis). None = load all.

    Returns
    -------
    dict with keys:
        params       : np.ndarray, shape (n1_sub*n2_sub, 2)  — flat, row-major
        params_grid  : np.ndarray, shape (n1_sub, n2_sub, 2) — 2-D grid view
        gstates      : list of length n1_sub*n2_sub           — flat, row-major
        gstates_grid : list of n1_sub lists, each of n2_sub states
        row_indices  : np.ndarray, shape (n1_sub,)            — original row indices
        col_indices  : np.ndarray, shape (n2_sub,)            — original col indices
        n1_sub       : int   — number of selected rows
        n2_sub       : int   — number of selected columns
        l, d, chi, model_name, dmrg_params, stats
    """
    mpath    = _manifest_path(path_to_tensor, base_filename)
    manifest = _gz_load(mpath)
    try:
        n1          = manifest["n1"]
        n2          = manifest["n2"]
    except:
        n           = manifest["n"]
        n1          = n
        n2          = n
    params_grid = manifest["params_grid"]   # (n1, n2, 2)

    # ── find which rows/cols fall inside the requested ranges ─────────────
    # λ₁ is stored in axis-1 (columns), λ₂ in axis-0 (rows)
    lam1_vals = params_grid[0, :, 0]   # shape (n2,)
    lam2_vals = params_grid[:, 0, 1]   # shape (n1,)

    if lambda1_range is None:
        col_mask = np.ones(n2, dtype=bool)
    else:
        lo, hi = lambda1_range
        lo_idx = np.argmin(np.abs(lam1_vals - lo))
        hi_idx = np.argmin(np.abs(lam1_vals - hi))
        if lo_idx > hi_idx:
            lo_idx, hi_idx = hi_idx, lo_idx
        col_mask = np.zeros(n2, dtype=bool)
        col_mask[lo_idx:hi_idx + 1] = True

    if lambda2_range is None:
        row_mask = np.ones(n1, dtype=bool)
    else:
        lo, hi = lambda2_range
        lo_idx = np.argmin(np.abs(lam2_vals - lo))
        hi_idx = np.argmin(np.abs(lam2_vals - hi))
        if lo_idx > hi_idx:
            lo_idx, hi_idx = hi_idx, lo_idx
        row_mask = np.zeros(n1, dtype=bool)
        row_mask[lo_idx:hi_idx + 1] = True

    row_indices = np.where(row_mask)[0]   # selected λ₂ indices (axis 0)
    col_indices = np.where(col_mask)[0]   # selected λ₁ indices (axis 1)

    if len(row_indices) == 0 or len(col_indices) == 0:
        raise ValueError("No grid points found in the requested parameter range.")

    n1_sub = len(row_indices)
    n2_sub = len(col_indices)

    print(f"[qupytex_io] sub-grid: {n1_sub}×{n2_sub} "
          f"(rows {row_indices[0]}–{row_indices[-1]}, "
          f"cols {col_indices[0]}–{col_indices[-1]})")

    # ── find which chunks overlap the needed rows ─────────────────────────
    needed_rows    = set(row_indices.tolist())
    chunks_to_load = [
        c for c in manifest["chunks"]
        if set(range(c["row_start"], c["row_end"])) & needed_rows
    ]

    print(f"[qupytex_io] loading {len(chunks_to_load)} / "
          f"{len(manifest['chunks'])} chunk(s)...")

    # ── load and assemble ─────────────────────────────────────────────────
    row_cache = {}    # global_row_idx → list of n2 states (full row)

    for chunk_meta in chunks_to_load:
        chunk_path = _chunk_path(path_to_tensor, base_filename, chunk_meta["chunk_idx"])
        cdata = _gz_load(chunk_path)
        for local_row, global_row in enumerate(
                range(chunk_meta["row_start"], chunk_meta["row_end"])):
            if global_row in needed_rows:
                row_cache[global_row] = cdata["gstates"][local_row]

    # ── extract sub-grid ──────────────────────────────────────────────────
    gstates_grid = []
    for ri in row_indices:
        full_row = row_cache[ri]               # list of n2 states
        sub_row  = [full_row[ci] for ci in col_indices]
        gstates_grid.append(sub_row)

    # flat lists / arrays — row-major
    gstates_flat = [s for row in gstates_grid for s in row]
    params_sub   = params_grid[np.ix_(row_indices, col_indices)]  # (n1_sub, n2_sub, 2)
    params_flat  = params_sub.reshape(-1, 2)

    # recover stats from the last loaded chunk
    last_chunk_meta = chunks_to_load[-1] if chunks_to_load else None
    last_chunk_path = _chunk_path(path_to_tensor, base_filename, last_chunk_meta["chunk_idx"]) if last_chunk_meta else None
    last_chunk_data = _gz_load(last_chunk_path) if last_chunk_path else {}
    stats = last_chunk_data.get("stats")

    return dict(
        params       = params_flat,
        params_grid  = params_sub,
        gstates      = gstates_flat,
        gstates_grid = gstates_grid,
        row_indices  = row_indices,
        col_indices  = col_indices,
        n1_sub       = n1_sub,
        n2_sub       = n2_sub,
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

def find_manifest(path_to_tensor, model_name=None, l=None,
                  n1=None, n2=None, chi=None):
    """
    Find manifest files in a directory, optionally filtering by model/params.
    Useful when you're not sure of the exact filename.

    Filenames are expected to contain '_npoints_{n1}x{n2}_'.
    For square grids you can pass n1=n2=n; for rectangular ones pass both
    n1 and n2 separately (both must match if provided).
    """
    pattern = os.path.join(path_to_tensor, "*.manifest.pkl.gz")
    files   = glob.glob(pattern)

    if not files:
        print(f"No manifest files found in {path_to_tensor}")
        return []

    matches = []
    for f in files:
        name = os.path.basename(f)
        if model_name and model_name not in name:          continue
        if l           and f"_L_{l}_"        not in name: continue
        if n1 and n2:
            if f"_npoints_{n1}x{n2}_" not in name: continue
        elif n1:
            if f"_npoints_{n1}x"      not in name: continue
        elif n2:
            if f"x{n2}_"              not in name: continue
        if chi         and f"_chi_{chi}_"    not in name: continue
        matches.append(f)

    for f in matches:
        print(f"  {os.path.basename(f)}")
    return matches


def describe_manifest(path_to_tensor, base_filename):
    """Print a summary of what's stored in a manifest."""
    mpath    = _manifest_path(path_to_tensor, base_filename)
    manifest = _gz_load(mpath)
    try:
        n1          = manifest["n1"]
        n2          = manifest["n2"]
    except:
        n           = manifest["n"]
        n1          = n
        n2          = n
    print(f"Model      : {manifest['model_name']}")
    print(f"Grid       : {n1}×{n2}")
    print(f"L, d, chi  : {manifest['l']}, {manifest['d']}, {manifest['chi']}")
    print(f"Chunks     : {len(manifest['chunks'])}")
    for c in manifest["chunks"]:
        print(f"  [{c['chunk_idx']:03d}] rows [{c['row_start']:3d},{c['row_end']:3d}) "
              f"λ₁∈[{c['lam1_range'][0]:.3f},{c['lam1_range'][1]:.3f}] "
              f"λ₂∈[{c['lam2_range'][0]:.3f},{c['lam2_range'][1]:.3f}]")

# ─────────────────────────────────────────────────────────────────────────────
# RDM save / load  (Zenodo-friendly .npz, no pickle)
# ─────────────────────────────────────────────────────────────────────────────

def _rdms_path(directory, base):
    return os.path.join(directory, f"{base}.rdms.npz")


def save_rdms(path_to_rdms, base_filename, rdms, params_grid,
              sites, model_name, l, chi, dmrg_params=None, extra=None):
    """
    Save a pre-computed RDM grid to a single compressed .npz file.

    Parameters
    ----------
    path_to_rdms : str
        Directory where the file is written (created if absent).
    base_filename : str
        Stem of the output file (no extension).
        Typically the same stem used for the MPS chunk files.
    rdms : np.ndarray, shape (n1, n2, D, D)
        Reduced density matrices on the 2-D parameter grid.
        D = d^len(sites).
    params_grid : np.ndarray, shape (n1, n2, 2)
        Parameter values; params_grid[i, j] = (λ₁, λ₂) for grid point (i,j).
    sites : list[int]
        Site indices used for the partial trace (stored as metadata).
    model_name : str
    l : int
        System size.
    chi : int
        Bond dimension of the MPS the RDMs were computed from.
    dmrg_params : dict or None
        Arbitrary DMRG hyper-parameters (JSON-serialised as a string).
    extra : dict or None
        Any additional arrays to store verbatim (keys must be str,
        values must be numpy-castable).  Stored with an 'extra_' prefix.

    Returns
    -------
    str  – path of the written file.
    """
    os.makedirs(path_to_rdms, exist_ok=True)
    path = _rdms_path(path_to_rdms, base_filename)

    import json
    payload = dict(
        rdms        = np.asarray(rdms),
        params_grid = np.asarray(params_grid),
        sites       = np.asarray(sites, dtype=np.int32),
        # scalar metadata packed into a tiny structured array
        meta_l      = np.array([l],    dtype=np.int32),
        meta_chi    = np.array([chi],  dtype=np.int32),
        meta_n1     = np.array([params_grid.shape[0]], dtype=np.int32),
        meta_n2     = np.array([params_grid.shape[1]], dtype=np.int32),
        meta_model  = np.frombuffer(model_name.encode(), dtype=np.uint8),
        meta_dmrg   = np.frombuffer(
            json.dumps(dmrg_params or {}).encode(), dtype=np.uint8
        ),
    )
    if extra:
        for k, v in extra.items():
            payload[f"extra_{k}"] = np.asarray(v)

    np.savez_compressed(path, **payload)
    n1, n2, D, _ = np.asarray(rdms).shape
    print(f"[qupytex_io] saved RDMs → {os.path.basename(path)}"
          f"  grid={n1}×{n2}  D={D}  sites={sites}")
    return path


def load_rdms(path_to_rdms, base_filename,
              lambda1_range=None, lambda2_range=None):
    """
    Load RDMs from a .npz file, optionally slicing to a parameter sub-region.

    Parameters
    ----------
    path_to_rdms : str
        Directory containing the .npz file.
    base_filename : str
        Stem shared with save_rdms.
    lambda1_range : (float, float) or None
        Inclusive [min, max] for λ₁.  None = load all columns.
    lambda2_range : (float, float) or None
        Inclusive [min, max] for λ₂.  None = load all rows.

    Returns
    -------
    dict with keys:
        rdms         : np.ndarray, shape (n1_sub, n2_sub, D, D)
        rdms_flat    : np.ndarray, shape (n1_sub*n2_sub, D, D)
        params_grid  : np.ndarray, shape (n1_sub, n2_sub, 2)
        params       : np.ndarray, shape (n1_sub*n2_sub, 2)
        sites        : list[int]
        row_indices  : np.ndarray  – original row indices selected
        col_indices  : np.ndarray  – original col indices selected
        n1_sub, n2_sub : int
        l, chi, model_name, dmrg_params
        extra        : dict  – any 'extra_*' arrays stored at save time
    """
    import json
    path = _rdms_path(path_to_rdms, base_filename)
    data = np.load(path, allow_pickle=False)

    params_grid_full = data["params_grid"]        # (n1, n2, 2)
    n1 = int(data["meta_n1"][0])
    n2 = int(data["meta_n2"][0])

    lam1_vals = params_grid_full[0, :, 0]         # shape (n2,)
    lam2_vals = params_grid_full[:, 0, 1]         # shape (n1,)

    def _make_mask(vals, rng):
        if rng is None:
            return np.ones(len(vals), dtype=bool)
        lo, hi = rng
        lo_i = np.argmin(np.abs(vals - lo))
        hi_i = np.argmin(np.abs(vals - hi))
        if lo_i > hi_i:
            lo_i, hi_i = hi_i, lo_i
        m = np.zeros(len(vals), dtype=bool)
        m[lo_i:hi_i + 1] = True
        return m

    col_mask = _make_mask(lam1_vals, lambda1_range)
    row_mask = _make_mask(lam2_vals, lambda2_range)

    row_indices = np.where(row_mask)[0]
    col_indices = np.where(col_mask)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        raise ValueError("No grid points in the requested parameter range.")

    n1_sub = len(row_indices)
    n2_sub = len(col_indices)

    rdms_full   = data["rdms"]                    # (n1, n2, D, D)
    rdms_sub    = rdms_full[np.ix_(row_indices, col_indices)]  # (n1_sub, n2_sub, D, D)
    params_sub  = params_grid_full[np.ix_(row_indices, col_indices)]

    print(f"[qupytex_io] loaded RDMs  grid={n1_sub}×{n2_sub}"
          f"  (rows {row_indices[0]}–{row_indices[-1]},"
          f" cols {col_indices[0]}–{col_indices[-1]})")

    extra = {k[len("extra_"):]: data[k]
             for k in data.files if k.startswith("extra_")}

    return dict(
        rdms        = rdms_sub,
        rdms_flat   = rdms_sub.reshape((-1,) + rdms_sub.shape[2:]),
        params_grid = params_sub,
        params      = params_sub.reshape(-1, 2),
        sites       = data["sites"].tolist(),
        row_indices = row_indices,
        col_indices = col_indices,
        n1_sub      = n1_sub,
        n2_sub      = n2_sub,
        l           = int(data["meta_l"][0]),
        chi         = int(data["meta_chi"][0]),
        model_name  = bytes(data["meta_model"]).decode(),
        dmrg_params = json.loads(bytes(data["meta_dmrg"]).decode()),
        extra       = extra,
    )