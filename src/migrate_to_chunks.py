"""
migrate_to_chunks.py
====================
Convert old single-file DMRG saves (.pkl, .pkl.gz, .h5) to the new
chunked + manifest format used by qupytex_io.

Usage
-----
    python migrate_to_chunks.py                        # interactive mode
    python migrate_to_chunks.py path/to/file.pkl.gz   # direct mode

The script:
  1. Loads the old file
  2. Infers missing metadata (d, chi) if possible, or asks you
  3. Writes chunks + manifest via save_gstates
  4. Optionally deletes or renames the old file
"""

import os
import sys
import glob
import gzip
import pickle
import numpy as np


# ── try to import h5py (optional) ────────────────────────────────────────────
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ── local import ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from qupytex_io import save_gstates


# ─────────────────────────────────────────────────────────────────────────────
# Loaders for old formats
# ─────────────────────────────────────────────────────────────────────────────

def load_old_pkl(path):
    opener = gzip.open if path.endswith(".gz") else open
    mode   = "rb"
    with opener(path, mode) as f:
        return pickle.load(f)


def load_old_h5(path):
    if not HAS_H5PY:
        raise ImportError("h5py is required to load .h5 files: pip install h5py")
    import h5py
    data = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            val = f[key][()]
            # bytes → str for string fields
            if isinstance(val, bytes):
                val = val.decode()
            data[key] = val
    return data


def load_old_file(path):
    ext = path.lower()
    if ext.endswith(".h5") or ext.endswith(".hdf5"):
        return load_old_h5(path)
    elif ext.endswith(".pkl.gz") or ext.endswith(".pkl"):
        return load_old_pkl(path)
    else:
        raise ValueError(f"Unrecognised extension: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Metadata inference
# ─────────────────────────────────────────────────────────────────────────────

def _ask_int(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  Please enter an integer.")


def _infer_chi(gstates):
    """Try to read bond dimension from the first MPS tensor."""
    try:
        first = gstates[0]
        if isinstance(first, (list, np.ndarray)):
            first = first[0]
        # MPS site tensors are typically (chi_left, d, chi_right)
        if hasattr(first, "shape") and len(first.shape) == 3:
            return int(max(first.shape[0], first.shape[2]))
    except Exception:
        pass
    return None


def _infer_d(gstates):
    """Try to read physical dimension from the first MPS tensor."""
    try:
        first = gstates[0]
        if isinstance(first, (list, np.ndarray)):
            first = first[0]
        if hasattr(first, "shape") and len(first.shape) == 3:
            return int(first.shape[1])
    except Exception:
        pass
    return None


def _infer_n(gstates, n_hint=None):
    """Infer n from len(gstates) assuming square grid."""
    total = len(gstates)
    n     = int(round(total ** 0.5))
    if n * n == total:
        return n
    if n_hint is not None:
        return n_hint
    return None


def complete_metadata(data, path):
    """
    Ensure data dict has all keys required by save_gstates.
    Infers what it can, asks the user for the rest.
    """
    print(f"\n── Metadata for: {os.path.basename(path)} ──")

    gstates = data.get("gstates")
    if gstates is None:
        raise KeyError("'gstates' key not found in the loaded file.")

    # ── n ─────────────────────────────────────────────────────────────────
    if "n" not in data or data["n"] is None:
        n_inferred = _infer_n(gstates)
        if n_inferred:
            print(f"  inferred n={n_inferred} (√{len(gstates)})")
            data["n"] = n_inferred
        else:
            data["n"] = _ask_int("  n (grid points per axis)")

    n = data["n"]
    assert len(gstates) == n * n, \
        f"len(gstates)={len(gstates)} but n*n={n*n} — check n."

    # ── l ─────────────────────────────────────────────────────────────────
    if "l" not in data or data["l"] is None:
        data["l"] = _ask_int("  l (chain length)")

    # ── d ─────────────────────────────────────────────────────────────────
    if "d" not in data or data["d"] is None:
        d_inferred = _infer_d(gstates)
        if d_inferred:
            print(f"  inferred d={d_inferred} from tensor shape")
            data["d"] = d_inferred
        else:
            data["d"] = _ask_int("  d (physical dimension)", default=2)

    # ── chi ───────────────────────────────────────────────────────────────
    if "chi" not in data or data["chi"] is None:
        chi_inferred = _infer_chi(gstates)
        if chi_inferred:
            print(f"  inferred chi={chi_inferred} from tensor shape")
            data["chi"] = chi_inferred
        else:
            data["chi"] = _ask_int("  chi (bond dimension)", default=50)

    # ── model_name ────────────────────────────────────────────────────────
    if "model_name" not in data or not data["model_name"]:
        data["model_name"] = input("  model_name (ANNNI/Cluster/Rydberg/tjv): ").strip()

    # ── dmrg_params ───────────────────────────────────────────────────────
    if "dmrg_params" not in data or data["dmrg_params"] is None:
        print("  dmrg_params not found — storing empty dict.")
        data["dmrg_params"] = {}

    # ── stats ─────────────────────────────────────────────────────────────
    if "stats" not in data:
        data["stats"] = None

    # ── params ────────────────────────────────────────────────────────────
    if "params" not in data or data["params"] is None:
        print("  'params' array not found in file.")
        print("  Please enter the parameter grid bounds used for this run.")
        lam1_min = float(input("    lambda1 min: "))
        lam1_max = float(input("    lambda1 max: "))
        lam2_min = float(input("    lambda2 min: "))
        lam2_max = float(input("    lambda2 max: "))
        n = data["n"]
        lam1 = np.linspace(lam1_min, lam1_max, n)
        lam2 = np.linspace(lam2_min, lam2_max, n)
        grid = np.meshgrid(lam1, lam2, indexing='xy')
        data["params"] = np.stack([m.flatten() for m in grid]).T

    data["params"] = np.asarray(data["params"])
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Base filename builder
# ─────────────────────────────────────────────────────────────────────────────

def build_base_filename(data):
    params  = np.asarray(data["params"])
    n       = data["n"]
    l       = data["l"]
    chi     = data["chi"]
    c1      = data.get("c1") or data.get("eps") or data["dmrg_params"].get("eps", "?")
    model   = data["model_name"]

    pe = np.concatenate([params.min(axis=0), params.max(axis=0)])
    pe = pe[[0, 2, 1, 3]]   # [lam1_min, lam1_max, lam2_min, lam2_max]

    return (
        f"{model}_L_{l}"
        f"_lambda_1_{pe[0]}-{pe[1]}"
        f"_lambda_2_{pe[2]}-{pe[3]}"
        f"_npoints_{n}x{n}_chi_{chi}_eps_{c1}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main migration logic
# ─────────────────────────────────────────────────────────────────────────────

def migrate_file(path, max_file_gb=45, keep_old=True):
    print(f"\n{'═'*60}")
    print(f"Migrating: {path}")
    print('═'*60)

    # ── load ──────────────────────────────────────────────────────────────
    print("  Loading old file...")
    data = load_old_file(path)
    print(f"  Keys found: {list(data.keys())}")

    # ── complete metadata ─────────────────────────────────────────────────
    data = complete_metadata(data, path)

    # ── output directory = same as old file ───────────────────────────────
    directory = os.path.dirname(os.path.abspath(path))

    # ── base filename: reuse old stem or rebuild from metadata ────────────
    stem = os.path.basename(path)
    for ext in (".pkl.gz", ".pkl", ".h5", ".hdf5"):
        stem = stem.replace(ext, "")

    # Check if stem already has the right naming convention
    if "lambda_1" in stem and "lambda_2" in stem:
        base_filename = stem
        print(f"  Reusing existing filename stem: {base_filename}")
    else:
        base_filename = build_base_filename(data)
        print(f"  Built new filename stem: {base_filename}")

    # ── check if manifest already exists ─────────────────────────────────
    manifest_path = os.path.join(directory, f"{base_filename}.manifest.pkl.gz")
    if os.path.exists(manifest_path):
        overwrite = input(f"  Manifest already exists. Overwrite? [y/N]: ").strip().lower()
        if overwrite != "y":
            print("  Skipping.")
            return

    # ── save in new format ────────────────────────────────────────────────
    print("  Writing chunks + manifest...")
    save_gstates(
        path_to_tensor = directory,
        base_filename  = base_filename,
        data           = data,
        max_file_gb    = max_file_gb,
    )

    # ── handle old file ───────────────────────────────────────────────────
    if not keep_old:
        os.rename(path, path + ".migrated_backup")
        print(f"  Old file renamed to: {os.path.basename(path)}.migrated_backup")
    else:
        print(f"  Old file kept at: {path}")

    print(f"  ✓ Done.")


def find_old_files(directory):
    """Find all pkl/h5 files that don't have a corresponding manifest."""
    patterns  = ["*.pkl", "*.pkl.gz", "*.h5", "*.hdf5"]
    old_files = []
    for pat in patterns:
        old_files.extend(glob.glob(os.path.join(directory, pat)))

    # exclude chunk and manifest files already in new format
    old_files = [
        f for f in old_files
        if ".manifest." not in f and ".chunk_" not in f
    ]
    return sorted(old_files)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) > 1:
        # direct mode: file paths passed as arguments
        paths = sys.argv[1:]
    else:
        # interactive mode: ask for directory
        directory = input("Directory containing old files (leave blank for current): ").strip()
        if not directory:
            directory = "."
        paths = find_old_files(directory)
        if not paths:
            print("No old .pkl / .h5 files found.")
            sys.exit(0)
        print(f"\nFound {len(paths)} file(s):")
        for p in paths:
            print(f"  {p}")

    max_gb   = float(input("\nMax chunk size in GB [45]: ").strip() or "45")
    keep_old = input("Keep old files (rename to .migrated_backup if no)? [Y/n]: ").strip().lower()
    keep_old = keep_old != "n"

    for path in paths:
        try:
            migrate_file(path, max_file_gb=max_gb, keep_old=keep_old)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    print("\nMigration complete.")
