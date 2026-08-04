"""
exec_save_rdms.py
=================
Compute reduced density matrices (RDMs) from DMRG ground-state tensors and
save them to a portable, Zenodo-friendly .npz archive.

Run this once after your DMRG sweep.  The resulting file is orders of
magnitude smaller than the MPS chunk files and is the only data the three
downstream analysis scripts need:

    exec_phase_diagram_from_rdms.py
    exec_order_parameter_discovery_from_rdms.py
    exec_finite_size_scaling_from_rdms.py
"""

import numpy as np
import gzip
import pickle

from qphaset.phases import gstates_to_rdms_matrix_qs_mps, sanitize_state
from qupytex_io import load_gstates, find_manifest, describe_manifest, save_rdms
import os

# ── Model config ──────────────────────────────────────────────────────────────
model_name = "ANNNI"
figname = "fig04"
l   = 50
n1  = 64
n2  = 64
chi = 50
c1  = 1e-2
lam1_i, lam1_f = None, None
lam2_i, lam2_f = None, None
old = False

# model_name = "Cluster"
# l   = 20; n1 = 30; n2 = 30; chi = 100; c1 = 1e-3
# lam1_i, lam1_f = 0.5, 1.5
# lam2_i, lam2_f = 0.5, 1.5
# old = True

# model_name = "Rydberg"
# l   = 20; n1 = 30; n2 = 30; chi = 100; c1 = 1e-3
# lam1_i, lam1_f = 1, 3
# lam2_i, lam2_f = 1, 3
# old = False

# model_name = "tjv"
# l   = 12; n1 = 5; n2 = 5; chi = 50; c1 = 1e-2

# ── Sites for the partial trace ───────────────────────────────────────────────
# Use the same sites as in your analysis scripts.
# Phase diagram  → 2-site RDM:  [l//2, l//2 + 1]
# OPD / FSS      → 3-site RDM:  [l//2 - 1, l//2, l//2 + 1]
# You can call save_rdms twice with different site lists if you need both.
sites = [l // 2, l // 2 + 1]

# ── Device ────────────────────────────────────────────────────────────────────
device = 'pc'
device = 'ngt'

if device == 'pc':
    device_path = "D:/work"
elif device == 'ngt':
    device_path = "/eos/user/f/fdimarca"

# ── Routing ───────────────────────────────────────────────────────────────────
# 
assert figname in ['fig04', 'fig05-06', 'fig07-08', 'fig09-10', 'fig12'], "Choose a valid image among 'fig04', 'fig05-06', 'fig07-08', 'fig09-10', 'fig12'"
path_to_tensor  = f"{device_path}/projects/OPD/{figname}"
path_to_rdms    = f"{device_path}/projects/OPD/{figname}/rdms"


if old:
    filename = f'{path_to_tensor}/{model_name}_L_{l}_lambda_1_{lam1_i}-{lam1_f}_lambda_2_{lam2_i}-{lam2_f}_npoints_{n1}x{n2}_chi_{chi}_eps_{c1}.pkl'
    # filename = f'{path_to_tensor}/{model_name}_L_{l}_lambda_1_{params_extent[2]}-{params_extent[3]}_lambda_2_{params_extent[0]}-{params_extent[1]}_npoints_{n}x{n}.pkl'


    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    params = data['params']
    l, n = data['l'], data['n']
    gstates = data['gstates']
    stats = data['stats']

    params_extent = np.concatenate([np.min(params, axis=0), np.max(params, axis=0)])
    params_extent = tuple(params_extent[[0, 2, 1, 3]])

    gstates = [
        sanitize_state(state)
        for row in gstates
        for state in (row if isinstance(row, (list, np.ndarray)) else [row])
    ]
else:
    # ── Find manifest automatically ───────────────────────────────────────────────
    manifests = find_manifest(path_to_tensor, model_name=model_name,
                            l=l, n1=n1, n2=n2, chi=chi)
    if not manifests:
        raise FileNotFoundError(f"No matching manifest in {path_to_tensor}")
    if len(manifests) > 1:
        print("Multiple matches — using last. Set base_filename manually if wrong.")

    manifest_path = manifests[-1]
    base_filename = os.path.basename(manifest_path).replace(".manifest.pkl.gz", "")
    print(f"Using: {base_filename}")

    describe_manifest(path_to_tensor, base_filename)

    # ── Load MPS ground states ────────────────────────────────────────────────────
    result = load_gstates(
        path_to_tensor = path_to_tensor,
        base_filename  = base_filename,
    )

    params_grid  = result["params_grid"]    # (n1, n2, 2)
    gstates_grid = result["gstates_grid"]
    n1_loaded    = result["n1_sub"]
    n2_loaded    = result["n2_sub"]
    l            = result["l"]
    dmrg_params  = result["dmrg_params"]

    gstates = [sanitize_state(s) for row in gstates_grid for s in row]

# ── Compute RDMs ──────────────────────────────────────────────────────────────
print(f"Computing RDMs for sites {sites} ...")
rdms = gstates_to_rdms_matrix_qs_mps(gstates, sites=sites, shape=(n1,n2),generalized=True)
# rdms shape: (n1*n2, D, D) — reshape to grid
# D = rdms.shape[-1]
# rdms = rdms.reshape(n1_loaded, n2_loaded, D, D)
print(f"RDMs shape: {rdms.shape}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = save_rdms(
    path_to_rdms = path_to_rdms,
    base_filename = base_filename,
    rdms          = rdms,
    params_grid   = params_grid,
    sites         = sites,
    model_name    = model_name,
    l             = l,
    chi           = chi,
    dmrg_params   = dmrg_params,
)
print(f"Done. Upload to Zenodo: {out_path}")