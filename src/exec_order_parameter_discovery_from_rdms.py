"""
exec_order_parameter_discovery_from_rdms.py
============================================
Order-parameter discovery (OPD) starting directly from the pre-computed RDM
archive (.npz) — no MPS tensors required.

Prerequisite: run exec_save_rdms.py (with sites = [l//2-1, l//2, l//2+1])
once to produce the .npz file, then share that file on Zenodo.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp

from qphaset.phases import phases_vfield, extract_submatrix
from qphaset.plotting import (plot_grad_g_angle_stream, plot_k_components,
                               plot_observable)

from qupytex_io import load_rdms

# ── Model config ──────────────────────────────────────────────────────────────
model_name = "ANNNI"
l   = 12
n1  = 5
n2  = 5
chi = 50
c1  = 1e-3

# model_name = "Cluster"
# l   = 20; n1 = 30; n2 = 30; chi = 100; c1 = 1e-3

# model_name = "Rydberg"
# l   = 20; n1 = 30; n2 = 30; chi = 100; c1 = 1e-3

# model_name = "tjv"
# l   = 20; n1 = 30; n2 = 30; chi = 100; c1 = 1e-3

# ── Optional: restrict to a sub-region ───────────────────────────────────────
lambda1_range = None
lambda2_range = None

# ── OPD config ────────────────────────────────────────────────────────────────
theta                = 0
obs_ev_idx           = 2
v0_first_schmidt_vec = False

# ── Device ────────────────────────────────────────────────────────────────────
device = 'pc'
# device = 'ngt'

if device == 'pc':
    device_path = "D:/work"
elif device == 'ngt':
    device_path = "/eos/user/f/fdimarca"

# ── Routing ───────────────────────────────────────────────────────────────────
if model_name == 'ANNNI':
    path_to_rdms    = f"{device_path}/projects/2_ANNNI/results/rdms"
    path_to_figures = f"{device_path}/projects/2_ANNNI/figures"
    axis_name = ('k', 'h')
elif model_name == 'Cluster':
    path_to_rdms    = f"{device_path}/projects/3_CLUSTER/results/rdms"
    path_to_figures = f"{device_path}/projects/3_CLUSTER/figures"
    axis_name = ('K', 'h')
elif model_name == 'Rydberg':
    path_to_rdms    = f"{device_path}/projects/4_RYDBERG/results/rdms"
    path_to_figures = f"{device_path}/projects/4_RYDBERG/figures"
    axis_name = ('$\\Delta/\\Omega$', '$R_b/a$')
elif model_name == 'tjv':
    path_to_rdms    = f"{device_path}/projects/6_TJ/results/rdms"
    path_to_figures = f"{device_path}/projects/6_TJ/figures"
    axis_name = ('$t$', '$V$')
else:
    raise SyntaxError("Choose a valid model among 'ANNNI', 'Cluster', 'Rydberg', 'tjv'")

# ── Reconstruct base filename ─────────────────────────────────────────────────
lambda1_i, lambda1_f = 0.5, 1.5
lambda2_i, lambda2_f = lambda1_f, lambda1_i   # reversed

lam1_min, lam1_max = min(lambda1_i, lambda1_f), max(lambda1_i, lambda1_f)
lam2_min, lam2_max = min(lambda2_i, lambda2_f), max(lambda2_i, lambda2_f)

base_filename = (
    f"{model_name}_L_{l}"
    f"_lambda_1_{lam1_min}-{lam1_max}"
    f"_lambda_2_{lam2_min}-{lam2_max}"
    f"_npoints_{n1}x{n2}_chi_{chi}_eps_{c1}"
)

# ── Load RDMs ─────────────────────────────────────────────────────────────────
result = load_rdms(
    path_to_rdms  = path_to_rdms,
    base_filename = base_filename,
    lambda1_range = lambda1_range,
    lambda2_range = lambda2_range,
)

rdms        = result["rdms"]            # (n1_sub, n2_sub, D, D)
params_grid = result["params_grid"]     # (n1_sub, n2_sub, 2)
params      = result["params"]          # (n1_sub*n2_sub, 2)
n1_sub      = result["n1_sub"]
n2_sub      = result["n2_sub"]
l           = result["l"]
sites       = result["sites"]

print(f"Loaded RDMs: {rdms.shape}  sites={sites}")

params_extent = (
    float(params[:, 0].min()), float(params[:, 0].max()),
    float(params[:, 1].min()), float(params[:, 1].max()),
)

# ── Optional sub-matrix trimming ─────────────────────────────────────────────
select_sub_mat = False
if select_sub_mat:
    x0, y0 = 1, 1
    x_vals  = np.linspace(lambda1_i, lambda1_f, n2_sub)
    y_vals  = np.linspace(lambda2_i, lambda2_f, n1_sub)
    print(f"rdms shape before trimming: {rdms.shape}")
    rdms, params_extent_red_idx = extract_submatrix(rdms, x_vals, y_vals, x0, y0, dx=1, dy=1)
    params_extent = np.array(
        [x_vals[i] for i in params_extent_red_idx[:2]] +
        [y_vals[i] for i in params_extent_red_idx[2:]]
    )
    params_extent = tuple(params_extent[[0, 1, 3, 2]])
    print(f"rdms shape after trimming:  {rdms.shape}")

# ── Order parameter discovery ─────────────────────────────────────────────────
grad_g = phases_vfield(rdms, scale=1)
ys     = np.sin(np.angle(grad_g) + theta)

rdms_inner    = rdms[1:-1, 1:-1]
lattice_shape = rdms_inner.shape[:2]
rdms_flat     = rdms_inner.reshape((-1,) + rdms_inner.shape[2:])
ys_flat       = ys.flatten()

rhoa = np.average(rdms_flat[np.nonzero(ys_flat > 0)], axis=0)
rhob = np.average(rdms_flat[np.nonzero(ys_flat < 0)], axis=0)
rhoa /= np.linalg.norm(rhoa)
rhob /= np.linalg.norm(rhob)

dot_ab = np.trace(rhoa @ rhob)
obs    = rhoa - dot_ab * rhob
obs   /= np.sqrt(1 - dot_ab ** 2)

obs_eval, obs_ev = np.linalg.eigh(obs)
print(f"Eigenvalues of the observable:\n{obs_eval}")
print(SparsePauliOp.from_operator(obs))

# ── Plots ─────────────────────────────────────────────────────────────────────
figure_name = (
    f"{path_to_figures}/{model_name}_L_{l}"
    f"_{n1_sub}x{n2_sub}_{len(sites)}-rdm_OPD"
)
plot_observable(obs, rdms_flat, sites, figure_name=figure_name,
                params_extent=params_extent, lattice_shape=lattice_shape)
plot_k_components(obs, rdms_flat, sites, figure_name=figure_name,
                  params_extent=params_extent, lattice_shape=lattice_shape,
                  v0_first_schmidt_vec=v0_first_schmidt_vec)
plot_grad_g_angle_stream(grad_g, params_extent=params_extent, theory_lines=False)
plt.savefig(
    f"{path_to_figures}/{model_name}_L_{l}_{n1_sub}x{n2_sub}_{len(sites)}-rdm.png"
)