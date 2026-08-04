"""
exec_phase_diagram_from_rdms.py
================================
Reproduce the phase diagram (gradient-g angle stream plot) starting directly
from the pre-computed RDM archive (.npz) — no MPS tensors required.

Prerequisite: run exec_save_rdms.py (with sites = [l//2, l//2+1]) once to
produce the .npz file, then share that file on Zenodo.
"""

import numpy as np
import matplotlib.pyplot as plt

from qphaset.phases import phases_vfield
from qphaset.plotting import plot_grad_g_angle_stream

from qupytex_io import load_rdms

# ── Model config ──────────────────────────────────────────────────────────────
model_name = "ANNNI"
l   = 20
n1  = 30
n2  = 30
chi = 100
c1  = 1e-3

# model_name = "Cluster"
# l   = 20; n1 = 30; n2 = 30; chi = 100; c1 = 1e-3

# model_name = "Rydberg"
# l   = 20; n1 = 30; n2 = 30; chi = 100; c1 = 1e-3

# model_name = "tjv"
# l   = 12; n1 = 5; n2 = 5; chi = 50; c1 = 1e-2

# ── Optional: restrict to a sub-region of the phase diagram ──────────────────
lambda1_range = None        # e.g. (0.21, 0.8)
lambda2_range = None        # e.g. (0.01, 0.6)

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
# ANNNI
lambda1_i, lambda1_f = 0.21, 0.8
lambda2_i, lambda2_f = 0.01, 0.6

# # tjv
# lambda1_i, lambda1_f = -4, 4
# lambda2_i, lambda2_f = lambda1_f, lambda1_i

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
n1_sub      = result["n1_sub"]
n2_sub      = result["n2_sub"]
l           = result["l"]
sites       = result["sites"]

lam1_min = float(params_grid[:, :, 0].min())
lam1_max = float(params_grid[:, :, 0].max())
lam2_min = float(params_grid[:, :, 1].min())
lam2_max = float(params_grid[:, :, 1].max())

print(f"Loaded RDMs: {rdms.shape}  sites={sites}")

# ── Phase diagram ─────────────────────────────────────────────────────────────
grad_g = phases_vfield(rdms)

plot_grad_g_angle_stream(
    grad_g,
    params_extent = [lam1_min, lam1_max, lam2_min, lam2_max],
    axis_name     = axis_name,
    theory_lines  = False,
)

out = (
    f"{path_to_figures}/{model_name}_L_{l}"
    f"_lambda_1_{lam1_min}-{lam1_max}"
    f"_lambda_2_{lam2_min}-{lam2_max}"
    f"_{n1_sub}x{n2_sub}_{len(sites)}-rdm.png"
)
plt.savefig(out)
print(f"Saved → {out}")