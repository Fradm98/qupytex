import numpy as np
import matplotlib.pyplot as plt

from qphaset.phases import gstates_to_rdms_matrix_qs_mps, phases_vfield, sanitize_state
from qphaset.plotting import plot_grad_g_angle_stream

from qupytex_io import load_gstates, describe_manifest, find_manifest
import os

# ── Model config ──────────────────────────────────────────────────────────────
model_name = "ANNNI"
l   = 20
n   = 30
chi = 100
c1  = 1e-3

# model_name = "Cluster"
# l   = 20
# n   = 30
# chi = 100
# c1  = 1e-3

# model_name = "Rydberg"
# l   = 20
# n   = 30
# chi = 100
# c1  = 1e-3

# model_name = "tjv"
# l   = 20
# n   = 30
# chi = 100
# c1  = 1e-3

# ── Optional: restrict to a sub-region of the phase diagram ──────────────────
# Set to None to load the full grid.
# These are *inclusive* bounds on the original parameter axes.
lambda1_range = (0.5,0.8)        # e.g. (1.0, 2.0)
lambda2_range = (0.3,0.6)     # e.g. (1.5, 3.0)

# ── Device ────────────────────────────────────────────────────────────────────
device = 'pc'
device = 'ngt'

if device == 'pc':
    device_path = "D:/work"
elif device == 'ngt':
    device_path = "/eos/user/f/fdimarca"

# ── Routing ───────────────────────────────────────────────────────────────────
if model_name == 'ANNNI':
    path_to_tensor  = f"{device_path}/projects/2_ANNNI/results/data"
    path_to_figures = f"{device_path}/projects/2_ANNNI/figures"
    axis_name = ('k', 'h')
elif model_name == 'Cluster':
    path_to_tensor  = f"{device_path}/projects/3_CLUSTER/results/data"
    path_to_figures = f"{device_path}/projects/3_CLUSTER/figures"
    axis_name = ('K', 'h')
elif model_name == 'Rydberg':
    path_to_tensor  = f"{device_path}/projects/4_RYDBERG/results/data"
    path_to_figures = f"{device_path}/projects/4_RYDBERG/figures"
    axis_name = ('$\\Delta/\\Omega$', '$R_b/a$')
elif model_name == 'tjv':
    path_to_tensor  = f"{device_path}/projects/6_TJ/results/data"
    path_to_figures = f"{device_path}/projects/6_TJ/figures"
    axis_name = ('$J_{perp}$', '$t$')
else:
    raise SyntaxError("Choose a valid model among 'ANNNI', 'Cluster', 'Rydberg', 'tjv'")

# ── Reconstruct base filename (must match what exec_dmrg.py wrote) ────────────
# ANNNI 
lambda1_i, lambda1_f      = 0.5, 1.5 
lambda2_i, lambda2_f      = lambda1_f, lambda1_i # reverse the indices

# ANNNI zoom on floating phase
lambda1_i, lambda1_f      = 0.21, 0.8 
lambda2_i, lambda2_f      = 0.01, 0.6

# # Cluster
# lambda1_i, lambda1_f      = 0.5, 1.5
# lambda2_i, lambda2_f      = lambda1_f, lambda1_i # reverse the indices

# # Rydberg
# lambda1_i, lambda1_f      = 1, 3
# lambda2_i, lambda2_f      = lambda1_f, lambda1_i # reverse the indices

# # tjv
# lambda1_i, lambda1_f      = 0.1, 5
# lambda2_i, lambda2_f      = lambda1_f, lambda1_i # reverse the indices

# # tjv zoom (which phase is this?)
# lambda1_i, lambda1_f      = 0.01, 2
# lambda2_i, lambda2_f      = 4, 0.01 # reverse the indices


params_tmp    = np.linspace(lambda1_i, lambda1_f, n), np.linspace(lambda2_i, lambda2_f, n)
params_tmp    = map(lambda m: m.flatten(), np.meshgrid(*params_tmp, indexing='xy'))
params_tmp    = np.stack(tuple(params_tmp)).T

params_grid = params_tmp.reshape(n, n, 2)
lam1_min = float(params_grid[:, :, 0].min())
lam1_max = float(params_grid[:, :, 0].max())
lam2_min = float(params_grid[:, :, 1].min())
lam2_max = float(params_grid[:, :, 1].max())

base_filename = (
    f"{model_name}_L_{l}"
    f"_lambda_1_{lam1_min}-{lam1_max}"
    f"_lambda_2_{lam2_min}-{lam2_max}"
    f"_npoints_{n}x{n}_chi_{chi}_eps_{c1}"
)

# ── Find manifest automatically ───────────────────────────────────────────────
manifests = find_manifest(path_to_tensor, model_name=model_name, l=l, n=n, chi=chi)

if not manifests:
    raise FileNotFoundError(f"No matching manifest in {path_to_tensor}")
if len(manifests) > 1:
    print("Multiple matches — using first. Set base_filename manually if wrong.")

manifest_path = manifests[-1]
# extract base_filename from the path
base_filename = os.path.basename(manifest_path).replace(".manifest.pkl.gz", "")
print(f"Using: {base_filename}")

# ── (Optional) inspect what is stored before loading ─────────────────────────
describe_manifest(path_to_tensor, base_filename)

# ── Load (full grid or sub-region) ────────────────────────────────────────────
result = load_gstates(
    path_to_tensor = path_to_tensor,
    base_filename  = base_filename,
    lambda1_range  = lambda1_range,
    lambda2_range  = lambda2_range,
)

params       = result["params"]          # (n'*m', 2)  flat
params_grid  = result["params_grid"]     # (n', m', 2)
gstates_grid = result["gstates_grid"]   # list[n'] of list[m']
n_sub        = result["n_sub"]
m_sub        = result["m_sub"]
l            = result["l"]

lam1_min = float(params_grid[:, :, 0].min())
lam1_max = float(params_grid[:, :, 0].max())
lam2_min = float(params_grid[:, :, 1].min())
lam2_max = float(params_grid[:, :, 1].max())

# flat list in row-major order, matching original convention
gstates = [s for row in gstates_grid for s in row]
gstates = [sanitize_state(s) for s in gstates]


# ── Sites for partial trace ───────────────────────────────────────────────────
sites = [l // 2, (l // 2) + 1]

# ── Compute RDMs and phase diagram ───────────────────────────────────────────
rdms    = gstates_to_rdms_matrix_qs_mps(gstates, sites=sites, generalized=True)
grad_g  = phases_vfield(rdms)

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
    f"_{n_sub}x{m_sub}_{len(sites)}-rdm.png"
)
plt.savefig(out)
print(f"Saved → {out}")
