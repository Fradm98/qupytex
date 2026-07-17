import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp

from qs_mps.applications.ISING.utils import discrete_fidelity_susceptibility

from qphaset.fidelity import uhlmann_fidelity
from qphaset.phases import (gstates_to_rdms_matrix_qs_mps,
                             sanitize_state, extract_submatrix)

from qupytex_io import load_gstates, describe_manifest, find_manifest

# ── Model config ──────────────────────────────────────────────────────────────
model_name = "ANNNI"
l   = 30
n1  = 3
n2  = 81
chi = 50
c1  = 1e-2

# model_name = "Cluster"
# l   = 20
# n1  = 30
# n2  = 30
# chi = 100
# c1  = 1e-3

# model_name = "Rydberg"
# l   = 20
# n1  = 30
# n2  = 30
# chi = 100
# c1  = 1e-3

# model_name = "tjv"
# l   = 20
# n1  = 30
# n2  = 30
# chi = 100
# c1  = 1e-3

# ── Optional: restrict to a sub-region ───────────────────────────────────────
# Set to None to load the full grid.
lambda1_range = None        # e.g. (0.5, 1.2)
lambda2_range = None        # e.g. (0.3, 1.0)

# ── Device ────────────────────────────────────────────────────────────────────
device = 'pc'
# device = 'ngt'

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
else:
    raise SyntaxError("Choose a valid model among 'ANNNI', 'Cluster', 'Rydberg'")

# ── Reconstruct base filename (must match what exec_dmrg.py wrote) ────────────
# ANNNI 
lambda1_i, lambda1_f      = 0.5, 1.5 
lambda2_i, lambda2_f      = lambda1_f, lambda1_i # reverse the indices

# ANNNI (Ising-like) 
lambda1_i, lambda1_f      = 0.0, 0.02 
lambda1_i, lambda1_f      = 0.005, 0.015 
lambda2_i, lambda2_f      = 0.95, 1.05 # reverse the indices

# # ANNNI zoom on floating phase
# lambda1_i, lambda1_f      = 0.5, 0.8 
# lambda2_i, lambda2_f      = 0.01, 1.5

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


lam1_min, lam1_max = min(lambda1_i, lambda1_f), max(lambda1_i, lambda1_f)
lam2_min, lam2_max = min(lambda2_i, lambda2_f), max(lambda2_i, lambda2_f)

base_filename = (
    f"{model_name}_L_{l}"
    f"_lambda_1_{lam1_min}-{lam1_max}"
    f"_lambda_2_{lam2_min}-{lam2_max}"
    f"_npoints_{n1}x{n2}_chi_{chi}_eps_{c1}"
)


# ── (Optional) inspect what is stored ────────────────────────────────────────
describe_manifest(path_to_tensor, base_filename)

# ── Load ──────────────────────────────────────────────────────────────────────
result = load_gstates(
    path_to_tensor = path_to_tensor,
    base_filename  = base_filename,
    lambda1_range  = lambda1_range,
    lambda2_range  = lambda2_range,
)

params       = result["params"]
params_grid  = result["params_grid"]
gstates_grid = result["gstates_grid"]
n1_sub       = result["n1_sub"]
n2_sub       = result["n2_sub"]
l            = result["l"]

gstates = [s for row in gstates_grid for s in row]
gstates = [sanitize_state(s) for s in gstates]

params_extent = np.concatenate([np.min(params, axis=0), np.max(params, axis=0)])
params_extent = tuple(params_extent[[0, 2, 1, 3]])

# ── OPD config ────────────────────────────────────────────────────────────────
theta               = 0
obs_ev_idx          = 2
v0_first_schmidt_vec = False
a = abs(params_grid[0, 0, 0] - params_grid[0, 1, 0]) if n2_sub > 1 else 1.0

# ── Sites for partial trace ───────────────────────────────────────────────────
sites = [l // 2 - 1, l // 2]

# ── RDMs ─────────────────────────────────────────────────────────────────────
# rdms = gstates_to_rdms_matrix_qs_mps(gstates, sites=sites, generalized=True)
rdms = gstates_to_rdms_matrix_qs_mps(gstates, sites=sites, shape=(n1_sub, n2_sub), generalized=True)

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

fidelity_rdms = []
for i in range(n2_sub):
    row = []
    for j in range(n1_sub - 1):
        row.append(uhlmann_fidelity(rdms[j, i], rdms[j + 1, i]))
    fidelity_rdms.append(row)

dfss_rdms = [discrete_fidelity_susceptibility(fid=row, a=a) for row in fidelity_rdms]
print(dfss_rdms)
plt.plot(dfss_rdms[2])
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

im0 = ax.matshow(np.asarray(dfss_rdms), origin='lower',
                    extent=params_extent, aspect='auto')
ax.set_title("reduced fidelity susceptibility")
ax.set_xlabel(axis_name[0])
ax.set_ylabel(axis_name[1])
fig.colorbar(im0, ax=ax)
plt.show()
