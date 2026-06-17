import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp

from qphaset.phases import (phases_vfield, gstates_to_rdms_matrix_qs_mps,
                             sanitize_state, extract_submatrix)
from qphaset.plotting import (plot_grad_g_angle_stream, plot_k_components,
                               plot_observable)

from qupytex_io import load_gstates, describe_manifest

# ── Model config ──────────────────────────────────────────────────────────────
model_name = "ANNNI"
l   = 12
n   = 5
chi = 50
c1  = 1e-3

model_name = "Cluster"
l   = 20
n   = 30
chi = 100
c1  = 1e-3

model_name = "Rydberg"
l   = 20
n   = 30
chi = 100
c1  = 1e-3

model_name = "tjv"
l   = 20
n   = 30
chi = 100
c1  = 1e-3

# ── Optional: restrict to a sub-region ───────────────────────────────────────
# Set to None to load the full grid.
lambda1_range = None        # e.g. (0.5, 1.2)
lambda2_range = None        # e.g. (0.3, 1.0)

# ── Device ────────────────────────────────────────────────────────────────────
device = 'pc'
# device = 'ngt'

if device == 'pc':
    device_path = "D:/code"
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


params_tmp    = np.linspace(lambda1_i, lambda1_f, n), np.linspace(lambda2_i, lambda2_f, n)
params_tmp    = map(lambda m: m.flatten(), np.meshgrid(*params_tmp, indexing='xy'))
params_tmp    = np.stack(tuple(params_tmp)).T
params_extent = np.concatenate([np.min(params_tmp, axis=0), np.max(params_tmp, axis=0)])
params_extent = tuple(params_extent[[0, 2, 1, 3]])

base_filename = (
    f"{model_name}_L_{l}"
    f"_lambda_1_{params_extent[2]}-{params_extent[3]}"
    f"_lambda_2_{params_extent[0]}-{params_extent[1]}"
    f"_npoints_{n}x{n}_chi_{chi}_eps_{c1}"
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
n_sub        = result["n_sub"]
m_sub        = result["m_sub"]
l            = result["l"]

gstates = [s for row in gstates_grid for s in row]
gstates = [sanitize_state(s) for s in gstates]

params_extent = np.concatenate([np.min(params, axis=0), np.max(params, axis=0)])
params_extent = tuple(params_extent[[0, 2, 1, 3]])

# ── OPD config ────────────────────────────────────────────────────────────────
theta               = 0
obs_ev_idx          = 2
v0_first_schmidt_vec = False

# ── Sites for partial trace ───────────────────────────────────────────────────
sites = [l // 2 - 1, l // 2, l // 2 + 1]

# ── RDMs ─────────────────────────────────────────────────────────────────────
rdms = gstates_to_rdms_matrix_qs_mps(gstates, sites=sites, generalized=True)

# ── Optional sub-matrix trimming ─────────────────────────────────────────────
select_sub_mat = False
if select_sub_mat:
    x0, y0 = 1, 1
    x_vals  = np.linspace(lambda1_i, lambda1_f, n_sub)
    y_vals  = np.linspace(lambda2_i, lambda2_f, m_sub)
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

rdms_inner = rdms[1:-1, 1:-1]
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
print(f"eigenvalues of the observable:\n{obs_eval}")
print(SparsePauliOp.from_operator(obs))

# ── Plots ─────────────────────────────────────────────────────────────────────
figure_name = (
    f"{path_to_figures}/{model_name}_L_{l}"
    f"_{n_sub}x{m_sub}_{len(sites)}-rdm_OPD"
)
plot_observable(obs, rdms_flat, sites, figure_name=figure_name,
                params_extent=params_extent, lattice_shape=lattice_shape)
plot_k_components(obs, rdms_flat, sites, figure_name=figure_name,
                  params_extent=params_extent, lattice_shape=lattice_shape,
                  v0_first_schmidt_vec=v0_first_schmidt_vec)
plot_grad_g_angle_stream(grad_g, params_extent=params_extent, theory_lines=False)
plt.savefig(f"{path_to_figures}/{model_name}_L_{l}_{n_sub}x{m_sub}_{len(sites)}-rdm.png")
