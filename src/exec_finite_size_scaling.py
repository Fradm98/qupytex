import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp

from qs_mps.applications.ISING.utils import discrete_fidelity_susceptibility
from qs_mps.utils import create_sequential_colors

from qphaset.fidelity import uhlmann_fidelity
from qphaset.phases import (gstates_to_rdms_matrix_qs_mps, phases_vfield, constructing_order_parameter,
                             sanitize_state, extract_submatrix)

from qupytex_io import load_gstates, describe_manifest, find_manifest

# ── Model config ──────────────────────────────────────────────────────────────
model_name = "ANNNI"
l   = 20
Ls   = [12]
n1  = 1
n2  = 31
chi = 50
c1  = 1e-5

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
lambda1_i, lambda1_f      = 0.001, 0.001 
lambda2_i, lambda2_f      = 0.9, 1.1 # reverse the indices
lambda2_i, lambda2_f      = 0.1, 1.1 # reverse the indices
direction = "v"

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

colors = create_sequential_colors(len(Ls))
i = 0
theta = 0
for l in Ls:
    # ── Sites for partial trace ───────────────────────────────────────────────────
    sites = [l // 2 - 1, l // 2]


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


    if direction=="h":
        a = abs(params_grid[0, 0, 0] - params_grid[0, 1, 0]) if n1_sub > 1 else 1.0
    elif direction=="v":
        a = abs(params_grid[0, 0, 1] - params_grid[0, 1, 1]) if n2_sub > 1 else 1.0
        print(f"a: {a}")


    # ── RDMs ─────────────────────────────────────────────────────────────────────
    # rdms = gstates_to_rdms_matrix_qs_mps(gstates, sites=sites, generalized=True)
    rdms = gstates_to_rdms_matrix_qs_mps(gstates, sites=sites, shape=(n1_sub, n2_sub), generalized=True)

    theta = -np.pi/2
    obs_eval, obs_ev, obs, rdms_flat = constructing_order_parameter(rdms, theta=theta)
    print(f"eigenvalues of the observable:\n{obs_eval}")
    print(SparsePauliOp.from_operator(obs))

    # ── Plots ─────────────────────────────────────────────────────────────────────
    figure_name = (
        f"{path_to_figures}/{model_name}_L_{l}"
        f"_{n1_sub}x{n2_sub}_{len(sites)}-rdm_OPD"
    )

    meas = [np.trace(rdm @ obs) for rdm in rdms_flat]

    plt.plot(np.abs(meas))
    plt.show()
    plt.savefig(f"{figure_name}_fss.png", dpi=300)

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

    print(f"rdms shape: {rdms.shape}")
    fidelity_rdms = []
    for i in range(n1_sub):
        row = []
        for j in range(n2_sub-1):
            row.append(uhlmann_fidelity(rdms[i, j], rdms[i, j+1]))
        fidelity_rdms.append(row)

    dfss_rdms = [discrete_fidelity_susceptibility(fid=row, a=a) for row in fidelity_rdms]
    plt.plot(np.linspace(lambda2_i, lambda2_f, n2)[:-1], dfss_rdms[0], color=colors[i], label=f"L:{l}")

    i+=1

plt.legend()
plt.show()
plt.savefig(f"{path_to_figures}/{base_filename}_fss.png", dpi=300)