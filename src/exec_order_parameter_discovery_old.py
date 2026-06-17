import numpy as np
import pickle
import gzip

from qiskit.quantum_info import SparsePauliOp

from matplotlib import pyplot as plt

from qphaset.phases import phases_vfield, gstates_to_rdms_matrix_qs_mps, sanitize_state, extract_submatrix
from qphaset.plotting import plot_grad_g_angle_stream, plot_k_components, plot_observable

gamma = None

model_name = "Cluster"
l = 12
n = 20
params = np.linspace(0.1, 1.5, n), np.linspace(1.5, 0.1, n) # upside-down

model_name = "ANNNI"
l = 12
n = 30
params = np.linspace(0.5, 2.1, n), np.linspace(1.6, 0.01, n) # upside-down

params = map(lambda m: m.flatten(), np.meshgrid(*params, indexing='xy'))
params = tuple(params)
params = np.stack(params).T
params_extent = np.concatenate([np.min(params, axis=0), np.max(params, axis=0)])
params_extent = tuple(params_extent[[0, 2, 1, 3]])

theta = 0
#gamma = 50
obs_ev_idx = 2
v0_first_schmidt_vec = False

device = 'pc'
# device = 'ngt'

if device == 'pc':
    device_path = "D:/code"
elif device == 'ngt':
    device_path = "/eos/user/f/fdimarca"

# dmrg params
chi = 100 # bond dimension
c1 = 1e-3 # eps symm. break.
if model_name == 'ANNNI':
    path_to_tensor = f"{device_path}/projects/2_ANNNI/results/data"
    path_to_figures = f"{device_path}/projects/2_ANNNI/figures"
    axis_name = ('k', 'h')

elif model_name == 'Cluster':
    path_to_tensor = f"{device_path}/projects/3_CLUSTER/results/data"
    path_to_figures = f"{device_path}/projects/3_CLUSTER/figures"
    axis_name = ('k', 'h')

elif model_name == 'Rydberg':
    path_to_tensor = f"{device_path}/projects/4_RYDBERG/results/data"
    path_to_figures = f"{device_path}/projects/4_RYDBERG/figures"
    axis_name = ('$\\Delta/\\Omega$', '$R_b/a$')

else:
    raise SyntaxError("Choose a valid model among 'ANNNI', 'Cluster', and 'Rydberg'")


filename = f'{path_to_tensor}/{model_name}_L_{l}_lambda_1_{params_extent[2]}-{params_extent[3]}_lambda_2_{params_extent[0]}-{params_extent[1]}_npoints_{n}x{n}_chi_{chi}_eps_{c1}.pkl'
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
sites = [l // 2]
#sites = [2]
sites = [l // 2, l // 2 + 1]
sites = [l // 2 - 1, l // 2, l // 2 + 1]

rdms = gstates_to_rdms_matrix_qs_mps(gstates, sites=sites, generalized=True)

# set to True if you wanna select a sub matrix of params, x0, y0 (the final matrix will be a square matrix)
select_sub_mat = False

if select_sub_mat:
    x0, y0 = 1, 1

    # Parameter axes
    x_vals = np.linspace(0.5, 2.1, n)
    y_vals = np.linspace(1.6, 0.01, n)

    # Example matrix
    M = np.random.rand(n, n)

    print("\n-------------------------------------------")
    print(f"rdms shape before trimming: {rdms.shape}")
    print(f"params_extent before trimming: {params_extent}")
    rdms, params_extent_red_idx = extract_submatrix(rdms, x_vals, y_vals, x0, y0, dx=1, dy=1)
    params_extent = np.array([x_vals[i] for i in params_extent_red_idx[:2]] + [y_vals[i] for i in params_extent_red_idx[2:]])
    params_extent = tuple(params_extent[[0,1,3,2]])
    print("-------------------------------------------")
    print(f"rdms shape after trimming: {rdms.shape}")
    print(f"params_extent after trimming: {params_extent}")
    print("-------------------------------------------\n")


# theta = 0  # Adjust st phases have opposite signs.
# Most of the times theta=0 is good, however, use theta=pi to obtain the complementary
# order parameter. 

grad_g = phases_vfield(rdms, scale=1)
ys = np.sin(np.angle(grad_g) + theta)

# Labels plot
# plt.matshow(np.sign(ys), origin='lower', extent=params_extent, aspect='auto')

rdms = rdms[1:-1, 1:-1] # TODO fix

lattice_shape = rdms.shape[:2]
rdms = np.reshape(rdms, (-1, ) + rdms.shape[2:])
ys = ys.flatten()

# define the observable
rhoa = rdms[np.nonzero(ys > 0)]
rhob = rdms[np.nonzero(ys < 0)]
rhoa = np.average(rhoa, axis=0)
rhob = np.average(rhob, axis=0)

rhoa = rhoa / np.linalg.norm(rhoa)
rhob = rhob / np.linalg.norm(rhob)

dot_ab = np.trace(rhoa * rhob)
obs = rhoa - dot_ab * rhob
obs = obs / np.sqrt(1 - dot_ab ** 2)

# plt.matshow(np.abs(obs), vmin=0, vmax=1)
# plt.colorbar()

obs_eval, obs_ev = np.linalg.eigh(obs)
# Eigenvalues of the observable, check here the magnitudes (check also the sign!) and explore the specific projectors
# by setting the variable obs_ev_idx with the index of the selected eigenvector.
print(f"eigenvalues of the observable:\n {obs_eval}")

print(SparsePauliOp.from_operator(obs))


figure_name = f"{path_to_figures}/{model_name}_L_{l}_{n}x{n}_{len(sites)}-rdm_OPD"
plot_observable(obs, rdms, sites, figure_name=figure_name, params_extent=params_extent, lattice_shape=lattice_shape)
plot_k_components(obs, rdms, sites, figure_name=figure_name, params_extent=params_extent, lattice_shape=lattice_shape, v0_first_schmidt_vec=v0_first_schmidt_vec)
plot_grad_g_angle_stream(grad_g, params_extent=params_extent, theory_lines=False)
plt.savefig(f"{path_to_figures}/{model_name}_L_{l}_{n}x{n}_{len(sites)}-rdm.png")