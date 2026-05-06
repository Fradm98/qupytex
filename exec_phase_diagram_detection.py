
# fix relative imports
import os
cwd = os.path.normpath(os.getcwd())
cwd = cwd.split(os.sep)
find = cwd.index("fidelity-phase-tran")
newdir = f"{os.sep}".join(cwd[:find+1])
os.chdir(newdir)

# import known packages
import numpy as np
import pickle
import gzip

from matplotlib import pyplot as plt

from scipy import signal

from tenpy.networks.mps import MPS

# import adhoc packages
from qphaset.phases import gstates_to_rdms_matrix, gstates_to_rdms_matrix_qs_mps, phases_vfield, generalized_k_rdm
from qphaset.models import get_bond_dim_qs_mps
from qphaset.plotting import plot_grad_g_angle_stream, plot_grad_g_angle4, plot_grad_g_angle_sin_cos

# choose which tensor network package to use:
# tnpy, qsmps = True, False
tnpy, qsmps = False, True

# ## Phase transitions detection
# Implemementation of one of the main results of the paper.

"""
Type of filename to use:
filename = 'results/data/{model}_{identifier}.pkl'
"""


# francesco's qs-mps results

path_to_tensor = "/Users/fradm/Desktop/projects/fidelity-phase-tran/results/data"
#filename = f"{path_to_tensor}/Cluster-17d83de7-f497-4403-8306-5aee3bafad5f.pkl" # Cluster 10, c1=-1, upside down, 10 x 10
#filename = "results/data/Cluster-9b763977-c15f-48c6-b384-dbb7adad65bf.pkl" # Cluster 10, c1=1e-3, upside down, 10 x 10
filename =  f"{path_to_tensor}/Cluster-f59fbbd7-7b9a-417d-934c-4a0e6a3c238e.pkl"
filename = "D:/code/projects/3_CLUSTER/results/data/Cluster-6b22eed9-1e68-4cd0-a41c-f904407b657b.pkl"


model_name = "Rydberg"
l = 12
n = 25
# Rydberg params
params = np.linspace(3, 1, n), np.linspace(1, 3, n) # upside-down

params = map(lambda m: m.flatten(), np.meshgrid(*params, indexing='xy'))
params = tuple(params)
params = np.stack(params).T
params_extent = np.concatenate([np.min(params, axis=0), np.max(params, axis=0)])
params_extent = tuple(params_extent[[0, 2, 1, 3]])

if model_name == 'ANNNI':
    path_to_tensor = "D:/code/projects/2_ANNNI/results/data"
    path_to_figures = "D:/code/projects/2_ANNNI/figures"
    axis_name = ('k', 'h')
elif model_name == 'Cluster':
    path_to_tensor = "D:/code/projects/3_CLUSTER/results/data"
    path_to_figures = "D:/code/projects/3_CLUSTER/figures"
    axis_name = ('k', 'h')
elif model_name == 'Rydberg':
    path_to_tensor = "D:/code/projects/4_RYDBERG/results/data"
    path_to_figures = "D:/code/projects/4_RYDBERG/figures"
    axis_name = ('$\\Delta/\\Omega$', '$R_b/a$')
else:
    raise SyntaxError("Choose a valid model among 'ANNNI', 'Cluster', and 'Rydberg'")

filename = f'{path_to_tensor}/{model_name}_L_{l}_lambda_1_{params_extent[2]}-{params_extent[3]}_lambda_2_{params_extent[0]}-{params_extent[1]}_npoints_{n}x{n}.pkl' 


with gzip.open(filename, 'rb') as f:
    data = pickle.load(f)
params = data['params']
l, n = data['l'], data['n']
gstates = data['gstates']
stats = data['stats']

params_extent = np.concatenate([np.min(params, axis=0), np.max(params, axis=0)])
params_extent = tuple(params_extent[[0, 2, 1, 3]])

import types
if isinstance(gstates[0], (types.BuiltinFunctionType, types.BuiltinMethodType)):
    gstates = [gstate() for gstate in gstates]

# Select sites for the partial trace (gstates -> rdms, ie ground states to reduced density matrices).
# Note the concept of site depends on the model. For example in the case of models based on the
# class SpinChainNNN, the site corresponds to 2 qubits.

# sites = [(l // 2) - 2, (l // 2) - 1, l // 2, (l // 2) + 1, (l // 2) + 2]
# sites = [(l // 2) - 1, l // 2, (l // 2) + 1]
sites = [l // 2, (l // 2) + 1]
# sites = [l // 2]

if tnpy:
    rdms = gstates_to_rdms_matrix(gstates, sites=sites)
elif qsmps:
    rdms = gstates_to_rdms_matrix_qs_mps(gstates, sites=sites, generalized=True)


# TODO Fix this in the DMRG runs.
rdms = rdms[::-1]

# Plot RDMs ranks. This is useful to study the problem of the singularities
# of the Quantum Fisher Information Matrix.

rdm_ranks = np.reshape(rdms, (-1, ) + rdms.shape[2:])
rdm_ranks = [np.linalg.matrix_rank(mat, hermitian=True) for mat in rdm_ranks]
rdm_ranks = np.reshape(rdm_ranks, rdms.shape[:2])
plt.matshow(rdm_ranks, origin='lower', extent=params_extent)
plt.colorbar()
plt.close()

# Plot bond dimension.
if tnpy:
    bond_dim_map = np.array([np.mean(psi.chi) for psi in gstates])
    bond_dim_map = bond_dim_map.reshape((n, n))
    plt.matshow(bond_dim_map, origin='lower', extent=params_extent)
    plt.title('Bond dimension stat')
    plt.colorbar()

elif qsmps:
    # TODO for now we save arrays instead of objects of the class MPS
    bond_dim_map = np.array([np.mean(get_bond_dim_qs_mps(psi)) for psi in gstates])
    bond_dim_map = bond_dim_map.reshape((n, n))
    plt.matshow(bond_dim_map, origin='lower', extent=params_extent)
    plt.title('Bond dimension stat')
    plt.colorbar()
    plt.savefig(f"L_{l}_{model_name}_{n}x{n}_{len(sites)}-rdm_times.png")

plt.close()

eps = params[1, 0] - params[0, 0]

# Plot execution time per pixel (DMRG runtime).

plt.matshow(np.array(stats['times']).reshape((n, n))[::-1], cmap='hot', origin='lower', extent=params_extent)
plt.colorbar()
plt.savefig(f"{path_to_figures}/{model_name}_L_{l}_{n}x{n}_{len(sites)}-rdm_times.png")
plt.close()

# Grad laplacian filter with upsampling
grad_g = phases_vfield(rdms)
# TODO review params_extent, consider coordinates shift.


# *** Scaling behaviour (experimental) ***

dlog_g = phases_vfield(rdms, log_g=True, scale=1)

from qphaset.models import params_2d_lattice
xy = params_2d_lattice(params_extent[:2], params_extent[2:], n=len(dlog_g))
xy = np.reshape(xy, (len(dlog_g), ) * 2 + (2, ))
# TODO Double check real/imag association with x, y!
nu = xy[:, :, 0] * np.real(dlog_g) + xy[:, :, 1] * np.imag(dlog_g)

# Note the values should not be interpreted directly since the numerical derivative
# is mul by some const factor.
plt.matshow(nu, origin='lower', cmap='twilight')
plt.close()


plot_grad_g_angle_stream(grad_g, params_extent=params_extent, axis_name=axis_name, theory_lines=False);
plt.savefig(f"{path_to_figures}/{model_name}_L_{l}_{n}x{n}_{len(sites)}-rdm.png")
plt.close()

plot_grad_g_angle4(grad_g, params_extent=params_extent);
plt.close()

plot_grad_g_angle_sin_cos(grad_g, params_extent=params_extent);
plt.close()