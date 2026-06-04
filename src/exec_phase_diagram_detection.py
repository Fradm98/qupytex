# import known packages
import numpy as np
import pickle
import gzip

from matplotlib import pyplot as plt

from scipy import signal

from tenpy.networks.mps import MPS

# import adhoc packages
from qphaset.phases import gstates_to_rdms_matrix, gstates_to_rdms_matrix_qs_mps, phases_vfield, sanitize_state
from qphaset.plotting import plot_grad_g_angle_stream

# choose which tensor network package to use:
# tnpy, qsmps = True, False
tnpy, qsmps = False, True

# ## Phase transitions detection
# Implemementation of one of the main results of the paper.

model_name = "Rydberg"
l = 12
n = 25
params = np.linspace(3, 1, n), np.linspace(1, 3, n) # upside-down


model_name = "Cluster"
l = 15
n = 10
params = np.linspace(0.5, 0.6, n), np.linspace(0.6, 0.5, n) # upside-down

# model_name = "Cluster"
# l = 50
# n = 64
# params = np.linspace(0.5, 1.5, n), np.linspace(1.5, 0.5, n) # upside-down

model_name = "Rydberg"
l = 20
n = 30
params = np.linspace(1, 3, n), np.linspace(1.8, 3, n) # upside-down

model_name = "ANNNI"
l = 12
n = 20
params = np.linspace(0.01, 1.5, n), np.linspace(1.5, 0.01, n) # upside-down

model_name = "tjv"
l = 12
d = 3 # physical local dimension
n = 30
Jz = 10
params = np.linspace(1, 3, n), np.linspace(3, 1, n) # upside-down

params = map(lambda m: m.flatten(), np.meshgrid(*params, indexing='xy'))
params = tuple(params)
params = np.stack(params).T
params_extent = np.concatenate([np.min(params, axis=0), np.max(params, axis=0)])
params_extent = tuple(params_extent[[0, 2, 1, 3]])

device = 'pc'
# device = 'ngt'

if device == 'pc':
    device_path = "D:/work"
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

elif model_name == 'tj':
    path_to_tensor = f"{device_path}/projects/6_TJ/results/data"
    path_to_figures = f"{device_path}/projects/6_TJ/figures"
    axis_name = ('$J_{perp}$', '$t$')

else:
    raise SyntaxError("Choose a valid model among 'ANNNI', 'Cluster', and 'Rydberg'")


filename = f'{path_to_tensor}/{model_name}_L_{l}_lambda_1_{params_extent[2]}-{params_extent[3]}_lambda_2_{params_extent[0]}-{params_extent[1]}_npoints_{n}x{n}_chi_{chi}_eps_{c1}.pkl'


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

gstates = [
    sanitize_state(state)
    for row in gstates
    for state in (row if isinstance(row, (list, np.ndarray)) else [row])
]

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

grad_g = phases_vfield(rdms)

plot_grad_g_angle_stream(grad_g, params_extent=params_extent, axis_name=axis_name, theory_lines=False);
plt.savefig(f"{path_to_figures}/{model_name}_L_{l}_{n}x{n}_{len(sites)}-rdm.png")