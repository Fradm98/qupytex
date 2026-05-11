import numpy as np
import gzip
import pickle

from qs_mps.mps_class import MPS
from qs_mps.applications.ISING.utils import *

import matplotlib.pyplot as plt
# default parameters of the plot layout
# plt.rcParams["text.usetex"] = True  # use latex
# plt.rcParams["font.size"] = 13
# plt.rcParams["figure.dpi"] = 300
# plt.rcParams["figure.constrained_layout.use"] = True

model_name = "Cluster"
l = 15
n = 10
h_i, h_f = 0.5, 0.6

model_name = "Cluster"
l = 50
n = 64
h_i, h_f = 0.5, 1.5

model_name = "Rydberg"
l = 20
n = 30
h_i, h_f = 1, 3
params = np.linspace(1, 3, n), np.linspace(1.8, 3, n) # upside-down

params = np.linspace(h_i, h_f, n), np.linspace(h_f, h_i, n) # upside-down
couplings = np.linspace(h_i,h_f,n)
a = abs(couplings[0] - couplings[1])

params = map(lambda m: m.flatten(), np.meshgrid(*params, indexing='xy'))
params = tuple(params)
params = np.stack(params).T
params_extent = np.concatenate([np.min(params, axis=0), np.max(params, axis=0)])
params_extent = tuple(params_extent[[0, 2, 1, 3]])

device = 'pc'
device = 'ngt'

if device == 'pc':
    device_path = "D:/code"
elif device == 'ngt':
    device_path = "/eos/user/f/fdimarca"

# dmrg params
d = 2
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
filename = f'{path_to_tensor}/{model_name}_L_{l}_lambda_1_{params_extent[2]}-{params_extent[3]}_lambda_2_{params_extent[0]}-{params_extent[1]}_npoints_{n}x{n}.pkl'


with gzip.open(filename, 'rb') as f:
    data = pickle.load(f)
params = data['params']
l, n = data['l'], data['n']
gstates = data['gstates']
stats = data['stats']

fidelity = []
for i in range(n):
    fidelity_row = []
    for j in range(n-1):
        print(f"L: {l}, couplings: {params[i+j]}")
        heis_chain_g = MPS(L=l, d=d, model=model_name, chi=chi, h=None, eps=c1, J=None, bc='obc')
        
        heis_chain_g.sites = gstates[i+j+i*(n-1)]
        heis_chain_g.ancilla_sites = gstates[i+j+i*(n-1)+1]

        fidelity_row.append(heis_chain_g._compute_norm(site=1, mixed=True).copy())
    fidelity.append(fidelity_row)

dfss = []
for fid_row in fidelity:
    dfs = discrete_fidelity_susceptibility(fid=fid_row, a=a)
    dfss.append(dfs)
plt.imshow(np.asarray(dfss))
plt.colorbar()
plt.savefig(f"{path_to_figures}/{model_name}_L_{l}_{n}x{n}_total_state_no_convol.png")