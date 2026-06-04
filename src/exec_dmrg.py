import numpy as np
import pickle
import gzip
from functools import partial
import os

from qphaset.annni import model_annni_qs_mps
from qphaset.cluster import model_cluster_qs_mps
from qphaset.rydberg import model_rydberg_qs_mps
from qphaset.tjv_model import model_tjv_qs_mps
from qphaset.models import drmg_gstate_qs_mps
from qphaset.run import *


model_name = "Cluster"
l = 12
d = 2 # physical local dimension
n = 30
params = np.linspace(0.5, 1.5, n), np.linspace(1.5, 0.5, n) # upside-down

model_name = "Cluster"
l = 15
d = 2 # physical local dimension
n = 10
params = np.linspace(0.5, 0.6, n), np.linspace(0.6, 0.5, n) # upside-down

model_name = "Rydberg"
l = 12
d = 2 # physical local dimension
n = 30
params = np.linspace(1, 3, n), np.linspace(3, 1, n) # upside-down

model_name = "tjv"
l = 12
d = 3 # physical local dimension
n = 30
Jz = 10
params = np.linspace(1, 3, n), np.linspace(3, 1, n) # upside-down


params = map(lambda m: m.flatten(), np.meshgrid(*params, indexing='xy'))
params = tuple(params)
params = np.stack(params).T


# *** Config ***
device = 'pc'
# device = 'ngt'

if device == 'pc':
    device_path = "D:/work"
elif device == 'ngt':
    device_path = "/eos/user/f/fdimarca"

# dmrg params
chi = 10 # bond dimension
c1 = 1e-3 # symm- break.



estimate_storage = 16 * (n**2) * l * d * (chi**2)  # bytes (complex128)
estimate_gb = estimate_storage / 1e9

limit_pkl_storage = 50  # GB

if model_name == 'ANNNI':
    path_to_tensor = f"{device_path}/projects/2_ANNNI/results/data"
    path_to_figures = f"{device_path}/projects/2_ANNNI/figures"
    model_factory = partial(model_annni_qs_mps, c1=c1, chi=chi)

elif model_name == 'Cluster':
    model_factory = partial(model_cluster_qs_mps, c1=c1, chi=chi)
    path_to_tensor = f"{device_path}/projects/3_CLUSTER/results/data"
    path_to_figures = f"{device_path}/projects/3_CLUSTER/figures"

elif model_name == 'Rydberg':
    model_factory = partial(model_rydberg_qs_mps, c1=c1, chi=chi)
    path_to_tensor = f"{device_path}/projects/4_RYDBERG/results/data"
    path_to_figures = f"{device_path}/projects/4_RYDBERG/figures"

elif model_name == 'tjv':
    model_factory = partial(model_tjv_qs_mps, Jz=Jz, c1=c1, chi=chi)
    path_to_tensor = f"{device_path}/projects/4_RYDBERG/results/data"
    path_to_figures = f"{device_path}/projects/4_RYDBERG/figures"

else:
    raise SyntaxError("Choose a valid model among 'ANNNI', 'Cluster', and 'Rydberg'")

dmrg_params = {
    #'mixer': True,
    #'max_E_err': 1.e-16,
    #'chi_list': {0: 25, 10: 50, 20: 100},
    'type_shape': "rectangular",
    'trunc_params': {
       'trunc_tol': False,
       'trunc_chi': True
    #   'svd_min': 1.e-16
    },
    'd': d,
    'max_hours': 16 / 3600,
    #'combine': True
}

# *** Solver ***

print(f"model: {model_name}, L:{l}, parameter space:{n}x{n}")
print(f"bond dimension: {chi}, eps: {c1}, device: {device}")
gstates, stats = run_model(params,
                        model_factory=partial(model_factory, l=l),
                        gstate_solver=partial(drmg_gstate_qs_mps, dmrg_params=dmrg_params))



# ---------------- SAVE ----------------
params_extent = np.concatenate([np.min(params, axis=0), np.max(params, axis=0)])
params_extent = tuple(params_extent[[0, 2, 1, 3]])

filename = f'{path_to_tensor}/{model_name}_L_{l}_lambda_1_{params_extent[2]}-{params_extent[3]}_lambda_2_{params_extent[0]}-{params_extent[1]}_npoints_{n}x{n}_chi_{chi}_eps_{c1}.pkl'

data = dict(
    params=params,
    dmrg_params=dmrg_params,
    l=l,
    n=n,
    model_name=model_name,
    gstates=gstates,
    stats=stats
)

if estimate_gb < limit_pkl_storage:

    try:
        print(f"Saving pickle: {filename}")

        with gzip.open(filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        print(f"Pickle failed: {e}")
        print("Falling back to HDF5")
        if os.path.exists(filename):
            os.remove(filename)
        save_hdf5(filename, data)

else:
    print(f"Too large for pickle ({estimate_gb:.1f} GB), using HDF5")
    save_hdf5(filename, data)