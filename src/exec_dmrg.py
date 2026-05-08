import numpy as np
import time
import pickle
import gzip
from functools import partial

from tqdm import tqdm
from tenpy.tools import hdf5_io
import h5py

from qphaset.annni import model_annni_qs_mps
from qphaset.cluster import model_cluster_qs_mps
from qphaset.rydberg import model_rydberg_qs_mps
from qphaset.models import drmg_gstate_qs_mps

def guess_state(L):
    """
    guess_state

    This function gives you a product state tensor of all up spins
    for a chain of length L
    
    """
    up = np.array([[[1],[0]]])
    init_tensor = [up for _ in range(L)]
    return init_tensor

def run_model(params, model_factory, gstate_solver):
    t_tot = []
    g_states = []
    init_tensor = None
    pbar = tqdm(params, dynamic_ncols=True)

    for (x, y) in pbar:

        # This updates the SAME tqdm line continuously
        # pbar.set_postfix({
        #     "lambda1": f"{x:.6f}",
        #     "lambda2": f"{y:.6f}"
        # })
        pbar.set_description(f"lambda1={x:.4f}, lambda2={y:.4f}")

        model = model_factory(x, y)

        try:
            if y == params[0, -1]:
                init_tensor = guess_state(model.L)
                model.sites = init_tensor.copy()
                model.enlarge_chi(noise_std=1e-6)
            else:
                model.sites = init_tensor.copy()

            timer = time.monotonic()
            gstate = gstate_solver(model=model)
            timer = time.monotonic() - timer

        except Exception as e:
            tqdm.write(f"Random state fallback at x={x}, y={y}")
            tqdm.write(str(e))
            model._random_state(seed=3, type_shape="rectangular")
            model.canonical_form()
            timer = time.monotonic()
            gstate = gstate_solver(model=model)
            timer = time.monotonic() - timer

        t_tot.append(timer)
        g_states.append(model.sites.copy())
        init_tensor = model.sites.copy()

    statistics = dict(times=t_tot)
    return g_states, statistics

# def run_model(params, model_factory, gstate_solver):
#     t_tot = []
#     g_states = []
#     for (x,y) in tqdm(params):
#         print(f"x: {x}, y:{y}")
#         model = model_factory(x, y)
#         try:
#             print("try with guess state...")
#             if y == params[0,-1]:
#                 init_tensor = guess_state(model.L)
#                 model.sites = init_tensor.copy()
#                 model.enlarge_chi()
#             else:
#                 model.sites = init_tensor.copy()
#             timer = time.monotonic()
#             gstate = gstate_solver(model=model)
#             timer = time.monotonic() - timer
#         except:
#             print("try with random state...")
#             model._random_state(seed=3, type_shape="rectangular")
#             model.canonical_form()
#             timer = time.monotonic()
#             gstate = gstate_solver(model=model)
#             timer = time.monotonic() - timer
#         t_tot.append(timer)
#         g_states.append(model.sites.copy())
#         init_tensor = model.sites.copy()
#     statistics = dict(times=t_tot)
#     return g_states, statistics

# *** Data sampling (Hamitonian parameters grid) ***

n = 10  # Sampling grid size

model_name = "Rydberg"
l = 12
n = 30
params = np.linspace(3, 1, n), np.linspace(1, 3, n) # upside-down

model_name = "Cluster"
l = 12
n = 30
params = np.linspace(0.5, 1.5, n), np.linspace(1.5, 0.5, n) # upside-down

model_name = "Cluster"
l = 15
n = 10
params = np.linspace(0.5, 0.6, n), np.linspace(0.6, 0.5, n) # upside-down


params = map(lambda m: m.flatten(), np.meshgrid(*params, indexing='xy'))
params = tuple(params)
params = np.stack(params).T


# *** Config ***
device = 'pc'
# device = 'ngt'

if device == 'pc':
    device_path = "D:/code"
elif device == 'ngt':
    device_path = "eos/f/fdimarca"

# dmrg params
chi = 100 # bond dimension
c1 = 1e-3 # symm- break.

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
    'd': 2,
    'max_hours': 16 / 3600
    #'combine': True
}

# *** Solver ***

print(f"model: {model_name}, L:{l}, parameter space:{n}x{n}")
print(f"bond dimension: {chi}, eps: {c1}, device: {device}")
gstates, stats = run_model(params,
                        model_factory=partial(model_factory, l=l),
                        gstate_solver=partial(drmg_gstate_qs_mps, dmrg_params=dmrg_params))



# ## Save `pickle`

# *** Data export ***

params_extent = np.concatenate([np.min(params, axis=0), np.max(params, axis=0)])
params_extent = tuple(params_extent[[0, 2, 1, 3]])

filename = f'{path_to_tensor}/{model_name}_L_{l}_lambda_1_{params_extent[2]}-{params_extent[3]}_lambda_2_{params_extent[0]}-{params_extent[1]}_npoints_{n}x{n}_chi_{chi}_eps_{c1}.pkl' 
print(f"Saving pickle files at: {filename}")

data = dict(params=params, dmrg_params=dmrg_params,
            l=l, n=n, model_name=model_name,
            gstates=gstates, stats=stats)
with gzip.open(filename, 'wb') as f:
    pickle.dump(data, f)

# ## Save `hdf5`

data = dict(params=params, dmrg_params=dmrg_params,
            l=l, n=n, model_name=model_name,
            gstates=gstates, stats=stats)

# Looking at the extent of the parameters explored (all just double checking all is okay!)
params_extent = np.concatenate([np.min(params, axis=0), np.max(params, axis=0)])
params_extent = tuple(params_extent[[0, 2, 1, 3]])
print(params_extent)
print(np.shape(params))

# Taking the tensors
tensor_list =  data["gstates"]

# Convert the list to a 64x64 nested list
matrix_64x64 = [tensor_list[i*64:(i+1)*64] for i in range(64)]
# Flip the nested list vertically
flipped_matrix = matrix_64x64[::-1]
# Flatten the nested list back into a single list
corrected_list = [item for sublist in flipped_matrix for item in sublist]

# Creating data dictionary to be saved 
data_h5 = {#"gstates": gstates,  # list of MPS - tensors 
        "gstates": corrected_list, # Flipped list
        "params_extent": params_extent, # (2, N) evenly spaced - in future 
        "n": data["n"],
        "l": data["l"], # Number of qubits (L)
        "dmrg_params": data["dmrg_params"], # DMRG params used in calculation - converting? hashtable 
        "info": {"model_type":"Cluster",}, # Additional info 
        "times": np.array(data["stats"]["times"]), # Times for DMRG } # params swept 
}

# Saving your file, make sure to change the name to the name you want to store your file in! 
filename = f'{path_to_tensor}/{model_name}_L_{l}_lambda_1_{params_extent[2]}-{params_extent[3]}_lambda_2_{params_extent[0]}-{params_extent[1]}_npoints_{n}x{n}.h5' 
print(f"Saving hdf5 files at: {filename}")
with h5py.File(filename, 'w') as f:
    hdf5_io.save_to_hdf5(f, data_h5)