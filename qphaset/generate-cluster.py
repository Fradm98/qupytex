# fix relative imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# cwd = os.path.normpath(os.getcwd())
# cwd = cwd.split(os.sep)
# find = cwd.index("fidelity-phase-tran")
# newdir = f"{os.sep}".join(cwd[:find+1])
# os.chdir(newdir)


import numpy as np
import uuid
import time
import pickle
import gzip
from functools import partial

from tenpy.tools import hdf5_io
import h5py

from cluster import model_cluster_qs_mps
from models import drmg_gstate_qs_mps

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
    for (x,y) in params:
        print(f"x: {x}, y:{y}")
        model = model_factory(x, y)
        try:
            print("try with guess state...")
            if y == params[0,-1]:
                init_tensor = guess_state(model.L)
                model.sites = init_tensor.copy()
                model.enlarge_chi()
            else:
                model.sites = init_tensor.copy()
            timer = time.monotonic()
            gstate = gstate_solver(model=model)
            timer = time.monotonic() - timer
        except:
            print("try with random state...")
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

# *** Data sampling (Hamitonian parameters grid) ***

n = 64  # Sampling grid size

params = np.linspace(0.5, 1.5, n), np.linspace(1.5, 0.5, n) # upside-down
params = map(lambda m: m.flatten(), np.meshgrid(*params, indexing='xy'))
params = tuple(params)
params = np.stack(params).T

# *** Config ***

l = 50  # Number of spins

model_name = 'Cluster'
model_factory = partial(model_cluster_qs_mps, c1=1e-3, chi=64)
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

gstates, stats = run_model(params,
                        model_factory=partial(model_factory, l=l),
                        gstate_solver=partial(drmg_gstate_qs_mps, dmrg_params=dmrg_params))

# *** Data export ***

filename = f'/Users/fradm/Desktop/projects/fidelity-phase-tran/results/data/{model_name}-{str(uuid.uuid4())}.pkl' 
data = dict(params=params, dmrg_params=dmrg_params,
            l=l, n=n, model_name=model_name,
            gstates=gstates, stats=stats)
with gzip.open(filename, 'wb') as f:
    pickle.dump(data, f)

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
matrix_64x64 = [tensor_list[i*n:(i+1)*n] for i in range(n)]
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
        "info": {"model_type":"ANNNI",}, # Additional info 
        "times": np.array(data["stats"]["times"]), # Times for DMRG } # params swept 
}

# Saving your file, make sure to change the name to the name you want to store your file in! 
filename_to_save = f"/Users/fradm/Desktop/projects/fidelity-phase-tran/results/data/{model_name}-{str(uuid.uuid4())}.h5"
with h5py.File(filename_to_save, 'w') as f:
    hdf5_io.save_to_hdf5(f, data_h5)

