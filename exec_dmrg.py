# fix relative imports
import os
cwd = os.path.normpath(os.getcwd())
cwd = cwd.split(os.sep)
find = cwd.index("fidelity-phase-tran")
newdir = f"{os.sep}".join(cwd[:find+1])
os.chdir(newdir)

import numpy as np
import uuid
import time
import pickle
import gzip
from functools import partial

from tqdm.notebook import tqdm
from tenpy.tools import hdf5_io
import h5py

import matplotlib.pyplot as plt

from qphaset.cluster import model_cluster_qs_mps
from qphaset.annni import model_annni_qs_mps
from qphaset.rydberg import model_rydberg_qs_mps
from qphaset.models import drmg_gstate_qs_mps

# ## Template notebook for running the DMRG for the Cluster model with `qs-mps`

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
    for (x,y) in tqdm(params):
        print(f"lambda 1: {x}, lambda 2:{y}")
        model = model_factory(y, x)
        print(f"L: {model.L}")
        try:
            print("try with guess state...")
            if x == params[0,0]:
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

n = 25    # Sampling grid size

# params = np.linspace(0.01, 1.5, n), np.linspace(0.01, 1.5, n)
params = np.linspace(0.8, 0.5, n), np.linspace(0.5, 0.8, n) # upside-down
# params = np.linspace(0.01, 1., n), np.linspace(1., 0.01, n)   # Good one
# params = np.linspace(0.5, 1.5, n), np.linspace(1., 0.2, n)      # Floating phase detail
# params = np.linspace(0.5, 1.5, n), np.linspace(0.8, 0.01, n)      # Floating phase detail (lowered)
# params = np.linspace(0.01, 1., n), np.linspace(0.8, 0.01, n)      # Floating phase detail (lowered) + multi-critical point
# params = np.linspace(0.6, 1.5, n), np.linspace(0.1, 1.5, n)      # Floating phase detail ++

# Rydberg params
params = np.linspace(3, 1, n), np.linspace(1, 3, n) # upside-down

params = map(lambda m: m.flatten(), np.meshgrid(*params, indexing='xy'))
params = tuple(params)
params = np.stack(params).T

# *** Config ***

ls = [12]  # Number of spins

# choose a model between 'ANNNI', 'Cluster', and 'Rydberg'
model_name = 'Rydberg'

if model_name == 'ANNNI':
    model_factory = partial(model_annni_qs_mps, c1=0, chi=64)
elif model_name == 'Cluster':
    model_factory = partial(model_cluster_qs_mps, c1=0, chi=64)
elif model_name == 'Rydberg':
    model_factory = partial(model_rydberg_qs_mps, c1=0, chi=64)
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

# marcos
path_to_tensor = "/Users/fradm/Desktop/projects/fidelity-phase-tran/results/data"
# pc
path_to_tensor = "D:/Users/HP/Desktop/cluster_results"

if model_name == 'ANNNI':
    path_to_tensor = "D:/code/projects/2_ANNNI/results/data"
elif model_name == 'Cluster':
    path_to_tensor = "D:/code/projects/3_CLUSTER/results/data"
elif model_name == 'Rydberg':
    path_to_tensor = "D:/code/projects/4_RYDBERG/results/data"
else:
    raise SyntaxError("Choose a valid model among 'ANNNI', 'Cluster', and 'Rydberg'")

for l in ls:
    gstates, stats = run_model(params,
                            model_factory=partial(model_factory, l=l),
                            gstate_solver=partial(drmg_gstate_qs_mps, dmrg_params=dmrg_params))

    print(f"All the DMRG for the {model_name} model has been executed")

    # *** Execution time per pixel ***

    plt.matshow(np.array(stats['times']).reshape((n, n)), cmap='hot', origin='lower')
    plt.colorbar()
    plt.close()

    # ## Save `pickle`

    print("Saving pickle files...")
    # *** Data export ***

    params_extent = np.concatenate([np.min(params, axis=0), np.max(params, axis=0)])
    params_extent = tuple(params_extent[[0, 2, 1, 3]])

    filename = f'{path_to_tensor}/{model_name}_L_{l}_lambda_1_{params_extent[2]}-{params_extent[3]}_lambda_2_{params_extent[0]}-{params_extent[1]}_npoints_{n}x{n}.pkl' 
    data = dict(params=params, dmrg_params=dmrg_params,
                l=l, n=n, model_name=model_name,
                gstates=gstates, stats=stats)
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data, f)


    # ## Save `hdf5`
    print("Saving hdf5 files...")

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

    # Convert the list to a nxn nested list
    matrix_nxn = [tensor_list[i*n:(i+1)*n] for i in range(n)]
    # Flip the nested list vertically
    flipped_matrix = matrix_nxn[::-1]
    # Flatten the nested list back into a single list
    corrected_list = [item for sublist in flipped_matrix for item in sublist]

    # Creating data dictionary to be saved 
    data_h5 = {#"gstates": gstates,  # list of MPS - tensors 
            "gstates": corrected_list, # Flipped list
            "params_extent": params_extent, # (2, N) evenly spaced - in future 
            "n": data["n"],
            "l": data["l"], # Number of qubits (L)
            "dmrg_params": data["dmrg_params"], # DMRG params used in calculation - converting? hashtable 
            "info": {"model_type":f"{model_name}",}, # Additional info 
            "times": np.array(data["stats"]["times"]), # Times for DMRG } # params swept 
    }


    # Saving your file, make sure to change the name to the name you want to store your file in! 
    filename_to_save = f'{path_to_tensor}/{model_name}_L_{l}_lambda_1_{params_extent[2]}-{params_extent[3]}_lambda_2_{params_extent[0]}-{params_extent[1]}_npoints_{n}x{n}.h5' 
    with h5py.File(filename_to_save, 'w') as f:
        hdf5_io.save_to_hdf5(f, data_h5)



