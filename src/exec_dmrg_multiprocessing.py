import numpy as np
import time
import pickle
import gzip
from functools import partial
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def run_single_y(
    y,
    x_values,
    model_factory,
    gstate_solver
):

    init_tensor = None

    local_states = []
    local_times = []

    for i, x in enumerate(x_values):

        model = model_factory(x, y)

        try:

            if i == 0:
                init_tensor = guess_state(model.L)

                model.sites = init_tensor.copy()

                model.enlarge_chi(noise_std=1e-6)

            else:

                model.sites = init_tensor.copy()

            t0 = time.monotonic()

            gstate_solver(model=model)

            elapsed = time.monotonic() - t0

        except Exception as e:

            tqdm.write(f"Fallback at x={x}, y={y}")
            tqdm.write(str(e))

            model._random_state(seed=3, type_shape="rectangular")
            model.canonical_form()

            t0 = time.monotonic()

            gstate_solver(model=model)

            elapsed = time.monotonic() - t0

        init_tensor = model.sites.copy()

        local_states.append(init_tensor)
        local_times.append(elapsed)

    return y, local_states, local_times

def run_model_parallel(
    params,
    model_factory,
    gstate_solver,
    max_workers=None
):

    # -----------------------------------
    # Extract ordered unique axes
    # -----------------------------------

    x_unique = np.unique(params[:, 0])
    y_unique = np.unique(params[:, 1])

    nx = len(x_unique)
    ny = len(y_unique)

    # -----------------------------------
    # Group x by y
    # -----------------------------------

    grouped = defaultdict(list)

    for x, y in params:
        grouped[y].append(x)

    for y in grouped:
        grouped[y] = sorted(grouped[y])

    # -----------------------------------
    # Allocate result matrices
    # -----------------------------------

    gstates_matrix = np.empty((nx, ny), dtype=object)
    times_matrix = np.zeros((nx, ny))

    # maps physical values -> matrix index
    x_to_ix = {x: i for i, x in enumerate(x_unique)}
    y_to_iy = {y: i for i, y in enumerate(y_unique)}

    # -----------------------------------
    # Parallel execution over y
    # -----------------------------------

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        futures = []

        for y in y_unique:

            futures.append(
                executor.submit(
                    run_single_y,
                    y,
                    grouped[y],
                    model_factory,
                    gstate_solver
                )
            )

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="y batches"
        ):

            y, local_states, local_times = future.result()

            iy = y_to_iy[y]

            for x, state, timing in zip(
                grouped[y],
                local_states,
                local_times
            ):

                ix = x_to_ix[x]

                gstates_matrix[iy, ix] = state
                times_matrix[iy, ix] = timing

    statistics = dict(times=times_matrix)

    return gstates_matrix, statistics

# *** Data sampling (Hamitonian parameters grid) ***

model_name = "Cluster"
l = 12
n = 30
params = np.linspace(0.5, 1.5, n), np.linspace(1.5, 0.5, n) # upside-down

model_name = "Cluster"
l = 15
n = 10
params = np.linspace(0.5, 0.6, n), np.linspace(0.6, 0.5, n) # upside-down

model_name = "Rydberg"
l = 12
n = 10
params = np.linspace(1, 3, n), np.linspace(3, 1, n) # upside-down

model_name = "ANNNI"
l = 12
n = 20
params = np.linspace(0.01, 1.5, n), np.linspace(1.5, 0.01, n) # upside-down

params = map(lambda m: m.flatten(), np.meshgrid(*params, indexing='xy'))
params = tuple(params)
params = np.stack(params).T


# *** Config ***
device = 'pc'
# device = 'ngt'

if device == 'pc':
    device_path = "D:/code"
elif device == 'ngt':
    device_path = "/eos/user/f/fdimarca"

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

def main():
    print(f"model: {model_name}, L:{l}, parameter space:{n}x{n}")
    print(f"bond dimension: {chi}, eps: {c1}, device: {device}")

    gstates, stats = run_model_parallel(
        params,
        model_factory=partial(model_factory, l=l),
        gstate_solver=partial(
            drmg_gstate_qs_mps,
            dmrg_params=dmrg_params
        ),
        max_workers=4
    )

    # ---------------- SAVE ----------------
    params_extent = np.concatenate([np.min(params, axis=0), np.max(params, axis=0)])
    params_extent = tuple(params_extent[[0, 2, 1, 3]])

    filename = f'{path_to_tensor}/{model_name}_L_{l}_lambda_1_{params_extent[2]}-{params_extent[3]}_lambda_2_{params_extent[0]}-{params_extent[1]}_npoints_{n}x{n}_chi_{chi}_eps_{c1}.pkl'

    print(f"Saving pickle files at: {filename}")

    data = dict(
        params=params,
        dmrg_params=dmrg_params,
        l=l,
        n=n,
        model_name=model_name,
        gstates=gstates,
        stats=stats
    )

    with gzip.open(filename, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()