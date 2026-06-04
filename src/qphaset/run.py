import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
from tenpy.tools import hdf5_io

def guess_state(L, d):
    """
    guess_state

    This function gives you a product state tensor of all up spins
    for a chain of length L
    
    """
    state = [0]*d
    state[0] = 1
    up = np.array([state]).reshape((1,d,1))
    init_tensor = [up for _ in range(L)]
    return init_tensor

def neel_prod_state(L, d):
    spin_up_tn = np.array([1,0,0]).reshape((1,d,1))
    hole_tn = np.array([0,1,0]).reshape((1,d,1))
    spin_down_tn = np.array([0,0,1]).reshape((1,d,1))
    tn_list = [spin_up_tn if (i%2) == 0 else spin_down_tn for i in range(L)]
    return tn_list

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
                if model.model == "tj":
                    init_tensor = neel_prod_state(model.L, model.d)
                else:
                    init_tensor = guess_state(model.L, model.d)
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
                if model.model == "tj":
                    init_tensor = neel_prod_state(model.L, model.d)
                else:
                    init_tensor = guess_state(model.L, model.d)

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

def save_hdf5(filename, data):

        params_extent = np.concatenate([
            np.min(data["params"], axis=0),
            np.max(data["params"], axis=0)
        ])

        params_extent = tuple(params_extent[[0, 2, 1, 3]])

        tensor_list = data["gstates"]

        matrix_nxn = [tensor_list[i*data["n"]:(i+1)*data["n"]] for i in range(data["n"])]
        corrected_list = [x for row in matrix_nxn[::-1] for x in row]

        data_h5 = {
            "gstates": corrected_list,
            "params_extent": params_extent,
            "n": data["n"],
            "l": data["l"],
            "dmrg_params": data["dmrg_params"],
            "info": {"model_type": "Cluster"},
            "times": np.array(data["stats"]["times"]),
        }

        with h5py.File(f"{filename}.h5", "w") as f:
            hdf5_io.save_to_hdf5(f, data_h5)
