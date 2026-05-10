# TenPy related helper functions.
# fix relative imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tenpy.models.tf_ising import TFIChain
from tenpy.models.spins_nnn import SpinChainNNN
from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS
from tenpy.networks.site import Site
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.algorithms import dmrg
from qs_mps.mps_class import MPS as mps
from .fidelity import compute_norm
import numpy as np
import scipy.linalg as la
from ncon import ncon



def params_2d_lattice(ext_x, ext_y, *, n):
    params = np.linspace(*ext_x, n), np.linspace(*ext_y, n)

    params = map(lambda m: m.flatten(), np.meshgrid(*params, indexing='xy'))
    params = tuple(params)
    return np.stack(params).T


def model_ising(g, j, *, l, params={}) -> TFIChain:
    """Prepare the ISING model with parameters g and j.
    The length of the chain is given in l."""
    model_params = dict(L=l,                # Length
                        bc_MPS='finite',    # Boundary condition
                        J=j, g=g,
                        conserve=None)
    model_params.update(params)
    return TFIChain(model_params)


def model_annni(k, h, *, l, j1=1, params={}) -> SpinChainNNN:
    """Prepare the ANNNI model with parameters k and h.
    The half-length of the chain is given in l."""
    # -k * j1 = j2
    # h * j1 = b
    model_params = dict(L=l,    # Length
                        Jy=0, Jz=0, Jyp=0, Jzp=0,
                        hx=0, hy=0,
                        bc_MPS='finite',  # Boundary condition
                        Jx=-j1,
                        Jxp=k * j1,     # J_2
                        hz=-(h * j1),    # B
                        conserve=None)
    model_params.update(params)
    return SpinChainNNN(model_params)


def model_xxz(d, h, *, l, params={}) -> XXZChain:
    model_params = dict(L=l,              # Length
                        bc_MPS='finite',  # Boundary condition
                        sort_charge=False,
                        Jxx = -2,
                        Jz = -d / 2,
                        hz = h)
    model_params.update(params)
    return XXZChain(model_params)


def get_rdm(psi: MPS, *, sites) -> np.ndarray:
    """Obtain the reduced density matrix from the input MPS
    on the selected sites."""
    # get_rho_segment assumes the sites are sorted.
    sites = np.sort(sites)
    rdm = psi.get_rho_segment(sites).to_ndarray()
    sz = int(np.sqrt(np.product(rdm.shape)))
    return np.reshape(rdm, (sz, sz))


def get_rdm_qs_mps(psi: mps, *, sites) -> np.ndarray:
    """Obtain the reduced density matrix from the input MPS
    on the selected sites."""
    # reduced_densirt_matrix() assumes the sites are sorted.
    sites = np.sort(sites)
    # state = mps(L=len(psi), d=2, model="ANNNI")
    # state.sites = psi
    rdm = psi.reduced_density_matrix(sites)
    sz = int(np.sqrt(np.product(rdm.shape)))
    return np.reshape(rdm, (sz, sz))


def reduced_density_matrix(mps: list, sites: list):
    """
    reduced_density_matrix

    This function allows us to get the reduced density matrix (rdm) of a mps.
    We trace out all the sites not specified in the argument sites.
    The algorithm only works for consecutive sites, e.g., [1,2,3],
    [56,57], etc. To implement a rdm of sites [5,37] we need another middle
    environment that manages the contractions between the specified sites.

    mps: list - list of np.ndarrays, the tensors of our mps
    sites: list - list of ints, the sites of the complementary subsystem we are tracing out

    """
    kets = mps
    L = len(mps)
    d = kets[0].shape[1]
    k = len(sites)
    bras = [ket.conjugate() for ket in kets]
    a = np.array([1])
    env = ncon([a, a], [[-1], [-2]])
    up = [int(-elem) for elem in np.linspace(1, 0, 0)]
    down = [int(-elem) for elem in np.linspace(L + 1, 0, 0)]
    mid_up = [1]
    mid_down = [2]
    label_env = up + down + mid_up + mid_down
    # left env:
    env_l = env
    for i in range(sites[0] - 1):
        label_ket = [1, 3, -1]
        label_bra = [2, 3, -2]
        env_l = ncon([env_l, kets[i], bras[i]], [label_env, label_ket, label_bra])
    # right env:
    env_r = env
    for i in range(L - 1, sites[-1]-1, -1):
        label_ket = [-1, 3, 1]
        label_bra = [-2, 3, 2]
        env_r = ncon([env_r, kets[i], bras[i]], [label_env, label_ket, label_bra])
    # central env
    # idx = 0
    for i in range(k):
        label_ket = [1, -1 - i, -k * 100]
        label_bra = [2, -k - 1 - i, -k * 100 - 1]
        env_l = ncon([env_l, kets[sites[i]-1], bras[sites[i]-1]], [label_env, label_ket, label_bra])
        up = [int(-elem) for elem in np.linspace(1, i + 1, i + 1)]
        down = [
            int(-elem)
            for elem in np.linspace(k + 1, k + 1 + i, i + 1)
        ]
        label_env = up + down + mid_up + mid_down
    
    rdm = ncon([env_l, env_r], [label_env, [1, 2]]).reshape((d**k,d**k))

    return rdm


def left_canonical_form(
        mps: list, stop: int, schmidt_tol: float=1e-15,
    ):
        """
        right_svd

        This function transforms the states in mps in a canonical
        form using svd. We start from the first site and sweeping through
        site up to stop we save the Gamma tensors on each site and the Schmidt values on the bonds

        mps: list - tensors list for our chain of spins
        stop: int - site where we want to stop the left canonical form (by default the states
            after dmrg are in right canonical form so we need to put them in lcf up to a specific
            site fi we want to obtain a mixed canonical form for the computation of rdms)
        schmidt_tol: float - tolerance used to cut the schmidt values after svd

        """
        new_mps = mps.copy()
        bonds = []
        s_init = np.array([1])
        psi = np.diag(s_init)
        d = mps[0].shape[1]
        bonds.append(s_init)
        chis = [array.shape[2] for array in new_mps]
        chis.pop()

        for i in range(stop):
            new_site = ncon(
                [psi, new_mps[i]],
                [
                    [-1, 1],
                    [1, -2, -3],
                ],
            )
            new_site = new_site.reshape(new_site.shape[0] * d, new_site.shape[2])

            original_matrix = new_site
            scaled_matrix = original_matrix / np.max(np.abs(original_matrix))
            lambda_ = 1e-15
            regularized_matrix = scaled_matrix + lambda_ * np.eye(
                scaled_matrix.shape[0], scaled_matrix.shape[1]
            )
            u, s, v = la.svd(
                regularized_matrix,
                full_matrices=False,
            )

            bond_l = u.shape[0] // d
            u = u.reshape(bond_l, d, u.shape[1])

            u = u[:, :, : chis[i]]
            s = s[: chis[i]]
            v = v[: chis[i], :]
            s = s / la.norm(s)

            # condition = s >= schmidt_tol
            # s_trunc = np.extract(condition, s)
            # s = s_trunc / la.norm(s_trunc)
            # u = u[:, :, : len(s)]
            # v = v[: len(s), :]

            new_mps[i] = u
            bonds.append(s)
            psi = ncon(
                [np.diag(s), v],
                [
                    [-1, 1],
                    [1, -2],
                ],
            )

        return new_mps


def generalized_k_rdm(mps, sites, prnt: bool=False) -> np.array:

    # # print("right canonical form: ")
    # # tensor_shapes(mps)
    # psi_mcf = left_canonical_form(mps=mps, stop=sites[0])
    # # print("mixed canonical form: ")
    # # tensor_shapes(psi_mcf)
    # norm = compute_norm(psi_mcf, len(psi_mcf), prnt=False)
    # psi_mcf[sites[0]] = psi_mcf[sites[0]]/(norm**(1/2))

    # kets = psi_mcf.copy(
    kets = mps.copy()
    bras = [ket.conjugate() for ket in kets]
    sites = np.sort(sites)

    d = kets[0].shape[1]
    k_tot = len(sites)
    mid_up = [1]
    mid_down = [2]
    up = [int(-elem) for elem in np.linspace(1, 0, 0)]
    down = [int(-elem) for elem in np.linspace(len(mps) + 1, 0, 0)]
    mid_up = [1]
    mid_down = [2]
    label_env = up + down + mid_up + mid_down
    
    a = np.array([1])
    env = ncon([a, a], [[-1], [-2]])
    
    # # left env:
    # env_l = np.identity(n=mps[sites[0]-1].shape[0])
    # label_env = mid_up + mid_down
    # if prnt:
    #     print(label_env, env_l.shape)
    # # right env:
    # env_r = np.identity(n=mps[sites[-1]-1].shape[-1])
    
    # left env:
    env_l = env
    for i in range(sites[0] - 1):
        label_ket = [1, 3, -1]
        label_bra = [2, 3, -2]
        env_l = ncon([env_l, kets[i], bras[i]], [label_env, label_ket, label_bra])
    # right env:
    env_r = env
    for i in range(len(mps) - 1, sites[-1]-1, -1):
        label_ket = [-1, 3, 1]
        label_bra = [-2, 3, 2]
        env_r = ncon([env_r, kets[i], bras[i]], [label_env, label_ket, label_bra])

    # central env
    i = 0
    for j in range(sites[0],sites[-1]+1):
        if j in sites:
            if prnt:
                print("j in sites: ", j)
            label_ket = [1, -1 - i, -2*k_tot -1]
            label_bra = [2, -k_tot - 1 - i, -2*k_tot - 2]
            if prnt:
                print(label_ket, label_bra)
            env_l = ncon(
                [env_l, kets[j - 1], bras[j - 1]],
                [label_env, label_ket, label_bra],
            )
            up = [int(-elem) for elem in np.linspace(1, i + 1, i + 1)]
            down = [int(-elem) for elem in np.linspace(k_tot + 1, k_tot + 1 + i, i + 1)]
            label_env = up + down + mid_up + mid_down
            if prnt:
                print(label_env, env_l.shape)

            i += 1
        else:
            if prnt:
                print("j out sites: ", j)
            label_ket = [1, 3, -2*k_tot-1]
            label_bra = [2, 3, -2*k_tot-2]
            if prnt:
                print(label_ket, label_bra)
            env_l = ncon([env_l, kets[j-1], bras[j-1]], [label_env, label_ket, label_bra])
        
    rdm = ncon([env_l, env_r], [label_env, [1, 2]]).reshape((d**k_tot, d**k_tot))

    return rdm


def get_bond_dim_qs_mps(psi: list):
        bond_dims = []
        for i in range(len(psi)-1):
            bond_dims.append(psi[i].shape[-1])

        return bond_dims


def exact_gstate_mps(model, *, guess_psi=None, sparse=False, max_size=16e6) -> MPS:
    """Compute the ground state (MPS) of the given model using
    exact diagonalization."""
    # Param guess_psi, just for compatibility.
    solver = ExactDiag(model, sparse=sparse, max_size=max_size)
    solver.build_full_H_from_mpo()
    solver.full_diagonalization()
    eigval, psi = solver.groundstate()
    # TODO Return stats (containing eigval).
    return solver.full_to_mps(psi)


def _get_labels(site: Site):
    l = {v: k for k, v in site.state_labels.items()}
    return list(l.values())


def drmg_gstate_mps(model, *, drmg_params={}, reps=3, guess_psi=None) -> MPS:
    """Compute the ground state (MPS) of the given model using
    the DMRG algorithm."""
    l = model.lat.N_sites
    # Assume the sites are all of the same nature.
    state_labels = _get_labels(model.lat.mps_sites()[0])

    for _ in range(reps):
        if guess_psi is None:
            # Randomized guess (MPS) state based on site labels.
            product_state = np.random.choice(state_labels, l)
            psi = MPS.from_product_state(model.lat.mps_sites(), product_state,
                                         bc=model.lat.bc_MPS)
        else:
            assert isinstance(guess_psi, MPS)
            psi = guess_psi.copy()
            # Use the guess once only.
            guess_psi = None
        stats = dmrg.run(psi, model, drmg_params)
        if not stats['shelve']:
            break
    # TODO Ideally we should return the stats
    # and cosider them in the serialization of the results.
    return psi

def drmg_gstate_qs_mps(model:mps, dmrg_params={}) -> mps:
    """Compute the ground state (MPS) of the given model using
    the DMRG algorithm."""
    
    # Assume the sites are all of the same nature.
    trunc_tol = dmrg_params.get('trunc_params').get('trunc_tol')
    trunc_chi = dmrg_params.get('trunc_params').get('trunc_chi')
    energy, entropy, schmidt_vals, t_dmrg = model.DMRG(trunc_chi=trunc_chi, trunc_tol=trunc_tol, where=model.L//2)
        
    return model.sites
