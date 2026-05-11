import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import tqdm
from .fidelity import fidelity_laplacian
from scipy import signal
import scipy.linalg as la
from .filters import SOBEL, bump_kernel, upsampling_base
from .models import get_rdm, get_rdm_qs_mps, reduced_density_matrix, generalized_k_rdm
from .qfim import rdms_lattice_tr_qfim
from .linalg import projh_psd as _projh_psd
from ncon import ncon
from qs_mps.utils import tensor_shapes

def gstates_to_rdms_matrix(gstates, *, sites=None, shape=None, proj_psd=False):
    """Given a list of ground states (TenPy MPS) ordered corresponding
    to a flattened lattice of ground states (row-major ordering),
    obtain a matrix of RDMs."""
    if shape is None:
        shape = (int(np.sqrt(len(gstates))), ) * 2
    if sites is None:
        # Default to the middle site.
        sites = (gstates[0].L // 2, )
    rdms = [get_rdm(psi, sites=sites) for psi in gstates]
    if proj_psd:
        # Project on PSD cone to correct minor numerical errors
        # that induce small negative eigenvalues.
        rdms = [_projh_psd(rdm) for rdm in rdms]
    rdms = np.array(rdms)
    rdms = rdms.reshape(shape + rdms.shape[1:])
    # [i, j, rdm_i, rdm_j]
    return rdms


def gstates_to_rdms_matrix_qs_mps(gstates, *, sites=None, shape=None, proj_psd=False, generalized=False):
    """Given a list of ground states (qs-mps MPS) ordered corresponding
    to a flattened lattice of ground states (row-major ordering),
    obtain a matrix of RDMs."""
    if shape is None:
        shape = (int(np.sqrt(len(gstates))), ) * 2
    if sites is None:
        # Default to the middle site.
        sites = (gstates[0].L // 2, )
    # for i, psi in enumerate(gstates):
    #     print(f"i: {i}")
    #     rdms = generalized_k_rdm(psi, sites=sites)
    if generalized:
        pbar = tqdm(range(len(gstates)), dynamic_ncols=True)
        rdms = []
        for idx in pbar:

            # This updates the SAME tqdm line continuously
            # pbar.set_postfix({
            #     "lambda1": f"{x:.6f}",
            #     "lambda2": f"{y:.6f}"
            # })
            pbar.set_description(f"rdm comp: {idx+1}")
            rdms.append(generalized_k_rdm(gstates[idx], sites=sites))
        # rdms = [generalized_k_rdm(psi, sites=sites) for psi in gstates]
    else:
        rdms = [reduced_density_matrix(psi, sites=sites) for psi in gstates]
    if proj_psd:
        # Project on PSD cone to correct minor numerical errors
        # that induce small negative eigenvalues.
        rdms = [_projh_psd(rdm) for rdm in rdms]
    rdms = np.array(rdms)
    rdms = rdms.reshape(shape + rdms.shape[1:])
    # [i, j, rdm_i, rdm_j]
    return rdms


def rdms_matrix_laplacian(rdms):
    """Entry-wise laplacian for a 2D lattice of RDMs."""
    rdms = np.array(rdms)
    assert rdms.ndim == 4
    rdms_dxx = -2 * rdms[:,1:-1] + rdms[:,:-2] + rdms[:,2:]
    rdms_dyy = -2 * rdms[1:-1] + rdms[:-2] + rdms[2:]
    return rdms_dxx[1:-1] + rdms_dyy[:,1:-1]


# def log_fidelity(a, b):
#    return np.log(uhlmann_fidelity(a, b))
# g = -fidelity_laplacian(rdms, fidelity=log_fidelity, log=True)

def phases_vfield(rdms_matrix, *, scale=2, grad=True, fidelity=None,
                  log_g=False, method='fidelity'):
    assert scale in {1, 2}
    # TODO Exclude boundaries, re-eval domain. Accept params_extend param
    # and return adjusted version of it.

    g = None
    if method == 'fidelity':
        g = fidelity_laplacian(rdms_matrix, fidelity=fidelity)
        g = np.log(np.maximum(-g, 1e-6)) if log_g else g
    elif method == 'tr_qfim':
        g = -rdms_lattice_tr_qfim(rdms_matrix)
    else:
        raise ValueError(f'Unknown method: {method}')

    if grad:
        kernel = None
        if scale > 1:
            assert scale == 2
            g = upsampling_base(g)
            # TODO Substitute bump with a possibly separable low-pass filter.
            kernel = bump_kernel(6, scale=scale)
            kernel = signal.convolve2d(kernel, SOBEL, boundary='symm', mode='same')
        else:
            kernel = SOBEL
        return signal.convolve2d(g, kernel, boundary='symm', mode='same')

    if scale > 1:
        assert scale == 2
        g = upsampling_base(g)
        kernel = bump_kernel(6, scale=scale)
        return signal.convolve2d(g, kernel, boundary='symm', mode='same')
    return g

# def phases_vfield(rdms_matrix, *, scale=2, grad=True, fidelity=None,
#                   log_g=False, method='fidelity'):
#     assert scale in {1, 2}
#     # TODO Exclude boundaries, re-eval domain. Accept params_extend param
#     # and return adjusted version of it.

#     g = None
#     if method == 'fidelity':
#         g = fidelity_laplacian(rdms_matrix, fidelity=fidelity)
#         g = np.log(np.maximum(-g, 1e-6)) if log_g else g
#     elif method == 'tr_qfim':
#         g = -rdms_lattice_tr_qfim(rdms_matrix)
#     else:
#         raise ValueError(f'Unknown method: {method}')

#     return g