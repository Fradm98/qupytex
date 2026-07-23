import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import tqdm
from .fidelity import fidelity_laplacian, fidelity_dxx
from scipy import signal
from .filters import SOBEL, bump_kernel, upsampling_base
from .models import get_rdm, reduced_density_matrix, generalized_k_rdm
from .qfim import rdms_lattice_tr_qfim
from .linalg import projh_psd as _projh_psd
from qphaset.linalg import schmidt_decomp_half
from qiskit.quantum_info import SparsePauliOp

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
    
    n_rows, n_cols = rdms_matrix.shape[:2]
    is_1d_rows = n_rows == 1   # trivial along rows, scan along cols
    is_1d_cols = n_cols == 1   # trivial along cols, scan along rows

    # ── 1D case: one axis is trivial ─────────────────────────────────────────
    if is_1d_rows or is_1d_cols:
        if method == 'tr_qfim':
            raise NotImplementedError("tr_qfim not supported for 1D grids")

        if is_1d_rows:
            # scan axis is columns → fidelity_dxx directly, no swap
            g = fidelity_dxx(rdms_matrix, fidelity=fidelity,
                             log=(log_g if log_g else False))
            # g has shape (1, n_cols-2); squeeze to 1D
            g = g[0]
        else:
            # scan axis is rows → swap so fidelity_dxx runs along cols,
            # then swap back
            g = fidelity_dxx(np.swapaxes(rdms_matrix, 0, 1),
                             fidelity=fidelity,
                             log=(log_g if log_g else False))
            g = g[0]   # shape (n_rows-2,) after squeeze

        if log_g:
            g = np.log(np.maximum(-g, 1e-6))

        if not grad:
            return g

        # 1D gradient: simple first difference (central where possible)
        g_grad = np.gradient(g)
        return g_grad

    # ── 2D case: original logic ───────────────────────────────────────────────
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

def make_obs_vec(obs_ev, obs_eval, obs_ev_idx, v0_first_schmidt_vec=False):
    v0 = obs_ev[:, obs_ev_idx]

    # Schmidt coefficients w.r.t. middle split
    if v0_first_schmidt_vec:
        v0 = schmidt_decomp_half(v0, contract_sigmas=1, normalize=True)

    v0 = np.reshape(v0, (-1, 1))
    obs0 = np.sign(obs_eval[obs_ev_idx]) * v0 @ np.conj(v0.T)
    return obs0

def get_obs_ev(obs):
    obs_eval, obs_ev = np.linalg.eigh(obs)
    return obs_eval, obs_ev

def decompose_obs(obs, k_sites=2):
    operators = SparsePauliOp.from_operator(obs)
    sorted_indices = np.argsort(operators.coeffs)[::-1]
    components = operators.paulis
    sorted_components = [components[i] for i in sorted_indices]
    return sorted_components[:2**(k_sites)]

def sanitize_state(state):
    # ensure flat list of arrays
    return [np.array(t) for t in state]

def extract_submatrix(matrix, x_vals, y_vals,
                        x0, y0,
                        dx=2, dy=2, to_end=True):
    """
    Extract submatrix centered around parameter point (x0, y0).

    Parameters
    ----------
    matrix : 2D array
    x_vals : 1D x-axis values
    y_vals : 1D y-axis values
    x0, y0 : target parameter values
    dx, dy : half-width in index space

    Returns
    -------
    submatrix, x_indices, y_indices
    """

    # Closest grid indices
    ix = np.argmin(np.abs(x_vals - x0))
    iy = np.argmin(np.abs(y_vals - y0))

    # Bounds
    if to_end:
        x_start = ix - dx
        x_end   = len(x_vals) - 1

        y_start = iy - dy
        y_end   = len(y_vals) - 1
    else:
        x_start = 0
        x_end = ix + dx

        y_start = 0
        y_end = iy + dy

    if x_end - x_start != y_end - y_start:
        min_len = min(x_end - x_start, y_end - y_start)
        if x_end - x_start > min_len:
            if to_end:
                x_end = x_start + min_len
            else:
                x_start = x_end - min_len
        elif y_end - y_start > min_len:
            if to_end:
                y_end = y_start + min_len
            else:
                y_start = y_end - min_len
            
    submatrix = matrix[y_start:y_end, x_start:x_end].copy()

    return submatrix, tuple([x_start, x_end, y_start, y_end])

def constructing_order_parameter(rdms, *, idxi=0, idxf=-1, theta=0.0, fidelity=None, log_g=False):
    """
    1D analogue of the phases_vfield pipeline for rdms with one trivial axis.

    Parameters
    ----------
    rdms    : np.ndarray of shape (1, n, rdm_sz, rdm_sz) or (n, 1, rdm_sz, rdm_sz)
    theta   : float – rotation angle for the phase partition (default 0)
    fidelity, log_g – forwarded to phases_vfield

    Returns
    -------
    obs_eval : np.ndarray   – eigenvalues of the observable
    obs_ev   : np.ndarray   – eigenvectors of the observable
    rhoa     : np.ndarray   – averaged RDM of region A (gradient > 0)
    rhob     : np.ndarray   – averaged RDM of region B (gradient < 0)
    or (None, None, None, None) if one region is empty.
    """
    n_rows, n_cols = rdms.shape[:2]
    is_1d_rows = n_rows == 1
    is_1d_cols = n_cols == 1
    if not (is_1d_rows or is_1d_cols):
        raise ValueError("phases_vfield_1d expects one axis of size 1, "
                         f"got shape {rdms.shape[:2]}. Use phases_vfield instead.")

    rdms = rdms[:,idxi:idxf,:,:]
    grad_g = phases_vfield(rdms, scale=1, grad=True,
                                fidelity=None, log_g=False)

    ys = np.sin(np.angle(grad_g.astype(complex)) + theta)
    
    # grad_g = phases_vfield(rdms, scale=1, grad=True,
    #                        fidelity=fidelity, log_g=log_g)
    
    # ys = np.sin(np.angle(grad_g.astype(complex)) + theta)

    if is_1d_rows:
        rdms_inner = rdms[0, 1:-1]   # (n_cols-2, rdm_sz, rdm_sz)
    else:
        rdms_inner = rdms[1:-1, 0]   # (n_rows-2, rdm_sz, rdm_sz)

    ys_flat   = ys.flatten()
    rdms_flat = rdms_inner            # already flat along scan axis


    idx_a = np.nonzero(ys_flat > 0)
    idx_b = np.nonzero(ys_flat < 0)
    print(idx_a, idx_b)

    if len(idx_a) == 0 or len(idx_b) == 0:
        print(f"[phases_vfield_1d] Warning: one region is empty "
              f"(|A|={len(idx_a)}, |B|={len(idx_b)}). "
              f"Try adjusting theta.")
        return None, None, None, None

    # rhoa = np.average(rdms_flat[idx_a], axis=0)
    # rhob = np.average(rdms_flat[idx_b], axis=0)
    # rhoa = rdms_flat[idx_a][0]
    # rhob = rdms_flat[idx_b][-1]

    # normalize all candidates
    rhos_a = rdms_flat[idx_a]
    rhos_b = rdms_flat[idx_b]
    rhos_a /= np.linalg.norm(rhos_a, axis=(1,2), keepdims=True)
    rhos_b /= np.linalg.norm(rhos_b, axis=(1,2), keepdims=True)

    # compute all pairwise dot products at once: shape (len_a, len_b)
    dots = np.einsum('aij,bji->ab', rhos_a, rhos_b).real

    # find the minimizing pair
    best_i, best_j = np.unravel_index(np.argmin(dots), dots.shape)
    rhoa = rhos_a[best_i]
    rhob = rhos_b[best_j]


    # rhoa /= np.linalg.norm(rhoa) # , ord='fro')
    # rhob /= np.linalg.norm(rhob) # , ord='fro')

    dot_ab = np.trace(rhoa @ rhob)
    obs    = rhoa - dot_ab * rhob
    obs   /= np.sqrt(1 - dot_ab ** 2)

    print(f"dot_ab = {dot_ab:.6f}, sqrt(1-dot_ab^2) = {np.sqrt(1 - dot_ab**2):.6f}")

    obs_eval, obs_ev = np.linalg.eigh(obs)
    return obs_eval, obs_ev, obs, rdms_flat