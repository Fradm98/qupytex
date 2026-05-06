# fix relative imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from .linalg import psd_sqrt_safe as sqrtm 
from functools import partial
from ncon import ncon

def _sqrt_safe(v):
    v = np.maximum(v, 0)   # Fix minor numerical errors.
    return np.sqrt(v)


def uhlmann_fidelity_1q(d1, d2, *, purity_only=False, superfidelity=False):
    """Compute the Uhlmann fidelity for single qubit density
    matrices."""
    d1, d2 = map(np.asarray, (d1, d2))
    assert d1.ndim == 2 and d2.ndim == 2
    if not superfidelity:
        assert d1.shape == (2, 2)
        assert d2.shape == (2, 2)
    p1, p2 = map(lambda d: np.real(np.trace(d @ d)), (d1, d2))
    p = _sqrt_safe((1 - p1) * (1 - p2))
    return p if purity_only else np.real(np.trace(d1 @ d2)) + p


def uhlmann_fidelity_svd(d1, d2, *, sqrt=True):
    """Nuclean norm-based implementation of the Uhlmann fidelity
    (obtained through SVD)."""
    d1, d2 = map(np.asarray, (d1, d2))
    d1, d2 = map(sqrtm, (d1, d2))
    v = np.sum(np.linalg.svd(d1 @ d2)[1])
    if sqrt:
        return v.astype(np.float64)
    return np.square(v).astype(np.float64)


def uhlmann_fidelity(d1, d2, *, sqrt=False, svd=True):
    """Uhlmann fidelity."""
    if svd:
        return uhlmann_fidelity_svd(d1, d2, sqrt=sqrt)
    d1, d2 = map(np.asarray, (d1, d2))
    d1s = sqrtm(d1)
    v = np.real(np.trace(sqrtm(d1s @ d2 @ d1s)))
    if sqrt:
        return v.astype(np.float64)
    return np.square(v).astype(np.float64)


def superfidelity(d1, d2):
    """Superfidelity. This is a Jozsa fidelity which bounds the Uhlmann's
    above."""
    return uhlmann_fidelity_1q(d1, d2, superfidelity=True)


def superfidelity_dist(d1, d2):
    """RDM distance based on superfidelity sqrt(1-F(a, b))."""
    v = 1 - superfidelity(d1, d2)
    return _sqrt_safe(v)


def bures_angle(d1, d2, *, svd=True):
    return np.arccos(uhlmann_fidelity(d1, d2, svd=svd, sqrt=True))


def bures(d1, d2, *, squared=False, svd=True):
    """Compute the Bures distance."""
    v = 2 - 2 * uhlmann_fidelity(d1, d2, sqrt=True, svd=svd)
    v = np.maximum(0, v)
    return v if squared else np.sqrt(v)


def bures_1q(d1, d2, *, squared=False):
    """Compute the Bures distance on 1-RDM."""
    dist = 2 * (1 - np.sqrt(uhlmann_fidelity_1q(d1, d2)))
    v = np.maximum(0, v)
    return dist if squared else np.sqrt(dist)


def fidelity_dxx(rdms: np.ndarray, *, fidelity=None, log=False) -> np.ndarray:
    """Compute the fidelity susceptibility along the x axis.
    The parameter rdms is expected to be a tensor structured as a matrix of RDM.
    Set param log=True when the fidelity function (see corresponding argument)
    is the log of a fidelity. Note that the resulting second derivative
    is not multipled by -1 as the common definition of fidelity susceptibility.
    """
    f_dx = fidelity_diffx(rdms, fidelity=fidelity)
    # Apply the second difference operator
    # (see https://en.wikipedia.org/wiki/Recurrence_relation).
    f_dxx = f_dx[:, :-1] + f_dx[:, 1:] - (0 if log else 2)
    return f_dxx


def fidelity_diffx(rdms: np.ndarray, *, fidelity=None) -> np.ndarray:
    """Compute the fidelity differential along the x axis."""
    if fidelity is None:
        fidelity = partial(uhlmann_fidelity, sqrt=True)
    mn, rdm_sz = rdms.shape[:2], rdms.shape[2]
    mat1, mat2 = rdms[:, :-1], rdms[:, 1:]
    mat1, mat2 = mat1.reshape((-1, rdm_sz, rdm_sz)), mat2.reshape((-1, rdm_sz, rdm_sz))
    f_dx = np.array([fidelity(d1, d2) for d1, d2 in zip(mat1, mat2)])
    f_dx = f_dx.reshape((mn[0], mn[1] - 1))
    return f_dx


def fidelity_diff_1d(rdms: np.ndarray, *, fidelity=None) -> np.ndarray:
    if fidelity is None:
        fidelity = partial(uhlmann_fidelity, sqrt=True)
    rdms = np.asarray(rdms)
    mat1, mat2 = rdms[:-1], rdms[1:]
    return np.array([fidelity(d1, d2) for d1, d2 in zip(mat1, mat2)])


def fidelity_laplacian_1d(rdms: np.ndarray, *, fidelity=None, log=False) -> np.ndarray:
    dx = fidelity_diff_1d(rdms, fidelity=fidelity)
    return dx[:-1] + dx[1:] - (0 if log else 2)


def fidelity_diff_xy(rdms, *, fidelity=None):
    """Compute the fidelity differential along the x, y axes."""
    diffx = fidelity_diffx(rdms, fidelity=fidelity)
    diffy = fidelity_diffx(np.swapaxes(rdms, 0, 1), fidelity=fidelity)
    diffy = np.swapaxes(diffy, 0, 1)

    # Assume square lattice.
    n1 = diffx.shape[0] * 2 - 1
    mat = np.zeros((n1, n1))
    mat[0::2, 1::2] = diffx
    mat[1::2, ::2] = diffy
    return mat


def fidelity_laplacian(rdms: np.ndarray, *, fidelity=None, log=False) -> np.ndarray:
    """
    Compute the Laplacian of the fidelity. See fidelity_dxx for more information
    on the parameter rdms.
    """
    dxx_params = dict(fidelity=fidelity, log=log)
    f1 = fidelity_dxx(rdms, **dxx_params)
    f2 = fidelity_dxx(np.swapaxes(rdms, 0, 1), **dxx_params)
    f2 = np.swapaxes(f2, 0, 1)
    return f2[:, 1:-1] + f1[1:-1, :]


def _compute_norm(mps, site, prnt=False):
    """
    _compute_norm

    This function computes the norm of our quantum state which is represented in mps.
    It takes the attributes .sites and .bonds of the class which gives us
    mps in canonical form (Vidal notation).

    site: int - sites in the chain
    ancilla: bool - if True we compute the norm of the ancilla_sites mps. By default False
    mixed: bool - if True we compute the braket between the ancilla_sites and the sites mps. By default False

    """
    a = np.array([1])

    array = mps.copy()

    ten = ncon([a, a],[[-1], [-2]])
    for i in range(site):
        
        ten = ncon(
            [ten, array[i], array[i].conjugate()],
            [[1, 2], [1, 3, -1], [2, 3, -2]],
        )
        if prnt:
            print(f"site {i+1}: ", ten.shape, array[i].shape)

    N = ncon([ten, a, a], [[1, 2], [1], [2]]).real

    return N
