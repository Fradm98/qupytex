import numpy as np
from .linalg import mat_lattice_vec


def _lyap_mat(rho):
    """Prepare matrix for the solution of the Lyapunov equation
    to obtain the symmetric logarithmic derivative."""
    rho = np.asarray(rho)
    assert rho.ndim == 2
    iden = np.eye(len(rho))
    rho = np.kron(np.conj(rho), iden) + np.kron(iden, rho)
    return np.linalg.pinv(rho)


_EINSUM_LATTICE_VT_MAT_V = 'ija,ijab,ijb->ij'


def rdms_lattice_tr_qfim(rdms):
    """Compute the trace of the quantum fisher information matrix
    for a 2D lattice of density matrices."""
    rdms = np.asarray(rdms)
    assert rdms.ndim == 4

    lattice_shape = rdms.shape[:2]
    rdms_ksum = rdms.reshape((-1, ) + rdms.shape[2:])
    rdms_ksum = np.array([_lyap_mat(rho) for rho in rdms_ksum])
    rdms_ksum = rdms_ksum.reshape(lattice_shape + rdms_ksum.shape[1:])
    # Remove 1 row and col to adapt to the derivatives below.
    rdms_ksum = rdms_ksum[1:-1, 1:-1]

    rdms_vdy = mat_lattice_vec(rdms[:, :-2] - rdms[:, 2:])
    rdms_vdy = rdms_vdy[1:-1]
    rdms_vdx = mat_lattice_vec(rdms[:-2] - rdms[2:])
    rdms_vdx = rdms_vdx[:, 1:-1]

    h_xx = np.einsum(_EINSUM_LATTICE_VT_MAT_V, np.conj(rdms_vdx), rdms_ksum, rdms_vdx)
    h_yy = np.einsum(_EINSUM_LATTICE_VT_MAT_V, np.conj(rdms_vdy), rdms_ksum, rdms_vdy)
    return h_xx + h_yy
