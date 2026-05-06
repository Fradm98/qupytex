import numpy as np


def psd_sqrt_safe(mat) -> np.ndarray:
    """Compute a numerically safe sqrt of a given PSD matrix."""
    d, u = np.linalg.eigh(mat)
    d = np.sqrt(np.maximum(d, 0))
    return u @ np.diag(d) @ np.conj(u).T


def projh_psd(m) -> np.ndarray:
    """Project a given hermitian matrix into the cone
    of positive semidefinite and hermitian matrices."""
    d, u = np.linalg.eigh(m)
    d = np.where(d >= 0, d, 0)
    return u @ np.diag(d) @ np.conj(u).T


def proj_centering(mat):
    mat = np.asarray(mat)
    assert mat.ndim == 2
    p = np.eye(len(mat)) - np.ones_like(mat) / len(mat)
    return p @ mat @ p


def mat_lattice_vec(mat):
    """Vectorize (column-major) the matrices of a lattice of matrices."""
    mat = np.swapaxes(mat, -1, -2)
    return np.reshape(mat, mat.shape[:-2] + (-1, ))


def schmidt_decomp_half(v, *, contract_sigmas=None, normalize=False):
    v = np.asarray(v).flatten()
    v = np.reshape(v, (int(np.sqrt(len(v))), -1))
    mat_u, sigmas, mat_v = np.linalg.svd(v)
    mat_v = np.transpose(mat_v) # Note no conj!

    if contract_sigmas is not None: # The number of terms to be used.
        assert contract_sigmas > 0
        mat_u = mat_u[:, :contract_sigmas]
        mat_v = mat_v[:, :contract_sigmas]
        sigmas = sigmas[:contract_sigmas]
        v = np.einsum('aj,bj,j->ab', mat_u, mat_v, sigmas)
        v = v.flatten()
        # The flag normalize assumes the resulting vector is non-zero.
        return v / np.linalg.norm(v) if normalize else v

    if normalize:
        sigmas = np.square(sigmas)
        sigmas = np.sqrt(sigmas / np.sum(sigmas))
    return mat_u, sigmas, mat_v


def qubit_state_to_angles(v):
    """Given a qubit state obtain the angles theta and phi corresponding to
    v=cos(theta) + i e^{i phi} sin(theta). See Nielsen-Chuang eq (1.4).
    """
    v = np.asarray(v).flatten()
    assert len(v) == 2
    a = np.angle(v[1]) - np.angle(v[0])
    v = np.abs(v)
    assert np.isclose(np.sum(np.square(v)), 1)
    return np.arccos(v[0]), a


def schmidt_decomp_2q_angles(v):
    v = np.asarray(v).flatten()
    assert len(v) == 4  # Expected 2 qubits state.
    mat_u, sigmas, mat_v = schmidt_decomp_half(v)
    angles = []
    for i in range(len(sigmas)):
        a1 = qubit_state_to_angles(mat_u[:, i])
        a2 = qubit_state_to_angles(mat_v[:, i])
        angles.append(a1 + a2)
    return np.array(angles), sigmas