# Global and reduced (Uhlmann) fidelities between two MPS given as lists of
# (chi_l, d, chi_r) tensors, as produced by the qs-mps DMRG.

import numpy as np
from qs_mps.mps_class import MPS

from qphaset.fidelity import uhlmann_fidelity


def global_overlap(s1, s2, l, chi, d=2, model_name="ANNNI"):
    """<s2|s1> and the two norms, via the same contraction used in exec_comparison."""
    mps = MPS(L=l, d=d, model=model_name, chi=chi, h=None, eps=None, J=None, bc='obc')
    mps.sites, mps.ancilla_sites = s2, s1
    ov = complex(mps._compute_norm(site=1, mixed=True))
    n1 = mps._compute_norm(site=1, ancilla=True).real
    n2 = mps._compute_norm(site=1).real
    return ov, n1, n2


def mixed_canonical(sites, start, stop):
    """Left-canonical on [0, start), right-canonical on [stop, L); norm absorbed in the block."""
    s = [t.copy() for t in sites]
    for i in range(start):
        cl, dd, cr = s[i].shape
        q, r = np.linalg.qr(s[i].reshape(cl * dd, cr))
        s[i] = q.reshape(cl, dd, -1)
        s[i + 1] = np.tensordot(r, s[i + 1], axes=(1, 0))
    for i in range(len(s) - 1, stop - 1, -1):
        cl, dd, cr = s[i].shape
        q, r = np.linalg.qr(s[i].reshape(cl, dd * cr).T)
        s[i] = q.T.reshape(-1, dd, cr)
        s[i - 1] = np.tensordot(s[i - 1], r.T, axes=(2, 0))
    return s


def block_tensor(s, start, stop):
    """Block of a mixed-canonical MPS as a (d^l, chi_l * chi_r) purification matrix, normalized."""
    T = s[start]
    for i in range(start + 1, stop):
        T = np.tensordot(T, s[i], axes=(2, 0)).reshape(T.shape[0], -1, s[i].shape[2])
    M = T.transpose(1, 0, 2).reshape(T.shape[1], -1)
    return M / np.linalg.norm(M)


def block_norm(s, start, stop):
    E = np.eye(s[start].shape[0], dtype=complex)
    for i in range(start, stop):
        E = np.einsum('ab,asx,bsy->xy', E, s[i].conj(), s[i], optimize=True)
    return np.trace(E).real


def reduced_fidelity(s1, s2, start, stop, max_elems=2 ** 25):
    """Uhlmann fidelity F (squared) of the block [start, stop), returned with the method used.

    "purif": sqrt(F) = || M1^dag M2 ||_1 with M_i the block in the orthonormal (left, right)
             Schmidt bases of state i (exact, no sqrt of near-zero eigenvalues: needed when the
             RDMs are almost pure, e.g. deep in the ferro phase). Builds a chi^4 tensor.
    "rdm":   rho_i = M_i M_i^dag (d^l x d^l), then uhlmann_fidelity as in exec_comparison.
             Used when chi^4 is too large; fine away from pure RDMs.
    Returns NaN when neither fits in max_elems."""
    a = mixed_canonical(s1, start, stop)
    b = mixed_canonical(s2, start, stop)
    da, db = a[start].shape[0], b[start].shape[0]
    ra, rb = a[stop - 1].shape[2], b[stop - 1].shape[2]
    dl = a[start].shape[1] ** (stop - start)
    if da * db * ra * rb <= max_elems:
        # X[a, a', c, c'] = sum_s conj(A)[a s.. c] B[a' s.. c']
        X = np.einsum('ac,bd->abcd', np.eye(da), np.eye(db)).astype(complex)
        for i in range(start, stop):
            X = np.einsum('abcd,csx,dsy->abxy', X, a[i].conj(), b[i], optimize=True)
        O = X.transpose(0, 2, 1, 3).reshape(da * ra, db * rb)
        sqrtF = np.sum(np.linalg.svd(O, compute_uv=False))
        sqrtF /= np.sqrt(block_norm(a, start, stop) * block_norm(b, start, stop))
        return float(sqrtF ** 2), "purif"
    if dl * max(da * ra, db * rb) <= max_elems:
        M1, M2 = block_tensor(a, start, stop), block_tensor(b, start, stop)
        rho1, rho2 = M1 @ M1.conj().T, M2 @ M2.conj().T
        return float(uhlmann_fidelity(rho1, rho2)), "rdm"
    return float('nan'), "skip"


def mid_magnetization(s, l):
    """(<sigma^x>, <sigma^z>) on the central site: tells whether the state is symmetry broken."""
    i = l // 2
    M = block_tensor(mixed_canonical(s, i, i + 1), i, i + 1)
    rho = M @ M.conj().T
    return float(np.real(rho[0, 1] + rho[1, 0])), float(np.real(rho[0, 0] - rho[1, 1]))
