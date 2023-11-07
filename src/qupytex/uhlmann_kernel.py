import numpy as np
from scipy.linalg import sqrtm

def uhlmann_kernel(rdm_1, rdm_2):
    n = rdm_2.shape[0]
    assert rdm_2.shape == (n, n)
    assert rdm_1.shape == (n, n)
    # √rho1
    rdm_1_sqrt = np.real(sqrtm(rdm_1))
    # √(√rho1 * rho2 * √rho1)
    dist_sqrt = sqrtm(rdm_1_sqrt @ rdm_2 @ rdm_1_sqrt)
    # Equation 11
    f11_body = np.trace(dist_sqrt)
    f11 = f11**2
    return f11

def uhlmann_fidelity_1q(d1, d2, *, purity_only=False):
    """
    uhlmann_fidelity_1q

    This function computes the Uhlmann fidelity for single qubit 
    reduced density matrices.

    d1: first rdm
    d2: second rdm
    purity_only: choose which formula to use according to the purity of the rdms
    """
    d1, d2 = map(np.asarray, (d1, d2))
    assert d1.shape == (2, 2)
    assert d2.shape == (2, 2)
    p1, p2 = map(lambda d: np.real(np.trace(d @ d)), (d1, d2))
    p = np.sqrt((1 - p1) * (1 - p2))
    return p if purity_only else np.real(np.trace(d1 @ d2)) + p