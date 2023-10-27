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