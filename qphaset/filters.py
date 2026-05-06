import numpy as np
from scipy import signal


# Scharr filter kernel
SCHARR_X = np.array([[3, 0, -3],
                     [10, 0, -10],
                     [3, 0, -3]])
SCHARR = SCHARR_X + SCHARR_X.T * 1j


# Sobel filter kernel
SOBEL_X = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
SOBEL = SOBEL_X + SOBEL_X.T * 1j


def bump(x, *, _sqrt=False):
    """Evaluate the bump function at the scalar point x."""
    x = np.where(np.abs(x) >= 1, 1 - np.finfo(float).eps, x)
    x = np.exp(-1/(1 - x)) if _sqrt else np.exp(-1/(1 - x**2))
    return x


def bump_rbf(x):
    """A radial basis function (RBF) based on the bump function."""
    x = np.sum(np.square(x), axis=-1)
    return bump(x, _sqrt=True)


def bump_kernel(n, *, scale=1.):
    """Prepare the bump kernel filter. This is used as a low-pass filter.
    The parameter scale is used when upsampling."""
    xy = np.linspace(-1, 1, n), np.linspace(-1, 1, n)
    xy = map(lambda m: m.flatten(), np.meshgrid(*xy, indexing='xy'))
    xy = tuple(xy)
    xy = np.stack(xy).T

    kernel = bump_rbf(xy)
    kernel = np.reshape(kernel, (n, n))
    kernel = scale * kernel / np.sum(kernel)
    return kernel


def upsampling_base(mat):
    """Given a matrix, prepare the pattern for the scale doubling upsampling."""
    mat = np.asarray(mat)
    assert mat.ndim == 2
    mat1 = np.zeros(np.array(mat.shape) * 2)
    mat1[::2, ::2] = mat
    return mat1
