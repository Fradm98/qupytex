
import numpy as np
from matplotlib import pyplot as plt
from .models import params_2d_lattice
from .annni import th_lines, PT_tran, BKT_tran, I_tran


def plot_grad_g_angle_stream(grad_g, *, params_extent=(0, 1, 0, 1), axis_name=('k', 'h'),
                             figsize=(12, 5), theory_lines=True):
    grad_g = np.asarray(grad_g)
    assert grad_g.ndim == 2
    # Assume same sampling rate for rows and columns.
    assert grad_g.shape[0] == grad_g.shape[1]
    n = len(grad_g)

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].matshow(np.angle(-grad_g), origin='lower', cmap='twilight',
                   extent=params_extent, aspect='equal',
                   vmin=-np.pi, vmax=np.pi)
    # The minus in front of grad_g is there just to keep the same coloring scheme.
    axs[0].set_xlabel(axis_name[0])
    axs[0].set_ylabel(axis_name[1])
    if theory_lines:
        th_lines(PT_tran, axs[0], params_extent, color='red', linestyle='--')
        th_lines(BKT_tran, axs[0], params_extent, color='red', linestyle='--')
        th_lines(I_tran, axs[0], params_extent, color='red', linestyle='--')

    xy = params_2d_lattice(params_extent[:2], params_extent[2:], n=n)
    X, Y = xy[:, 0].reshape((n, n)), xy[:, 1].reshape((n, n))
    axs[1].streamplot(X, Y, np.real(grad_g), np.imag(grad_g), density=3)
    axs[1].set_xlabel(axis_name[0])
    axs[1].set_ylabel(axis_name[1])
    return axs


def plot_grad_g_angle4(grad_g, *, params_extent=(0, 1, 0, 1), axis_name = ('k', 'h'),
                       figsize=(12, 5)):
    """Angle plots with different rotations of the colormap."""
    fig, axs = plt.subplots(1, 4, figsize=figsize)
    for i, c in enumerate((1, -1, 1j, -1j)):
        axs[i].matshow(np.angle(-grad_g * c), origin='lower', cmap='twilight',
                       extent=params_extent, aspect='equal',
                       vmin=-np.pi, vmax=np.pi)
        axs[i].set_xlabel(axis_name[0])
        axs[i].set_ylabel(axis_name[1])
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    return axs


def plot_grad_g_angle_sin_cos(grad_g, *, params_extent=(0, 1, 0, 1), axis_name = ('k', 'h'),
                              figsize=(12, 5)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    # Note we do not need cyclic colormaps for sin/cos of angle.
    axs[0].matshow(np.cos(np.angle(grad_g)), origin='lower', cmap='plasma',
                   extent=params_extent, aspect='equal', vmin=-1, vmax=1)
    lastax = axs[1].matshow(np.sin(np.angle(grad_g)), origin='lower', cmap='plasma',
                            extent=params_extent, aspect='equal', vmin=-1, vmax=1)
    plt.colorbar(lastax)
    for i in (0, 1):
        axs[i].set_xlabel(axis_name[0])
        axs[i].set_ylabel(axis_name[1])
    return axs
