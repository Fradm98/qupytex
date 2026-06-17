
import numpy as np
from matplotlib import pyplot as plt
from .models import params_2d_lattice
from .annni import th_lines, PT_tran, BKT_tran, I_tran
from .phases import decompose_obs, make_obs_vec, get_obs_ev

def plot_grad_g_angle_stream(grad_g, *, params_extent=(0, 1, 0, 1), axis_name=('k', 'h'),
                             figsize=(12, 5), theory_lines=True):
    grad_g = np.asarray(grad_g)
    assert grad_g.ndim == 2

    if grad_g.shape[0] == grad_g.shape[1]:
        n1 = len(grad_g)
        n2 = n1
    else:
        n1 = grad_g.shape[0]
        n2 = grad_g.shape[1]

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].matshow(np.angle(-grad_g), cmap='twilight',
                   extent=params_extent, origin='upper', aspect='equal',
                   vmin=-np.pi, vmax=np.pi)
    # The minus in front of grad_g is there just to keep the same coloring scheme.
    axs[0].set_xlabel(axis_name[0])
    axs[0].set_ylabel(axis_name[1])
    if theory_lines:
        th_lines(PT_tran, axs[0], params_extent, color='red', linestyle='--')
        th_lines(BKT_tran, axs[0], params_extent, color='red', linestyle='--')
        th_lines(I_tran, axs[0], params_extent, color='red', linestyle='--')

    xy = params_2d_lattice(params_extent[:2], params_extent[2:], n1=n1, n2=n2)
    X, Y = xy[:, 0].reshape((n1, n2)), xy[:, 1].reshape((n1, n2))
    axs[1].streamplot(X, Y, np.real(grad_g), np.imag(grad_g), origin='upper', density=3)
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

def plot_observable(obs, rdms, sites=None,  figure_name=None, lattice_shape=None, params_extent=(0, 1, 0, 1), component=None):
    meas = [np.trace(rdm @ obs) for rdm in rdms]
    meas = np.reshape(meas, lattice_shape)

    # Note we plot the absolute value to avoid misunderstanding in the interpretation of the
    # colormap.
    fig, ax = plt.subplots(1,2, figsize=(10,4.5))
    ax0 = ax[0].matshow(np.abs(meas), origin='lower', cmap='plasma', aspect='auto', extent=params_extent)
    fig.colorbar(ax0, ax=ax[0])
    ax[0].set_xlabel('$\\kappa$')
    ax[0].set_ylabel('h')
    if component is None:
        title_str = "Whole Observable"
        figure_name = f"{figure_name}.png"
    else:
        figure_name = f"{figure_name}_component_{component}.png"
        title_str = f"Eigvec: {component}"
    ax[0].set_title(title_str)

    sorted_components = decompose_obs(obs, len(sites))
    # ax1 = ax[1].matshow(obs.real, aspect='auto')
    ax1 = ax[1].matshow(np.abs(obs), aspect='auto', vmin=0, vmax=1)
    fig.colorbar(ax1, ax=ax[1])
    ax[1].set_xticks(range(len(sorted_components)), sorted_components)
    ax[1].set_yticks(range(len(sorted_components)), sorted_components)
    ax[1].set_title("Pauli Decomposition")
    plt.tight_layout()

    plt.savefig(figure_name)
    plt.close()

def plot_k_components(obs, rdms, sites, figure_name, params_extent, lattice_shape, v0_first_schmidt_vec=False):
    obs_eval, obs_ev = get_obs_ev(obs)
    for obs_ev_idx in range(2**len(sites)):
        obs_vec = make_obs_vec(obs_ev, obs_eval, obs_ev_idx, v0_first_schmidt_vec)
        plot_observable(obs_vec, rdms, sites, figure_name=figure_name, lattice_shape=lattice_shape, params_extent=params_extent, component=obs_ev_idx)