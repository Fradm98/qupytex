"""
exec_finite_size_scaling_from_rdms.py
======================================
Finite-size scaling (FSS) analysis starting directly from the pre-computed RDM
archives (.npz) — no MPS tensors required.

Prerequisite: for each system size L in `Ls`, run exec_save_rdms.py with
sites = [l//2] (or [l//2-1, l//2]) to produce one .npz per L, then upload
the whole set to Zenodo.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from qiskit.quantum_info import SparsePauliOp

from qphaset.phases import constructing_order_parameter, phases_vfield
from qs_mps.utils import create_sequential_colors

from qupytex_io import load_rdms

# ── Plot style (matches publication layout) ───────────────────────────────────
FIGWIDTH = 7.0
ASPECT   = 1.0
COLS     = 2
fig_height = (FIGWIDTH / COLS) * ASPECT

plt.rcParams.update({
    'figure.figsize'  : (FIGWIDTH, fig_height),
    'figure.dpi'      : 300,
    'font.size'       : 9,
    'font.family'     : 'sans-serif',
    'axes.labelsize'  : 10,
    'axes.titlesize'  : 10,
    'axes.linewidth'  : 0.6,
    'xtick.labelsize' : 9,
    'ytick.labelsize' : 9,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.fontsize' : 8,
    'legend.framealpha': 0.8,
    'lines.linewidth' : 1.0,
    'lines.markersize': 3,
})

# ── Model config ──────────────────────────────────────────────────────────────
model_name = "ANNNI"
Ls   = [50, 70, 90, 110, 130, 150]     # system sizes for FSS
n1   = 1
n2   = 401
chi  = 50
c1   = 1e-4

# ── Scan parameters (must match what was used in exec_save_rdms.py) ───────────
lambda1_i, lambda1_f = 0.001, 0.001
lambda2_i, lambda2_f = 0.8,   1.2
direction = "v"                         # "v" → sweep along λ₂, "h" → along λ₁

lam1_min, lam1_max = min(lambda1_i, lambda1_f), max(lambda1_i, lambda1_f)
lam2_min, lam2_max = min(lambda2_i, lambda2_f), max(lambda2_i, lambda2_f)

# ── Optional sub-region restriction ──────────────────────────────────────────
lambda1_range = None
lambda2_range = None

# ── Observable construction window ───────────────────────────────────────────
idxi  = 10
idxf  = -1
theta = -np.pi / 2

# ── Device ────────────────────────────────────────────────────────────────────
device = 'pc'
# device = 'ngt'

if device == 'pc':
    device_path = "D:/work"
elif device == 'ngt':
    device_path = "/eos/user/f/fdimarca"

# ── Routing ───────────────────────────────────────────────────────────────────
if model_name == 'ANNNI':
    path_to_rdms    = f"{device_path}/projects/2_ANNNI/results/rdms"
    path_to_figures = f"{device_path}/projects/2_ANNNI/figures"
    axis_name = ('k', 'h')
elif model_name == 'Cluster':
    path_to_rdms    = f"{device_path}/projects/3_CLUSTER/results/rdms"
    path_to_figures = f"{device_path}/projects/3_CLUSTER/figures"
    axis_name = ('K', 'h')
elif model_name == 'Rydberg':
    path_to_rdms    = f"{device_path}/projects/4_RYDBERG/results/rdms"
    path_to_figures = f"{device_path}/projects/4_RYDBERG/figures"
    axis_name = ('$\\Delta/\\Omega$', '$R_b/a$')
else:
    raise SyntaxError("Choose a valid model among 'ANNNI', 'Cluster', 'Rydberg'")

# ── FSS fit models ────────────────────────────────────────────────────────────
def fss_model(L, a_pp, nu, b_pp, theta_exp):
    """Peak susceptibility scaling: a'' * L^(1/ν) * (1 + b'' * L^(-θ/ν))"""
    return a_pp * L ** (1.0 / nu) * (1.0 + b_pp * L ** (-theta_exp / nu))

def pow_law(x, a, b, c):
    return a + b * (x ** c)

def power_law(logL, log_A, bnu):
    return log_A - bnu * logL

# ── Collect results across system sizes ──────────────────────────────────────
colors       = create_sequential_colors(len(Ls) + 1)
peak_vals    = []
peak_lambdas = []
lam_vals     = []
mag_vals     = []
peak_idxs    = []

fig_07, (ax_op, ax_sus) = plt.subplots(1, 2,
                                        figsize=(FIGWIDTH, fig_height),
                                        constrained_layout=True)

for color, l in zip(colors, Ls):

    base_filename = (
        f"{model_name}_L_{l}"
        f"_lambda_1_{lam1_min}-{lam1_max}"
        f"_lambda_2_{lam2_min}-{lam2_max}"
        f"_npoints_{n1}x{n2}_chi_{chi}_eps_{c1}"
    )

    # ── Load RDMs for this L ─────────────────────────────────────────────────
    result = load_rdms(
        path_to_rdms  = path_to_rdms,
        base_filename = base_filename,
        lambda1_range = lambda1_range,
        lambda2_range = lambda2_range,
    )

    rdms        = result["rdms"]            # (n1_sub, n2_sub, D, D)
    params_grid = result["params_grid"]
    n1_sub      = result["n1_sub"]
    n2_sub      = result["n2_sub"]
    l           = result["l"]
    sites       = result["sites"]

    print(f"L={l}  RDMs shape: {rdms.shape}  sites={sites}")

    # ── d_lambda for the finite-difference susceptibility ────────────────────
    if direction == "h":
        d_lambda = abs(params_grid[0, 0, 0] - params_grid[0, 1, 0]) if n2_sub > 1 else 1.0
    else:  # "v"
        d_lambda = abs(params_grid[0, 0, 1] - params_grid[1, 0, 1]) if n1_sub > 1 else 1.0
    print(f"  d_lambda={d_lambda:.6f}")

    # ── Build observable M ───────────────────────────────────────────────────
    obs_eval, obs_ev, obs, rdms_flat = constructing_order_parameter(
        rdms, idxi=idxi, idxf=idxf, theta=theta
    )
    print(f"  Observable eigenvalues: {obs_eval}")
    print(SparsePauliOp.from_operator(obs))

    # ── λ axis aligned with rdms_flat ────────────────────────────────────────
    lambdas_full   = np.linspace(lambda2_i, lambda2_f, n2_sub)
    lambdas_window = lambdas_full[idxi:idxf]
    n_flat         = len(rdms_flat)
    lambdas_window = lambdas_window[:n_flat]

    order_param = np.array([np.trace(rdm @ obs).real for rdm in rdms_flat])
    mag_vals.append(order_param)

    # ── Susceptibility (centred finite difference) ───────────────────────────
    susceptibility = np.abs(
        (order_param[2:] - order_param[:-2]) / (2.0 * d_lambda)
    )
    lambdas_inner = lambdas_window[1:-1]

    # ── Peak location ────────────────────────────────────────────────────────
    i_peak     = np.argmax(susceptibility)
    lambda_c_L = lambdas_inner[i_peak]
    chi_peak   = susceptibility[i_peak]
    print(f"  lambda_c(L)={lambda_c_L:.5f}  chi_peak={chi_peak:.5f}")

    peak_vals.append(chi_peak)
    peak_lambdas.append(lambda_c_L)
    lam_vals.append(lambdas_window)
    peak_idxs.append(i_peak)

    ax_op.plot(lambdas_window, order_param,  color=color, label=f"$L={l}$")
    ax_sus.plot(lambdas_inner, susceptibility, color=color, label=f"$L={l}$")

# ── Dress op / susceptibility subplots ────────────────────────────────────────
ax_op.set_xlabel("$h$")
ax_op.set_ylabel("$\\langle M \\rangle$")
ax_op.text(0.74, 0.7, "(a)")
ax_op.legend()
ax_op.grid(True, alpha=0.3)
ax_op.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])

ax_sus.set_xlabel("$h$")
ax_sus.set_ylabel("$\\chi_M = \\frac{\\partial \\langle M \\rangle}{\\partial h}$")
ax_sus.text(0.74, 26.5, "(b)")
ax_sus.legend()
ax_sus.grid(True, alpha=0.3)
ax_sus.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])

fig_07.savefig(f"{path_to_figures}/fig07.pdf", dpi=300, bbox_inches='tight')
plt.close()

# ── Critical-point extrapolation (1/L → 0) ────────────────────────────────────
y_err          = (lambda2_f - lambda2_i) / n2_sub
crit_vals_err  = np.array([y_err] * len(Ls))

xdata = [1 / L for L in Ls]
ydata = peak_lambdas
p_opt, co_opt = curve_fit(pow_law, xdata, ydata, sigma=crit_vals_err)
a_opt, b_opt, c_opt = p_opt
perr = np.sqrt(np.diag(co_opt))
a_err, b_err, c_err = perr

print(
    f"\nCritical point extrapolation:\n"
    f"  g_c∞  = {a_opt:.4f} ± {a_err:.4f}\n"
    f"  ν     = {1/c_opt:.4f} ± {c_err/c_opt**2:.4f}"
)

xfit = np.linspace(0, 1.0 / min(Ls), 100)
yfit = pow_law(xfit, a_opt, b_opt, c_opt)

fig, ax = plt.subplots(1, 2, figsize=(FIGWIDTH, fig_height), constrained_layout=True)

ax[0].scatter(xdata, peak_lambdas, s=40, marker='o', color='k')
ax[0].errorbar(xdata, peak_lambdas, yerr=crit_vals_err,
               fmt='none', ecolor='k', capsize=7)
ax[0].plot(xfit, yfit, '--', color='red', linewidth=1.5)
ax[0].scatter([0], [pow_law(0, a_opt, b_opt, c_opt)],
              marker='x', color='darkred', s=70)
ax[0].errorbar([0], [pow_law(0, a_opt, b_opt, c_opt)], yerr=[a_err],
               fmt='none', ecolor='darkred', capsize=7, zorder=-1)
ax[0].text(0.009, 0.995,
           f"$g_c^{{\\infty}} = {pow_law(0, a_opt, b_opt, c_opt):.4f} \\pm {a_err:.4f}$")
ax[0].text(0.009, 0.985,
           f"$\\nu = {1/c_opt:.4f} \\pm {c_err/c_opt**2:.4f}$")
ax[0].set_xlabel("$1/L$")
ax[0].set_ylabel("$h_c^L$")
ax[0].grid(True, alpha=0.3)
ax[0].text(-0.0025, 1.014, "(a)")
ax[0].set_xticks([0, 0.005, 0.01, 0.015, 0.02])

# ── β/ν extrapolation ────────────────────────────────────────────────────────
print("\nM at λ_c(L):")
m_crit = []
for L, lam, M, lc in zip(Ls, lam_vals, mag_vals, peak_lambdas):
    M_interp = float(interp1d(lam, M, kind='cubic')(lc))
    m_crit.append(M_interp)
    print(f"  L={L}: lambda_c={lc:.4f}, M={M_interp:.6f}")

xdata_log = np.log(Ls[2:])
ydata_log = np.log(m_crit[2:])
p_opt2, co_opt2 = curve_fit(power_law, xdata_log, ydata_log)
a_opt2, bnu_opt = p_opt2
perr2 = np.sqrt(np.diag(co_opt2))
a_err2, bnu_err = perr2

print(f"\nβ/ν extrapolation:\n"
      f"  β/ν = {bnu_opt:.4f} ± {bnu_err:.4f}")

xfit2 = np.linspace(np.log(min(Ls[2:])) - 0.1, np.log(max(Ls[2:])) + 0.1, 100)
yfit2 = power_law(xfit2, a_opt2, bnu_opt)

ax[1].scatter(xdata_log, ydata_log, s=40, marker='o', color='k')
ax[1].plot(xfit2, yfit2, '--', color='red', linewidth=1.5)
ax[1].text(4.7, -0.93,
           f"$\\beta/\\nu = {bnu_opt:.4f} \\pm {bnu_err:.4f}$")
ax[1].set_xlabel("$\\log(L)$")
ax[1].set_ylabel("$\\log(M(h_c^L))$")
ax[1].grid(True, alpha=0.3)
ax[1].text(4.31, -0.914, "(b)")

plt.savefig(f"{path_to_figures}/{model_name}_fss_critical_extrapolation.png",
            dpi=300, bbox_inches='tight')
fig.savefig(f"{path_to_figures}/fig08.pdf", dpi=300)
plt.show()