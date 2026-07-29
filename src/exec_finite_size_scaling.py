import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit.quantum_info import SparsePauliOp

from qs_mps.utils import create_sequential_colors

from qphaset.phases import (gstates_to_rdms_matrix_qs_mps, constructing_order_parameter, make_obs_vec, phases_vfield, decompose_obs,
                             sanitize_state)

from qupytex_io import load_gstates, describe_manifest

import matplotlib.pyplot as plt

# Two square-ish subplots side by side, fitting a two-column page
FIGWIDTH = 7.0        # inches — full text width of a two-column paper
ASPECT   = 1.0        # square subplots
COLS     = 2
HSPACE   = 0.35       # vertical space reserved for labels

fig_height = (FIGWIDTH / COLS) * ASPECT  # = 3.5 inches

plt.rcParams.update({
    # Figure
    'figure.figsize'        : (FIGWIDTH, fig_height),  # (7.0, 3.5)
    'figure.dpi'            : 300,                     # print quality

    # Font — match your paper's body font size (usually 9–10 pt)
    'font.size'             : 9,    # base fallback
    'font.family'           : 'sans-serif',

    # Axes
    'axes.labelsize'        : 10,    # x/y labels
    'axes.titlesize'        : 10,    # subplot titles
    'axes.linewidth'        : 0.6,  # thinner spines look cleaner at small size

    # Ticks
    'xtick.labelsize'       : 9,
    'ytick.labelsize'       : 9,
    'xtick.major.width'     : 0.6,
    'ytick.major.width'     : 0.6,
    'xtick.major.size'      : 3,
    'ytick.major.size'      : 3,

    # Legend
    'legend.fontsize'       : 8,
    'legend.framealpha'     : 0.8,

    # Lines & markers — thinner lines scale better when printed small
    'lines.linewidth'       : 1.0,
    'lines.markersize'      : 3,
})


# -- Pauli matrices ---
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# ── Model config ──────────────────────────────────────────────────────────────
model_name = "ANNNI"
Ls   = [50, 70, 90, 110, 130, 150]         # system sizes for FSS
n1   = 1
n2   = 401
chi  = 50
c1   = 1e-4

# ── Optional: restrict to a sub-region ───────────────────────────────────────
lambda1_range = None
lambda2_range = None

# ── Device ────────────────────────────────────────────────────────────────────
device = 'pc'
device = 'ngt'

if device == 'pc':
    device_path = "D:/work"
elif device == 'ngt':
    device_path = "/eos/user/f/fdimarca"

# ── Routing ───────────────────────────────────────────────────────────────────
if model_name == 'ANNNI':
    path_to_tensor  = f"{device_path}/projects/2_ANNNI/results/data"
    path_to_figures = f"{device_path}/projects/2_ANNNI/figures"
    axis_name = ('k', 'h')
elif model_name == 'Cluster':
    path_to_tensor  = f"{device_path}/projects/3_CLUSTER/results/data"
    path_to_figures = f"{device_path}/projects/3_CLUSTER/figures"
    axis_name = ('K', 'h')
elif model_name == 'Rydberg':
    path_to_tensor  = f"{device_path}/projects/4_RYDBERG/results/data"
    path_to_figures = f"{device_path}/projects/4_RYDBERG/figures"
    axis_name = ('$\\Delta/\\Omega$', '$R_b/a$')
else:
    raise SyntaxError("Choose a valid model among 'ANNNI', 'Cluster', 'Rydberg'")

# ── Scan parameters ───────────────────────────────────────────────────────────
lambda1_i, lambda1_f = 0.001, 0.001
lambda2_i, lambda2_f = 0.8,   1.2
direction = "v"

lam1_min, lam1_max = min(lambda1_i, lambda1_f), max(lambda1_i, lambda1_f)
lam2_min, lam2_max = min(lambda2_i, lambda2_f), max(lambda2_i, lambda2_f)

# ── Observable construction parameters ───────────────────────────────────────
idxi  = 10
idxf  = -1
theta = -np.pi/2

# ── FSS fit model ─────────────────────────────────────────────────────────────
# max_h { d<M>/dh } = a'' * L^(1/nu) * (1 + b'' * L^(-theta_exp/nu))
def fss_model(L, a_pp, nu, b_pp, theta_exp):
    return a_pp * L ** (1.0 / nu) * (1.0 + b_pp * L ** (-theta_exp / nu))

# ── Collect results across system sizes ──────────────────────────────────────
colors       = create_sequential_colors(len(Ls)+1)
peak_vals    = []   # max susceptibility for each L
peak_lambdas = []   # lambda at peak for each L
lam_vals     = []   # lambda values for each L
mag_vals     = []
peak_idxs    = []

fig_07,  (ax_op, ax_sus)  = plt.subplots(1, 2,
                         figsize=(FIGWIDTH, fig_height),
                         constrained_layout=True)

for color, l in zip(colors, Ls):

    sites = [l // 2 - 1, l // 2]
    sites = [l // 2]

    base_filename = (
        f"{model_name}_L_{l}"
        f"_lambda_1_{lam1_min}-{lam1_max}"
        f"_lambda_2_{lam2_min}-{lam2_max}"
        f"_npoints_{n1}x{n2}_chi_{chi}_eps_{c1}"
    )

    describe_manifest(path_to_tensor, base_filename)

    result = load_gstates(
        path_to_tensor = path_to_tensor,
        base_filename  = base_filename,
        lambda1_range  = lambda1_range,
        lambda2_range  = lambda2_range,
    )

    params       = result["params"]
    params_grid  = result["params_grid"]
    gstates_grid = result["gstates_grid"]
    n1_sub       = result["n1_sub"]
    n2_sub       = result["n2_sub"]
    l            = result["l"]

    gstates = [sanitize_state(s) for row in gstates_grid for s in row]

    if direction == "h":
        d_lambda = abs(params_grid[0, 0, 0] - params_grid[0, 1, 0]) if n1_sub > 1 else 1.0
    elif direction == "v":
        d_lambda = abs(params_grid[0, 0, 1] - params_grid[0, 1, 1]) if n2_sub > 1 else 1.0
    print(f"L={l}  d_lambda={d_lambda:.6f}")

    # ── RDMs ─────────────────────────────────────────────────────────────────
    rdms = gstates_to_rdms_matrix_qs_mps(
        gstates, sites=sites, shape=(n1_sub, n2_sub), generalized=True
    )
    print(f"rdms shape: {rdms.shape}")

    # rdms_red = rdms[:,idxi:idxf,:,:]
    # grad_g = phases_vfield(rdms_red, scale=1, grad=True,
    #                             fidelity=None, log_g=False)

    # plt.plot(grad_g)
    # plt.show()

    print(rdms[:,idxi,:,:])
    print(rdms[:,idxf,:,:])

    # ── Build observable M ───────────────────────────────────────────────────
    obs_eval, obs_ev, obs, rdms_flat = constructing_order_parameter(
        rdms, idxi=idxi, idxf=idxf, theta=theta
    )
    print(f"Observable eigenvalues: {obs_eval}")
    print(SparsePauliOp.from_operator(obs))

    # ── λ axis for the flat rdms (idxi..idxf window) ─────────────────────────
    lambdas_full   = np.linspace(lambda2_i, lambda2_f, n2_sub)
    lambdas_window = lambdas_full[idxi:idxf]          # matches rdms_flat length
    n_flat         = len(rdms_flat)
    # constructing_order_parameter may return fewer points than idxf-idxi
    # (e.g. it drops boundary points); align from the left
    lambdas_window = lambdas_window[:n_flat]

    # # ── Order parameter vec <M>(λ) ──────────────────────────────────────────────
    # obs_vec = make_obs_vec(obs_ev=obs_ev, obs_eval=obs_eval, obs_ev_idx=0)
    # sorted_components, sorted_coeffs = decompose_obs(obs=obs_vec, k_sites=len(sites))
    # print(f"sorted pauli components of the vector observable: ", sorted_components, sorted_coeffs)

    # obs = sigma_x / np.sqrt(3) + (np.eye(2) - sigma_z) / (2 * np.sqrt(3))
    # obs = sigma_x / np.sqrt(2)

    order_param = np.array([np.trace(rdm @ obs).real for rdm in rdms_flat])

    mag_vals.append(order_param)
    # ── Susceptibility vec χ(λ) = d<M>/dλ  (centered finite difference) ─────────
    susceptibility  = (order_param[2:] - order_param[:-2]) / (2.0 * d_lambda)
    susceptibility  = np.abs(susceptibility)
    lambdas_inner   = lambdas_window[1:-1]

    # ── Locate peak with sub-grid precision (parabolic interpolation) ─────────
    i_peak = np.argmax(susceptibility)
    # if 1 <= i_peak <= len(susceptibility) - 2:
    #     coeffs     = np.polyfit(lambdas_inner[i_peak-1:i_peak+2],
    #                             susceptibility[i_peak-1:i_peak+2], 2)
    #     lambda_c_L = -coeffs[1] / (2.0 * coeffs[0])
    #     chi_peak   = np.polyval(coeffs, lambda_c_L)
    # else:
    #     lambda_c_L = lambdas_inner[i_peak]
    #     chi_peak   = susceptibility[i_peak]

    lambda_c_L = lambdas_inner[i_peak]
    chi_peak   = susceptibility[i_peak]
    print(f"L={l}  lambda_c(L)={lambda_c_L:.5f}  chi_peak={chi_peak:.5f}")
    peak_vals.append(chi_peak)
    peak_lambdas.append(lambda_c_L)
    lam_vals.append(lambdas_window)
    peak_idxs.append(i_peak)

    # ── Plots per L ───────────────────────────────────────────────────────────
    ax_op.plot(lambdas_window, order_param,
               color=color, label=f"$L={l}$")
    ax_sus.plot(lambdas_inner, susceptibility,
                color=color, label=f"$L={l}$")
    # ax_sus.axvline(lambda_c_L, color=color, ls='--', lw=0.8)

# ── Dress order-parameter plot ────────────────────────────────────────────────
ax_op.set_xlabel(f"$h$")
ax_op.set_ylabel("$\\langle M \\rangle$")
ax_op.text(0.74, 0.7, "(a)")
ax_op.legend()
ax_op.grid(True, alpha=0.3)
ax_op.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])

ax_sus.set_xlabel(f"$h$")
ax_sus.set_ylabel("$\\chi_M = \\frac{\\partial \\langle M \\rangle}{\\partial h}$")
ax_sus.text(0.74, 26.5, "(b)")
ax_sus.legend()
ax_sus.grid(True, alpha=0.3)
ax_sus.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])

fig_07.savefig(f"{path_to_figures}/fig07.pdf", dpi=300, bbox_inches='tight')
# print("Saved order-parameter figure.")

# # ── Dress susceptibility plot ─────────────────────────────────────────────────
# ax_sus.set_xlabel(f"$h$ ({axis_name[1]})")
# ax_sus.set_ylabel(r"$\\chi = \\partial \\langle M \\rangle / \\partial h$")
# ax_sus.text(0.72, 26, "b)")
# ax_sus.legend(fontsize=10)
# fig_sus.tight_layout()
# fig_sus.savefig(f"{path_to_figures}/{model_name}_susceptibility_fss.png", dpi=300)
# print("Saved susceptibility figure.")

plt.close()

# Power fit function
def pow_law(x,a,b,c):
    return a + b*(x**c)

###### G CRIT AND NU EXTRAPOLATION ######
# Error on y
y_err = (lambda2_f - lambda2_i) / n2_sub
crit_vals_err = np.array([y_err] * len(Ls))

# Perform the linear fit
xdata = [1/L for L in Ls]
ydata = peak_lambdas
p_opt, co_opt = curve_fit(pow_law, xdata, ydata, sigma=crit_vals_err) # , sigma=crit_vals_err, absolute_sigma=True, maxfev=2000, bounds=([-10,-np.inf,-10],[10,np.inf,10]))

# Extract the optimal parameters
a_opt, b_opt, c_opt = p_opt

# Extract the standard errors of the parameters
perr = np.sqrt(np.diag(co_opt))
a_err, b_err, c_err = perr

# Print the results
print(f"Optimal parameters: crit g = {a_opt:.4f} ± {a_err:.4f}, amplitude = {b_opt:.4f} ± {b_err:.4f}, nu = {1/c_opt:.4f} ± {(c_err / c_opt**2):.4f}")

xfit = np.linspace(0, 1.0/min(Ls), 100)
yfit = pow_law(xfit, a_opt, b_opt, c_opt)
fig, ax = plt.subplots(1, 2,
                         figsize=(FIGWIDTH, fig_height),
                         constrained_layout=True)
ax[0].scatter(xdata, peak_lambdas, s=40, marker='o', color='k')
ax[0].errorbar(xdata, peak_lambdas, yerr=crit_vals_err, fmt='none', ecolor='k', capsize=7)
ax[0].plot(xfit, yfit, '--', color='red', linewidth=1.5)
# ax[0].fill_between(xfit, yfit - pow_law(xfit, a_opt-a_err, b_opt-b_err, c_opt-c_err), yfit + pow_law(xfit, a_opt+a_err, b_opt+b_err, c_opt+c_err), color='red', alpha=0.2)
ax[0].scatter([0], [pow_law(0, a_opt, b_opt, c_opt)], marker='x', color='darkred', s=70)
ax[0].errorbar([0], [pow_law(0, a_opt, b_opt, c_opt)], yerr=[a_err], fmt='none', ecolor='darkred', capsize=7, zorder=-1)

ax[0].text(0.009, 0.995, f"$g_c^{{\\infty}} = {pow_law(0, a_opt, b_opt, c_opt):.4f} \\pm {a_err:.4f}$")
ax[0].text(0.009, 0.985, f"$\\nu = {1/c_opt:.4f} \\pm {(c_err / c_opt**2):.4f}$")
ax[0].set_xlabel("$1/L$")
ax[0].set_ylabel(f"$h_c^L$")
ax[0].grid(True, alpha=0.3)
ax[0].text(-0.0025, 1.014, "(a)")
ax[0].set_xticks([0, 0.005, 0.01, 0.015, 0.02])
###### BETA EXTRAPOLATION ######
from scipy.interpolate import interp1d

print("M at lambda_c(L) for each L:")
m_crit = []
for L, lam, M, lc in zip(Ls, lam_vals, mag_vals, peak_lambdas):
    M_interp = float(interp1d(lam, M, kind='cubic')(lc))
    m_crit.append(M_interp)
    print(f"  L={L}: lambda_c={lc:.4f}, M={M_interp:.6f}")


def power_law(logL, log_A, bnu):
    return log_A - bnu * logL

# Perform the linear fit
xdata = np.log(Ls[2:])
ydata = np.log(m_crit[2:])
p_opt, co_opt = curve_fit(power_law, xdata, ydata) # , sigma=crit_vals_err, absolute_sigma=True, maxfev=2000, bounds=([-10,-np.inf,-10],[10,np.inf,10]))

# Extract the optimal parameters
a_opt, bnu_opt = p_opt

# Extract the standard errors of the parameters
perr = np.sqrt(np.diag(co_opt))
a_err, bnu_err = perr

# Print the results
print(f"Optimal parameters: constant = {a_opt:.4f} ± {a_err:.4f}, beta/nu = {bnu_opt:.4f} ± {bnu_err:.4f}")

xfit = np.linspace(np.log(min(Ls[2:]))-0.1, np.log(max(Ls[2:]))+0.1, 100)
yfit = power_law(xfit, a_opt, bnu_opt)
ax[1].scatter(np.log(Ls[2:]), np.log(m_crit[2:]), s=40, marker='o', color='k')
ax[1].plot(xfit, yfit, '--', color='red', linewidth=1.5)
# ax[1].fill_between(xfit, yfit - power_law(xfit, a_opt-a_err, bnu_opt-bnu_err), yfit + power_law(xfit, a_opt+a_err, bnu_opt+bnu_err), color='red', alpha=0.2)
ax[1].text(4.7, -0.93, f"$\\beta/\\nu = {bnu_opt:.4f} \\pm {bnu_err:.4f}$")
ax[1].set_xlabel("$\\log(L)$")
ax[1].set_ylabel(f"$\\log(M(h_c^L))$")
ax[1].grid(True, alpha=0.3)
ax[1].text(4.31, -0.914, "(b)")

plt.show()
plt.savefig(f"{path_to_figures}/{model_name}_fss_critical_extrapolation.png", dpi=300, bbox_inches='tight')
fig.savefig(f"{path_to_figures}/fig08.pdf", dpi=300)