import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit.quantum_info import SparsePauliOp

from qs_mps.utils import create_sequential_colors

from qphaset.phases import (gstates_to_rdms_matrix_qs_mps, constructing_order_parameter, make_obs_vec, phases_vfield, decompose_obs,
                             sanitize_state)

from qupytex_io import load_gstates, describe_manifest

# -- Pauli matrices ---
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# ── Model config ──────────────────────────────────────────────────────────────
model_name = "ANNNI"
Ls   = [50, 70]         # system sizes for FSS
n1   = 1
n2   = 201
chi  = 50
c1   = 1e-4

# ── Optional: restrict to a sub-region ───────────────────────────────────────
lambda1_range = None
lambda2_range = None

# ── Device ────────────────────────────────────────────────────────────────────
device = 'pc'
# device = 'ngt'

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
lambda2_i, lambda2_f = 0.9,   1.1
direction = "v"

lam1_min, lam1_max = min(lambda1_i, lambda1_f), max(lambda1_i, lambda1_f)
lam2_min, lam2_max = min(lambda2_i, lambda2_f), max(lambda2_i, lambda2_f)

# ── Observable construction parameters ───────────────────────────────────────
idxi  = 50
idxf  = -1
theta = -np.pi/2

# ── FSS fit model ─────────────────────────────────────────────────────────────
# max_h { d<M>/dh } = a'' * L^(1/nu) * (1 + b'' * L^(-theta_exp/nu))
def fss_model(L, a_pp, nu, b_pp, theta_exp):
    return a_pp * L ** (1.0 / nu) * (1.0 + b_pp * L ** (-theta_exp / nu))

# ── Collect results across system sizes ──────────────────────────────────────
colors       = create_sequential_colors(len(Ls))
peak_vals    = []   # max susceptibility for each L
peak_lambdas = []   # lambda at peak for each L
mag_vals     = []
peak_idxs    = []

fig_op,  ax_op  = plt.subplots(figsize=(7, 4))
fig_sus, ax_sus = plt.subplots(figsize=(7, 4))

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

    # # ── Order parameter  <M>(λ) ──────────────────────────────────────────────
    #order_param = np.array([np.trace(rdm @ obs).real for rdm in rdms_flat])

    # # ── Susceptibility  χ(λ) = d<M>/dλ  (centered finite difference) ─────────
    # susceptibility  = (order_param[2:] - order_param[:-2]) / (2.0 * d_lambda)
    # susceptibility  = np.abs(susceptibility)
    # lambdas_inner   = lambdas_window[1:-1]


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
    peak_idxs.append(i_peak)

    # ── Plots per L ───────────────────────────────────────────────────────────
    ax_op.plot(lambdas_window, order_param,
               color=color, label=f"$L={l}$")
    ax_sus.plot(lambdas_inner, susceptibility,
                color=color, label=f"$L={l}$")
    ax_sus.axvline(lambda_c_L, color=color, ls='--', lw=0.8)

# ── Dress order-parameter plot ────────────────────────────────────────────────
ax_op.set_xlabel(f"$\\lambda$ ({axis_name[1]})", fontsize=13)
ax_op.set_ylabel(r"$\langle M \rangle$", fontsize=13)
ax_op.set_title(f"{model_name} — order parameter", fontsize=13)
ax_op.legend(fontsize=10)
fig_op.tight_layout()
fig_op.savefig(f"{path_to_figures}/{model_name}_order_parameter_fss.png", dpi=300)
print("Saved order-parameter figure.")

# ── Dress susceptibility plot ─────────────────────────────────────────────────
ax_sus.set_xlabel(f"$\\lambda$ ({axis_name[1]})", fontsize=13)
ax_sus.set_ylabel(r"$\chi = \partial\langle M\rangle / \partial\lambda$", fontsize=13)
ax_sus.set_title(f"{model_name} — magnetic susceptibility", fontsize=13)
ax_sus.legend(fontsize=10)
fig_sus.tight_layout()
fig_sus.savefig(f"{path_to_figures}/{model_name}_susceptibility_fss.png", dpi=300)
print("Saved susceptibility figure.")

plt.close()
plt.close()


# # ── FSS fit: chi_peak ~ a'' L^(1/nu) (1 + b'' L^(-theta/nu)) ────────────────
# Ls_arr    = np.array(Ls, dtype=float)
# peaks_arr = np.array(peak_vals, dtype=float)

# if len(Ls) >= 4:
#     # Ising 2D priors: nu=1, theta~2
#     p0 = [peaks_arr[0] / Ls_arr[0], 1.0, 0.1, 2.0]
#     try:
#         popt, pcov = curve_fit(fss_model, Ls_arr, peaks_arr, p0=p0,
#                                maxfev=10_000)
#         perr = np.sqrt(np.diag(pcov))
#         a_pp, nu_fit, b_pp, theta_fit = popt
#         print("\n── FSS fit results ──────────────────────────────")
#         print(f"  a''    = {a_pp:.4f}  ±  {perr[0]:.4f}")
#         print(f"  nu     = {nu_fit:.4f}  ±  {perr[1]:.4f}   (Ising 2D: 1.0)")
#         print(f"  b''    = {b_pp:.4f}  ±  {perr[2]:.4f}")
#         print(f"  theta  = {theta_fit:.4f}  ±  {perr[3]:.4f}")

#         # ── FSS plot: chi_peak vs L ───────────────────────────────────────────
#         fig_fss, ax_fss = plt.subplots(figsize=(5, 4))
#         L_fine = np.linspace(Ls_arr[0] * 0.9, Ls_arr[-1] * 1.1, 200)
#         ax_fss.plot(Ls_arr, peaks_arr, 'o', color='steelblue',
#                     ms=7, label="data")
#         ax_fss.plot(L_fine, fss_model(L_fine, *popt), '-', color='tomato',
#                     label=(rf"fit: $\nu={nu_fit:.3f}\pm{perr[1]:.3f}$,"
#                            rf" $\theta={theta_fit:.3f}\pm{perr[3]:.3f}$"))
#         ax_fss.set_xlabel("$L$", fontsize=13)
#         ax_fss.set_ylabel(r"$\max_\lambda\,\chi(L)$", fontsize=13)
#         ax_fss.set_title(f"{model_name} — FSS of peak susceptibility", fontsize=12)
#         ax_fss.legend(fontsize=9)
#         fig_fss.tight_layout()
#         fig_fss.savefig(f"{path_to_figures}/{model_name}_fss_fit.png", dpi=300)
#         print("Saved FSS fit figure.")

#         # ── lambda_c extrapolation plot ───────────────────────────────────────
#         # lambda_c(L) - lambda_c(inf) ~ L^{-1/nu}  =>  plot vs L^{-1/nu_fit}
#         lc_arr = np.array(peak_lambdas)
#         x_fss  = Ls_arr ** (-1.0 / nu_fit)
#         coeffs_lc = np.polyfit(x_fss, lc_arr, 1)
#         lambda_c_inf = coeffs_lc[1]
#         print(f"\n  lambda_c(inf) ~ {lambda_c_inf:.5f}  (linear extrap. in L^{{-1/nu}})")

#         fig_lc, ax_lc = plt.subplots(figsize=(5, 4))
#         ax_lc.plot(x_fss, lc_arr, 'o', color='steelblue', ms=7, label="data")
#         x_fit = np.linspace(0, x_fss.max() * 1.05, 100)
#         ax_lc.plot(x_fit, np.polyval(coeffs_lc, x_fit), '--', color='tomato',
#                    label=rf"extrap. $\lambda_c(\infty)={lambda_c_inf:.4f}$")
#         ax_lc.set_xlabel(rf"$L^{{-1/\nu}}$  ($\nu={nu_fit:.3f}$)", fontsize=13)
#         ax_lc.set_ylabel(r"$\lambda_c(L)$", fontsize=13)
#         ax_lc.set_title(f"{model_name} — critical point extrapolation", fontsize=12)
#         ax_lc.legend(fontsize=9)
#         fig_lc.tight_layout()
#         fig_lc.savefig(f"{path_to_figures}/{model_name}_lambda_c_extrap.png", dpi=300)
#         print("Saved lambda_c extrapolation figure.")

#     except RuntimeError as e:
#         print(f"FSS fit did not converge: {e}")
#         print("  Try providing better initial guesses in p0.")
# else:
#     print(f"\nOnly {len(Ls)} system sizes — need at least 4 for a reliable FSS fit.")
#     print("Peak susceptibilities:", dict(zip([int(l) for l in Ls_arr], peaks_arr.tolist())))

# plt.show()

# # Your given data
# Ls_inv = [1/L for L in Ls]

# Power fit function
def pow_law(x,a,b,c):
    return a + b*(x**c)

###### G CRIT AND NU EXTRAPOLATION ######
# Error on y
y_err = (lambda2_f - lambda2_i) / n2_sub
crit_vals_err = np.array([y_err] * len(Ls))

# Perform the linear fit
# xdata = Ls
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

# Theoretical and fitted critical points
h_th = 1
h_c = pow_law(x=1e-6, a=p_opt[0], b=p_opt[1], c=p_opt[2])
print(f"exp value of g_critical: {h_c}")


pppp, ccc = curve_fit(pow_law, Ls, peak_vals)
x_fit = np.linspace(Ls[0], Ls[-1], 50)
y_fit = pow_law(x_fit, *pppp)

# print("exponent: ", pppp[2], np.sqrt(np.diag(ccc))[2])
# fig, ax = plt.subplots(1,2)
# ax[0].scatter(Ls, peak_vals, s=10, marker='+')
# ax[0].plot(x_fit, y_fit, '--', color='red')

# ax[1].scatter(Ls, np.asarray(peak_vals)/np.asarray(Ls), s=10, marker='+')
# ax[1].plot(x_fit, y_fit/x_fit, '--', color='red')
# plt.show()

fig, ax = plt.subplots(1,2)
ax[0].scatter(Ls, peak_vals/pppp[0], s=10, marker='+')
ax[0].plot(x_fit, y_fit/pppp[0], '--', color='red')

ax[1].scatter(Ls, np.asarray(peak_vals)/np.asarray(Ls), s=10, marker='+')
ax[1].plot(x_fit, y_fit/x_fit, '--', color='red')
plt.show()

###### BETA EXTRAPOLATION ######
from scipy.interpolate import UnivariateSpline
# M_at_crit = []
# for order_param in mag_vals:
#     # interpolate <M>(lambda) for this L
#     spl = UnivariateSpline(lambdas_window, order_param, s=0, k=3)
#     M_at_crit.append(spl(a_opt).item())  # evaluate at lambda_c(inf)

M_at_crit = []
for order_param, x_peak in zip(mag_vals, peak_idxs):
    M_at_crit.append(order_param[x_peak])


xdata = [1/L for L in Ls]
ydata = np.array(M_at_crit)

p_opt, co_opt = curve_fit(pow_law, xdata, ydata, maxfev=2000, bounds=([-np.inf,-np.inf,0],[np.inf,np.inf,1]))

# Extract the optimal parameters
a_opt, b_opt, c_opt = p_opt

# Extract the standard errors of the parameters
perr = np.sqrt(np.diag(co_opt))
a_err, b_err, c_err = perr

# Print the results
print(f"Optimal parameters: const = {a_opt:.4f} ± {a_err:.4f}, amplitude = {b_opt:.4f} ± {b_err:.4f}, beta/nu = {c_opt:.4f} ± {c_err:.4f}")

###### GAMMA EXTRAPOLATION ######
xdata = peak_lambdas - a_opt
xdata = [1/x for x in xdata]
ydata = peak_vals

p_opt, co_opt = curve_fit(pow_law, xdata, ydata) # , sigma=crit_vals_err, absolute_sigma=True, maxfev=2000, bounds=([-10,-np.inf,-10],[10,np.inf,10]))

# Extract the optimal parameters
a_opt, b_opt, c_opt = p_opt

# Extract the standard errors of the parameters
perr = np.sqrt(np.diag(co_opt))
a_err, b_err, c_err = perr

# Print the results
print(f"Optimal parameters: const = {a_opt:.4f} ± {a_err:.4f}, amplitude = {b_opt:.4f} ± {b_err:.4f}, gamma = {c_opt:.4f} ± {c_err:.4f}")

# def plotting(p_opt, perr, xdata, ydata, crit_vals_err, colors):
#     # Data for the fit line and error bounds
#     xs = np.linspace(0, 0.3)
#     y_fit = pow_law(xs, p_opt[0], p_opt[1], -p_opt[2])
#     # error over nu
#     nu_err = (1/(p_opt[2]-perr[2]) - 1/(p_opt[2]+perr[2]))/2
#     y_err_plus = pow_law(xs, p_opt[0] + perr[0], p_opt[1] + perr[1], -p_opt[2] + nu_err)
#     y_err_minus = pow_law(xs, p_opt[0] - perr[0], p_opt[1] - perr[1], -p_opt[2] - nu_err)

#     # Plotting
#     fig, ax = plt.subplots()
#     ax.plot(xs, y_fit, color="k", linewidth=1, linestyle='--', label='Fit: $aL^{-1/\\nu} + b$', zorder=1)
#     # ax.plot(xs, y_fit, color=colors[0], linewidth=1, label='Fit: $aL^{-1/\\nu} + b$')
#     # ax.fill_between(xs, y_err_minus, y_err_plus, color=colors[1], alpha=0.5, label="Fit Uncertainty")
#     ax.errorbar(xdata, ydata, yerr=crit_vals_err, fmt='o', elinewidth=1, capsize=8, markersize=8, mfc="white", mec="red", ecolor="red", label="Data", zorder=2)

#     ax.scatter([0], [h_th], marker='x', color=colors[3], s=70, label="$g_c^{th}$",zorder=4)
#     ax.errorbar([0], [h_c], yerr=perr[0], fmt='o', elinewidth=1, capsize=8, markersize=8, mfc="white", color=colors[4], label="$g_c^{fit}$", zorder=3)
#     ax.set_xlabel("$1/L$", fontsize=14)
#     ax.set_ylabel("electric coupling $(g)$", fontsize=14)
#     # ax.grid(True, alpha=0.5)
#     ax.legend(loc='upper right', fontsize=14)
#     ax.set_ylim((0.42,0.64))
    
#     # Inset plot
#     inset_ax = inset_axes(ax, width="42%", height="45%", loc="lower left", bbox_to_anchor=(0.1, 0.09, 0.95, 0.9), bbox_transform=ax.transAxes)
#     colors = ['#99d98c', '#52b69a', '#168aad', '#1e6091', '#d9ed92']
#     colors = ["#4688CE","#9B4DB7","#DC4563"]
#     colors = ["#FC4778","#BB4BA2","#7A4ECB","#3952F5"]
#     colors = ["#DC4563","#AD5A85","#90679A", "#4688CE"]


#     i = 0
#     Ls = [4,5,6,7]
#     for L, chi, lx, ly in zip(Ls, chis, lxs, lys):
#         l = L
#         string = np.load(f"{parent_path}/results/thooft/thooft_string_first_moment_{lx}-{ly}_horizontal_Z2_dual_direct_lattice_{l}x{L}_{sector}_bc_{boundcond}_{cx}-{cy}_h_{lambda2_i}-{lambda2_f}_delta_{n2_sub}_chi_{chi}.npy")
#         d_string_dh = np.gradient(string, interval)
#         g_max = round(interval[np.argmax(d_string_dh)], 4)
#         inset_ax.plot(interval, d_string_dh, color=colors[i], label=f"${l}$x${L}$", zorder=1)
#         inset_ax.scatter(g_max, np.max(d_string_dh), marker='o', facecolors="white", edgecolors="red", zorder=2)
#         i += 1

#     # inset_ax.grid(True, alpha=0.5)
#     inset_ax.legend(fontsize=10, loc='upper right')
#     inset_ax.set_xlabel("electric coupling $(g)$",fontsize=14)
#     inset_ax.set_ylabel("$d\langle M \\rangle / dg$", fontsize=14)
#     inset_ax.tick_params(axis='both', which='major', labelsize=8)
#     inset_ax.tick_params(axis='both', which='minor', labelsize=8)

#     # Save the plot
#     plt.savefig(f"{parent_path}/figures/magnetization/critical_point_Z2_dual_L_{Ls}_{sector}_bc_{boundcond}_None-None_h_{lambda2_i}-{lambda2_f}_delta_{n2_sub}_facecolor_white.png")

#     plt.show()