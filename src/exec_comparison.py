import numpy as np
import matplotlib.pyplot as plt
import types

from qs_mps.mps_class import MPS
from qs_mps.applications.ISING.utils import discrete_fidelity_susceptibility

from qphaset.phases import gstates_to_rdms_matrix_qs_mps, sanitize_state
from qphaset.fidelity import uhlmann_fidelity

from qupytex_io import load_gstates, describe_manifest

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.titlesize": 9, "lines.linewidth": 1.5,
})

# ── Model config ──────────────────────────────────────────────────────────────
model_name = "ANNNI"
l   = 20
n   = 30
chi = 100
c1  = 1e-3
d   = 2

model_name = "Cluster"
l   = 20
n   = 30
chi = 100
c1  = 1e-3
d   = 2

model_name = "Rydberg"
l   = 20
n   = 30
chi = 100
c1  = 1e-3
d   = 2

model_name = "tjv"
l   = 20
n   = 30
chi = 100
c1  = 1e-3
d   = 3

# ── Optional: restrict to a sub-region ───────────────────────────────────────
lambda1_range = None
lambda2_range = None

# ── Device ────────────────────────────────────────────────────────────────────
# device = 'pc'
device = 'ngt'

if device == 'pc':
    device_path = "D:/code"
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

# ── Reconstruct base filename ─────────────────────────────────────────────────
# ANNNI 
lambda1_i, lambda1_f      = 0.01, 1.5 
lambda2_i, lambda2_f      = lambda1_f, lambda1_i # reverse the indices

# ANNNI zoom on floating phase
lambda1_i, lambda1_f      = 0.5, 0.8 
lambda2_i, lambda2_f      = 0.01, 1.5

# Cluster
lambda1_i, lambda1_f      = 0.5, 1.5
lambda2_i, lambda2_f      = lambda1_f, lambda1_i # reverse the indices

# Rydberg
lambda1_i, lambda1_f      = 1, 3
lambda2_i, lambda2_f      = lambda1_f, lambda1_i # reverse the indices

# tjv
lambda1_i, lambda1_f      = 0.1, 5
lambda2_i, lambda2_f      = lambda1_f, lambda1_i # reverse the indices

# tjv zoom (which phase is this?)
lambda1_i, lambda1_f      = 0.01, 2
lambda2_i, lambda2_f      = 4, 0.01 # reverse the indices


params_tmp    = np.linspace(lambda1_i, lambda1_f, n), np.linspace(lambda2_i, lambda2_f, n)
params_tmp = map(lambda m: m.flatten(), np.meshgrid(*params_tmp, indexing='xy'))
params_tmp = np.stack(tuple(params_tmp)).T
pe         = np.concatenate([np.min(params_tmp, axis=0), np.max(params_tmp, axis=0)])
pe         = tuple(pe[[0, 2, 1, 3]])

base_filename = (
    f"{model_name}_L_{l}"
    f"_lambda_1_{pe[2]}-{pe[3]}"
    f"_lambda_2_{pe[0]}-{pe[1]}"
    f"_npoints_{n}x{n}_chi_{chi}_eps_{c1}"
)

describe_manifest(path_to_tensor, base_filename)

# ── Load ──────────────────────────────────────────────────────────────────────
result = load_gstates(
    path_to_tensor = path_to_tensor,
    base_filename  = base_filename,
    lambda1_range  = lambda1_range,
    lambda2_range  = lambda2_range,
)

params       = result["params"]
params_grid  = result["params_grid"]
gstates_grid = result["gstates_grid"]
n_sub        = result["n_sub"]
m_sub        = result["m_sub"]
l            = result["l"]

gstates = [s for row in gstates_grid for s in row]
gstates = [sanitize_state(s) for s in gstates]
if isinstance(gstates[0], (types.BuiltinFunctionType, types.BuiltinMethodType)):
    gstates = [gs() for gs in gstates]

params_extent = np.concatenate([np.min(params, axis=0), np.max(params, axis=0)])
params_extent = tuple(params_extent[[0, 2, 1, 3]])

a = abs(params_grid[0, 0, 0] - params_grid[0, 1, 0]) if m_sub > 1 else 1.0

# ── Sites ─────────────────────────────────────────────────────────────────────
sites = [(l // 2) - 1, l // 2, (l // 2) + 1]

# ── RDM-based fidelity susceptibility ────────────────────────────────────────
rdms = gstates_to_rdms_matrix_qs_mps(gstates, sites=sites, generalized=True)

fidelity_rdms = []
for i in range(n_sub):
    row = []
    for j in range(m_sub - 1):
        row.append(uhlmann_fidelity(rdms[i, j], rdms[i, j + 1]))
    fidelity_rdms.append(row)

dfss_rdms = [discrete_fidelity_susceptibility(fid=row, a=a) for row in fidelity_rdms]

# ── Global fidelity susceptibility ───────────────────────────────────────────
fidelity = []
for i in range(n_sub):
    row = []
    for j in range(m_sub - 1):
        mps = MPS(L=l, d=d, model=model_name, chi=chi,
                  h=None, eps=c1, J=None, bc='obc')
        mps.sites         = gstates[i * m_sub + j]
        mps.ancilla_sites = gstates[i * m_sub + j + 1]
        row.append(mps._compute_norm(site=1, mixed=True).copy())
    fidelity.append(row)

dfss = [discrete_fidelity_susceptibility(fid=row, a=a) for row in fidelity]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

im0 = ax[0].matshow(np.asarray(dfss_rdms), origin='lower',
                    extent=params_extent, aspect='auto')
ax[0].set_title("reduced fidelity susceptibility")
ax[0].set_xlabel(axis_name[0])
ax[0].set_ylabel(axis_name[1])
fig.colorbar(im0, ax=ax[0])

im1 = ax[1].matshow(np.asarray(dfss), origin='lower',
                    extent=params_extent, aspect='auto')
ax[1].set_title("global fidelity susceptibility")
ax[1].set_xlabel(axis_name[0])
ax[1].set_ylabel(axis_name[1])
fig.colorbar(im1, ax=ax[1])

plt.tight_layout()
out = f"{path_to_figures}/{model_name}_L_{l}_{n_sub}x{m_sub}_comparison_no_convol.png"
fig.savefig(out, dpi=300)
print(f"Saved → {out}")
