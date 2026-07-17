import numpy as np
from functools import partial
import os

from qphaset.annni import model_annni_qs_mps
from qphaset.cluster import model_cluster_qs_mps
from qphaset.rydberg import model_rydberg_qs_mps
from qphaset.tjv_model import model_tjv_qs_mps
from qphaset.models import drmg_gstate_qs_mps
from qphaset.run import run_model

from qupytex_io import save_gstates

# ── Model config ──────────────────────────────────────────────────────────────

model_name = "ANNNI"
l = 30
d = 2 # physical local dimension

a = 0.005
Jz = None
k_study = 0.01
lam1_i, lam1_f = k_study - 2*a, k_study + 2*a 
lam1_i, lam1_f = k_study - a, k_study + a 
params = np.linspace(lam1_i, lam1_f,round((lam1_f - lam1_i) / a) + 1), np.linspace(0.95,1.05,round((1.2 - 0.8) / a) + 1) # upside-down
n1 = len(params[0])
n2 = len(params[1])
# params = np.linspace(0.001, 0.001, n1), np.linspace(0.1, 1.2, n2) # upside-down

# model_name = "Cluster"
# l = 15
# d = 2 # physical local dimension
# n = 10
# params = np.linspace(0.5, 0.6, n), np.linspace(0.6, 0.5, n) # upside-down

# model_name = "Rydberg"
# l = 12
# d = 2 # physical local dimension
# n = 30
# params = np.linspace(1, 3, n), np.linspace(3, 1, n) # upside-down

# model_name = "tjv"
# l  = 12
# d  = 3   # physical local dimension
# n  = 5
# Jz = -1/2
# params = np.linspace(-4, 4, n), np.linspace(4, -4, n)

# ── Device ────────────────────────────────────────────────────────────────────
device = 'pc'
# device = 'ngt'

if device == 'pc':
    device_path = "D:/work"
elif device == 'ngt':
    device_path = "/eos/user/f/fdimarca"

# ── DMRG params ───────────────────────────────────────────────────────────────
chi = 50
c1  = 1e-2
defect = True

dmrg_params = {
    'type_shape': "rectangular",
    'trunc_params': {
        'trunc_tol': False,
        'trunc_chi': True,
    },
    'd':         d,
    'max_hours': 16 / 3600,
    'defect': defect,
}

# ── Routing ───────────────────────────────────────────────────────────────────
if model_name == 'ANNNI':
    path_to_tensor  = f"{device_path}/projects/2_ANNNI/results/data"
    model_factory   = partial(model_annni_qs_mps, c1=c1, chi=chi)
elif model_name == 'Cluster':
    path_to_tensor  = f"{device_path}/projects/3_CLUSTER/results/data"
    model_factory   = partial(model_cluster_qs_mps, c1=c1, chi=chi)
elif model_name == 'Rydberg':
    path_to_tensor  = f"{device_path}/projects/4_RYDBERG/results/data"
    model_factory   = partial(model_rydberg_qs_mps, c1=c1, chi=chi)
elif model_name == 'tjv':
    path_to_tensor  = f"{device_path}/projects/6_TJ/results/data"
    model_factory   = partial(model_tjv_qs_mps, Jz=Jz, c1=c1, chi=chi)
else:
    raise SyntaxError("Choose a valid model among 'ANNNI', 'Cluster', 'Rydberg', 'tjv'")

# ── Build parameter grid ──────────────────────────────────────────────────────
params = map(lambda m: m.flatten(), np.meshgrid(*params, indexing='xy'))
params = np.stack(tuple(params)).T    # shape (n*n, 2)

# ── Run DMRG ──────────────────────────────────────────────────────────────────
print(f"model: {model_name}, L:{l}, parameter space:{n1}×{n2}")
print(f"bond dimension: {chi}, eps: {c1}, device: {device}")

gstates, stats = run_model(
    params,
    model_factory  = partial(model_factory, l=l),
    gstate_solver  = partial(drmg_gstate_qs_mps, dmrg_params=dmrg_params),
)

# ── Build base filename (same convention as before) ───────────────────────────
params_grid = params.reshape(n1, n2, 2)
lam1_min = float(params_grid[:, :, 0].min())
lam1_max = float(params_grid[:, :, 0].max())
lam2_min = float(params_grid[:, :, 1].min())
lam2_max = float(params_grid[:, :, 1].max())

base_filename = (
    f"{model_name}_L_{l}"
    f"_lambda_1_{lam1_min}-{lam1_max}"
    f"_lambda_2_{lam2_min}-{lam2_max}"
    f"_npoints_{n1}x{n2}_chi_{chi}_eps_{c1}"
)

# ── Save in EOS-safe chunks ───────────────────────────────────────────────────
data = dict(
    params      = params,
    gstates     = gstates,
    stats       = stats,
    l           = l,
    n1          = n1,
    n2          = n2,
    d           = d,
    chi         = chi,
    model_name  = model_name,
    dmrg_params = dmrg_params,
)

save_gstates(
    path_to_tensor = path_to_tensor,
    base_filename  = base_filename,
    data           = data,
    max_file_gb    = 45,    # safely under EOS 50 GB per-file cap
)