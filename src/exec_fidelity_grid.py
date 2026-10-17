"""
Global vs reduced fidelity susceptibility along a uniform h-grid of the ANNNI chain,
from neighbouring grid points (one DMRG per point, warm start from the previous one).

  H = -sum X_i X_{i+1} + k sum X_i X_{i+2} - h sum Z_i      (qs-mps ANNNI MPO, J = 1, eps = 0)

For consecutive grid points h_j, h_{j+1} = h_j + dh we store, at h_bar = h_j + dh/2,
  chi_F = -ln F / dh^2,          F   = |<psi_j|psi_{j+1}>|^2
  chi_A = (1 - F_A) / dh^2,      F_A = Uhlmann fidelity (squared) of a centred block of l_A sites
and, per point, E, S_mid, <sigma^x>, <sigma^z> on the central site and the sweep count.

States are parity symmetric (eps = 0, start from |up z>); <sigma^x>_mid flags any leak into a
symmetry-broken state. The run is resumable: results are saved after every point together with
the last MPS, and a restart continues from where it stopped.

Default: k = 0.75 (floating phase, Beccaria et al. PRB 76, 094410 (2007): B1 = 0.2574, B2 ~ 0.43),
h in [0.27, 0.47] with dh = 2e-3, chi = 64. Noise-floor test at h = 0.35 (N = 100, 200): DMRG floor
-ln F ~ 1e-7 (global), 1 - F_A <~ 4e-10 (reduced), i.e. ~1e-3 of the signal at dh = 2e-3.

Usage (from src/), one process per N, e.g. with 4 threads each:
  OMP_NUM_THREADS=4 python exec_fidelity_grid.py --L 200 --device ngt
  OMP_NUM_THREADS=4 python exec_fidelity_grid.py --L 150 --device ngt
  OMP_NUM_THREADS=4 python exec_fidelity_grid.py --L 100 --device ngt
  OMP_NUM_THREADS=4 python exec_fidelity_grid.py --L 50  --device ngt
"""
import os
import json
import time
import argparse
import numpy as np

import inspect
from qs_mps.mps_class import MPS

from qphaset.annni import model_annni_qs_mps
from qphaset import mps_fidelity as mf

# Older qs-mps releases accept (and silently ignore) conv_criterion through **kwargs, which would
# run DMRG with the 2-sweep default criterion. Refuse to start with such a version.
if "conv_criterion" not in inspect.signature(MPS.DMRG).parameters:
    import qs_mps
    raise RuntimeError(
        f"qs_mps at {qs_mps.__file__} is too old: DMRG has no conv_criterion option. "
        "Install the repo checkout at commit 09734f3 or later: pip install -e /path/to/qs-mps")

parser = argparse.ArgumentParser()
parser.add_argument("--L", type=int, required=True)
parser.add_argument("--kappa", type=float, default=0.75)
parser.add_argument("--hmin", type=float, default=0.27)
parser.add_argument("--hmax", type=float, default=0.47)
parser.add_argument("--dh", type=float, default=2e-3, help="grid step = finite-difference step")
parser.add_argument("--chi", type=int, default=64)
parser.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
parser.add_argument("--n-sweeps", type=int, default=12)
parser.add_argument("--conv-tol", type=float, default=1e-10)
parser.add_argument("--save-states", action="store_true", help="also keep every converged MPS (.npz)")
parser.add_argument("--device", choices=["pc", "ngt"], default="pc")
parser.add_argument("--project", default=None, help="override the project folder")
args = parser.parse_args()

# ── Device / paths (same routing as the other exec scripts) ───────────────────
if args.device == "pc":
    device_path = "C:/Users/HP/Desktop"
elif args.device == "ngt":
    device_path = "/eos/user/f/fdimarca"
project = args.project or f"{device_path}/projects/2_ANNNI"
L, k, chi, dh = args.L, args.kappa, args.chi, args.dh
tag = f"grid_k_{k:g}_eps_0_L_{L}_chi_{chi}_h_{args.hmin:g}-{args.hmax:g}_dh_{dh:g}"
path_to_data = f"{project}/results/data/annni_grid/{tag}"
os.makedirs(path_to_data, exist_ok=True)
json_out = f"{path_to_data}/{tag}.json"
last_state = f"{path_to_data}/last_state.npz"

# Two-site DMRG converged on the mid-chain energy. The qs-mps default "standard" criterion stops
# after 2 sweeps; with "mid_chain" the floating phase stalls at ~1e-10 relative (alternating sweep
# directions), so the tolerance is 1e-10 with a cap of 12 sweeps.
dmrg_params = {
    'n_sweeps': args.n_sweeps,
    'DMRG2': True,
    'conv_criterion': 'mid_chain',
    'conv_tol': args.conv_tol,
    'eigsh_params': {'ncv': 40, 'tol': 1e-13, 'maxiter': 5000},
}
hs = np.round(np.arange(args.hmin, args.hmax + dh / 2, dh), 10)


def ground_state(h, init_sites):
    model = model_annni_qs_mps(k, h, chi=chi, l=L, j1=1.0, c1=0.0, d=2)
    if init_sites is None:
        up = np.array([1, 0], dtype=complex).reshape((1, 2, 1))   # parity-symmetric |up z>
        model.sites = [up.copy() for _ in range(L)]
        model.enlarge_chi(noise_std=1e-6, seed=7)
    else:
        model.sites = [t.copy() for t in init_sites]
    model.canonical_form()
    _, entropy, _, _ = model.DMRG(
        trunc_chi=True, trunc_tol=False, where=L // 2,
        n_sweeps=dmrg_params['n_sweeps'], eigsh_params=dmrg_params['eigsh_params'],
        conv_criterion=dmrg_params['conv_criterion'], conv_tol=dmrg_params['conv_tol'],
        DMRG2=dmrg_params['DMRG2'],
    )
    e = np.real(np.array(model.mid_chain_energies))
    sites = [np.array(t) for t in model.sites]
    return sites, dict(E=float(e[-1]), dE_last=float(abs(e[-1] - e[-2])) if len(e) > 1 else None,
                       sweeps=len(e), S_mid=float(entropy[-1]) if len(entropy) else None,
                       m_mid=mf.mid_magnetization(sites, L))


def save_state(path, sites):
    np.savez(path, *sites)


def load_state(path):
    with np.load(path) as z:
        return [z[f"arr_{i}"] for i in range(len(z.files))]


# ── Resume ────────────────────────────────────────────────────────────────────
res = dict(L=L, kappa=k, chi=chi, dh=dh, eps=0.0, init="upz", blocks=args.blocks,
           dmrg_params=dmrg_params, h_grid=hs.tolist(), points=[], pairs=[])
prev = None
if os.path.exists(json_out) and os.path.exists(last_state):
    with open(json_out) as f:
        res = json.load(f)
    prev = load_state(last_state)
    print(f"Resuming {tag}: {len(res['points'])} / {len(hs)} points done", flush=True)

print(f"── {tag} ──", flush=True)
for h in hs[len(res["points"]):]:
    t0 = time.time()
    s, info = ground_state(float(h), prev)
    info["h"] = float(h)
    info["time"] = time.time() - t0
    line = (f"  h={h:.4f}  E={info['E']:.12f}  S_mid={info['S_mid']:.4f}  "
            f"<sx>={info['m_mid'][0]:+.1e}  sweeps={info['sweeps']}  t={info['time']:.0f}s")
    if prev is not None:
        ov, n1, n2 = mf.global_overlap(prev, s, L, chi)
        pair = dict(h_bar=float(h - dh / 2), chi_F=float(-np.log(abs(ov) ** 2 / (n1 * n2)) / dh ** 2), chi_A={})
        for b in args.blocks:
            start = L // 2 - b // 2
            Fr, _ = mf.reduced_fidelity(prev, s, start, start + b)
            pair["chi_A"][str(b)] = float((1 - Fr) / dh ** 2)
        res["pairs"].append(pair)
        line += f"  | chi_F={pair['chi_F']:.4g}  chi_A(1)={pair['chi_A']['1']:.4g}"
    res["points"].append(info)
    print(line, flush=True)

    # Persist after every point: results, the state to resume from, optionally every state
    save_state(last_state + ".tmp.npz", s)
    os.replace(last_state + ".tmp.npz", last_state)
    if args.save_states:
        save_state(f"{path_to_data}/state_h_{h:.4f}.npz", s)
    with open(json_out + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(json_out + ".tmp", json_out)
    prev = s

print(f"Saved → {json_out}", flush=True)
