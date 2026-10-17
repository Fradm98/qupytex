"""
N-dependence of the reduced fidelity susceptibility (RFS) and of block reduced density matrices
(RDMs) of the ANNNI chain at fixed h: a test of how fast a block of l_A sites stops feeling the
system size (power law in the critical floating phase, exponential in a gapped phase).

  H = -sum X_i X_{i+1} + k sum X_i X_{i+2} - h sum Z_i      (qs-mps ANNNI MPO, J = 1, eps = 0)

For each h in --hs and each even N in --Ns (default 40, 48, ..., 160):
  * cold-start two-site DMRG at h   (symmetric start |up z>, fixed seed), state s1
  * warm-start DMRG at h + dh from s1, state s2
  * RFS   chi_A(l_A) = (1 - F_A(rho_A(s1), rho_A(s2))) / dh^2   for the central blocks l_A in --blocks
  * global chi_F = -ln F / dh^2
  * the block RDMs of s1 for l_A in --rdm-blocks (saved, so that fidelities between different N,
    e.g. F_A(N, N + 8), can be computed offline)
No order-parameter observables are used. The only operator measured is <sigma^x> on the central site,
as a DIAGNOSTIC that the state did not fall into a symmetry-broken sector.

Central blocks are placed at sites N/2 - l_A/2 ... N/2 + l_A/2 - 1. The ANNNI floating phase is
modulated with a wave vector close to pi/2 (period ~4), so the position of the block modulo 4
relative to the boundary matters: with N a multiple of 8, N/2 = 0 mod 4 for every N.

Resumable: the JSON is written after every (h, N) and finished entries are skipped on restart.

Usage (from src/), e.g. one process per field:
  OMP_NUM_THREADS=4 python exec_rdm_vs_N.py --hs 0.38 --project ~/projects/2_ANNNI
  OMP_NUM_THREADS=4 python exec_rdm_vs_N.py --hs 0.50 --project ~/projects/2_ANNNI
"""
import os
import json
import time
import inspect
import argparse
import numpy as np

from qs_mps.mps_class import MPS

from qphaset.annni import model_annni_qs_mps
from qphaset import mps_fidelity as mf

# Older qs-mps releases accept (and silently ignore) conv_criterion through **kwargs.
if "conv_criterion" not in inspect.signature(MPS.DMRG).parameters:
    import qs_mps
    raise RuntimeError(
        f"qs_mps at {qs_mps.__file__} is too old: DMRG has no conv_criterion option. "
        "Install the repo checkout at commit 09734f3 or later: pip install -e /path/to/qs-mps")

parser = argparse.ArgumentParser()
parser.add_argument("--hs", type=float, nargs="+", default=[0.38, 0.50])
parser.add_argument("--Ns", type=int, nargs="+", default=list(range(40, 161, 8)))
parser.add_argument("--kappa", type=float, default=0.75)
parser.add_argument("--dh", type=float, default=2e-3)
parser.add_argument("--chi", type=int, default=64)
parser.add_argument("--blocks", type=int, nargs="+", default=[2, 4, 8], help="blocks for chi_A")
parser.add_argument("--rdm-blocks", type=int, nargs="+", default=[2, 4], help="blocks whose RDM is saved")
parser.add_argument("--n-sweeps", type=int, default=12)
parser.add_argument("--conv-tol", type=float, default=1e-10)
parser.add_argument("--device", choices=["pc", "ngt"], default="pc")
parser.add_argument("--project", default=None, help="override the project folder")
args = parser.parse_args()

if args.device == "pc":
    device_path = "C:/Users/HP/Desktop"
elif args.device == "ngt":
    device_path = "/eos/user/f/fdimarca"
project = os.path.expanduser(args.project) if args.project else f"{device_path}/projects/2_ANNNI"
k, chi, dh = args.kappa, args.chi, args.dh
assert all(N % 2 == 0 for N in args.Ns), "use even N"
# the fields are part of the file name, so that one process per field can share a project folder
tag = f"rdm_vs_N_k_{k:g}_eps_0_chi_{chi}_dh_{dh:g}_h_{'-'.join(f'{h:g}' for h in args.hs)}"
path_to_data = f"{project}/results/data/annni_rdm_vs_N"
os.makedirs(path_to_data, exist_ok=True)

dmrg_params = {
    'n_sweeps': args.n_sweeps, 'DMRG2': True, 'conv_criterion': 'mid_chain', 'conv_tol': args.conv_tol,
    'eigsh_params': {'ncv': 40, 'tol': 1e-13, 'maxiter': 5000},
}
sx_broken = 1e-3


def ground_state(h, N, init_sites):
    model = model_annni_qs_mps(k, h, chi=chi, l=N, j1=1.0, c1=0.0, d=2)
    if init_sites is None:
        up = np.array([1, 0], dtype=complex).reshape((1, 2, 1))          # parity-symmetric |up z>
        model.sites = [up.copy() for _ in range(N)]
        model.enlarge_chi(noise_std=1e-6, seed=7)
    else:
        model.sites = [t.copy() for t in init_sites]
    model.canonical_form()
    _, entropy, _, _ = model.DMRG(
        trunc_chi=True, trunc_tol=False, where=N // 2,
        n_sweeps=dmrg_params['n_sweeps'], eigsh_params=dmrg_params['eigsh_params'],
        conv_criterion=dmrg_params['conv_criterion'], conv_tol=dmrg_params['conv_tol'],
        DMRG2=dmrg_params['DMRG2'])
    e = np.real(np.array(model.mid_chain_energies))
    sites = [np.array(t) for t in model.sites]
    return sites, dict(E=float(e[-1]), dE_last=float(abs(e[-1] - e[-2])) if len(e) > 1 else None,
                       sweeps=len(e), S_mid=float(entropy[-1]) if len(entropy) else None)


def block_rdm(s, N, l):
    start = N // 2 - l // 2
    M = mf.block_tensor(mf.mixed_canonical(s, start, start + l), start, start + l)
    return M @ M.conj().T


json_out = f"{path_to_data}/{tag}.json"
res = dict(kappa=k, chi=chi, dh=dh, eps=0.0, blocks=args.blocks, rdm_blocks=args.rdm_blocks,
           dmrg_params=dmrg_params, entries={})
if os.path.exists(json_out):
    res = json.load(open(json_out))
    print(f"Resuming {tag}: {len(res['entries'])} entries done", flush=True)

print(f"── {tag} ──", flush=True)
for h in args.hs:
    for N in args.Ns:
        key = f"h={h:.5f}_N={N}"
        if key in res["entries"]:
            continue
        t0 = time.time()
        s1, i1 = ground_state(h, N, None)
        s2, i2 = ground_state(h + dh, N, s1)
        ov, n1, n2 = mf.global_overlap(s1, s2, N, chi)
        ent = dict(h=h, N=N, dh=dh, info_h=i1, info_hdh=i2,
                   sx_mid=mf.mid_magnetization(s1, N)[0],            # diagnostic only
                   chi_F=float(-np.log(abs(ov) ** 2 / (n1 * n2)) / dh ** 2), chi_A={}, rdm={})
        ent["broken"] = abs(ent["sx_mid"]) > sx_broken
        for l in args.blocks:
            start = N // 2 - l // 2
            Fr, _ = mf.reduced_fidelity(s1, s2, start, start + l)
            ent["chi_A"][str(l)] = float((1 - Fr) / dh ** 2)
        for l in args.rdm_blocks:
            r = block_rdm(s1, N, l)
            ent["rdm"][str(l)] = dict(re=r.real.tolist(), im=r.imag.tolist())
        ent["time"] = time.time() - t0
        res["entries"][key] = ent
        print(f"  h={h:.3f} N={N:3d}  S_mid={i1['S_mid']:.4f}  sweeps={i1['sweeps']},{i2['sweeps']}  "
              f"dE={i1['dE_last']:.1e}  chi_F/N={ent['chi_F'] / N:.4g}  "
              + "  ".join(f"chi_A({l})={ent['chi_A'][str(l)]:.4g}" for l in args.blocks)
              + f"  t={ent['time']:.0f}s{'  BROKEN' if ent['broken'] else ''}", flush=True)
        with open(json_out + ".tmp", "w") as f:
            json.dump(res, f)
        os.replace(json_out + ".tmp", json_out)
print(f"Saved → {json_out}", flush=True)
