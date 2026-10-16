"""
Fidelity-susceptibility scan of the transverse-field Ising chain (ANNNI k = 0) across h_c = 1,
for the finite-size scaling of the global and reduced fidelity susceptibilities.

For each h_bar on a grid and each L we compute the ground states at h_bar -+ dh/2
(two-site DMRG, symmetric states: eps = 0, initial |up z>, warm start from the previous
h_bar) and record
  chi_F(h_bar) = -ln F / dh^2                          (F = |<psi1|psi2>|^2)
  chi_A(h_bar) = (1 - F_A) / dh^2, centred blocks l_A  (Uhlmann, squared)
  E(h1), E(h2), S_mid, <sx>_mid, <sz>_mid, sweeps
A coarse grid is followed by a refinement around the peaks of chi_F and chi_A(l_A=1).

Usage (from src/):
  python exec_fidelity_scan.py --Ls 50 150 --chi 32
"""
import os
import json
import argparse
import numpy as np

from qphaset.annni import model_annni_qs_mps
from qphaset import mps_fidelity as mf

parser = argparse.ArgumentParser()
parser.add_argument("--Ls", type=int, nargs="+", default=[50, 100, 150, 200])
parser.add_argument("--chi", type=int, default=32)
parser.add_argument("--dh", type=float, default=1e-4)
parser.add_argument("--hmin", type=float, default=0.90)
parser.add_argument("--hmax", type=float, default=1.05)
parser.add_argument("--npts", type=int, default=51)
parser.add_argument("--nrefine", type=int, default=11, help="points in each refinement window")
parser.add_argument("--refine-range", type=float, nargs=3, default=None, metavar=("A", "B", "N"),
                    help="skip the coarse grid: only scan N points in [A, B] (saved with suffix _refine)")
parser.add_argument("--kappa", type=float, default=0.0, help="ANNNI frustration k (0: Ising)")
parser.add_argument("--hgrid", nargs="+", default=None, metavar="A:B:N",
                    help="h_bar grid as linspace segments, e.g. 0.15:0.60:46 0.25:0.266:17 (overrides hmin/hmax/npts)")
parser.add_argument("--cold-check", action="store_true",
                    help="at each point also run DMRG from the product state and keep the lower energy "
                         "(guards against a warm start following the wrong branch at level crossings)")
parser.add_argument("--project", default="C:/Users/HP/Desktop/projects/2_ANNNI")
args = parser.parse_args()

k, J, d, eps = args.kappa, 1.0, 2, 0.0
block_sizes = [1, 2, 4, 8]
# Same settings as exec_orthogonality_catastrophe.py: two-site DMRG converged on the mid-chain
# energy (the qs-mps default "standard" criterion stops after 2 sweeps).
dmrg_params = {
    'n_sweeps': 20,
    'DMRG2': True,
    'conv_criterion': 'mid_chain',
    'conv_tol': 1e-12,
    'eigsh_params': {'ncv': 40, 'tol': 1e-13, 'maxiter': 5000},
}
# Deep in the ferro phase the parity splitting (~h^L) drops below the DMRG resolution for
# L >~ 150 and the state can fall into a symmetry-broken one: such points are flagged.
sx_broken = 1e-3
path_to_data = (f"{args.project}/results/data/oc_ising/scan" if k == 0
                else f"{args.project}/results/data/annni_scan/k_{k:g}")
os.makedirs(path_to_data, exist_ok=True)


def ground_state(h, l, init_sites):
    """Two-site DMRG at field h; init_sites=None starts from the symmetric product state |up z>."""
    model = model_annni_qs_mps(k, h, chi=args.chi, l=l, j1=J, c1=eps, d=d)
    if init_sites is None:
        up = np.array([1, 0], dtype=complex).reshape((1, d, 1))
        model.sites = [up.copy() for _ in range(l)]
        model.enlarge_chi(noise_std=1e-6, seed=7)
    else:
        model.sites = [t.copy() for t in init_sites]
    model.canonical_form()
    energies, entropy, _, _ = model.DMRG(
        trunc_chi=True, trunc_tol=False, where=l // 2,
        n_sweeps=dmrg_params['n_sweeps'], eigsh_params=dmrg_params['eigsh_params'],
        conv_criterion=dmrg_params['conv_criterion'], conv_tol=dmrg_params['conv_tol'],
        DMRG2=dmrg_params['DMRG2'],
    )
    n_done = len(model.mid_chain_energies)
    if n_done >= dmrg_params['n_sweeps']:
        print(f"  WARNING: h={h}, L={l} hit the sweep cap ({n_done})")
    sites = [np.array(t) for t in model.sites]
    return sites, float(np.real(energies[-1])), float(entropy[-1]) if len(entropy) else float('nan'), n_done


def point(hb, l, init):
    h1, h2 = hb - args.dh / 2, hb + args.dh / 2
    s1, e1, S1, n1 = ground_state(h1, l, init)
    cold = None
    if args.cold_check and init is not None:
        c1_, ce1, cS1, cn1 = ground_state(h1, l, None)
        cold = ce1 - e1
        if ce1 < e1 - 1e-10 * abs(e1):
            print(f"  cold start lower by {e1 - ce1:.3e} at h={h1:.6f}: switching branch", flush=True)
            s1, e1, S1, n1 = c1_, ce1, cS1, cn1
    s2, e2, S2, n2 = ground_state(h2, l, s1)
    ov, nn1, nn2 = mf.global_overlap(s1, s2, l, args.chi)
    F = abs(ov) ** 2 / (nn1 * nn2)
    rec = dict(h_bar=hb, h1=h1, h2=h2, chi_F=-np.log(F) / args.dh ** 2, E=[e1, e2], S_mid=[S1, S2],
               m_mid=[mf.mid_magnetization(s1, l), mf.mid_magnetization(s2, l)], sweeps=[n1, n2], chi_A={},
               dE_cold=cold)
    for b in block_sizes:
        start = l // 2 - b // 2
        Fr, _ = mf.reduced_fidelity(s1, s2, start, start + b)
        rec["chi_A"][str(b)] = (1 - Fr) / args.dh ** 2
    rec["broken"] = max(abs(rec["m_mid"][0][0]), abs(rec["m_mid"][1][0])) > sx_broken
    print(f"  h_bar={hb:.5f}  chi_F={rec['chi_F']:.6e}  chi_A(1)={rec['chi_A']['1']:.6e}  "
          f"chi_A(4)={rec['chi_A']['4']:.6e}  <sx>={rec['m_mid'][0][0]:+.1e}  sweeps={n1},{n2}"
          f"{'  BROKEN' if rec['broken'] else ''}", flush=True)
    return rec, s2


def sweep(hbars, l, init=None):
    out = []
    for hb in hbars:
        rec, init = point(hb, l, init)
        out.append(rec)
    return out


for l in args.Ls:
    print(f"── L = {l}, chi = {args.chi}, dh = {args.dh} ──", flush=True)
    json_out = f"{path_to_data}/scan_eps_0_L_{l}_chi_{args.chi}_dh_{args.dh:g}.json"
    if k != 0:
        json_out = json_out.replace("scan_eps_0", f"scan_k_{k:g}_eps_0")
    if args.refine_range is not None:
        a, b, n = args.refine_range
        json_out = json_out.replace(".json", "_refine.json")
        res = dict(L=l, chi=args.chi, dh=args.dh, eps=eps, init="upz", block_sizes=block_sizes,
                   dmrg_params=dmrg_params, coarse=[], refine=[])
        for rec in sweep(np.linspace(a, b, int(n)), l):
            res["refine"].append(rec)
            with open(json_out, "w") as f:
                json.dump(res, f, indent=1)
        print(f"Saved → {json_out}", flush=True)
        continue
    if args.hgrid:
        segs = [tuple(map(float, g.split(":"))) for g in args.hgrid]
        coarse = np.unique(np.round(np.concatenate([np.linspace(a, b, int(n)) for a, b, n in segs]), 8))
    else:
        coarse = np.linspace(args.hmin, args.hmax, args.npts)
    recs = sweep(coarse, l)
    res = dict(L=l, chi=args.chi, dh=args.dh, kappa=k, eps=eps, init="upz", block_sizes=block_sizes,
               dmrg_params=dmrg_params, coarse=recs, refine=[])
    with open(json_out, "w") as f:
        json.dump(res, f, indent=1)
    if args.nrefine <= 0:
        print(f"Saved → {json_out}", flush=True)
        continue

    # Refinement windows around the coarse peaks of chi_F and chi_A(1)
    step = coarse[1] - coarse[0]
    sym = [not r["broken"] for r in recs]
    i_F = int(np.argmax([r["chi_F"] if s else -np.inf for r, s in zip(recs, sym)]))
    i_A = int(np.argmax([r["chi_A"]["1"] if s else -np.inf for r, s in zip(recs, sym)]))
    for name, i in [("chi_F", i_F), ("chi_A(1)", i_A)]:
        if i in (0, len(coarse) - 1):
            print(f"  WARNING: {name} peak on the grid edge (h_bar = {coarse[i]:.4f}); widen the window")
    peaks = {float(coarse[i_F]), float(coarse[i_A])}
    for hp in sorted(peaks):
        print(f"  refine around h_bar = {hp:.5f}", flush=True)
        fine = np.linspace(hp - step, hp + step, args.nrefine)
        res["refine"] += sweep(fine, l)
        with open(json_out, "w") as f:
            json.dump(res, f, indent=1)
    print(f"Saved → {json_out}", flush=True)
