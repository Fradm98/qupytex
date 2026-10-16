"""
Anderson orthogonality catastrophe check on the transverse-field Ising chain.

H = -J sum X_i X_{i+1} - h sum Z_i   (ANNNI with k = 0, J = 1, qs-mps conventions)

For two neighbouring points h1, h2 of the usual 64-point h grid on [0.01, 1.5]
and several system sizes L (and bond dimensions chi) we compute
  * the global fidelity   F(L)   = |<psi(h1)|psi(h2)>|^2
  * the reduced fidelity  F_A(L) = Uhlmann fidelity of the RDMs on a central block A
and check whether F(L) ~ exp(-alpha L) while F_A stays constant in L.

Exact bulk rate from free fermions (Bogoliubov angle theta_k = atan2(sin k, h - cos k)):
  alpha = -(1/2pi) int_0^pi ln cos^2((theta_k(h1) - theta_k(h2)) / 2) dk  ~=  (h2 - h1)^2 / 16
(Zanardi & Paunkovic, PRE 74, 031123 (2006); Damski, PRE 87, 052131 (2013)).

Usage (from src/):
  python exec_orthogonality_catastrophe.py                                # grid points 0,1 (deep ferro), chi 16
  python exec_orthogonality_catastrophe.py --idx 41 42 --chis 32 64 128   # across h = 1
  python exec_orthogonality_catastrophe.py --idx 41 42 --merge a.json b.json   # combine runs, plot only
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from qphaset.annni import model_annni_qs_mps
from qphaset.models import generalized_k_rdm
from qphaset.fidelity import uhlmann_fidelity
from qphaset import mps_fidelity

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.titlesize": 9, "lines.linewidth": 1.5,
})

parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, nargs=2, default=[0, 1],
                    help="indices of the two h points on np.linspace(0.01, 1.5, 64)")
parser.add_argument("--h", type=float, nargs=2, default=None,
                    help="explicit (h1, h2); overrides --idx")
parser.add_argument("--eps", type=float, default=1e-2,
                    help="edge longitudinal field pinning the symmetry-broken sector")
parser.add_argument("--init", choices=["plusx", "upz"], default="plusx",
                    help="initial product state: |+x> (symmetry broken) or |up z> (parity symmetric)")
parser.add_argument("--chis", type=int, nargs="+", default=[16])
parser.add_argument("--Ls", type=int, nargs="+", default=[50, 100, 150, 200])
parser.add_argument("--project", default="C:/Users/HP/Desktop/projects/2_ANNNI")
parser.add_argument("--merge", nargs="+", default=None,
                    help="JSON files of previous runs (same h pair) to combine and plot; no DMRG")
args = parser.parse_args()

# ── Model config ──────────────────────────────────────────────────────────────
model_name = "ANNNI"
k   = 0.0          # ANNNI with k=0 -> transverse-field Ising
J   = 1.0
h_grid = np.linspace(0.01, 1.5, 64)
if args.h is None:
    hs = (float(h_grid[args.idx[0]]), float(h_grid[args.idx[1]]))
else:
    hs = tuple(args.h)
Ls  = args.Ls
chis = args.chis
d   = 2
c1  = args.eps     # edge longitudinal field pinning the symmetry-broken sector

dmrg_params = {
    'trunc_params': {'trunc_tol': False, 'trunc_chi': True},
    'n_sweeps': 20,   # two-site usually converges in ~5; hitting the cap = truncation-limited chi
    # Two-site DMRG: single-site stalls ~5e-6 above the exact energy at h ~ 1
    # (L=100) for any chi; two-site reaches 1e-11 (chi 32) / 1e-14 (chi 64).
    'DMRG2': True,
    # "standard" compares the last two sites of the same sweep and stops after
    # 2 sweeps; "mid_chain" compares the mid-site energy across sweeps.
    'conv_criterion': 'mid_chain',
    'conv_tol': 1e-12,
    'eigsh_params': {'ncv': 40, 'tol': 1e-13, 'maxiter': 5000},
}

# ── Reduced-fidelity blocks (centred) ─────────────────────────────────────────
block_sizes = [1, 2, 4, 8, 16]
max_elems   = 2 ** 25   # largest dense tensor (complex entries) built for the reduced fidelity

h1, h2 = hs
tag = f"h_{h1:.4f}-{h2:.4f}" if args.h is None else f"h_{h1:.8f}-{h2:.8f}"
if c1 != parser.get_default("eps"):
    tag += f"_eps_{c1:g}"
if args.init != parser.get_default("init"):
    tag += f"_init_{args.init}"
path_to_figures = f"{args.project}/figures"
path_to_data    = f"{args.project}/results/data/oc_ising"
os.makedirs(path_to_figures, exist_ok=True)
os.makedirs(path_to_data, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def ground_state(h, l, chi):
    model = model_annni_qs_mps(k, h, chi=chi, l=l, j1=J, c1=c1, d=d)
    if args.init == "plusx":
        site0 = np.array([1, 1], dtype=complex).reshape((1, d, 1)) / np.sqrt(2)  # |+x>: broken ferro guess
    else:
        site0 = np.array([1, 0], dtype=complex).reshape((1, d, 1))               # |up z>: parity symmetric
    model.sites = [site0.copy() for _ in range(l)]
    model.enlarge_chi(noise_std=1e-6, seed=7)
    model.canonical_form()
    energies, entropy, _, _ = model.DMRG(
        trunc_chi=dmrg_params['trunc_params']['trunc_chi'],
        trunc_tol=dmrg_params['trunc_params']['trunc_tol'],
        where=l // 2, n_sweeps=dmrg_params['n_sweeps'],
        eigsh_params=dmrg_params['eigsh_params'],
        conv_criterion=dmrg_params['conv_criterion'],
        conv_tol=dmrg_params['conv_tol'],
        DMRG2=dmrg_params['DMRG2'],
    )
    s_mid = float(entropy[-1]) if len(entropy) else float('nan')
    n_done = len(model.mid_chain_energies)
    if n_done >= dmrg_params['n_sweeps']:
        print(f"  WARNING: h={h}, L={l}, chi={chi} hit the sweep cap ({n_done}) without converging")
    return [np.array(t) for t in model.sites], float(np.real(energies[-1])), s_mid, n_done


def global_overlap(s1, s2, l, chi):
    """<s2|s1> and the norms, via the same contraction used in exec_comparison."""
    return mps_fidelity.global_overlap(s1, s2, l, chi, d=d, model_name=model_name)


def reduced_fidelity(s1, s2, start, stop):
    return mps_fidelity.reduced_fidelity(s1, s2, start, stop, max_elems=max_elems)


mid_magnetization = mps_fidelity.mid_magnetization


def alpha_exact(h1, h2):
    # dtheta_k = arg(z1 conj(z2)), z = (h - cos k) + i sin k: no cancellation for tiny dh
    def f(kk):
        c, s = np.cos(kk), np.sin(kk)
        dth = np.arctan2(s * (h2 - h1), (h1 - c) * (h2 - c) + s * s)
        return np.log1p(-np.sin(dth / 2) ** 2)
    # ln F = sum_{k>0} ln cos^2(dtheta_k/2), with L/(2 pi) modes per unit k on (0, pi).
    # Near h = 1 the integrand lives at k ~ |h - 1|, dh: split the interval there.
    kc = min(np.pi / 2, 10 * max(abs(h1 - 1), abs(h2 - 1), abs(h2 - h1)))
    I = sum(quad(f, a, b, limit=400, epsabs=0, epsrel=1e-10)[0] for a, b in [(0, kc), (kc, np.pi)])
    return -I / (2 * np.pi)


def make_plot(results, out):
    Ls_a = np.array(results["Ls"], dtype=float)
    chis = sorted(int(c) for c in results["runs"])
    alpha_th = results["alpha_exact"]
    markers = ['o', 's', '^', 'D', 'v', 'P']
    chi_best = str(max(chis))

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    xx = np.linspace(0, max(Ls_a), 100)
    for m, chi in zip(markers, chis):
        lnF = np.log(np.array(results["runs"][str(chi)]["F_global"]))
        ax[0].plot(Ls_a, -lnF, m, mfc='none', label=rf"$\chi$={chi}")
        ax[1].plot(Ls_a, -lnF / Ls_a, m + '-', mfc='none', label=rf"$\chi$={chi}")
    run = results["runs"][chi_best]
    ax[0].plot(xx, alpha_th * xx, '--', lw=1, label=rf"exact bulk $\alpha$={alpha_th:.3e}")
    ax[0].set_xlabel("L"); ax[0].set_ylabel(r"$-\ln F_{\rm global}$")
    ax[0].set_title(rf"global fidelity, h={results['hs'][0]:.4f}$\to${results['hs'][1]:.4f}")
    ax[0].legend()

    ax[1].axhline(alpha_th, ls='--', color='gray', lw=1, label=r"exact bulk $\alpha$")
    ax[1].set_xlabel("L"); ax[1].set_ylabel(r"$-\ln F_{\rm global}/L$")
    ax[1].set_title("rate per site")
    ax[1].legend()

    for b in results["block_sizes"]:
        line, = ax[2].plot(Ls_a, 1 - np.array(run["F_red"][str(b)]), 'o-', label=rf"$\ell_A$={b}")
        for m, chi in zip(markers, chis):
            if str(chi) != chi_best:
                ax[2].plot(Ls_a, 1 - np.array(results["runs"][str(chi)]["F_red"][str(b)]),
                           m, mfc='none', color=line.get_color(), ms=7)
    ax[2].plot(Ls_a, 1 - np.array(run["F_red_rdm"]), 'x', color='k', label=r"$\ell_A$=4 (RDM+sqrtm)")
    ax[2].plot(Ls_a, 1 - np.array(run["F_global"]), 's--', color='gray', label="global")
    ax[2].set_yscale('log')
    ax[2].set_xlabel("L"); ax[2].set_ylabel(r"$1-F$")
    ax[2].set_title(rf"reduced vs global (lines $\chi$={chi_best}, open: lower $\chi$)")
    ax[2].legend(ncol=2)
    plt.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"Saved → {out}")


# ── Merge-only mode ───────────────────────────────────────────────────────────
if args.merge:
    merged = None
    for path in args.merge:
        with open(path) as f:
            r = json.load(f)
        if merged is None:
            merged = r
        else:
            assert np.allclose(r["hs"], merged["hs"]) and r["Ls"] == merged["Ls"], f"{path}: different hs/Ls"
            merged["runs"].update(r["runs"])
    chis_m = sorted(int(c) for c in merged["runs"])
    merged["chis"] = chis_m
    chi_tag = '-'.join(map(str, chis_m))
    with open(f"{path_to_data}/oc_ising_{tag}_chi_{chi_tag}.json", "w") as f:
        json.dump(merged, f, indent=1)
    make_plot(merged, f"{path_to_figures}/ISING_orthogonality_catastrophe_{tag}_chi_{chi_tag}.png")
    raise SystemExit


# ── Run ───────────────────────────────────────────────────────────────────────
alpha_th = alpha_exact(h1, h2)
print(f"h1 = {h1:.6f}, h2 = {h2:.6f}, dh = {h2 - h1:.6f}")
print(f"exact bulk rate alpha = {alpha_th:.6e}   (dh^2/16 = {(h2 - h1) ** 2 / 16:.6e})")

results = dict(Ls=Ls, hs=hs, chis=chis, c1=c1, init=args.init, k=k, J=J, block_sizes=block_sizes,
               dmrg_params=dmrg_params, alpha_exact=alpha_th, runs={})
chi_tag = '-'.join(map(str, chis))
if Ls != parser.get_default("Ls"):
    chi_tag += f"_L_{min(Ls)}-{max(Ls)}"
json_out = f"{path_to_data}/oc_ising_{tag}_chi_{chi_tag}.json"

for chi in chis:
    run = dict(F_global=[], F_red={str(b): [] for b in block_sizes}, F_red_method={},
               F_red_rdm=[], energy=[], S_mid=[], sweeps=[], m_mid=[])
    results["runs"][str(chi)] = run
    for l in Ls:
        print(f"── chi = {chi}, L = {l} ──", flush=True)
        s1, e1, S1, n1s = ground_state(h1, l, chi)
        s2, e2, S2, n2s = ground_state(h2, l, chi)
        run["energy"].append([e1, e2])
        run["S_mid"].append([S1, S2])
        run["sweeps"].append([n1s, n2s])
        m1, m2 = mid_magnetization(s1, l), mid_magnetization(s2, l)
        run["m_mid"].append([m1, m2])
        print(f"  E = {e1:.12f}, {e2:.12f}   S_mid = {S1:.5f}, {S2:.5f}   sweeps = {n1s}, {n2s}")
        print(f"  <sx>_mid = {m1[0]:+.6f}, {m2[0]:+.6f}   <sz>_mid = {m1[1]:+.6f}, {m2[1]:+.6f}")

        ov, n1, n2 = global_overlap(s1, s2, l, chi)
        F = abs(ov) ** 2 / (n1 * n2)
        run["F_global"].append(F)
        print(f"  global   F = {F:.12f}   -lnF = {-np.log(F):.6e}   -lnF/L = {-np.log(F) / l:.6e}")

        for b in block_sizes:
            start = l // 2 - b // 2
            Fr, method = reduced_fidelity(s1, s2, start, start + b)
            run["F_red"][str(b)].append(Fr)
            run["F_red_method"][str(b)] = method
            print(f"  reduced (l_A={b:2d}, {method:5s}) F = {Fr:.12f}   1-F = {1 - Fr:.6e}")

        # Same quantity through the RDM + sqrtm route of exec_comparison (1-indexed sites)
        sites = [(l // 2) - 1, l // 2, (l // 2) + 1, (l // 2) + 2]
        r1 = generalized_k_rdm(s1, sites=sites)
        r2 = generalized_k_rdm(s2, sites=sites)
        r1, r2 = r1 / np.trace(r1), r2 / np.trace(r2)
        Frdm = float(uhlmann_fidelity(r1, r2))
        run["F_red_rdm"].append(Frdm)
        print(f"  reduced (l_A= 4, RDM+sqrtm) F = {Frdm:.12f}   1-F = {1 - Frdm:.6e}", flush=True)

        with open(json_out, "w") as f:
            json.dump(results, f, indent=1)

    # No global linear fit: near h = 1 the data are in the crossover, where ln F is not linear in L.
    lnF = np.log(np.array(run["F_global"]))
    local = -np.diff(lnF) / np.diff(Ls)
    print(f"\nchi = {chi}: local slopes -dlnF/dL between consecutive L: "
          f"{np.array2string(local, precision=6)}   (exact bulk alpha = {alpha_th:.6e})\n")

with open(json_out, "w") as f:
    json.dump(results, f, indent=1)

make_plot(results, f"{path_to_figures}/ISING_orthogonality_catastrophe_{tag}_chi_{chi_tag}.png")
