"""
exec_annni_rfs_grid.py
======================
2D (kappa, h) DMRG scan of the ANNNI chain that stores the central k-site RDMs, as input
for the reduced-fidelity-susceptibility (RFS) phase diagram (Fig. 4 of Mariella et al.,
arXiv:2408.01400) and for the order-parameter discovery (OPD) of its Sec. 5.2.

  H = -sum X_i X_{i+1} + kappa sum X_i X_{i+2} - h sum Z_i - eps (X_1 + X_2)

(qs-mps ANNNI MPO, J = 1; eps pins the first two spins along x, as in the paper's runs).

What is stored per grid point: the RDM of the central `--block` sites (default 6, 0-indexed
sites [L/2-3, L/2+3), i.e. 1-indexed 23..28 for L = 50). Any smaller contiguous RDM, including
the paper's "two middle sites" (1-indexed L/2, L/2+1), is obtained from it by partial trace in
analyze_annni_rfs_opd.py. Also stored: energy, sweeps, mid-chain entropy, smallest kept
Schmidt value at the mid bond (chi adequacy), wall time.

Parallelization: each kappa column is an independent warm-started sweep in h, from h_max
(start |up z ... up z>, PM-like) down to h_min. Columns run in a process pool with one BLAS
thread per process (on this laptop one thread is as fast as four for chi = 64).

Resumable: every column is saved after each point (col_XXX.npz + its last MPS); a restart
continues where it stopped. Merge with analyze_annni_rfs_opd.py.

Usage (from src/):
  python exec_annni_rfs_grid.py --workers 4
  python exec_annni_rfs_grid.py --L 12 --chi 16 --n1 4 --n2 4 --workers 2 --tag-suffix _smoke
"""
import os
# One BLAS thread per worker. Must be set before numpy is imported (also in spawned children).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import time
import argparse
import traceback
import numpy as np
from multiprocessing import Pool


MODELS = {
    # columns = first parameter (--kmin/--kmax), sweep = second parameter (--hmin/--hmax).
    # sweep_from: the end each column starts from (the disordered side, initial state |0...0>).
    "annni": dict(project="2_ANNNI", params=("kappa", "h"), sweep_from="max",
                  hamiltonian="-sum XX + kappa sum X_i X_{i+2} - h sum Z - eps (X_1 + X_2)"),
    "cluster": dict(project="3_CLUSTER", params=("K", "h"), sweep_from="max",
                    hamiltonian="-h sum Z - K sum X_i Z_{i+1} X_{i+2} - eps X_1"),
    "rydberg": dict(project="4_RYDBERG", params=("Rb", "Delta"), sweep_from="min",
                    hamiltonian="sum X - Delta sum n + Rb^6 sum_{r <= range} r^-6 n_i n_{i+r} "
                                "(Omega = 1, Delta = Delta/Omega, Rb = R_b/a)"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), default="annni")
    p.add_argument("--L", type=int, default=50)
    p.add_argument("--chi", type=int, default=64)
    p.add_argument("--eps", type=float, default=0.01,
                   help="boundary x-pin: ANNNI sites 1, 2; cluster site 1; not used by rydberg")
    p.add_argument("--n-neighbours", type=int, default=4, help="rydberg: range of the 1/r^6 tail")
    p.add_argument("--kmin", type=float, default=0.1, help="first parameter (columns): kappa / K / Rb")
    p.add_argument("--kmax", type=float, default=1.0)
    p.add_argument("--hmin", type=float, default=0.05, help="second parameter (sweep): h / h / Delta")
    p.add_argument("--hmax", type=float, default=0.75)
    p.add_argument("--n1", type=int, default=64, help="number of points of the first parameter")
    p.add_argument("--n2", type=int, default=64, help="number of points of the second parameter")
    p.add_argument("--block", type=int, default=6, help="size of the stored central RDM")
    p.add_argument("--n-sweeps", type=int, default=12)
    p.add_argument("--conv-tol", type=float, default=1e-9)
    p.add_argument("--eig-tol", type=float, default=1e-8)
    p.add_argument("--ncv", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=5000, help="ARPACK restart cap per local eigsh")
    p.add_argument("--tol-schedule", type=float, nargs="+", default=None,
                   help="staged energy tolerances, e.g. 1e-5 1e-8 1e-12 (replaces --conv-tol); "
                        "n-sweeps is the budget split over the stages")
    p.add_argument("--eig-tol-schedule", type=float, nargs="+", default=None,
                   help="eigsh tol per stage (default min(1e-3, 0.1 sqrt(tol)))")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", choices=["pc", "ngt"], default="pc")
    p.add_argument("--project", default=None)
    p.add_argument("--tag-suffix", default="")
    p.add_argument("--cols", type=int, nargs="+", default=None,
                   help="run only these column indices (0-based); the rest can be resumed later")
    return p.parse_args(argv)


def make_tag(a):
    # The pinning field is part of the name: runs with eps = 0.01 and eps = 0 never share a folder.
    # (ANNNI tags are unchanged from the original script, so old runs can still be resumed.)
    p1, p2 = MODELS[a.model]["params"]
    if a.model == "rydberg":
        pin, n1, n2 = f"range_{a.n_neighbours}", p1, p2
    else:
        pin = f"pinned_eps_{a.eps:g}" if a.eps != 0 else "unpinned_eps_0"
        n1, n2 = ("k", "h") if a.model == "annni" else (p1, p2)
    return (f"rfs_grid_{pin}_L_{a.L}_chi_{a.chi}"
            f"_{n1}_{a.kmin:g}-{a.kmax:g}_{n2}_{a.hmin:g}-{a.hmax:g}_n_{a.n1}x{a.n2}"
            f"_block_{a.block}{a.tag_suffix}")


def out_dir(a):
    root = {"pc": "C:/Users/HP/Desktop", "ngt": "/eos/user/f/fdimarca"}[a.device]
    project = a.project or f"{root}/projects/{MODELS[a.model]['project']}"
    return f"{project}/results/data/{a.model}_rfs_grid/{make_tag(a)}"


def build_model(k, h, a):
    if a.model == "annni":
        from qphaset.annni import model_annni_qs_mps
        return model_annni_qs_mps(k, h, chi=a.chi, l=a.L, j1=1.0, c1=a.eps, d=2)
    if a.model == "cluster":
        from qphaset.cluster import model_cluster_qs_mps
        return model_cluster_qs_mps(k, h, chi=a.chi, l=a.L, c1=a.eps, d=2)
    from qphaset.rydberg import model_rydberg_qs_mps
    return model_rydberg_qs_mps(h, k, chi=a.chi, l=a.L, d=2, n_neighbours=a.n_neighbours)


def ground_state(k, h, init_sites, a):
    model = build_model(k, h, a)
    if init_sites is None:
        up = np.array([1, 0], dtype=complex).reshape((1, 2, 1))
        model.sites = [up.copy() for _ in range(a.L)]
        model.enlarge_chi(noise_std=1e-6, seed=7)
    else:
        model.sites = [t.copy() for t in init_sites]
    model.canonical_form()
    staged = dict(tol_schedule=a.tol_schedule, eig_tol_schedule=a.eig_tol_schedule) if a.tol_schedule else {}
    _, entropy, schmidt, _ = model.DMRG(
        trunc_chi=True, trunc_tol=False, where=a.L // 2, n_sweeps=a.n_sweeps,
        eigsh_params={"ncv": a.ncv, "tol": a.eig_tol, "maxiter": a.maxiter},
        conv_criterion="mid_chain", conv_tol=a.conv_tol, DMRG2=True, **staged,
    )
    e = np.real(np.array(model.mid_chain_energies))
    sites = [np.array(t) for t in model.sites]
    s_mid = np.asarray(schmidt[-1]) if len(schmidt) else np.array([np.nan])
    dE = float(abs(e[-1] - e[-2]) / abs(e[-1])) if len(e) > 1 else np.nan
    final_tol = a.tol_schedule[-1] if a.tol_schedule else a.conv_tol
    info = dict(E=float(e[-1]), dE=dE, sweeps=len(e),
                S_mid=float(entropy[-1]) if len(entropy) else np.nan,
                smin=float(np.min(np.abs(s_mid))),
                # 1 = last-stage tolerance reached (staged: and no local eigsh hit maxiter in that stage)
                converged=float(model.dmrg_info["converged"] if a.tol_schedule else dE < final_tol),
                eig_nc=float(getattr(model, "eig_no_conv", 0)))      # local eigsh calls that hit maxiter
    return sites, info


def central_rdm(sites, L, block):
    from qphaset.mps_fidelity import mixed_canonical, block_tensor
    start = L // 2 - block // 2
    stop = start + block
    M = block_tensor(mixed_canonical(sites, start, stop), start, stop)
    rho = M @ M.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    return rho / np.trace(rho).real


def save_npz(path, *args, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, *args, **arrays)
    os.replace(tmp, path)


def run_column(job):
    j, k, hs, argv = job
    a = parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    d = out_dir(a)
    col_path = f"{d}/col_{j:03d}.npz"
    state_path = f"{d}/col_{j:03d}_last_state.npz"
    keys = ("E", "dE", "sweeps", "S_mid", "smin", "time", "converged", "eig_nc")
    D = 2 ** a.block

    rdms, rec, prev = [], {kk: [] for kk in keys}, None
    if os.path.exists(col_path):
        with np.load(col_path) as z:
            if bool(z["done"]):
                return j, len(hs), 0.0
            rdms = list(z["rdms"])
            # columns written before the converged / eig_nc fields existed: fill with nan
            rec = {kk: (list(z[kk]) if kk in z.files else [np.nan] * len(rdms)) for kk in keys}
        if os.path.exists(state_path) and rdms:
            with np.load(state_path) as z:
                prev = [z[f"arr_{i}"] for i in range(len(z.files))]
        else:
            rdms, rec = [], {kk: [] for kk in keys}

    t_col = time.time()
    for i in range(len(rdms), len(hs)):
        h = float(hs[i])
        t0 = time.time()
        try:
            s, info = ground_state(k, h, prev, a)
        except Exception:
            print(f"[col {j}] DMRG failed at k={k:.4f} h={h:.4f}, restarting from product state\n"
                  f"{traceback.format_exc()}", flush=True)
            s, info = ground_state(k, h, None, a)
        info["time"] = time.time() - t0
        rdms.append(central_rdm(s, a.L, a.block))
        for kk in keys:
            rec[kk].append(info[kk])
        prev = s
        save_npz(state_path, *s)
        save_npz(col_path, rdms=np.array(rdms).reshape(-1, D, D), kappa=k, h=hs[: i + 1],
                 done=(i == len(hs) - 1), **{kk: np.array(v) for kk, v in rec.items()})
    if os.path.exists(state_path):
        os.remove(state_path)
    return j, len(hs), time.time() - t_col


def main():
    a = parse_args()
    d = out_dir(a)
    os.makedirs(d, exist_ok=True)
    spec = MODELS[a.model]
    kappas = np.round(np.linspace(a.kmin, a.kmax, a.n1), 10)
    if spec["sweep_from"] == "max":
        hs = np.round(np.linspace(a.hmax, a.hmin, a.n2), 10)    # descending: row 0 = h_max
    else:
        hs = np.round(np.linspace(a.hmin, a.hmax, a.n2), 10)    # ascending: row 0 = min (rows in sweep order)
    p1, p2 = spec["params"]
    meta = dict(vars(a), tag=make_tag(a), kappas=kappas.tolist(), hs=hs.tolist(),
                param_names=[p1, p2], row_order=("descending" if spec["sweep_from"] == "max" else "ascending"),
                block_sites_0idx=[a.L // 2 - a.block // 2 + i for i in range(a.block)],
                hamiltonian=spec["hamiltonian"],
                dmrg=f"qs-mps two-site, mid_chain convergence, start |0...0> (up z) at {p2}_{spec['sweep_from']} "
                     f"per column, warm start along the column")
    with open(f"{d}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)

    # Interleave columns so a partial run already covers the whole kappa window.
    order = sorted(range(a.n1), key=lambda j: (j % 8, j))
    if a.cols:
        order = [j for j in a.cols if 0 <= j < a.n1]
    jobs = [(j, float(kappas[j]), hs, sys.argv[1:]) for j in order]
    n_jobs = len(jobs)
    print(f"── {make_tag(a)} ── {a.n1}x{a.n2} grid, running {n_jobs} columns, "
          f"{a.workers} workers → {d}", flush=True)
    t0, n_done = time.time(), 0
    with Pool(a.workers) as pool:
        for j, n, t in pool.imap_unordered(run_column, jobs):
            n_done += 1
            el = time.time() - t0
            print(f"  col {j:3d} ({p1}={kappas[j]:.4f}) done in {t / 60:.1f} min | "
                  f"{n_done}/{n_jobs} cols, elapsed {el / 3600:.2f} h", flush=True)
    print(f"Saved → {d}", flush=True)


if __name__ == "__main__":
    main()
