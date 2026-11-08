"""
test_annni_rfs_opd.py
=====================
Checks for qphaset/opd.py and analyze_annni_rfs_opd.py without any DMRG.

1. Known-answer unit tests:
   - partial trace of product / Bell states,
   - Ising example of App. D: rho+ = |up z><up z|, rho- = |+x><+x|  ->  M* = (I + 2Z - X) / (2 sqrt 3),
     from the full QCQP (min eigenvector of A) and from the closed form Eq. (17),
   - A = 0 (not indefinite) when I+ and I- hold the same RDMs,
   - signed angles of product states and the Bell-state Schmidt spectrum,
   - orientation of the RFS field: P = -grad g from the library equals an independent
     finite-difference gradient in physical (kappa, h) coordinates.
2. A mock run: 6-site RDMs on a (kappa, h) grid built as smooth mixtures of phase-like
   product states (FM tilted along x, PM along z, AP with the period-4 ++-- pattern,
   FP as an incoherent mixture of spirals), switched across the App. F theory lines.
   It is written in the exec_annni_rfs_grid.py format and analyzed end to end.

Usage (from src/):  python test_annni_rfs_opd.py [--out <folder>]
"""
import os
import sys
import json
import argparse
import shutil
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qphaset import opd
import analyze_annni_rfs_opd as an

X, Y, Z, I2 = (opd.PAULI[c] for c in "XYZI")


def ket(alpha):
    """Spin in the xz plane at angle alpha from +z: cos(alpha/2)|0> + sin(alpha/2)|1>."""
    return np.array([np.cos(alpha / 2), np.sin(alpha / 2)], dtype=complex)


def product(vs):
    out = np.array([1.0 + 0j])
    for v in vs:
        out = np.kron(out, v)
    return out


def dm(v):
    return np.outer(v, v.conj())


# ── 1. unit tests ─────────────────────────────────────────────────────────────
def test_ptrace():
    rng = np.random.default_rng(1)
    alphas = rng.uniform(-np.pi, np.pi, 6)
    rho = dm(product([ket(t) for t in alphas]))
    for keep in ([0], [2], [2, 3], [1, 2, 3], [4, 5]):
        exp = dm(product([ket(alphas[i]) for i in keep]))
        assert np.allclose(opd.ptrace(rho, 6, keep), exp), keep
    bell = np.array([0, 1, 1, 0]) / np.sqrt(2)
    assert np.allclose(opd.ptrace(dm(bell), 2, [0]), I2 / 2)
    batch = np.stack([rho, rho])
    assert opd.ptrace(batch, 6, [2, 3]).shape == (2, 4, 4)
    print("ok  partial trace")


def test_ising_appendix_d():
    """App. D numbers, and the fact that Eq. (17) is NOT the minimizer of Eq. (16).

    For unit-norm pure rho+, rho- with c = <rho+, rho-> the quadratic form of Eq. (16) is
    x^T (b b^T - a a^T) x, whose minimum over ||x|| = 1 is -sqrt(1 - c^2). Eq. (17) gives
    Tr(rho- M) = 0 and objective -(1 - c^2), which is larger for 0 < c < 1: it is the optimum
    only with the extra constraint Tr(rho- M) = 0."""
    rp, rm = dm(ket(0)), dm(ket(np.pi / 2))          # |up z>, |+x>, c = 1/2
    expected = (I2 + 2 * Z - X) / (2 * np.sqrt(3))
    Mcf = opd.closed_form_observable(rp, rm)
    assert np.allclose(Mcf, expected), Mcf
    assert np.isclose(np.trace(rm @ Mcf).real, 0, atol=1e-12)
    assert np.isclose(np.trace(rp @ Mcf).real, np.sqrt(3) / 2)

    def objective(M):
        return -np.trace(rp @ M).real ** 2 + np.trace(rm @ M).real ** 2

    R = opd.pauli_coeffs(np.stack([rp, rm]), 1)
    sol = opd.solve_opd(R, np.array([1, -1]), 1)
    c = 0.5
    assert sol["indefinite"] and np.isclose(np.linalg.norm(sol["M"]), 1)
    assert np.isclose(objective(sol["M"]), -np.sqrt(1 - c ** 2))
    assert np.isclose(objective(Mcf), -(1 - c ** 2))
    # brute force over all unit-norm real 2x2 Hermitians spanned by I, X, Z
    best = min(objective(np.cos(t) * (np.cos(p) * I2 + np.sin(p) * Z) / np.sqrt(2) + np.sin(t) * X / np.sqrt(2))
               for t in np.linspace(0, np.pi, 361) for p in np.linspace(0, 2 * np.pi, 721))
    assert objective(sol["M"]) <= best + 1e-6
    print("ok  App. D Ising example: Eq. (17) = (I + 2Z - X)/(2 sqrt 3) with objective -0.750;"
          f" the true QCQP minimum is {objective(sol['M']):.3f} (= -sqrt(1-c^2)), Tr(rho- M) = "
          f"{np.trace(rm @ sol['M']).real:.3f}")


def test_not_indefinite():
    rng = np.random.default_rng(2)
    rhos = np.stack([dm(ket(t)) for t in rng.uniform(0, np.pi, 10)])
    R = opd.pauli_coeffs(np.concatenate([rhos, rhos]), 1)
    sol = opd.solve_opd(R, np.r_[np.ones(10), -np.ones(10)], 1)
    assert not sol["indefinite"] and np.allclose(sol["A_eigs"], 0)
    print("ok  identical label sets give A = 0 (indefiniteness fails, as App. B requires)")


def test_ladder_fail_then_pass(fig_dir):
    """I+ and I- differ only in ZZ correlations: identical 1-site marginals (I/2).

    Expected: k = 1 FAIL (A is DMRG-size noise), k = 2 PASS, smallest passing k = 2. Also shows
    why the absolute floor is needed: with a purely relative threshold the k = 1 noise matrix
    would be declared indefinite."""
    rng = np.random.default_rng(3)
    corr = (dm(product([ket(0), ket(0)])) + dm(product([ket(np.pi), ket(np.pi)]))) / 2      # (|00><00|+|11><11|)/2
    anti = (dm(product([ket(0), ket(np.pi)])) + dm(product([ket(np.pi), ket(0)]))) / 2      # (|01><01|+|10><10|)/2
    ts = np.linspace(0, 0.3, 20)

    def noisy(r):
        G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        r = r + 1e-9 * (G + G.conj().T)
        return r / np.trace(r).real

    rho2 = np.array([noisy((1 - t) * corr + t * np.eye(4) / 4) for t in ts] +
                    [noisy((1 - t) * anti + t * np.eye(4) / 4) for t in ts])
    ys = np.r_[np.ones(20), -np.ones(20)]
    R = {1: opd.pauli_coeffs(opd.ptrace(rho2, 2, [0]), 1), 2: opd.pauli_coeffs(rho2, 2)}
    lad = opd.indefiniteness_ladder(R, ys)
    c1, c2 = lad["per_k"][1], lad["per_k"][2]
    assert not c1["indefinite"] and c1["status"].startswith("FAIL"), c1
    assert c2["indefinite"] and lad["k_min_pass"] == 2, c2
    assert c1["scale"] < 1e-7                                   # A at k = 1 is pure noise
    relative_only = opd.indefiniteness(c1["eigs"], abs_tol=0.0)
    assert relative_only["indefinite"], "noise should fool a purely relative threshold"
    os.makedirs(fig_dir, exist_ok=True)
    an.plot_ladder([lad], ["ZZ-only mock"], os.path.join(fig_dir, "indefiniteness_ladder_FAIL_example"),
                   "Mock: labels differ only in ZZ correlations (1-site marginals identical)")
    an.print_ladder(lad, "ZZ-only mock")
    print(f"ok  indefiniteness ladder: k = 1 {c1['status'][:4]} (max|eig| = {c1['scale']:.1e}), "
          f"k = 2 PASS; relative-only tolerance would wrongly pass k = 1")


def test_angles():
    for t1, t2 in [(0.095, 0.034), (0.381, -0.331), (-0.25, 0.4)]:
        v = np.kron(np.array([np.cos(t1 * np.pi), np.sin(t1 * np.pi)]),
                    np.array([np.cos(t2 * np.pi), np.sin(t2 * np.pi)]))
        c = opd.decompose_observable(opd.projector(v), 2)[0]
        assert np.isclose(c["theta1"] / np.pi, t1) and np.isclose(c["theta2"] / np.pi, t2), (t1, t2, c)
        assert np.isclose(c["product_overlap"], 1)
        # global sign of the eigenvector must not change the angles
        c2 = opd.decompose_observable(opd.projector(-v), 2)[0]
        assert np.isclose(c2["theta1"], c["theta1"]) and np.isclose(c2["theta2"], c["theta2"])
    psi = np.array([0, 1, 1, 0]) / np.sqrt(2)
    c = opd.decompose_observable(opd.projector(psi), 2)[0]
    assert np.isclose(c["product_overlap"], 0.5) and np.isclose(c["bell_overlaps"]["Psi+"], 1)
    print("ok  signed (theta1, theta2) and Bell-state Schmidt spectrum")


def test_field_orientation():
    """g with a ridge along the line h = 0.5 + 0.3 (k - 0.5): P must point away from it."""
    ks = np.linspace(0.1, 1.0, 40)
    hs = np.linspace(0.9, 0.1, 30)                          # descending rows
    Kg, Hg = np.meshgrid(ks, hs)
    # fake RDMs whose fidelity susceptibility peaks on the line: the state rotates fast there
    s = np.tanh((Hg - 0.5 - 0.3 * (Kg - 0.5)) / 0.05)
    alpha = np.pi / 4 * (1 + s)
    rh = np.empty(Kg.shape + (2, 2), complex)
    for idx in np.ndindex(Kg.shape):
        v = ket(alpha[idx])
        rh[idx] = 0.98 * dm(v) + 0.01 * I2
    dk, dh = ks[1] - ks[0], hs[0] - hs[1]
    g, P = an.rfs_field(rh, dk, dh, scale=1)
    # independent gradient of g in physical coordinates
    gk = np.gradient(g, dk, axis=1)
    gh = -np.gradient(g, dh, axis=0)                        # rows descend in h
    Pref = -(gk + 1j * gh)
    m = np.abs(Pref) > 0.05 * np.abs(Pref).max()
    cos = np.real(P[m] * np.conj(Pref[m])) / (np.abs(P[m]) * np.abs(Pref[m]))
    assert np.mean(cos > 0.9) > 0.95, np.mean(cos > 0.9)
    # sources on the ridge: above the line P points up (+h), below it points down
    Ki, Hi = Kg[1:-1, 1:-1], Hg[1:-1, 1:-1]
    d = Hi - 0.5 - 0.3 * (Ki - 0.5)
    band = (np.abs(d) > 0.06) & (np.abs(d) < 0.2) & m
    assert np.mean(np.sign(P.imag[band]) == np.sign(d[band])) > 0.95
    print("ok  RFS field orientation: P = -grad g points away from the transition (sources)")


def test_parity_tools():
    """(rho + P rho P)/2 maps every mixture |a|^2 rho_+ + |b|^2 rho_- to the same cat-limit RDM."""
    assert np.allclose(np.diag(opd.parity_operator(2)), [1, -1, -1, 1])
    P6 = opd.parity_operator(6)
    rng = np.random.default_rng(3)
    alphas = rng.uniform(0.3, 1.2, 6)
    rp = dm(product([ket(t) for t in alphas]))             # broken: tilted towards +x
    rm = P6 @ rp @ P6                                       # its parity partner (tilted towards -x)
    assert np.allclose(rm, dm(product([ket(-t) for t in alphas])))
    cat = 0.5 * (rp + rm)
    for a in (1.0, 0.8, 0.5, 0.1, 0.0):
        assert np.allclose(opd.parity_symmetrize(a * rp + (1 - a) * rm, 6), cat), a
    assert np.allclose(opd.parity_symmetrize(cat, 6), cat)                       # idempotent
    assert np.isclose(opd.parity_asymmetry(rp, 6), np.linalg.norm(rp - rm))
    assert opd.parity_asymmetry(cat, 6) < 1e-14
    st = np.array([rp, cat])
    assert np.allclose(opd.parity_asymmetry(st, 6), [np.linalg.norm(rp - rm), 0])
    for keep in ([2], [2, 3], [1, 2, 3, 4]):                # commutes with the partial trace
        lhs = opd.ptrace(opd.parity_symmetrize(rp, 6), 6, keep)
        rhs = opd.parity_symmetrize(opd.ptrace(rp, 6, keep), len(keep))
        assert np.allclose(lhs, rhs), keep
    print("ok  parity: symmetrization is independent of the broken weights, idempotent, commutes with ptrace")


def annni_dense(L, k, h, eps=0.0):
    def op(o, i):
        out = np.array([[1.0]])
        for s in range(L):
            out = np.kron(out, o if s == i else np.eye(2))
        return out
    Xs, Zs = [op(X.real, i) for i in range(L)], [op(Z.real, i) for i in range(L)]
    H = -sum(Xs[i] @ Xs[i + 1] for i in range(L - 1)) + k * sum(Xs[i] @ Xs[i + 2] for i in range(L - 2))
    return H - h * sum(Zs) - eps * (Xs[0] + Xs[1])


def test_exact_gs_is_parity_symmetric():
    """Exact diagonalization, L = 10, eps = 0: the central-block RDM of the ground state commutes
    with the block parity in FM, PM, AP and FP alike (the theorem behind the phase-agnostic step);
    a pin breaks it in the ordered phases."""
    L, nb = 10, 4
    keep = list(range(L // 2 - nb // 2, L // 2 + nb // 2))
    for name, k, h in [("FM", 0.2, 0.3), ("PM", 0.3, 0.9), ("AP", 0.9, 0.1), ("FP", 0.8, 0.45)]:
        out = {}
        for eps in (0.0, 0.01):
            w, v = np.linalg.eigh(annni_dense(L, k, h, eps))
            out[eps] = opd.parity_asymmetry(opd.ptrace(dm(v[:, 0]), L, keep), nb)
        assert out[0.0] < 1e-10, (name, out)
        print(f"ok  exact GS, L={L}, {name} (k={k}, h={h}): asymmetry eps=0 {out[0.0]:.1e}, eps=0.01 {out[0.01]:.1e}")


# ── 2. mock grid ──────────────────────────────────────────────────────────────
def sig(x):
    return 0.5 * (1 + np.tanh(x))


def mock_rdm6(k, h, sites, width=0.025):
    """Smooth mixture of phase-like 6-site product states across the App. F lines."""
    ap_sign = np.array([1 if (j // 2) % 2 == 0 else -1 for j in sites])
    pm = dm(product([ket(0.15 + 0.1 * (0.75 - h)) for _ in sites]))
    if k < 0.5:
        hi = max(an.h_ising(k), 1e-3)
        fm = dm(product([ket(np.pi / 2 * (1 - 0.6 * min(h / hi, 1))) for _ in sites]))
        w_fm = sig((hi - h) / width)
        rho = w_fm * fm + (1 - w_fm) * pm
    else:
        hpt, hkt = an.h_pt(k), an.h_kt(k)
        ap = dm(product([ket(s * np.pi / 2 * (1 - 0.5 * min(h / max(hpt, 1e-3), 1))) for s in ap_sign]))
        x = np.clip((h - hpt) / max(hkt - hpt, 1e-3), 0, 1)
        q = np.pi / 2 * (1 - 0.6 * x)                        # wavevector: pi/2 at PT -> smaller at KT
        fp = np.mean([dm(product([ket(0.9 * np.cos(q * j + ph)) for j in sites]))
                      for ph in np.linspace(0, 2 * np.pi, 8, endpoint=False)], axis=0)
        w_ap = sig((hpt - h) / width)
        w_fp = (1 - w_ap) * sig((hkt - h) / width)
        rho = w_ap * ap + w_fp * fp + (1 - w_ap - w_fp) * pm
    rho = (1 - 1e-3) * rho + 1e-3 * np.eye(64) / 64
    return 0.5 * (rho + rho.conj().T)


def eps0_collapse(rd, k, hs, rng):
    """Imitate DMRG at eps = 0: cat (parity-symmetric) state from h_max down to a column-dependent
    collapse height inside the ordered phase, then a broken state with a random sign per column."""
    h_line = an.h_ising(k) if k < 0.5 else an.h_pt(k)
    h_c = h_line * rng.uniform(0.35, 0.75)
    sign = rng.choice([1, -1])
    P6 = opd.parity_operator(6)
    out = []
    for r, h in zip(rd, hs):
        r_sym = opd.parity_symmetrize(r, 6)
        if h > h_c or not np.isfinite(h_line):
            out.append(r_sym)
        else:
            out.append(r if sign > 0 else P6 @ r @ P6)
    return np.array(out)


def write_mock(out, n1=24, n2=24, L=50, block=6, eps=0.01, collapse_seed=None):
    os.makedirs(out, exist_ok=True)
    kappas = np.round(np.linspace(0.1, 1.0, n1), 10)
    hs = np.round(np.linspace(0.75, 0.05, n2), 10)
    sites = [L // 2 - block // 2 + i for i in range(block)]
    meta = dict(L=L, chi=0, eps=eps, block=block, n_sweeps=12, kappas=kappas.tolist(),
                hs=hs.tolist(), block_sites_0idx=sites, tag="mock")
    with open(f"{out}/meta.json", "w") as f:
        json.dump(meta, f)
    rng = np.random.default_rng(collapse_seed)
    for j, k in enumerate(kappas):
        rd = np.array([mock_rdm6(k, h, sites) for h in hs])
        if collapse_seed is not None:
            rd = eps0_collapse(rd, k, hs, rng)
        one = np.ones(n2)
        np.savez(f"{out}/col_{j:03d}.npz", rdms=rd, kappa=k, h=hs, done=True, E=-one, dE=1e-10 * one,
                 sweeps=3 * one, S_mid=one, smin=1e-9 * one, time=one)
    return out


def check_mock_summary(s):
    reg = s["regions"]["right_of_LP"]
    k1, k2 = reg["k"]["1"], reg["k"]["2"]
    # 1-site RDMs of real states: no Y component; at most I, X, Z vary
    assert s["one_site"]["max_abs_Y"] < 1e-10
    assert s["separability"]["results"]["1"]["effective_dimension"] <= 2
    assert k1["n_neg"] + k1["n_pos"] <= 3, "1-site A cannot have more than 3 non-zero eigenvalues (I, X, Z)"
    # the 2-site problem must be indefinite on this mock, and its M must have 4 components
    assert k2["indefinite"] and len(k2["components"]) == 4
    for c in k2["components"]:
        assert abs(c["complex_residual"]) < 1e-8
    print("ok  mock end-to-end: summary structure and 1-site rank bounds")


def test_eps0_collapse(out):
    """Mock eps = 0 run with a cat -> broken collapse inside FM / AP (random height and sign per
    column). Raw RDMs: the collapse is a fake RFS line far from every transition. After the
    phase-agnostic parity projection it is gone. The analysis must apply the projection on its
    own (meta eps = 0, --parity auto) and record it."""
    data = write_mock(os.path.join(out, "data_eps0"), eps=0.0, collapse_seed=11)
    meta, kappas, hs, rdm6, _ = an.load_grid(data)
    dk, dh = kappas[1] - kappas[0], hs[0] - hs[1]
    K, H = np.meshgrid(kappas[1:-1], hs[1:-1])
    far = an.distance_to_lines(K, H, kappas[0], kappas[-1]) > 0.08
    pos = an.central_positions(meta, 2)
    ratio = {}
    for name, r6 in (("raw", rdm6), ("symmetrized", opd.parity_symmetrize(rdm6, 6))):
        g, _ = an.rfs_field(opd.ptrace(r6, 6, pos), dk, dh)
        ratio[name] = float(g[far].max() / g[~far].max())
    print(f"    max g far from all lines / max g near the lines: raw {ratio['raw']:.2f}, "
          f"symmetrized {ratio['symmetrized']:.3f}")
    assert ratio["raw"] > 0.5, "the mock collapse should produce a fake RFS line"
    assert ratio["symmetrized"] < 0.1, "the parity projection should remove the fake line"
    s = an.main(["--data-dir", data, "--fig-dir", os.path.join(out, "figures_eps0"), "--n-eta", "24",
                 "--ks", "1", "2"])
    par = s["parity"]
    assert par["hamiltonian_parity_symmetric"] and par["applied"] and par["n_broken"] > 0, par
    r1 = s["one_site"]
    assert max(abs(r1["X_range"][0]), abs(r1["X_range"][1])) < 1e-12, "eps = 0: <X> must vanish after projection"
    print(f"ok  eps=0 mock: {par['n_broken']} broken points found, projection applied automatically, <X> = 0")


def test_join(full, out):
    """Split a mock grid like the Rydberg runs: A = low columns x top rows, B = low columns x
    bottom rows (swept upwards), C = high columns x all rows (swept upwards); A/B share one row
    and A, B / C share one column. Points repeated in a later run get a small energy offset.
    Joining must reproduce the full grid, with repeated points taken from the earlier run, and
    a missing block must be refused."""
    meta, kappas, hs, rdm6, diag = an.load_grid(full)
    n2, n1 = rdm6.shape[:2]
    layout = {"A": (slice(0, 13), slice(0, 12), False), "B": (slice(12, n2), slice(0, 12), True),
              "C": (slice(0, n2), slice(11, n1), True)}
    seen = np.zeros((n2, n1), bool)
    dirs = []
    for name, (rs, cs, upward) in layout.items():
        d = os.path.join(out, f"join_{name}")
        shutil.rmtree(d, ignore_errors=True)                 # no stale columns from earlier runs
        os.makedirs(d)
        h = hs[rs][::-1] if upward else hs[rs]
        with open(f"{d}/meta.json", "w") as f:
            json.dump(dict(meta, kappas=kappas[cs].tolist(), hs=h.tolist()), f)
        for jj, j in enumerate(range(n1)[cs]):
            rd, E = rdm6[rs, j], diag["E"][rs, j] + 1e-9 * seen[rs, j]   # repeated points differ slightly
            if upward:
                rd, E = rd[::-1], E[::-1]
            one = np.ones(len(h))
            np.savez(f"{d}/col_{jj:03d}.npz", rdms=rd, kappa=kappas[j], h=h, done=True, E=E,
                     dE=1e-10 * one, sweeps=3 * one, S_mid=one, smin=1e-9 * one, time=one)
        seen[rs, cs] = True
        dirs.append(d)
    m2, k2, h2, r2, d2 = an.load_grids(dirs)
    assert np.allclose(k2, kappas) and np.allclose(h2, hs) and np.array_equal(r2, rdm6)
    assert np.array_equal(d2["E"], diag["E"]), "repeated points must be kept from the earlier run"
    assert m2["hs"] == h2.tolist() and m2["kappas"] == k2.tolist() and len(m2["joined_from"]) == 3
    try:
        an.load_grids(dirs[:1] + dirs[2:])                   # B missing: a hole in the union
        raise AssertionError("a join with an uncovered block must be refused")
    except RuntimeError as e:
        assert "uncovered" in str(e)
    print("ok  join of three runs along both parameters (two swept upwards) reproduces the full grid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    a = p.parse_args()
    test_ptrace()
    test_ising_appendix_d()
    test_not_indefinite()
    test_angles()
    test_field_orientation()
    test_parity_tools()
    test_exact_gs_is_parity_symmetric()
    out = a.out or os.path.join(tempfile.gettempdir(), "annni_rfs_opd_mock")
    test_eps0_collapse(out)
    test_ladder_fail_then_pass(os.path.join(out, "figures"))
    data = write_mock(os.path.join(out, "data"))
    test_join(data, out)
    s = an.main(["--data-dir", data, "--fig-dir", os.path.join(out, "figures"), "--n-eta", "36"])
    check_mock_summary(s)
    assert not s["parity"]["applied"], "eps = 0.01 run must not be symmetrized under --parity auto"
    print(f"\nall checks passed; mock figures in {os.path.join(out, 'figures')}")


if __name__ == "__main__":
    main()
