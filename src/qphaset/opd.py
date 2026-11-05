"""
Order-parameter discovery (OPD) tools for lattices of k-site RDMs, following
Mariella et al., arXiv:2408.01400, Sec. 4 and Appendices B-C.

Conventions
-----------
* Qubit basis |0> = sigma^z = +1 ("up", field direction), |1> = sigma^z = -1.
* k-site RDMs are (2^k, 2^k) with the first site as the most significant qubit
  (same ordering as np.kron and as qphaset.mps_fidelity.block_tensor).
* Hermitian observables are expanded in the orthonormal Pauli basis
  {P / sqrt(2^k)}, so the paper's vectorization x = vec(M) becomes a real vector
  and ||M||_F = ||x||_2. Restricting to real coefficients is the same as the
  constraint (44c), which the paper shows is automatically satisfied.

Main entry points
-----------------
ptrace                 partial trace of a batch of RDMs onto a subset of sites
pauli_coeffs           r_i = vec(rho_i) in the Pauli basis
qcqp_matrix            matrix A of Eq. (45)
solve_opd              min eigenvector of A -> optimal M, plus the indefiniteness check
closed_form_observable two-state solution, Eq. (17)
decompose_observable   M = sum_i alpha_i M^(i) (Eq. 20), with Schmidt angles theta_1, theta_2
"""
import itertools
import numpy as np

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


# ── RDM manipulation ──────────────────────────────────────────────────────────
def ptrace(rho, n_sites, keep):
    """Partial trace of rho (..., 2^n, 2^n) keeping the sites in `keep` (0-indexed, in order)."""
    keep = list(keep)
    lead = rho.shape[:-2]
    t = rho.reshape(lead + (2,) * (2 * n_sites))
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ket = list(letters[:n_sites])
    bra = list(letters[n_sites:2 * n_sites])
    for i in range(n_sites):
        if i not in keep:
            bra[i] = ket[i]
    out = [ket[i] for i in keep] + [bra[i] for i in keep]
    expr = "..." + "".join(ket) + "".join(bra) + "->..." + "".join(out)
    red = np.einsum(expr, t)
    D = 2 ** len(keep)
    return red.reshape(lead + (D, D))


def pauli_strings(k):
    labels = ["".join(p) for p in itertools.product("IXYZ", repeat=k)]
    mats = []
    for lab in labels:
        m = np.array([[1.0 + 0j]])
        for c in lab:
            m = np.kron(m, PAULI[c])
        mats.append(m)
    return labels, np.array(mats)


def pauli_coeffs(rhos, k, paulis=None):
    """Real coordinates of rhos (N, D, D) in the orthonormal basis P / sqrt(D): r_P = Tr(rho P) / sqrt(D)."""
    if paulis is None:
        _, paulis = pauli_strings(k)
    D = 2 ** k
    c = np.einsum("nij,pji->np", rhos, paulis) / np.sqrt(D)
    if np.abs(c.imag).max() > 1e-8:
        raise ValueError(f"RDMs are not Hermitian: max |Im Tr(rho P)| = {np.abs(c.imag).max():.2e}")
    return c.real


def observable_from_coeffs(x, k, paulis=None):
    if paulis is None:
        _, paulis = pauli_strings(k)
    return np.einsum("p,pij->ij", x, paulis) / np.sqrt(2 ** k)


# ── QCQP of Eq. (15) / (44)-(45) ──────────────────────────────────────────────
def qcqp_matrix(R, ys):
    """A = -1/|I+| sum_{I+} r r^T / p + 1/|I-| sum_{I-} r r^T / p   (Eq. 45), with p_i = Tr(rho_i^2)."""
    R = np.asarray(R)
    ys = np.asarray(ys).ravel()
    p = np.sum(R ** 2, axis=1)
    plus, minus = ys > 0, ys < 0
    if not plus.any() or not minus.any():
        raise ValueError(f"empty label set: |I+| = {plus.sum()}, |I-| = {minus.sum()}")
    Ap = (R[plus] / p[plus, None]).T @ R[plus] / plus.sum()
    Am = (R[minus] / p[minus, None]).T @ R[minus] / minus.sum()
    return -Ap + Am


# An eigenvalue of A counts as non-zero when |lambda| > max(INDEF_ABS_TOL, INDEF_REL_TOL * max|lambda|).
# Every term r r^T / p of Eq. (45) has unit trace, so the spectrum of A lies in [-1, 1] and an
# absolute floor is meaningful: it stops a matrix made only of DMRG noise (|A| ~ 1e-9) from
# passing as "indefinite", which a purely relative threshold would allow.
INDEF_ABS_TOL = 1e-6
INDEF_REL_TOL = 1e-6


def indefiniteness(w, rel_tol=INDEF_REL_TOL, abs_tol=INDEF_ABS_TOL):
    """Indefiniteness check of App. B on the spectrum w (ascending) of A.

    PASS  : A has a numerically significant negative AND positive eigenvalue; the minimizer
            of Eq. (15) is then a Hermitian M that makes the labelled sets differ.
    FAIL  : A is (numerically) PSD or NSD. With no significant negative eigenvalue the
            objective cannot go below 0: the chosen k-site RDMs carry no linear observable
            separating I+ from I-, and the paper's prescription is to increase k.
    """
    w = np.sort(np.asarray(w, dtype=float))
    scale = float(np.max(np.abs(w)))
    tol = max(abs_tol, rel_tol * scale)
    n_neg, n_pos = int(np.sum(w < -tol)), int(np.sum(w > tol))
    passed = bool(n_neg > 0 and n_pos > 0)
    if passed:
        status = "PASS"
    elif n_neg == 0 and n_pos == 0:
        status = "FAIL: A = 0 within tolerance (I+ and I- carry the same k-site RDM statistics): increase k"
    elif n_neg == 0:
        status = "FAIL: A is PSD (no significant negative eigenvalue): increase k"
    else:
        status = "FAIL: A is NSD (no significant positive eigenvalue)"
    return dict(n_neg=n_neg, n_pos=n_pos, n_zero=int(len(w) - n_neg - n_pos),
                lam_min=float(w[0]), lam_max=float(w[-1]), indefinite=passed, status=status,
                tol=float(tol), rel_tol=float(rel_tol), abs_tol=float(abs_tol), scale=scale,
                rel_neg=float(-w[0] / scale) if (scale > 0 and w[0] < 0) else 0.0)


def solve_opd(R, ys, k, paulis=None, rel_tol=INDEF_REL_TOL):
    """Optimal observable of Eq. (15): x* = min eigenvector of A (Eq. 47), M = sum_P x_P P / sqrt(D).

    The sign of M is fixed so that <M> averaged over I+ is positive. Returns a dict with M,
    the spectrum of A and the indefiniteness check of App. B (keys `indefinite`, `status`).
    When the check fails, `valid` is False: M is still returned for inspection but is not an
    order parameter."""
    A = qcqp_matrix(R, ys)
    w, V = np.linalg.eigh(A)
    x = V[:, 0]
    plus = np.asarray(ys).ravel() > 0
    if np.mean(R[plus] @ x) < 0:
        x = -x
    M = observable_from_coeffs(x, k, paulis)
    chk = indefiniteness(w, rel_tol)
    deg = int(np.sum(np.abs(w - w[0]) <= chk["tol"]))
    return dict(M=M, x=x, A_eigs=w, min_degeneracy=deg, valid=chk["indefinite"], k=k, **chk)


def indefiniteness_ladder(R_by_k, ys, rel_tol=INDEF_REL_TOL):
    """Run the App. B check for every k in R_by_k (same labels), smallest k first.

    Returns {"per_k": {k: check}, "k_min_pass": smallest passing k or None,
             "performed_for_k": [...]} so a report can show that the lower k were tested."""
    per_k = {}
    for k in sorted(R_by_k):
        w = np.linalg.eigvalsh(qcqp_matrix(R_by_k[k], ys))
        per_k[k] = dict(eigs=w, **indefiniteness(w, rel_tol))
    passing = [k for k, c in per_k.items() if c["indefinite"]]
    return dict(per_k=per_k, k_min_pass=min(passing) if passing else None,
                performed_for_k=sorted(per_k), rel_tol=rel_tol, abs_tol=INDEF_ABS_TOL)


def closed_form_observable(rho_p, rho_m):
    """Two-state solution, Eq. (17): M = (r+ - <r+, r-> r-) / sqrt(1 - <r+, r->^2), r = rho / ||rho||_F."""
    a = rho_p / np.linalg.norm(rho_p)
    b = rho_m / np.linalg.norm(rho_m)
    dot = np.real(np.trace(a @ b))
    return (a - dot * b) / np.sqrt(1 - dot ** 2)


def expectations(rhos, M):
    return np.real(np.einsum("nij,ji->n", rhos, M))


# ── Eigendecomposition and angles (Eqs. 20-21) ────────────────────────────────
def signed_qubit_angle(u):
    """theta in (-pi/2, pi/2] such that u ~ cos(theta)|0> + sin(theta)|1> up to a global phase.
    Returns (theta, residual_phase): residual_phase != 0 flags a complex (non-xz-plane) state."""
    u = np.asarray(u, dtype=complex).ravel()
    u = u / np.linalg.norm(u)
    ph = np.angle(u[0]) if abs(u[0]) > 1e-12 else np.angle(u[1])
    u = u * np.exp(-1j * ph)
    rel = float(np.angle(u[1]) if abs(u[1]) > 1e-12 else 0.0)
    theta = np.arctan2(np.real(u[1]), np.real(u[0]))
    if theta > np.pi / 2:
        theta -= np.pi
    elif theta <= -np.pi / 2:
        theta += np.pi
    resid = abs(np.sin(rel)) if abs(u[1]) > 1e-12 else 0.0
    return float(theta), float(resid)


def schmidt_split(v, n_left, n_sites):
    m = np.asarray(v).reshape(2 ** n_left, 2 ** (n_sites - n_left))
    U, s, Vh = np.linalg.svd(m)
    return U, s, Vh


def decompose_observable(M, k):
    """M = sum_i alpha_i |v_i><v_i|, sorted by |alpha_i| (largest first), as in Eq. (20).

    For every eigenvector: Schmidt coefficients across the middle cut; for k = 2 also the
    angles (theta_1, theta_2) of the best product approximation B(theta_1, theta_2) of
    Eq. (21), whose overlap with v_i is the leading Schmidt coefficient squared."""
    alpha, V = np.linalg.eigh(M)
    order = np.argsort(-np.abs(alpha))
    comps = []
    for rank, i in enumerate(order):
        v = V[:, i]
        U, s, Vh = schmidt_split(v, k // 2, k)
        c = dict(index=rank + 1, alpha=float(alpha[i]), v=v, schmidt=s,
                 product_overlap=float(s[0] ** 2))
        if k == 2:
            t1, r1 = signed_qubit_angle(U[:, 0])
            t2, r2 = signed_qubit_angle(Vh[0, :])
            c.update(theta1=t1, theta2=t2, complex_residual=max(r1, r2))
            bell = {
                "Phi+": np.array([1, 0, 0, 1]) / np.sqrt(2), "Phi-": np.array([1, 0, 0, -1]) / np.sqrt(2),
                "Psi+": np.array([0, 1, 1, 0]) / np.sqrt(2), "Psi-": np.array([0, 1, -1, 0]) / np.sqrt(2),
            }
            c["bell_overlaps"] = {n: float(abs(np.vdot(b, v)) ** 2) for n, b in bell.items()}
        comps.append(c)
    return comps


# ── Z2 parity of the block (eps = 0 runs) ─────────────────────────────────────
# At eps = 0 the ANNNI Hamiltonian commutes with P = prod_i Z_i. The exact finite-L ground
# state is a P eigenstate, so its block RDM obeys P_A rho P_A = rho in EVERY phase
# (P = P_A (x) P_rest and P_rest is traced out). DMRG can still return a broken state
# a|+> + b|->; its block RDM is |a|^2 rho_+ + |b|^2 rho_- (cross terms ~ e^{-L}), and the
# group average below maps it back to (rho_+ + rho_-)/2 for any a, b. The projection is
# therefore applied phase-agnostically, chosen by the symmetry of H, never by the phase.
def parity_operator(n_sites):
    return np.diag(np.array([(-1) ** bin(i).count("1") for i in range(2 ** n_sites)], dtype=float))


def parity_symmetrize(rhos, n_sites):
    """(rho + P rho P) / 2 for one RDM or a stack (..., D, D). P is diagonal, so this keeps
    exactly the entries between basis states of equal parity."""
    d = np.diag(parity_operator(n_sites))
    keep = (np.outer(d, d) > 0).astype(float)
    return np.asarray(rhos) * keep


def parity_asymmetry(rhos, n_sites):
    """Frobenius norm of rho - P rho P, per RDM. 0 for a parity eigenstate; for a fully broken
    state it equals ||rho_+ - rho_-||_F (about sqrt(2) for a 6-site block deep in FM/AP)."""
    rhos = np.asarray(rhos)
    sym = parity_symmetrize(rhos, n_sites)
    return 2 * np.linalg.norm((rhos - sym).reshape(*rhos.shape[:-2], -1), axis=-1)


def projector(v):
    v = np.asarray(v).reshape(-1, 1)
    return v @ v.conj().T


def top_pauli_terms(M, k, n=6, paulis=None, labels=None):
    if paulis is None:
        labels, paulis = pauli_strings(k)
    c = np.real(np.einsum("pij,ji->p", paulis, M)) / 2 ** k        # M = sum c_P P
    idx = np.argsort(-np.abs(c))[:n]
    return [(labels[i], float(c[i])) for i in idx]


# ── Separability diagnostics ──────────────────────────────────────────────────
def auc(a, b):
    """Probability that a random element of a exceeds one of b (ties count 1/2)."""
    a, b = np.asarray(a), np.asarray(b)
    allv = np.concatenate([a, b])
    ranks = np.empty(len(allv))
    order = np.argsort(allv, kind="mergesort")
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks over ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    ra = ranks[: len(a)].sum()
    return float((ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def fisher_lda(Ra, Rb, reg=1e-10):
    """Best linear functional w (on Pauli coordinates) separating two point clouds."""
    mu = Ra.mean(0) - Rb.mean(0)
    S = np.cov(Ra.T, bias=True) + np.cov(Rb.T, bias=True)
    S = np.atleast_2d(S) + reg * np.trace(np.atleast_2d(S)) / max(len(mu), 1) * np.eye(len(mu)) + 1e-14 * np.eye(len(mu))
    w = np.linalg.solve(S, mu)
    return w / np.linalg.norm(w)


def cross_phase_collision(Ra, Rb):
    """min_{a, b} ||r_a - r_b||_2 (Hilbert-Schmidt distance between RDMs of the two phases),
    and the arg-min pair indices."""
    from scipy.spatial.distance import cdist
    d = cdist(Ra, Rb)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return float(d[i, j]), int(i), int(j)


def grid_step_scale(R_grid, mask=None):
    """Median HS distance between RDMs at adjacent grid points (R_grid: (n_rows, n_cols, n_coeff)).
    This is the RDM change produced by one grid step, the natural unit for collision distances."""
    dx = np.linalg.norm(R_grid[:, 1:] - R_grid[:, :-1], axis=-1)
    dy = np.linalg.norm(R_grid[1:] - R_grid[:-1], axis=-1)
    if mask is not None:
        dx = dx[mask[:, 1:] & mask[:, :-1]]
        dy = dy[mask[1:] & mask[:-1]]
    d = np.concatenate([np.ravel(dx), np.ravel(dy)])
    return float(np.median(d)) if d.size else np.nan


def trace_distance(a, b):
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(a - b))))


def effective_dimension(R, rel_tol=1e-3):
    """Number of significant directions spanned by the RDM cloud (centred Pauli coordinates)."""
    s = np.linalg.svd(R - R.mean(0), compute_uv=False)
    return int(np.sum(s > rel_tol * s[0])), s
