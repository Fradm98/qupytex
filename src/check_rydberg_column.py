"""Crystal order along one Rydberg column from the stored central 6-site RDMs.

usage: python check_rydberg_column.py <data_dir> [col_index]

For every point: densities <n_i> on the block sites, the Z2 / Z3 structure factors
|sum_j n_j e^{2 pi i j / N}| / (n_sites) for N = 2, 3, and the populations of the 4-site
projectors |0101>, |1010> (Z2) and |1001> (Z3; with L = 3m + 1 the edge-pinned pattern
occupies sites 0, 3, 6, ...). Basis: |0> ground, |1> Rydberg, n = |1><1|.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qphaset import opd


def basis_projector(bits):
    v = np.zeros(2 ** len(bits))
    v[int("".join(map(str, bits)), 2)] = 1.0
    return np.outer(v, v)


def main(data_dir, col=0):
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    sites = meta["block_sites_0idx"]
    nb = meta["block"]
    with np.load(os.path.join(data_dir, f"col_{col:03d}.npz")) as z:
        rd, hs, t, conv, smin = z["rdms"], z["h"], z["time"], z["converged"], z["smin"]
        Rb = float(z["kappa"])
    n_site = np.stack([np.real(opd.ptrace(rd, nb, [p])[:, 1, 1]) for p in range(nb)], 1)
    j = np.array(sites)
    S = {N: np.abs((n_site * np.exp(2j * np.pi * j / N)).sum(1)) / nb for N in (2, 3)}
    first4 = [sites.index(s) for s in range(sites[0], sites[0] + 4)]     # sites 21-24 for L = 49
    rho4 = opd.ptrace(rd, nb, first4)
    proj = {b: np.real(opd.expectations(rho4, basis_projector([int(c) for c in b])))
            for b in ("1001", "0101", "1010")}
    print(f"Rb = {Rb:.4f}, block sites {sites}, 4-site window {[sites[i] for i in first4]}")
    print(f"{'Delta':>6} " + " ".join(f"n{s:<4d}" for s in sites)
          + f" {'S_Z2':>5} {'S_Z3':>5} {'|1001>':>6} {'|0101>':>6} {'|1010>':>6}  conv  smin     t[s]")
    for i, h in enumerate(hs):
        print(f"{h:6.3f} " + " ".join(f"{x:5.2f}" for x in n_site[i])
              + f" {S[2][i]:5.2f} {S[3][i]:5.2f} {proj['1001'][i]:6.2f} {proj['0101'][i]:6.2f} "
              f"{proj['1010'][i]:6.2f}  {conv[i]:4.0f}  {smin[i]:.1e} {t[i]:5.0f}")


if __name__ == "__main__":
    d = sys.argv[1]
    main(d, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
