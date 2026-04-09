"""
demo_1d_cft.py — Verify entanglement entropy on 1D free fermion chain against CFT.

For the XX model (free fermions) on an open chain of length L at half-filling,
the half-chain entanglement entropy follows the Calabrese-Cardy formula:

    S = (c/6) * ln(2L/pi) + c'_1,    c = 1.

This script computes S for various L and verifies the logarithmic scaling
with coefficient 1/6.

References
----------
[1] Calabrese & Cardy, J. Phys. A 42, 504005 (2009).
[2] Vidal et al., PRL 90, 227902 (2003).
"""

import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, '..')

from ee import ee_corr_matrix
from lattices import chain_1d


def main():
    print("=" * 60)
    print("1D Free Fermion Chain (OBC, Half-filling)")
    print("CFT Prediction: S = (c/6) * ln(2L/pi), c = 1")
    print("=" * 60)
    print(f"{'L':>6s}  {'S':>10s}  {'(1/6)ln(2L/pi)':>16s}  {'diff':>10s}")
    print("-" * 48)

    Ls, Ss = [], []
    for L in [10, 14, 20, 30, 50, 80, 100, 200, 400, 800]:
        G = chain_1d(L, pbc=False)
        S = ee_corr_matrix(G, list(range(L // 2)))
        cft = (1 / 6) * np.log(2 * L / np.pi)
        Ls.append(L)
        Ss.append(S)
        print(f"{L:6d}  {S:10.6f}  {cft:16.6f}  {S - cft:10.6f}")

    # Linear fit: S = a * ln(L) + b
    A = np.column_stack([np.log(Ls), np.ones(len(Ls))])
    coeffs, *_ = np.linalg.lstsq(A, Ss, rcond=None)

    # Use only large L for better asymptotic estimate
    mask = [i for i, l in enumerate(Ls) if l >= 50]
    A2 = np.column_stack([np.log([Ls[i] for i in mask]), np.ones(len(mask))])
    c2, *_ = np.linalg.lstsq(A2, [Ss[i] for i in mask], rcond=None)

    print()
    print(f"Fit (all L):   S = {coeffs[0]:.6f} ln(L) + {coeffs[1]:.6f}")
    print(f"Fit (L >= 50): S = {c2[0]:.6f} ln(L) + {c2[1]:.6f}")
    print(f"Expected coefficient: c/6 = 1/6 = {1/6:.6f}")
    print(f"Relative error (L>=50): {abs(c2[0] - 1/6) / (1/6) * 100:.2f}%")
    print()

    # Save data for plotting
    data = np.column_stack([Ls, Ss, [(1/6)*np.log(2*L/np.pi) for L in Ls]])
    np.savetxt('../data/1d_cft_data.txt', data, 
               header='L  S  S_cft', comments='',
               fmt=['%4d', '%10.6f', '%10.6f'])
    print("Data saved to ../data/1d_cft_data.txt")


if __name__ == "__main__":
    main()
