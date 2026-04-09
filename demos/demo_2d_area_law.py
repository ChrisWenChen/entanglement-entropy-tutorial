"""
demo_2d_area_law.py — Entanglement entropy on 2D lattices (square + honeycomb).

This script demonstrates the different area law behaviors:

1. Square lattice at half-filling:
   - Has a square Fermi surface
   - Widom conjecture: S ~ alpha * L_y * ln(L_y)
   - Logarithmic enhancement due to extended Fermi surface

2. Honeycomb lattice at half-filling:
   - Dirac points (no extended Fermi surface)
   - S ~ alpha * L_y + subleading log corrections
   - Strict area law with log corrections only

References
----------
[1] Gioev & Klich, PRL 96, 100503 (2006).
[2] D'Emidio et al., PRL 132, 076502 (2024).
"""

import sys
import numpy as np

sys.path.insert(0, '..')

from ee import ee_corr_matrix
from lattices import square_2d, honeycomb_2d, subsystem_left_half


def demo_square():
    """Square lattice: logarithmic enhancement of area law."""
    print("=" * 60)
    print("2D Square Lattice (OBC, Half-filling)")
    print("Prediction: S/L ~ ln(L) (logarithmic enhancement)")
    print("=" * 60)
    print(f"{'L':>4s}  {'N':>6s}  {'S':>10s}  {'S/L':>8s}  {'S/(L*ln L)':>12s}")
    print("-" * 48)

    results = []
    for L in [4, 6, 8, 10, 12, 16, 20]:
        G = square_2d(L, L, pbc=False)
        sub = subsystem_left_half(L, L, sites_per_cell=1)
        S = ee_corr_matrix(G, sub)
        N = L * L
        s_per_l = S / L
        s_normalized = S / (L * np.log(max(L, 2)))  # S/(L*ln L)
        results.append([L, N, S, s_per_l])
        print(f"{L:4d}  {N:6d}  {S:10.4f}  {s_per_l:8.4f}  {s_normalized:12.4f}")
    
    print()
    return np.array(results)


def demo_honeycomb():
    """Honeycomb lattice: strict area law."""
    print()
    print("=" * 60)
    print("Honeycomb Lattice (OBC, Half-filling)")
    print("Prediction: S/L -> const (strict area law)")
    print("=" * 60)
    print(f"{'L':>4s}  {'N':>6s}  {'S':>10s}  {'S/L':>8s}")
    print("-" * 34)

    results = []
    for L in [4, 6, 8, 10, 12, 16, 20]:
        G = honeycomb_2d(L, L, pbc=False)
        sub = subsystem_left_half(L, L, sites_per_cell=2)
        S = ee_corr_matrix(G, sub)
        N = 2 * L * L
        s_per_l = S / L
        results.append([L, N, S, s_per_l])
        print(f"{L:4d}  {N:6d}  {S:10.4f}  {s_per_l:8.4f}")
    
    print()
    return np.array(results)


def main():
    square_data = demo_square()
    honeycomb_data = demo_honeycomb()

    # Save data
    np.savetxt('../data/square_2d_data.txt', square_data,
               header='L  N  S  S_over_L', comments='',
               fmt=['%4d', '%6d', '%10.4f', '%10.4f'])
    np.savetxt('../data/honeycomb_2d_data.txt', honeycomb_data,
               header='L  N  S  S_over_L', comments='',
               fmt=['%4d', '%6d', '%10.4f', '%10.4f'])
    print("Data saved to ../data/square_2d_data.txt and honeycomb_2d_data.txt")

    # Summary comparison
    print("=" * 60)
    print("Summary: Convergence of S/L")
    print("=" * 60)
    print(f"{'L':>4s}  {'Square S/L':>12s}  {'Honeycomb S/L':>15s}")
    print("-" * 36)
    for i in range(len(square_data)):
        L = int(square_data[i, 0])
        s_sq = square_data[i, 3]
        s_hc = honeycomb_data[i, 3]
        print(f"{L:4d}  {s_sq:12.4f}  {s_hc:15.4f}")
    
    print()
    print("Observation:")
    print("- Square: S/L increases slowly (logarithmic enhancement)")
    print("- Honeycomb: S/L converges quickly to ~0.67 (strict area law)")


if __name__ == "__main__":
    main()
