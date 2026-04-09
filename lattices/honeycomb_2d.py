"""
lattices/honeycomb_2d.py — Honeycomb lattice tight-binding model.
"""

import numpy as np


def honeycomb_2d(Lx, Ly, pbc=False, t=1.0):
    """Honeycomb lattice tight-binding model with nearest-neighbor hopping.

    Each unit cell (x, y) contains two sites: A (sublattice 0) and B (sublattice 1).
    Site index: (x, y, s) -> 2 * (x * Ly + y) + s

    Nearest-neighbor bonds (for OBC):
      - A(x, y) -- B(x, y)         intra-cell (same unit cell)
      - A(x, y) -- B(x-1, y)       inter-cell, -x direction
      - A(x, y) -- B(x, y-1)       inter-cell, -y direction

    This creates the hexagonal coordination of the honeycomb lattice.

    Parameters
    ----------
    Lx, Ly : int
        Number of unit cells in x and y directions.
    pbc : bool, optional
        Periodic boundary conditions (default False).
    t : float, optional
        Hopping amplitude (default 1.0).

    Returns
    -------
    G : ndarray, shape (N, N) where N = 2 * Lx * Ly
        Correlation matrix at half-filling.

    Notes
    -----
    The honeycomb lattice has two Dirac points in the Brillouin zone
    (at K and K' points). At half-filling, the Fermi level is exactly
    at the Dirac points. Unlike the square lattice, there is no extended
    Fermi surface, just two points. This leads to different entanglement
    entropy scaling: S ~ alpha * L_y + subleading log corrections,
    without the L_y * log(L_y) term seen in the square lattice.
    """
    N = 2 * Lx * Ly
    H = np.zeros((N, N))
    
    def idx(x, y, s):
        """Map (x, y, sublattice) to linear index."""
        return 2 * ((x % Lx) * Ly + (y % Ly)) + s
    
    for x in range(Lx):
        for y in range(Ly):
            iA = idx(x, y, 0)  # A sublattice
            
            # Bond 1: A(x,y) -- B(x,y) [intra-cell]
            iB = idx(x, y, 1)
            H[iA, iB] = H[iB, iA] = -t
            
            # Bond 2: A(x,y) -- B(x-1,y) [inter-cell, -x]
            if x > 0 or pbc:
                iB2 = idx(x - 1, y, 1)
                H[iA, iB2] = H[iB2, iA] = -t
            
            # Bond 3: A(x,y) -- B(x,y-1) [inter-cell, -y]
            if y > 0 or pbc:
                iB3 = idx(x, y - 1, 1)
                H[iA, iB3] = H[iB3, iA] = -t
    
    evals, evecs = np.linalg.eigh(H)
    n_fill = N // 2
    return evecs[:, :n_fill] @ evecs[:, :n_fill].conj().T
