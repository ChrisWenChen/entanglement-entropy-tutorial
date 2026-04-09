"""
lattices/square_2d.py — 2D square lattice tight-binding model.
"""

import numpy as np


def square_2d(Lx, Ly, pbc=False, t=1.0):
    """2D square lattice tight-binding model with nearest-neighbor hopping.

    Hamiltonian: H = -t * sum_{<i,j>} (c_i^dag c_j + h.c.)

    Site index mapping: (x, y) -> x * Ly + y
    where x = 0,...,Lx-1 and y = 0,...,Ly-1.

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions in x and y directions.
    pbc : bool, optional
        Periodic boundary conditions (default False).
    t : float, optional
        Hopping amplitude (default 1.0).

    Returns
    -------
    G : ndarray, shape (N, N) where N = Lx * Ly
        Correlation matrix at half-filling.

    Notes
    -----
    At half-filling, the Fermi surface is a square in the Brillouin zone
    with |k_x| + |k_y| = pi (for t=1). This leads to a logarithmic
    enhancement of the area law: S ~ alpha * L_y * log(L_y).
    """
    N = Lx * Ly
    H = np.zeros((N, N))
    
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            
            # Right neighbor (x+1, y)
            if x + 1 < Lx:
                j = (x + 1) * Ly + y
                H[i, j] = H[j, i] = -t
            elif pbc and Lx > 1:
                j = y  # (0, y)
                H[i, j] = H[j, i] = -t
            
            # Up neighbor (x, y+1)
            if y + 1 < Ly:
                j = x * Ly + (y + 1)
                H[i, j] = H[j, i] = -t
            elif pbc and Ly > 1:
                j = x * Ly  # (x, 0)
                H[i, j] = H[j, i] = -t
    
    evals, evecs = np.linalg.eigh(H)
    n_fill = N // 2
    return evecs[:, :n_fill] @ evecs[:, :n_fill].conj().T
