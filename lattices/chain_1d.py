"""
lattices/chain_1d.py — 1D tight-binding chain.
"""

import numpy as np


def chain_1d(L, pbc=False, t=1.0):
    """1D tight-binding chain with nearest-neighbor hopping.

    Hamiltonian: H = -t * sum_i (c_i^dag c_{i+1} + h.c.)

    Parameters
    ----------
    L : int
        Number of sites.
    pbc : bool, optional
        Periodic boundary conditions (default False).
    t : float, optional
        Hopping amplitude (default 1.0). Units are arbitrary energy units.

    Returns
    -------
    G : ndarray, shape (L, L)
        Correlation matrix at half-filling (L//2 electrons).
        G_ij = <c_i^dag c_j>.

    Notes
    -----
    For the open chain (OBC), the single-particle states are standing waves
    with energies epsilon_k = -2t * cos(k), k = n*pi/(L+1), n=1,...,L.

    For the periodic chain (PBC), k = 2*pi*n/L.
    """
    H = np.zeros((L, L))
    for i in range(L - 1):
        H[i, i + 1] = H[i + 1, i] = -t
    if pbc:
        H[0, L - 1] = H[L - 1, 0] = -t
    
    # Diagonalize and fill lowest states
    evals, evecs = np.linalg.eigh(H)
    n_fill = L // 2
    return evecs[:, :n_fill] @ evecs[:, :n_fill].conj().T
