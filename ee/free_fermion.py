"""
ee/free_fermion.py — Free fermion entanglement via correlation matrix.

For non-interacting fermions (Slater determinant states), the entanglement
entropy can be computed much more efficiently from the single-particle
correlation matrix, avoiding the exponential cost of the full Hilbert space.

Theorem (Peschel 2003): For a free fermion state, the reduced density matrix
rho_A is Gaussian and completely determined by the restricted correlation
matrix G_A. The entanglement entropy is:

    S = -sum_k [xi_k ln(xi_k) + (1-xi_k) ln(1-xi_k)]

where xi_k are the eigenvalues of G_A.

References
----------
[1] Peschel, J. Phys. A 36, L205 (2003).
[2] Gioev & Klich, PRL 96, 100503 (2006).
"""

import numpy as np


def ee_corr_matrix(G, subsystem):
    """Entanglement entropy from the single-particle correlation matrix.

    For non-interacting fermions at zero temperature, the many-body ground
    state is a Slater determinant. The entanglement entropy of a subsystem
    can be computed directly from the eigenvalues of the restricted
    correlation matrix G_A.

    Parameters
    ----------
    G : ndarray, shape (N, N)
        Single-particle correlation matrix G_ij = <c_i^dag c_j>,
        where i, j run over all N lattice sites.
    subsystem : array-like of int
        Site indices belonging to subsystem A. These select the rows/columns
        to extract from G.

    Returns
    -------
    S : float
        Von Neumann entanglement entropy S_A.

    Notes
    -----
    The correlation matrix G for a Slater determinant at zero temperature is:
        G_ij = sum_{k in occupied} phi_k(i)* phi_k(j)
    where phi_k are the occupied single-particle orbitals.

    The eigenvalues xi_k of G_A satisfy 0 <= xi_k <= 1 and represent
    the occupation probabilities of the "natural orbitals" (entanglement
    modes) in subsystem A. Each mode contributes:
        -xi ln(xi) - (1-xi) ln(1-xi)
    to the entropy.

    Complexity: O(|A|^3) where |A| is the subsystem size.
    This is exponentially faster than the generic SVD method for large systems.

    Important limitation: This method assumes number-conserving fermions
    (no pairing/anomalous correlations). For BCS-type states with pairing,
    one needs the full Nambu covariance matrix (see Li & Haldane 2008).

    Examples
    --------
    >>> # G is a correlation matrix from diagonalizing a tight-binding Hamiltonian
    >>> subsystem = list(range(L//2))  # First half of the chain
    >>> S = ee_corr_matrix(G, subsystem)
    """
    sub = np.asarray(subsystem)
    G_A = G[np.ix_(sub, sub)]
    xi = np.linalg.eigvalsh(G_A)
    # Clip to avoid log(0) numerical issues
    xi = np.clip(xi, 1e-14, 1 - 1e-14)
    return -np.sum(xi * np.log(xi) + (1 - xi) * np.log(1 - xi))


def ee_corr_matrix_renyi(G, subsystem, n):
    """Rényi entropy from the correlation matrix.

    For integer Rényi index n, the Rényi entropy can be computed from
    the eigenvalues xi_k of G_A:

        S_n = (1/(1-n)) * sum_k ln[xi_k^n + (1-xi_k)^n]

    Parameters
    ----------
    G : ndarray, shape (N, N)
        Single-particle correlation matrix.
    subsystem : array-like of int
        Site indices of subsystem A.
    n : float
        Rényi index. n=1 gives von Neumann (use ee_corr_matrix instead).

    Returns
    -------
    S_n : float
        Rényi entropy of order n.
    """
    if abs(n - 1) < 1e-10:
        return ee_corr_matrix(G, subsystem)
    
    sub = np.asarray(subsystem)
    G_A = G[np.ix_(sub, sub)]
    xi = np.linalg.eigvalsh(G_A)
    xi = np.clip(xi, 1e-14, 1 - 1e-14)
    
    # S_n = (1/(1-n)) * sum_k ln[xi_k^n + (1-xi_k)^n]
    term = xi**n + (1 - xi)**n
    return np.sum(np.log(term)) / (1 - n)


def build_corr_matrix_from_hamiltonian(H, filling=0.5):
    """Build correlation matrix from single-particle Hamiltonian at given filling.

    Parameters
    ----------
    H : ndarray, shape (N, N)
        Single-particle Hamiltonian matrix.
    filling : float, optional
        Filling fraction (default 0.5 = half-filling).

    Returns
    -------
    G : ndarray, shape (N, N)
        Correlation matrix G_ij = sum_{k < n_fill} phi_k(i)* phi_k(j)
    """
    N = H.shape[0]
    evals, evecs = np.linalg.eigh(H)
    n_fill = int(N * filling)
    # G = sum_{occupied} |phi_k><phi_k|
    return evecs[:, :n_fill] @ evecs[:, :n_fill].conj().T
