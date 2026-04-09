"""
ee/core.py — Entanglement entropy computation: three methods.

Method 1: Density matrix eigendecomposition (rho_A = C C^dag, then eigh)
Method 2: Direct SVD of coefficient matrix C (recommended)
Method 3: Randomized SVD for low-rank approximation (area-law states)

References
----------
[1] Halko, Martinsson & Tropp, SIAM Rev. 53, 217 (2011) — randomized SVD.
"""

import numpy as np


def reshape_psi(psi, N, NA=None):
    """Reshape a state vector into the coefficient matrix C[i_A, j_B].

    Parameters
    ----------
    psi : ndarray, shape (d,)
        State vector in the computational basis |0...0>, |0...1>, ...
        For a spin-1/2 chain, d = 2^N.
    N : int
        Total number of sites.
    NA : int or None
        Number of sites in subsystem A (default: N // 2).

    Returns
    -------
    C : ndarray, shape (d_A, d_B)
        Coefficient matrix, where d_A = 2^NA, d_B = 2^(N - NA).

    Notes
    -----
    The indexing follows NumPy's C-order (row-major): the last index changes fastest.
    For a bipartition where A is the first NA sites and B is the remaining N-NA sites,
    this corresponds to interpreting the state vector as a matrix where rows are
    A-basis states and columns are B-basis states.
    """
    if NA is None:
        NA = N // 2
    dA = 2 ** NA
    return psi.reshape(dA, -1)


def ee_method1(psi, N, NA=None):
    """Entanglement entropy via rho_A = C C^dag, then eigendecomposition.

    This is the most direct method: construct the reduced density matrix
    explicitly and diagonalize it.

    Parameters
    ----------
    psi : ndarray, shape (2^N,)
        State vector in computational basis.
    N : int
        Total number of sites.
    NA : int or None, optional
        Number of sites in subsystem A. Default is N//2.

    Returns
    -------
    S : float
        Von Neumann entanglement entropy S_A = -Tr(rho_A log rho_A).

    Notes
    -----
    Numerical stability: This method squares the condition number of C when
    forming rho_A = C C^dag. For ill-conditioned C (e.g., area-law states with
    rapidly decaying Schmidt spectrum), this can lose precision.

    Complexity: O(d_A^2 * d_B) for matrix multiplication + O(d_A^3) for
    eigendecomposition, where d_A = 2^NA, d_B = 2^(N-NA).
    """
    C = reshape_psi(psi, N, NA)
    rho_A = C @ C.conj().T          # d_A x d_A
    evals = np.linalg.eigvalsh(rho_A)
    evals = evals[evals > 1e-14]    # Remove numerical zeros
    return -np.dot(evals, np.log(evals))


def ee_method2(psi, N, NA=None):
    """Entanglement entropy via SVD of the coefficient matrix C (recommended).

    This is the preferred method for general use. It avoids explicitly
    constructing rho_A and directly computes the Schmidt coefficients.

    Parameters
    ----------
    psi : ndarray, shape (2^N,)
        State vector in computational basis.
    N : int
        Total number of sites.
    NA : int or None, optional
        Number of sites in subsystem A. Default is N//2.

    Returns
    -------
    S : float
        Von Neumann entanglement entropy.

    Notes
    -----
    Numerical stability: Much better than Method 1. The SVD works directly
    on C without squaring the condition number.

    Complexity: O(d_A^2 * d_B) for SVD (with d_A <= d_B).

    The entropy can be computed directly from singular values s_k as:
        S = -sum_k s_k^2 * log(s_k^2)
    """
    C = reshape_psi(psi, N, NA)
    sv = np.linalg.svd(C, compute_uv=False)   # Singular values only
    lam = sv ** 2
    lam = lam[lam > 1e-14]
    return -np.dot(lam, np.log(lam))


def rsvd(C, k, p=5, n_iter=2):
    """Randomized SVD: approximate top-k singular values of C (hand-written).

    Implements the basic randomized SVD algorithm from Halko et al. (2011).
    
    Algorithm:
      1. Random projection: Y = C @ Omega,  Omega ~ N(0,1) of shape (n, k+p)
      2. Power iteration (n_iter rounds) to sharpen the range approximation
      3. QR decomposition: Y = Q R
      4. Project: B = Q^H @ C  (small matrix, (k+p) x n)
      5. SVD of B

    Parameters
    ----------
    C : ndarray, shape (m, n)
        Input matrix.
    k : int
        Target rank (number of singular values to compute).
    p : int, optional
        Oversampling parameter (default 5). More oversampling improves accuracy
        at the cost of slightly more computation.
    n_iter : int, optional
        Number of power iterations (default 2). More iterations help when
        the singular value spectrum decays slowly.

    Returns
    -------
    sv : ndarray, shape (k,)
        Approximate top-k singular values in descending order.

    Notes
    -----
    This is our hand-written implementation. For production use, consider
    sklearn.utils.extmath.randomized_svd which has more optimizations.

    Power iteration: Each iteration multiplies by (C @ C.H) which sharpens
    the spectral gap between the top k singular values and the rest.
    """
    m, n = C.shape
    if k > min(m, n):
        raise ValueError(f"k={k} exceeds matrix dimensions ({m}, {n})")
    
    # Step 1: Random projection
    Omega = np.random.randn(n, k + p)
    Y = C @ Omega                       # m x (k+p)
    
    # Step 2: Power iteration to sharpen the approximation
    # Use conj().T to support complex matrices
    for _ in range(n_iter):
        Y = C @ (C.conj().T @ Y)
    
    # Step 3: QR decomposition to get orthonormal basis
    Q, _ = np.linalg.qr(Y)             # m x (k+p)
    
    # Step 4: Project to low-dimensional subspace
    B = Q.conj().T @ C                  # (k+p) x n
    
    # Step 5: SVD of small matrix
    _, s, _ = np.linalg.svd(B, full_matrices=False)
    return s[:k]


def rsvd_sklearn(C, k, n_oversamples=5, n_iter=2):
    """Randomized SVD using scikit-learn (library version).

    This wraps sklearn.utils.extmath.randomized_svd for comparison.

    Parameters
    ----------
    C : ndarray, shape (m, n)
        Input matrix.
    k : int
        Target rank.
    n_oversamples : int, optional
        Oversampling parameter (default 5). Equivalent to 'p' in rsvd().
    n_iter : int, optional
        Number of power iterations (default 2).

    Returns
    -------
    sv : ndarray, shape (k,)
        Top-k singular values in descending order.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    """
    try:
        from sklearn.utils.extmath import randomized_svd
    except ImportError:
        raise ImportError(
            "scikit-learn is required for rsvd_sklearn. "
            "Install with: pip install scikit-learn"
        )
    
    U, s, Vt = randomized_svd(
        C, 
        n_components=k,
        n_oversamples=n_oversamples,
        n_iter=n_iter,
        power_iteration_normalizer='QR'
    )
    return s


def ee_method3(psi, N, k, NA=None, use_sklearn=False, **rsvd_kwargs):
    """Entanglement entropy via randomized SVD with target rank k.

    Parameters
    ----------
    psi : ndarray, shape (2^N,)
        State vector in computational basis.
    N : int
        Total number of sites.
    k : int
        Target Schmidt rank (number of singular values to approximate).
    NA : int or None, optional
        Number of sites in subsystem A. Default is N//2.
    use_sklearn : bool, optional
        If True, use sklearn's randomized_svd. If False (default), use
        our hand-written implementation.
    **rsvd_kwargs
        Additional arguments passed to rsvd() or rsvd_sklearn().
        Common options: p (oversampling), n_iter (power iterations).

    Returns
    -------
    S : float
        Approximate von Neumann entanglement entropy.

    Notes
    -----
    Randomized SVD is most effective for area-law states where the Schmidt
    rank is much smaller than d_A. For volume-law states, you need k ≈ d_A
    and there's no advantage over full SVD.

    Because we truncate to k singular values, the result may underestimate
    the true entropy. We renormalize the probabilities to sum to 1, which
    partially compensates for this.

    Examples
    --------
    >>> psi = np.random.randn(2**10)
    >>> psi /= np.linalg.norm(psi)
    >>> S_approx = ee_method3(psi, 10, k=8)  # Much faster than full SVD
    """
    C = reshape_psi(psi, N, NA)
    
    if use_sklearn:
        sv = rsvd_sklearn(C, k, **rsvd_kwargs)
    else:
        sv = rsvd(C, k, **rsvd_kwargs)
    
    lam = sv ** 2
    lam = lam[lam > 1e-14]
    # Renormalize because truncation changes the norm
    lam = lam / lam.sum()
    return -np.dot(lam, np.log(lam))


def compute_entanglement_spectrum(psi, N, NA=None):
    """Compute the full entanglement spectrum (eigenvalues of rho_A).

    Parameters
    ----------
    psi : ndarray, shape (2^N,)
        State vector.
    N : int
        Total number of sites.
    NA : int or None, optional
        Number of sites in subsystem A.

    Returns
    -------
    spec : ndarray
        Entanglement spectrum (Schmidt weights) in descending order.
        These are the eigenvalues of rho_A, i.e., lambda_k = s_k^2.
    """
    C = reshape_psi(psi, N, NA)
    sv = np.linalg.svd(C, compute_uv=False)
    return sv ** 2


def compute_renyi_entropy(psi, N, n, NA=None):
    """Compute the Rényi entropy of order n.

    S_n = (1/(1-n)) * log(Tr(rho_A^n))

    For n=1, this reduces to von Neumann entropy (computed via SVD).

    Parameters
    ----------
    psi : ndarray, shape (2^N,)
        State vector.
    N : int
        Total number of sites.
    n : float
        Rényi index. n=1 is von Neumann, n=2 is second Rényi, etc.
    NA : int or None, optional
        Number of sites in subsystem A.

    Returns
    -------
    S_n : float
        Rényi entropy of order n.

    Notes
    -----
    For integer n >= 2, can be computed directly from Schmidt coefficients:
        Tr(rho_A^n) = sum_k lambda_k^n = sum_k s_k^(2n)
    """
    if abs(n - 1) < 1e-10:
        return ee_method2(psi, N, NA)
    
    spec = compute_entanglement_spectrum(psi, N, NA)
    spec = spec[spec > 1e-14]
    
    if n == np.inf:
        # Min-entropy: S_inf = -log(lambda_max)
        return -np.log(spec[0])
    
    trace_rho_n = np.sum(spec ** n)
    return np.log(trace_rho_n) / (1 - n)
