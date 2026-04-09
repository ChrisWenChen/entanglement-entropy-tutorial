"""
ee — Entanglement entropy computation package.

This package provides methods to compute the entanglement entropy of
quantum many-body states.

Submodules
----------
core
    General methods for any quantum state:
    - Method 1: Density matrix eigendecomposition
    - Method 2: Direct SVD (recommended)
    - Method 3: Randomized SVD (for area-law states)

free_fermion
    Efficient methods for free fermion systems:
    - Correlation matrix method (Peschel 2003)

Examples
--------
>>> import numpy as np
>>> from ee import ee_method2, reshape_psi
>>>
>>> # Bell state (|00> + |11>)/sqrt(2)
>>> psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
>>> S = ee_method2(psi, N=2, NA=1)
>>> print(f"Bell state EE = {S:.4f}")  # Should be ln(2) ≈ 0.6931
"""

from .core import (
    reshape_psi,
    ee_method1,
    ee_method2,
    ee_method3,
    rsvd,
    rsvd_sklearn,
    compute_entanglement_spectrum,
    compute_renyi_entropy,
)

from .free_fermion import (
    ee_corr_matrix,
    ee_corr_matrix_renyi,
    build_corr_matrix_from_hamiltonian,
)

__version__ = "0.1.0"

__all__ = [
    # Core methods
    "reshape_psi",
    "ee_method1",
    "ee_method2", 
    "ee_method3",
    "rsvd",
    "rsvd_sklearn",
    "compute_entanglement_spectrum",
    "compute_renyi_entropy",
    # Free fermion methods
    "ee_corr_matrix",
    "ee_corr_matrix_renyi",
    "build_corr_matrix_from_hamiltonian",
]