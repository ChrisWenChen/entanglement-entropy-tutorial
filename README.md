# Entanglement Entropy Tutorial

Tutorial codebase for computing entanglement entropy of quantum many-body systems, accompanying the article *Computing Entanglement Entropy: From Partial Trace to Efficient Algorithms and 2D Extensions*.

## Quick Start

```bash
# Install dependencies
pip install numpy scipy scikit-learn pytest

# Run demos
python demos/demo_1d_cft.py        # 1D CFT verification
python demos/demo_2d_area_law.py   # 2D area law comparison
python demos/demo_compare_methods.py  # Three-method comparison

# Run tests
pytest tests/ -v
```

## Project Structure

```
code/entanglement-entropy-tutorial/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── ee/                         # Core computation module
│   ├── __init__.py
│   ├── core.py                # Three entanglement entropy methods
│   │   ├── reshape_psi        # Reshape state vector into coefficient matrix
│   │   ├── ee_method1         # Density matrix eigendecomposition
│   │   ├── ee_method2         # Direct SVD (recommended)
│   │   ├── ee_method3         # Randomized SVD
│   │   ├── rsvd               # Hand-written randomized SVD
│   │   ├── rsvd_sklearn       # sklearn randomized SVD
│   │   ├── compute_entanglement_spectrum  # Entanglement spectrum
│   │   └── compute_renyi_entropy          # Renyi entropy
│   └── free_fermion.py        # Free fermion correlation matrix method
│       ├── ee_corr_matrix     # EE from correlation matrix
│       ├── ee_corr_matrix_renyi
│       └── build_corr_matrix_from_hamiltonian
├── lattices/                   # Lattice construction module
│   ├── __init__.py
│   ├── chain_1d.py            # 1D chain
│   ├── square_2d.py           # 2D square lattice
│   ├── honeycomb_2d.py        # Honeycomb lattice
│   └── subsystems.py          # Subsystem construction utilities
├── demos/                      # Demo scripts
│   ├── demo_1d_cft.py         # 1D CFT verification
│   ├── demo_2d_area_law.py    # 2D area law comparison
│   └── demo_compare_methods.py # Comprehensive three-method comparison
├── tests/                      # Unit tests
│   ├── test_methods.py        # Computation method tests
│   └── test_lattices.py       # Lattice tests
├── data/                       # Data output directory
└── figs/                       # Figure output directory
```

## Core Features

### 1. General Methods (Any Quantum State)

```python
import numpy as np
from ee import ee_method1, ee_method2, ee_method3

# Bell state: (|00⟩ + |11⟩)/√2
psi = np.array([1, 0, 0, 1]) / np.sqrt(2)

# Three methods for computing entanglement entropy
S1 = ee_method1(psi, N=2, NA=1)  # Density matrix eigendecomposition
S2 = ee_method2(psi, N=2, NA=1)  # Direct SVD (recommended)
S3 = ee_method3(psi, N=2, NA=1, k=2)  # Randomized SVD

print(f"S = {S2:.4f}")  # Should be ln(2) ≈ 0.6931
```

### 2. Randomized SVD: Hand-Written vs sklearn

```python
from ee import rsvd, rsvd_sklearn
import numpy as np

C = np.random.randn(100, 200)

# Hand-written implementation
s_custom = rsvd(C, k=10, p=5, n_iter=2)

# sklearn implementation
s_sklearn = rsvd_sklearn(C, k=10, n_oversamples=5, n_iter=2)
```

### 3. Free Fermion Correlation Matrix Method

```python
from ee import ee_corr_matrix
from lattices import chain_1d

# Build correlation matrix for a 1D chain
L = 100
G = chain_1d(L, pbc=False)

# Compute half-chain entanglement entropy
S = ee_corr_matrix(G, list(range(L // 2)))
print(f"Half-chain EE for L={L}: S = {S:.4f}")
```

### 4. 2D Lattice Systems

```python
from ee import ee_corr_matrix
from lattices import square_2d, honeycomb_2d, subsystem_left_half

# Square lattice
G_sq = square_2d(10, 10, pbc=False)
sub_sq = subsystem_left_half(10, 10, sites_per_cell=1)
S_sq = ee_corr_matrix(G_sq, sub_sq)

# Honeycomb lattice
G_hc = honeycomb_2d(10, 10, pbc=False)
sub_hc = subsystem_left_half(10, 10, sites_per_cell=2)
S_hc = ee_corr_matrix(G_hc, sub_hc)
```

## Mapping to the Tutorial

| Tutorial Section | Code Location |
|-----------------|---------------|
| Sec. 2-3: Partial trace and Schmidt decomposition | `ee/core.py::reshape_psi` |
| Sec. 4: Hand-calculation exercises | `tests/test_methods.py` |
| Sec. 5.1: Density matrix eigendecomposition | `ee/core.py::ee_method1` |
| Sec. 5.2: Direct SVD | `ee/core.py::ee_method2` |
| Sec. 5.3: Advantages of Method 2 | `demos/demo_compare_methods.py` |
| Sec. 5.4: Randomized SVD | `ee/core.py::rsvd`, `rsvd_sklearn` |
| Sec. 6: Correlation matrix method | `ee/free_fermion.py::ee_corr_matrix` |
| Sec. 7: 1D CFT verification | `demos/demo_1d_cft.py` |
| Sec. 8: 2D extensions | `demos/demo_2d_area_law.py` |

## Method Comparison

| Method | Time Complexity | Space Complexity | Stability | Use Case |
|--------|----------------|-----------------|-----------|----------|
| Method 1 (rho eigenvalues) | O(d_A^3) | O(d_A^2) | Poor (condition number squared) | When rho_A is needed |
| Method 2 (SVD) | O(d_A^2 d_B) | O(d_A d_B) | Good | **General default** |
| Method 3 (rSVD) | O(d_A d_B k) | O(d_A k) | Good | Area-law states |

## Verification Results

### 1D CFT Verification (Calabrese-Cardy)

For the XX chain with open boundaries, the half-chain entanglement entropy should satisfy:
```
S = (c/6) ln(2L/pi) + const,  c = 1
```

Numerical result: fitted coefficient ~ 0.164, within < 2% of the theoretical value 1/6 ~ 0.167.

### 2D Area Law Comparison

- **Square lattice** (has a Fermi surface): S/L ~ ln(L) logarithmic divergence
- **Honeycomb lattice** (Dirac points): S/L -> constant (strict area law)

## References

1. Eisert, Cramer & Plenio, Rev. Mod. Phys. 82, 277 (2010).
2. Calabrese & Cardy, J. Phys. A 42, 504005 (2009).
3. Vidal et al., PRL 90, 227902 (2003).
4. Peschel, J. Phys. A 36, L205 (2003).
5. Gioev & Klich, PRL 96, 100503 (2006).
6. Halko, Martinsson & Tropp, SIAM Rev. 53, 217 (2011).
7. D'Emidio et al., PRL 132, 076502 (2024).

## Author

Based on the tutorial *Computing Entanglement Entropy: From Partial Trace to Efficient Algorithms and 2D Extensions*.
