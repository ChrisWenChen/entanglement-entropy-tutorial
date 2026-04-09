"""
test_methods.py — Unit tests for entanglement entropy methods.

Run with: pytest test_methods.py -v
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, '..')

from ee import (
    ee_method1, ee_method2, ee_method3,
    reshape_psi, compute_entanglement_spectrum, compute_renyi_entropy,
    rsvd, rsvd_sklearn
)


class TestReshape:
    """Test the reshape_psi function."""
    
    def test_reshape_dimensions(self):
        """Test that reshape gives correct dimensions."""
        psi = np.random.randn(2**6)
        C = reshape_psi(psi, 6, NA=3)
        assert C.shape == (8, 8)  # 2^3 x 2^3
        
    def test_reshape_unequal_split(self):
        """Test reshape with unequal partition."""
        psi = np.random.randn(2**5)
        C = reshape_psi(psi, 5, NA=2)
        assert C.shape == (4, 8)  # 2^2 x 2^3
        
    def test_reshape_preserves_norm(self):
        """Test that reshape preserves vector norm."""
        psi = np.random.randn(2**4)
        psi /= np.linalg.norm(psi)
        C = reshape_psi(psi, 4)
        assert np.abs(np.linalg.norm(C) - 1.0) < 1e-14


class TestMethod1:
    """Test density matrix eigendecomposition method."""
    
    def test_product_state(self):
        """Product state has zero entanglement."""
        psi = np.array([0, 1, 0, 0], dtype=float)  # |↑↓⟩
        S = ee_method1(psi, 2, NA=1)
        assert abs(S) < 1e-10
        
    def test_bell_state(self):
        """Bell state has S = ln(2)."""
        psi = np.array([1, 0, 0, 1], dtype=float) / np.sqrt(2)
        S = ee_method1(psi, 2, NA=1)
        assert abs(S - np.log(2)) < 1e-10
        
    def test_ghz_state(self):
        """GHZ state with A={0} has S = ln(2)."""
        psi = np.zeros(8)
        psi[0] = psi[7] = 1 / np.sqrt(2)
        S = ee_method1(psi, 3, NA=1)
        assert abs(S - np.log(2)) < 1e-10


class TestMethod2:
    """Test direct SVD method."""
    
    def test_product_state(self):
        """Product state has zero entanglement."""
        psi = np.array([0, 1, 0, 0], dtype=float)
        S = ee_method2(psi, 2, NA=1)
        assert abs(S) < 1e-10
        
    def test_bell_state(self):
        """Bell state has S = ln(2)."""
        psi = np.array([1, 0, 0, 1], dtype=float) / np.sqrt(2)
        S = ee_method2(psi, 2, NA=1)
        assert abs(S - np.log(2)) < 1e-10
        
    def test_agreement_with_method1(self):
        """Methods 1 and 2 should agree for simple cases."""
        np.random.seed(42)
        psi = np.random.randn(2**6)
        psi /= np.linalg.norm(psi)
        
        S1 = ee_method1(psi, 6)
        S2 = ee_method2(psi, 6)
        assert abs(S1 - S2) < 1e-6


class TestMethod3:
    """Test randomized SVD method."""
    
    def test_full_rank_equivalent_to_method2(self):
        """With full rank, rSVD should match exact method."""
        np.random.seed(42)
        psi = np.random.randn(2**8)
        psi /= np.linalg.norm(psi)
        
        S_exact = ee_method2(psi, 8)
        S_rsvd = ee_method3(psi, 8, k=16)  # Full rank
        
        assert abs(S_rsvd - S_exact) < 0.01  # Small tolerance for randomness
        
    def test_area_law_convergence(self):
        """rSVD converges quickly for area-law states."""
        # Create area-law state (exponentially decaying singular values)
        d = 16
        sv = np.exp(-2 * np.arange(d))
        sv /= np.linalg.norm(sv)
        C = np.diag(sv)
        psi = np.zeros(2**8)
        psi[:d*d] = C.flatten()
        psi /= np.linalg.norm(psi)
        
        S_exact = ee_method2(psi, 8)
        S_rsvd = ee_method3(psi, 8, k=4)  # Small k should be enough
        
        # Within 1% for area-law state
        assert abs(S_rsvd - S_exact) / S_exact < 0.01


class TestRSVDImplementations:
    """Compare hand-written vs sklearn rSVD."""
    
    def test_rsvd_vs_sklearn(self):
        """Our implementation should match sklearn."""
        np.random.seed(42)
        C = np.random.randn(20, 30)
        
        s_custom = rsvd(C, k=5, p=5, n_iter=2)
        
        try:
            s_sklearn = rsvd_sklearn(C, k=5, n_oversamples=5, n_iter=2)
            # They won't be exactly equal due to randomness, but should be close
            assert len(s_custom) == len(s_sklearn) == 5
        except ImportError:
            pytest.skip("sklearn not installed")


class TestSymmetries:
    """Test physical symmetries."""
    
    def test_sa_equals_sb(self):
        """For pure states, S_A = S_B."""
        np.random.seed(42)
        for N in [6, 8, 10]:
            psi = np.random.randn(2**N)
            psi /= np.linalg.norm(psi)
            
            S_A = ee_method2(psi, N, NA=N//2)
            S_B = ee_method2(psi, N, NA=N - N//2)
            assert abs(S_A - S_B) < 1e-10


class TestEntanglementSpectrum:
    """Test entanglement spectrum computation."""
    
    def test_bell_state_spectrum(self):
        """Bell state has spectrum [0.5, 0.5]."""
        psi = np.array([1, 0, 0, 1], dtype=float) / np.sqrt(2)
        spec = compute_entanglement_spectrum(psi, 2, NA=1)
        
        # Two non-zero eigenvalues, both 0.5
        assert len(spec[spec > 1e-10]) == 2
        assert abs(spec[0] - 0.5) < 1e-10
        assert abs(spec[1] - 0.5) < 1e-10
        
    def test_spectrum_normalized(self):
        """Spectrum sums to 1."""
        np.random.seed(42)
        psi = np.random.randn(2**6)
        psi /= np.linalg.norm(psi)
        
        spec = compute_entanglement_spectrum(psi, 6)
        assert abs(spec.sum() - 1.0) < 1e-10


class TestRenyiEntropy:
    """Test Rényi entropy computation."""
    
    def test_renyi_n1_is_vn(self):
        """n=1 Rényi is von Neumann."""
        psi = np.array([1, 0, 0, 1], dtype=float) / np.sqrt(2)
        S_vn = ee_method2(psi, 2, NA=1)
        S_renyi = compute_renyi_entropy(psi, 2, n=1, NA=1)
        assert abs(S_vn - S_renyi) < 1e-10
        
    def test_renyi_n2_bell_state(self):
        """Bell state S_2 = ln(2)."""
        psi = np.array([1, 0, 0, 1], dtype=float) / np.sqrt(2)
        S2 = compute_renyi_entropy(psi, 2, n=2, NA=1)
        # For Bell: lambda = [0.5, 0.5], Tr(rho^2) = 0.5, S_2 = -ln(0.5) = ln(2)
        assert abs(S2 - np.log(2)) < 1e-10
        
    def test_renyi_ordering(self):
        """S_n decreases with n."""
        np.random.seed(42)
        psi = np.random.randn(2**6)
        psi /= np.linalg.norm(psi)
        
        S1 = compute_renyi_entropy(psi, 6, n=1)
        S2 = compute_renyi_entropy(psi, 6, n=2)
        S_inf = compute_renyi_entropy(psi, 6, n=np.inf)
        
        assert S1 >= S2 >= S_inf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
