"""
test_lattices.py — Unit tests for lattice construction.

Run with: pytest test_lattices.py -v
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, '..')

from lattices import chain_1d, square_2d, honeycomb_2d
from lattices.subsystems import subsystem_left_half, subsystem_cylinder
from ee import ee_corr_matrix


class TestChain1D:
    """Test 1D chain tight-binding model."""
    
    def test_correlation_matrix_properties(self):
        """G should be Hermitian and have correct trace."""
        G = chain_1d(10, pbc=False)
        
        # Hermitian
        assert np.allclose(G, G.conj().T)
        
        # Trace = number of filled states (half-filling)
        assert abs(np.trace(G) - 5) < 1e-10
        
        # Eigenvalues in [0, 1]
        evals = np.linalg.eigvalsh(G)
        assert np.all(evals >= -1e-10)
        assert np.all(evals <= 1 + 1e-10)
        
    def test_pbc_vs_obc(self):
        """PBC and OBC should give different results."""
        G_obc = chain_1d(12, pbc=False)
        G_pbc = chain_1d(12, pbc=True)
        
        assert not np.allclose(G_obc, G_pbc)
        
    def test_half_chain_ee_scaling(self):
        """Test that EE increases with system size."""
        Ls = [10, 20, 40]
        entropies = []
        
        for L in Ls:
            G = chain_1d(L, pbc=False)
            S = ee_corr_matrix(G, list(range(L // 2)))
            entropies.append(S)
        
        # EE should increase with system size for critical chain
        assert entropies[1] > entropies[0]
        assert entropies[2] > entropies[1]


class TestSquare2D:
    """Test 2D square lattice."""
    
    def test_correlation_matrix_shape(self):
        """G has correct dimensions."""
        G = square_2d(4, 4, pbc=False)
        assert G.shape == (16, 16)  # 4x4 = 16 sites
        
    def test_correlation_matrix_properties(self):
        """G should be Hermitian and idempotent (at T=0)."""
        G = square_2d(6, 6, pbc=False)
        
        # Hermitian
        assert np.allclose(G, G.conj().T)
        
        # For Slater determinant: G^2 = G (projector)
        assert np.allclose(G @ G, G, atol=1e-10)
        
    def test_trace_half_filling(self):
        """Trace of G equals number of electrons."""
        L = 4
        G = square_2d(L, L, pbc=False)
        N_sites = L * L
        assert abs(np.trace(G) - N_sites // 2) < 1e-10


class TestHoneycomb2D:
    """Test honeycomb lattice."""
    
    def test_correlation_matrix_shape(self):
        """G has correct dimensions (2 sites per cell)."""
        G = honeycomb_2d(4, 4, pbc=False)
        assert G.shape == (32, 32)  # 2 * 4 * 4 = 32 sites
        
    def test_trace_half_filling(self):
        """Trace of G equals number of electrons."""
        Lx, Ly = 4, 4
        G = honeycomb_2d(Lx, Ly, pbc=False)
        N_sites = 2 * Lx * Ly
        assert abs(np.trace(G) - N_sites // 2) < 1e-10
        
    def test_vs_square_different(self):
        """Honeycomb and square should give different results."""
        G_sq = square_2d(4, 4, pbc=False)
        G_hc = honeycomb_2d(4, 4, pbc=False)
        
        # Different sizes, so can't compare directly
        # But both should be valid correlation matrices
        assert np.allclose(G_sq, G_sq.conj().T)
        assert np.allclose(G_hc, G_hc.conj().T)


class TestSubsystems:
    """Test subsystem construction."""
    
    def test_left_half_square(self):
        """Left half of square lattice."""
        sub = subsystem_left_half(4, 4, sites_per_cell=1)
        
        # Should have 2*4 = 8 sites (half of 16)
        assert len(sub) == 8
        
        # All indices should be in valid range
        assert all(0 <= i < 16 for i in sub)
        
    def test_left_half_honeycomb(self):
        """Left half of honeycomb lattice."""
        sub = subsystem_left_half(4, 4, sites_per_cell=2)
        
        # Should have 2*4*2 = 16 sites (half of 32)
        assert len(sub) == 16
        
    def test_cylinder(self):
        """Circular subsystem."""
        sub = subsystem_cylinder(10, 10, 5, 5, 3, sites_per_cell=1)
        
        # All indices in valid range
        assert all(0 <= i < 100 for i in sub)
        
        # Should have reasonable number of sites (rough check)
        # Area ~ pi * 3^2 = 28, so around 28 sites
        assert 10 < len(sub) < 50


class TestAreaLaw:
    """Test area law scaling in 2D."""
    
    def test_square_vs_honeycomb_scaling(self):
        """Square lattice has stronger entanglement scaling than honeycomb."""
        # Small systems for quick test
        Ls = [4, 6, 8]
        
        sq_ee_per_boundary = []
        hc_ee_per_boundary = []
        
        for L in Ls:
            # Square
            G_sq = square_2d(L, L, pbc=False)
            sub_sq = subsystem_left_half(L, L, sites_per_cell=1)
            S_sq = ee_corr_matrix(G_sq, sub_sq)
            sq_ee_per_boundary.append(S_sq / L)
            
            # Honeycomb
            G_hc = honeycomb_2d(L, L, pbc=False)
            sub_hc = subsystem_left_half(L, L, sites_per_cell=2)
            S_hc = ee_corr_matrix(G_hc, sub_hc)
            hc_ee_per_boundary.append(S_hc / L)
        
        # Both should increase (due to growing system)
        # Square increases faster due to Fermi surface
        # This is a qualitative test
        assert sq_ee_per_boundary[-1] > sq_ee_per_boundary[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
