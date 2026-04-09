"""
demo_compare_methods.py — Compare the three entanglement entropy methods.

This script demonstrates:
1. Accuracy comparison on hand-calculated examples
2. S_A = S_B verification for pure states
3. Condition number test (method1 vs method2 stability)
4. Speed comparison
5. Randomized SVD convergence for area-law vs volume-law states
"""

import sys
import time
import numpy as np

sys.path.insert(0, '..')

from ee import ee_method1, ee_method2, ee_method3, reshape_psi


def test_hand_calculated():
    """Test on examples with known exact answers."""
    print("=" * 65)
    print("1. Hand-calculated Examples")
    print("=" * 65)

    states = {}

    # Product state |↑↓⟩ = |10⟩, S = 0
    psi = np.array([0, 1, 0, 0], dtype=float)
    states["Product |↑↓⟩"] = (psi, 2, 0.0)

    # Bell state (|00⟩+|11⟩)/√2, S = ln2
    psi = np.array([1, 0, 0, 1], dtype=float) / np.sqrt(2)
    states["Bell |Φ+⟩"] = (psi, 2, np.log(2))

    # Partial entanglement cos(π/6)|00⟩ + sin(π/6)|11⟩
    theta = np.pi / 6
    psi = np.array([np.cos(theta), 0, 0, np.sin(theta)], dtype=float)
    S_exact = -(np.cos(theta)**2 * np.log(np.cos(theta)**2)
                + np.sin(theta)**2 * np.log(np.sin(theta)**2))
    states["Partial (θ=π/6)"] = (psi, 2, S_exact)

    # GHZ state (|000⟩+|111⟩)/√2, A={0}, B={1,2}
    psi = np.zeros(8)
    psi[0] = psi[7] = 1 / np.sqrt(2)
    states["GHZ (A={0})"] = (psi, 3, np.log(2))

    # W state (|100⟩+|010⟩+|001⟩)/√3, A={0}, B={1,2}
    psi = np.zeros(8)
    psi[4] = psi[2] = psi[1] = 1 / np.sqrt(3)  # |100⟩=4, |010⟩=2, |001⟩=1
    S_W = np.log(3) - (2 / 3) * np.log(2)
    states["W (A={0})"] = (psi, 3, S_W)

    print(f"{'State':<20s}  {'Exact':>8s}  {'M1(ρ eigh)':>10s}  "
          f"{'M2(SVD)':>10s}  {'M3(rSVD)':>10s}")
    print("-" * 68)

    for name, (psi, N, S_exact) in states.items():
        S1 = ee_method1(psi, N, NA=1)
        S2 = ee_method2(psi, N, NA=1)
        dA = 2 ** 1
        S3 = ee_method3(psi, N, k=dA, NA=1)
        print(f"{name:<20s}  {S_exact:8.5f}  {S1:10.5f}  {S2:10.5f}  {S3:10.5f}")


def test_sa_equals_sb():
    """Verify S_A = S_B for pure states."""
    print(f"\n{'=' * 65}")
    print("2. S_A = S_B for Pure States")
    print("=" * 65)

    np.random.seed(42)
    for N in [8, 10, 12]:
        psi = np.random.randn(2**N) + 1j * np.random.randn(2**N)
        psi /= np.linalg.norm(psi)
        psi_real = psi.real
        psi_real /= np.linalg.norm(psi_real)

        S_A = ee_method2(psi_real, N, NA=N // 2)
        S_B = ee_method2(psi_real, N, NA=N - N // 2)
        print(f"N={N:2d}:  S_A = {S_A:.10f},  S_B = {S_B:.10f},  "
              f"|S_A - S_B| = {abs(S_A - S_B):.2e}")


def test_condition_number():
    """Test numerical stability: method1 vs method2."""
    print(f"\n{'=' * 65}")
    print("3. Condition Number Test")
    print("=" * 65)
    print("Constructing states with exponentially decaying singular values.")
    print(f"{'α':>6s}  {'κ(C)':>10s}  {'S_exact':>10s}  "
          f"{'err(M1)':>10s}  {'err(M2)':>10s}")
    print("-" * 56)

    N_test = 12
    dA_test = 2 ** (N_test // 2)

    for alpha in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]:
        sv = np.exp(-alpha * np.arange(dA_test))
        sv /= np.linalg.norm(sv)
        C = np.diag(sv)
        psi_test = np.zeros(2**N_test)
        psi_test[:dA_test * dA_test] = C.flatten()
        psi_test /= np.linalg.norm(psi_test)

        lam = sv**2
        lam_nz = lam[lam > 1e-300]
        S_exact = -np.sum(lam_nz * np.log(lam_nz))

        S1 = ee_method1(psi_test, N_test)
        S2 = ee_method2(psi_test, N_test)

        kappa_C = sv[0] / sv[sv > 1e-15][-1]

        print(f"{alpha:6.1f}  {kappa_C:10.1e}  {S_exact:10.6f}  "
              f"{abs(S1 - S_exact):10.2e}  {abs(S2 - S_exact):10.2e}")


def test_speed():
    """Compare speed of three methods."""
    print(f"\n{'=' * 65}")
    print("4. Speed Comparison")
    print("=" * 65)

    print(f"{'N':>4s}  {'d_A':>6s}  {'M1 (ms)':>10s}  {'M2 (ms)':>10s}  "
          f"{'M3 k=8 (ms)':>12s}  {'M2/M1':>8s}")
    print("-" * 56)

    np.random.seed(42)
    for N in [10, 12, 14, 16, 18]:
        dA = 2 ** (N // 2)
        psi = np.random.randn(2**N)
        psi /= np.linalg.norm(psi)

        # Warm up
        ee_method1(psi, N)
        ee_method2(psi, N)

        n_rep = max(1, 100 // (2 ** (N - 10)))

        t0 = time.perf_counter()
        for _ in range(n_rep):
            ee_method1(psi, N)
        t1 = (time.perf_counter() - t0) / n_rep * 1000

        t0 = time.perf_counter()
        for _ in range(n_rep):
            ee_method2(psi, N)
        t2 = (time.perf_counter() - t0) / n_rep * 1000

        t0 = time.perf_counter()
        for _ in range(n_rep):
            ee_method3(psi, N, k=8)
        t3 = (time.perf_counter() - t0) / n_rep * 1000

        print(f"{N:4d}  {dA:6d}  {t1:10.3f}  {t2:10.3f}  {t3:12.3f}  {t2/t1:8.2f}")


def test_rsvd_convergence():
    """Test randomized SVD convergence for area-law vs volume-law states."""
    print(f"\n{'=' * 65}")
    print("5. Randomized SVD Convergence")
    print("=" * 65)

    N = 14
    dA = 2 ** (N // 2)
    print(f"N={N}, d_A={dA}")

    # Volume-law state (random)
    psi_rand = np.random.randn(2**N)
    psi_rand /= np.linalg.norm(psi_rand)
    S_full = ee_method2(psi_rand, N)

    # Area-law state (exponentially decaying singular values)
    C_area = np.zeros((dA, dA))
    for i in range(dA):
        C_area[i, i] = np.exp(-i * 2.0)
    C_area /= np.linalg.norm(C_area)
    psi_area = C_area.flatten()
    S_area_full = ee_method2(psi_area, N)

    print(f"\n{'k':>4s}  {'S_rand':>10s}  {'err%':>8s}  "
          f"{'S_area':>10s}  {'err%':>8s}")
    print("-" * 48)

    for k in [2, 4, 8, 16, 32, 64, dA]:
        k_use = min(k, dA)
        S_r_list, S_a_list = [], []
        for _ in range(5):
            S_r_list.append(ee_method3(psi_rand, N, k=k_use))
            S_a_list.append(ee_method3(psi_area, N, k=k_use))
        S_r = np.mean(S_r_list)
        S_a = np.mean(S_a_list)
        err_r = abs(S_r - S_full) / S_full * 100
        err_a = abs(S_a - S_area_full) / max(S_area_full, 1e-10) * 100
        print(f"{k_use:4d}  {S_r:10.4f}  {err_r:7.1f}%  "
              f"{S_a:10.4f}  {err_a:7.1f}%")

    print(f"\nExact: S_rand = {S_full:.4f}, S_area = {S_area_full:.4f}")
    print("Volume-law needs k ≈ d_A; area-law converges at small k.")


def main():
    test_hand_calculated()
    test_sa_equals_sb()
    test_condition_number()
    test_speed()
    test_rsvd_convergence()


if __name__ == "__main__":
    main()
