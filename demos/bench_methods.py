"""
bench_methods.py — Benchmark four entanglement entropy methods.

Method 1: rho_A eigendecomposition
Method 2: Direct SVD (recommended)
Method 3: Randomized SVD (hand-written, Halko et al. 2011)
Method 4: Randomized SVD (sklearn implementation)

Outputs data/bench_methods.json with timing, accuracy, and rSVD convergence.
"""

import json
import time
import sys

import numpy as np

sys.path.insert(0, "..")

from ee import ee_method1, ee_method2, ee_method3, rsvd, rsvd_sklearn, reshape_psi


def bench_timing():
    """Benchmark timing of all four methods across system sizes."""
    np.random.seed(42)
    Ns = [8, 10, 12, 14, 16, 18, 20]
    results = {"N": Ns, "t1": [], "t2": [], "t3": [], "t4": []}

    for N in Ns:
        print(f"  N={N} ...", end=" ", flush=True)
        psi = np.random.randn(2 ** N)
        psi /= np.linalg.norm(psi)
        n_rep = max(3, 200 // (2 ** (N - 8)))

        # Warm up
        ee_method1(psi, N)
        ee_method2(psi, N)
        ee_method3(psi, N, k=8, use_sklearn=False)
        ee_method3(psi, N, k=8, use_sklearn=True)

        t0 = time.perf_counter()
        for _ in range(n_rep):
            ee_method1(psi, N)
        results["t1"].append((time.perf_counter() - t0) / n_rep * 1e6)

        t0 = time.perf_counter()
        for _ in range(n_rep):
            ee_method2(psi, N)
        results["t2"].append((time.perf_counter() - t0) / n_rep * 1e6)

        t0 = time.perf_counter()
        for _ in range(n_rep):
            ee_method3(psi, N, k=8, use_sklearn=False)
        results["t3"].append((time.perf_counter() - t0) / n_rep * 1e6)

        t0 = time.perf_counter()
        for _ in range(n_rep):
            ee_method3(psi, N, k=8, use_sklearn=True)
        results["t4"].append((time.perf_counter() - t0) / n_rep * 1e6)

        print(f"t1={results['t1'][-1]:.0f}  t2={results['t2'][-1]:.0f}  "
              f"t3={results['t3'][-1]:.0f}  t4={results['t4'][-1]:.0f} us")

    return {"timing": results}


def bench_rsvd_convergence():
    """rSVD convergence for area-law vs volume-law states, hand-written vs sklearn."""
    np.random.seed(42)
    N = 16
    dA = 2 ** (N // 2)

    # Volume-law: random state
    psi_rand = np.random.randn(2 ** N)
    psi_rand /= np.linalg.norm(psi_rand)
    S_rand_exact = ee_method2(psi_rand, N)

    # Area-law: exponentially decaying singular values
    C_area = np.zeros((dA, dA))
    for i in range(dA):
        C_area[i, i] = np.exp(-i * 2.0)
    C_area /= np.linalg.norm(C_area)
    psi_area = C_area.flatten()
    S_area_exact = ee_method2(psi_area, N)

    ks = [2, 4, 8, 16, 32, 64, 128, dA]
    results = {
        "N": N, "dA": dA, "ks": ks,
        "S_rand_exact": S_rand_exact,
        "S_area_exact": S_area_exact,
        "S_rand_hand": [], "S_area_hand": [],
        "S_rand_sk": [], "S_area_sk": [],
    }

    n_trial = 10
    print(f"  rSVD convergence N={N}, dA={dA}")
    for k in ks:
        kk = min(k, dA)
        sr_h, sa_h, sr_s, sa_s = [], [], [], []
        for _ in range(n_trial):
            sr_h.append(ee_method3(psi_rand, N, k=kk, use_sklearn=False))
            sa_h.append(ee_method3(psi_area, N, k=kk, use_sklearn=False))
            sr_s.append(ee_method3(psi_rand, N, k=kk, use_sklearn=True))
            sa_s.append(ee_method3(psi_area, N, k=kk, use_sklearn=True))
        results["S_rand_hand"].append(float(np.mean(sr_h)))
        results["S_area_hand"].append(float(np.mean(sa_h)))
        results["S_rand_sk"].append(float(np.mean(sr_s)))
        results["S_area_sk"].append(float(np.mean(sa_s)))
        print(f"    k={kk:4d}  rand_h={np.mean(sr_h):.4f}  area_h={np.mean(sa_h):.6f}  "
              f"rand_sk={np.mean(sr_s):.4f}  area_sk={np.mean(sa_s):.6f}")

    return {"rsvd_convergence": results}


def bench_accuracy():
    """Compare accuracy and S_A = S_B for all four methods."""
    np.random.seed(42)
    results = {"accuracy": []}

    for N in [8, 10, 12, 14]:
        psi = np.random.randn(2 ** N) + 1j * np.random.randn(2 ** N)
        psi /= np.linalg.norm(psi)
        psi_r = psi.real.copy()
        psi_r /= np.linalg.norm(psi_r)

        SA = ee_method2(psi_r, N, NA=N // 2)
        SB = ee_method2(psi_r, N, NA=N - N // 2)
        S1 = ee_method1(psi_r, N)
        S2 = ee_method2(psi_r, N)
        dA = 2 ** (N // 2)
        S3 = ee_method3(psi_r, N, k=dA, use_sklearn=False)
        S4 = ee_method3(psi_r, N, k=dA, use_sklearn=True)

        results["accuracy"].append({
            "N": N, "SA": SA, "SB": SB,
            "S1": S1, "S2": S2, "S3": S3, "S4": S4,
        })
        print(f"  N={N}: SA={SA:.10f} SB={SB:.10f} |SA-SB|={abs(SA-SB):.2e}")
        print(f"         S1={S1:.10f} S2={S2:.10f} S3={S3:.10f} S4={S4:.10f}")

    return results


def bench_condition_number():
    """Condition number sensitivity: method 1 vs method 2."""
    N = 16
    dA = 2 ** (N // 2)
    alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    results = {
        "N": N, "alphas": alphas,
        "kappa": [], "S_exact": [], "err1": [], "err2": [],
    }

    print(f"  Condition number test N={N}, dA={dA}")
    for alpha in alphas:
        sv = np.exp(-alpha * np.arange(dA))
        sv /= np.linalg.norm(sv)
        C = np.zeros((dA, 2 ** (N - N // 2)))
        np.fill_diagonal(C, sv)
        psi_test = C.flatten()
        psi_test /= np.linalg.norm(psi_test)

        lam = sv ** 2
        lam_nz = lam[lam > 1e-300]
        S_exact = -np.sum(lam_nz * np.log(lam_nz))

        S1 = ee_method1(psi_test, N)
        S2 = ee_method2(psi_test, N)

        kappa = sv[0] / max(sv[sv > 1e-15][-1], 1e-300)
        results["kappa"].append(float(kappa))
        results["S_exact"].append(float(S_exact))
        results["err1"].append(float(abs(S1 - S_exact)))
        results["err2"].append(float(abs(S2 - S_exact)))
        print(f"    alpha={alpha:.2f}  kappa={kappa:.1e}  S={S_exact:.6f}  "
              f"err1={abs(S1-S_exact):.2e}  err2={abs(S2-S_exact):.2e}")

    return {"condition": results}


def main():
    np.random.seed(42)
    print("=" * 60)
    print("Benchmarking four EE methods")
    print("=" * 60)

    print("\n[1/4] Timing ...")
    r1 = bench_timing()

    print("\n[2/4] rSVD convergence ...")
    r2 = bench_rsvd_convergence()

    print("\n[3/4] Accuracy ...")
    r3 = bench_accuracy()

    print("\n[4/4] Condition number ...")
    r4 = bench_condition_number()

    results = {}
    results.update(r1)
    results.update(r2)
    results.update(r3)
    results.update(r4)

    outpath = "../data/bench_methods.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
