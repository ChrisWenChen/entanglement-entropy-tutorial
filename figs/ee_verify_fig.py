"""
ee_verify_fig.py — Benchmark of four entanglement entropy methods (PRB style).

Two panels:
  (a) Wall-clock time vs system size N for four methods:
      M1 (rho_A eigh), M2 (full SVD), M3 (hand-written rSVD), M4 (sklearn rSVD).
  (b) rSVD convergence: S(k)/S_exact for area-law vs volume-law states.
      Hand-written and sklearn produce identical results, so only one curve each.

Reads:  data/bench_methods.json
Writes: figs/ee_verify_fig.{png,pdf}
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker

# ── PRB style ───────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Colors ──────────────────────────────────────────────────────────────────
C1 = "#D62728"   # method 1 — red
C2 = "#1F77B4"   # method 2 — blue
C3 = "#2CA02C"   # method 3 — green
C4 = "#9467BD"   # method 4 — purple
C_vol = "#D62728"   # volume-law
C_area = "#2CA02C"  # area-law

# ── Load data ───────────────────────────────────────────────────────────────
with open("../data/bench_methods.json") as f:
    data = json.load(f)

tm = data["timing"]
rv = data["rsvd_convergence"]

# ── Figure ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8),
                                gridspec_kw={"wspace": 0.35})

# ═══════════════════════════════════════════════════════════════════════════
# Panel (a): Timing
# ═══════════════════════════════════════════════════════════════════════════
ax = ax1
Ns = np.array(tm["N"])
# Convert μs to ms
t1 = np.array(tm["t1"]) / 1e3
t2 = np.array(tm["t2"]) / 1e3
t3 = np.array(tm["t3"]) / 1e3
t4 = np.array(tm["t4"]) / 1e3

ax.semilogy(Ns, t1, "o-",  color=C1, label=r"M1: $\rho_A$ eigh", zorder=3)
ax.semilogy(Ns, t2, "s-",  color=C2, label="M2: full SVD", zorder=3)
ax.semilogy(Ns, t3, "^-",  color=C3, label=r"M3: rSVD (hand, $k{=}8$)", zorder=3)
ax.semilogy(Ns, t4, "D-",  color=C4, label=r"M4: rSVD (sklearn, $k{=}8$)", zorder=3)

ax.set_xlabel(r"Number of qubits $N$")
ax.set_ylabel("Wall time (ms)")
ax.legend(loc="lower right", frameon=False, handlelength=1.8, borderpad=0.2,
          labelspacing=0.3, fontsize=6.5)
ax.set_xlim(7, 21)
ax.set_xticks(Ns)
ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, fontweight="bold",
        va="top", fontsize=10)

# ═══════════════════════════════════════════════════════════════════════════
# Panel (b): rSVD convergence
# ═══════════════════════════════════════════════════════════════════════════
ax = ax2

ks = np.array(rv["ks"])
# Hand-written and sklearn give identical convergence; use hand-written
S_rand = np.array(rv["S_rand_hand"])
S_area = np.array(rv["S_area_hand"])
S_rand_ex = rv["S_rand_exact"]
S_area_ex = rv["S_area_exact"]
dA = rv["dA"]
N_rv = rv["N"]

ax.semilogx(ks, S_rand / S_rand_ex, "o-", color=C_vol, base=2,
            label="Volume law (random)")
ax.semilogx(ks, S_area / S_area_ex, "s-", color=C_area, base=2,
            label="Area law (exp. decay)")
ax.axhline(1.0, color="gray", ls="--", lw=0.5, alpha=0.5)

# Shade "converged" band
ax.axhspan(0.99, 1.01, color="gray", alpha=0.06)

ax.set_xlabel(r"Target rank $k$")
ax.set_ylabel(r"$S(k)\, /\, S_{\mathrm{exact}}$")
ax.legend(loc="lower right", frameon=False, handlelength=1.8, fontsize=6.5)
ax.set_ylim(0, 1.15)
ax.set_xlim(1.5, 360)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

ax.text(0.03, 0.97, "(b)", transform=ax.transAxes, fontweight="bold",
        va="top", fontsize=10)
ax.text(0.97, 0.97, rf"$N = {N_rv}$,  $d_A = {dA}$",
        transform=ax.transAxes, va="top", ha="right", fontsize=8)

# ── Save ────────────────────────────────────────────────────────────────────
fig.savefig("ee_verify_fig.png")
fig.savefig("ee_verify_fig.pdf")
print("Saved ee_verify_fig.png and ee_verify_fig.pdf")
