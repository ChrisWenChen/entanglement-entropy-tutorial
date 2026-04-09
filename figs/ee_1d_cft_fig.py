"""
ee_1d_cft_fig.py — 1D free fermion CFT verification (PRB style).

Two panels:
  (a) Half-chain S vs ln(L) with linear fit and CFT reference
  (b) S(ℓ) vs ℓ for fixed L=200, compared to CFT formula

Reads:  data/1d_cft_data.txt
Computes: S(ℓ) data on the fly for panel (b)
Writes: figs/ee_1d_cft_fig.{png,pdf}
"""

import sys
sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from ee import ee_corr_matrix
from lattices import chain_1d

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

C1 = "#2166AC"   # data
C2 = "#D62728"   # fit
C3 = "#2CA02C"   # CFT exact

# ── Load data ───────────────────────────────────────────────────────────────
data = np.loadtxt("../data/1d_cft_data.txt",
                  skiprows=1)
Ls = data[:, 0].astype(int)
Ss = data[:, 1]
lnLs = np.log(Ls)

# ── Figure ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8),
                                gridspec_kw={"wspace": 0.35})

# ═══════════════════════════════════════════════════════════════════════════
# Panel (a): Half-chain S vs ln(L)
# ═══════════════════════════════════════════════════════════════════════════
ax = ax1
ax.plot(lnLs, Ss, "o", color=C1, ms=5, zorder=5, label="Numerical (OBC)")

# Fit on L >= 50
mask = Ls >= 50
A = np.column_stack([lnLs[mask], np.ones(mask.sum())])
coeffs, *_ = np.linalg.lstsq(A, Ss[mask], rcond=None)

x_fit = np.linspace(lnLs.min() - 0.1, lnLs.max() + 0.1, 100)
ax.plot(x_fit, coeffs[0] * x_fit + coeffs[1], "-", color=C2, lw=1.2,
        label=rf"Fit ($L\geq 50$): $a = {coeffs[0]:.4f}$")

# CFT reference line (shifted to match data offset)
cft_offset = coeffs[1] - (coeffs[0] - 1/6) * np.mean(lnLs[mask])
ax.plot(x_fit, (1/6) * x_fit + cft_offset, "--", color=C3, lw=1.0,
        label=r"CFT: $c/6 = 0.1667$")

ax.set_xlabel(r"$\ln L$")
ax.set_ylabel(r"$S_{L/2}$")
ax.legend(loc="lower right", frameon=False, fontsize=6.5)
ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, fontweight="bold",
        va="top", fontsize=10)

# ═══════════════════════════════════════════════════════════════════════════
# Panel (b): S(ℓ) vs ℓ for fixed L = 200
# ═══════════════════════════════════════════════════════════════════════════
ax = ax2
L = 200
G = chain_1d(L, pbc=False)

ells = list(range(2, L - 1, 4))
S_ell = [ee_corr_matrix(G, list(range(ell))) for ell in ells]

# CFT: S = (c/6) ln((2L/π) sin(πℓ/L)) + c'_1
cft_vals = [(1/6) * np.log((2*L/np.pi) * np.sin(np.pi*ell/L)) for ell in ells]
c1_prime = np.mean(np.array(S_ell) - np.array(cft_vals))
ells_smooth = np.arange(2, L - 1)
cft_curve = [(1/6) * np.log((2*L/np.pi) * np.sin(np.pi*ell/L)) + c1_prime
             for ell in ells_smooth]

ax.plot(ells, S_ell, "o", color=C1, ms=2, alpha=0.7, label="Numerical")
ax.plot(ells_smooth, cft_curve, "-", color=C2, lw=1.2,
        label=r"CFT: $\frac{c}{6}\ln\!\left(\frac{2L}{\pi}\sin\frac{\pi\ell}{L}\right) + c_1'$")

ax.set_xlabel(r"Subsystem size $\ell$")
ax.set_ylabel(r"$S(\ell)$")
ax.legend(loc="lower center", frameon=False, fontsize=6.5)
ax.text(0.03, 0.97, "(b)", transform=ax.transAxes, fontweight="bold",
        va="top", fontsize=10)
ax.text(0.50, 0.35, f"$L = {L}$, OBC", transform=ax.transAxes,
        va="top", ha="center", fontsize=8)

# ── Save ────────────────────────────────────────────────────────────────────
fig.savefig("ee_1d_cft_fig.png")
fig.savefig("ee_1d_cft_fig.pdf")
print("Saved ee_1d_cft_fig.png and ee_1d_cft_fig.pdf")
