"""
ee_2d_lattice_fig.py — 2D lattice entanglement entropy comparison (PRB style).

Two panels:
  (a) S/L vs L for square and honeycomb lattices
  (b) S vs L with area-law fit for honeycomb and S ~ L ln L fit for square

Reads:  data/square_2d_data.txt
        data/honeycomb_2d_data.txt
Writes: figs/ee_2d_lattice_fig.{png,pdf}
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

C_sq = "#1F77B4"   # square — blue
C_hc = "#2CA02C"   # honeycomb — green

# ── Load data ───────────────────────────────────────────────────────────────
sq = np.loadtxt("../data/square_2d_data.txt",
                skiprows=1)
hc = np.loadtxt("../data/honeycomb_2d_data.txt",
                skiprows=1)

Ls_sq = sq[:, 0].astype(int)
Ss_sq = sq[:, 2]
SoL_sq = sq[:, 3]

Ls_hc = hc[:, 0].astype(int)
Ss_hc = hc[:, 2]
SoL_hc = hc[:, 3]

# ── Figure ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8),
                                gridspec_kw={"wspace": 0.35})

# ═══════════════════════════════════════════════════════════════════════════
# Panel (a): S/L vs L
# ═══════════════════════════════════════════════════════════════════════════
ax = ax1
ax.plot(Ls_sq, SoL_sq, "o-", color=C_sq, lw=1.0, ms=4, zorder=3,
        label=r"Square ($S/L \sim \ln L$)")
ax.plot(Ls_hc, SoL_hc, "s-", color=C_hc, lw=1.0, ms=4, zorder=3,
        label=r"Honeycomb ($S/L \to$ const)")

ax.set_xlabel("$L$")
ax.set_ylabel("$S / L$")
ax.legend(loc="lower right", frameon=False, fontsize=6.5)
ax.set_xlim(3, 21)
ax.set_ylim(0.4, 0.75)
ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, fontweight="bold",
        va="top", fontsize=10)

# ═══════════════════════════════════════════════════════════════════════════
# Panel (b): S vs L with fits
# ═══════════════════════════════════════════════════════════════════════════
ax = ax2
ax.plot(Ls_sq, Ss_sq, "o", color=C_sq, ms=5, zorder=5, label="Square")
ax.plot(Ls_hc, Ss_hc, "s", color=C_hc, ms=5, zorder=5, label="Honeycomb")

# Fit square: S = a * L * ln(L) + b * L
A_sq = np.column_stack([Ls_sq * np.log(Ls_sq), Ls_sq])
c_sq, *_ = np.linalg.lstsq(A_sq, Ss_sq, rcond=None)
x_smooth = np.linspace(3, 21, 100)
ax.plot(x_smooth, c_sq[0] * x_smooth * np.log(x_smooth) + c_sq[1] * x_smooth,
        "-", color=C_sq, lw=1.0, alpha=0.7,
        label=rf"$S = {c_sq[0]:.3f}\,L\ln L {c_sq[1]:+.3f}\,L$")

# Fit honeycomb: S = a * L + b
A_hc = np.column_stack([Ls_hc, np.ones(len(Ls_hc))])
c_hc, *_ = np.linalg.lstsq(A_hc, Ss_hc, rcond=None)
ax.plot(x_smooth, c_hc[0] * x_smooth + c_hc[1], "--", color=C_hc, lw=1.0,
        alpha=0.7, label=rf"$S = {c_hc[0]:.3f}\,L {c_hc[1]:+.2f}$")

ax.set_xlabel("$L$")
ax.set_ylabel("$S$")
ax.legend(loc="lower right", frameon=False, fontsize=5.5)
ax.text(0.03, 0.55, "OBC, half-filling", transform=ax.transAxes,
        va="top", ha="left", fontsize=8)
ax.text(0.03, 0.97, "(b)", transform=ax.transAxes, fontweight="bold",
        va="top", fontsize=10)

# ── Save ────────────────────────────────────────────────────────────────────
fig.savefig("ee_2d_lattice_fig.png")
fig.savefig("ee_2d_lattice_fig.pdf")
print("Saved ee_2d_lattice_fig.png and ee_2d_lattice_fig.pdf")
