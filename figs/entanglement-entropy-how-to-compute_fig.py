"""
Figures for: 纠缠熵的计算：从偏迹到高效算法与二维推广

Five panels:
  (a) S(theta) for partial entanglement
  (b) Schmidt decomposition / SVD flow diagram
  (c) EE scaling comparison (area/volume/log/MBL/scar)
  (d) 1D free fermion: S vs ln(L) with CFT fit
  (e) 2D comparison: square vs honeycomb S/L
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import sys, os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ee import ee_corr_matrix
from lattices import chain_1d, square_2d, honeycomb_2d, subsystem_left_half

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 10, "axes.labelsize": 12})
C1, C2, C3, C4, C5 = "#2166ac", "#d6604d", "#1a9850", "#984ea3", "#e6ab02"

fig = plt.figure(figsize=(16, 9))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# ═════════════════════════════════════════════════════════════════════════════
# (a) S(theta)
# ═════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0])
theta = np.linspace(0, np.pi / 2, 400)
lam1 = np.cos(theta) ** 2
lam2 = np.sin(theta) ** 2
eps = 1e-14
S = np.where(lam1 > eps, -lam1 * np.log(lam1), 0) + \
    np.where(lam2 > eps, -lam2 * np.log(lam2), 0)

ax1.plot(theta, S / np.log(2), color=C1, lw=2.2)
ax1.axhline(1, color="gray", ls="--", lw=1, label=r"$\ln 2$ (Bell)")
ax1.set_xlabel(r"$\theta$")
ax1.set_ylabel(r"$S_A / \ln 2$")
ax1.set_title(r"(a) $\cos\theta|\uparrow\uparrow\rangle + \sin\theta|\downarrow\downarrow\rangle$")
ax1.set_xlim(0, np.pi / 2)
ax1.set_xticks([0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2])
ax1.set_xticklabels(["0", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"])
ax1.legend(fontsize=9)

# ═════════════════════════════════════════════════════════════════════════════
# (b) SVD flow diagram
# ═════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.axis("off")
ax2.set_title("(b) SVD workflow")

def draw_box(ax, x, y, w, h, color, label, fontsize=9):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor="k", lw=1.2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize)

def arrow(ax, x0, x1, y, color="black"):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4))

draw_box(ax2, 0.2, 6.0, 2.6, 1.4, "#d4e6f1",
         r"$|\psi\rangle$" + "\n" + r"$2^N$ vector", fontsize=8.5)
arrow(ax2, 2.8, 3.5, 6.7)

draw_box(ax2, 3.6, 6.0, 2.8, 1.4, "#d5f5e3",
         r"reshape $\to C$" + "\n" + r"$d_A \times d_B$ matrix", fontsize=8.5)
arrow(ax2, 6.4, 7.1, 6.7)

draw_box(ax2, 7.2, 6.0, 2.6, 1.4, "#fef9e7",
         "SVD\n" + r"$C = U\Sigma V^\dagger$", fontsize=8.5)
ax2.annotate("", xy=(8.5, 5.1), xytext=(8.5, 6.0),
             arrowprops=dict(arrowstyle="->", color="k", lw=1.4))

draw_box(ax2, 6.5, 3.8, 3.8, 1.1, "#fde8d8",
         r"$\lambda_k = \sigma_k^2$", fontsize=9)
ax2.annotate("", xy=(8.5, 3.0), xytext=(8.5, 3.8),
             arrowprops=dict(arrowstyle="->", color="k", lw=1.4))

draw_box(ax2, 6.5, 1.8, 3.8, 1.1, "#e8daef",
         r"$S = -\sum_k \lambda_k \ln \lambda_k$", fontsize=9)

ax2.text(0.2, 5.0, "Method 2 (recommended):\nDirect SVD\n\nMethod 1:\n" + r"$\rho_A = CC^\dagger \to$ eigh"
         + "\n(cond. number squared!)", fontsize=7.5, color="#555", va="top")

# ═════════════════════════════════════════════════════════════════════════════
# (c) EE scaling comparison
# ═════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[0, 2])
N_vals = np.arange(4, 50, 2)
S_area = np.ones_like(N_vals, dtype=float) * 1.5
S_vol = 0.35 * N_vals / 2
S_log = (1 / 3) * np.log(N_vals)
S_mbl = 0.8 * np.ones_like(N_vals, dtype=float)
S_scar = 0.5 * np.log(N_vals) / np.log(2)

ax3.plot(N_vals, S_vol, color=C2, lw=2, label="Volume law (ETH)")
ax3.plot(N_vals, S_log, color=C4, lw=2, ls="--", label=r"Log ($c\!=\!1$)")
ax3.plot(N_vals, S_scar, color=C3, lw=2, ls="-.", label="Scar")
ax3.plot(N_vals, S_mbl, color="gray", lw=2, ls=":", label="MBL")
ax3.plot(N_vals, S_area, color=C1, lw=2, ls=(0, (5, 2)), label="Area law (GS)")
ax3.set_xlabel("System size $N$")
ax3.set_ylabel(r"$S_{N/2}$")
ax3.set_title("(c) EE scaling in different phases")
ax3.legend(fontsize=8, loc="upper left")
ax3.set_xlim(4, 48)
ax3.set_ylim(0, 9.5)

# ═════════════════════════════════════════════════════════════════════════════
# (d) 1D benchmark: S vs ln(L) with CFT fit
# ═════════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 0:2])

Ls_1d = [10, 14, 20, 30, 50, 80, 100, 200, 400, 800]
Ss_1d = []
for L in Ls_1d:
    G = chain_1d(L, pbc=False)
    Ss_1d.append(ee_corr_matrix(G, list(range(L // 2))))

lnL = np.log(Ls_1d)
ax4.scatter(lnL, Ss_1d, color=C1, s=50, zorder=5, label="Numerical (OBC)")

# CFT fit (L >= 50)
mask = [i for i, l in enumerate(Ls_1d) if l >= 50]
A_fit = np.column_stack([np.log([Ls_1d[i] for i in mask]), np.ones(len(mask))])
c_fit, *_ = np.linalg.lstsq(A_fit, [Ss_1d[i] for i in mask], rcond=None)
x_fit = np.linspace(np.log(8), np.log(1000), 100)
ax4.plot(x_fit, c_fit[0] * x_fit + c_fit[1], color=C2, ls="--", lw=1.5,
         label=f"Fit: $S = {c_fit[0]:.4f}\\ln L + {c_fit[1]:.3f}$")

# Exact CFT line
ax4.plot(x_fit, (1 / 6) * x_fit + c_fit[1] - (c_fit[0] - 1/6) * np.mean(x_fit),
         color=C3, ls=":", lw=1.5, label=r"CFT: $S = \frac{1}{6}\ln L + c_1'$")

ax4.set_xlabel(r"$\ln L$")
ax4.set_ylabel(r"$S_{\mathrm{half}}$")
ax4.set_title(r"(d) 1D XX chain: EE vs CFT ($c = 1$, OBC)")
ax4.legend(fontsize=9)

# ═════════════════════════════════════════════════════════════════════════════
# (e) 2D: square vs honeycomb S/L
# ═════════════════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[1, 2])

Ls_2d = [4, 6, 8, 10, 12, 16, 20]
Ss_sq, Ss_hc = [], []
for L in Ls_2d:
    G_sq = square_2d(L, L, pbc=False)
    sub_sq = subsystem_left_half(L, L, sites_per_cell=1)
    Ss_sq.append(ee_corr_matrix(G_sq, sub_sq))

    G_hc = honeycomb_2d(L, L, pbc=False)
    sub_hc = subsystem_left_half(L, L, sites_per_cell=2)
    Ss_hc.append(ee_corr_matrix(G_hc, sub_hc))

Ls_arr = np.array(Ls_2d)
ax5.plot(Ls_arr, np.array(Ss_sq) / Ls_arr, "o-", color=C2, lw=1.5, ms=5,
         label=r"Square ($S/L \sim \ln L$)")
ax5.plot(Ls_arr, np.array(Ss_hc) / Ls_arr, "s-", color=C4, lw=1.5, ms=5,
         label=r"Honeycomb ($S/L \to$ const)")

ax5.set_xlabel("$L$")
ax5.set_ylabel("$S / L$")
ax5.set_title("(e) 2D lattices: area law vs Widom")
ax5.legend(fontsize=9)
ax5.set_xlim(3, 21)

plt.savefig("entanglement-entropy-how-to-compute_fig.png", dpi=150, bbox_inches="tight")
plt.savefig("entanglement-entropy-how-to-compute_fig.pdf", bbox_inches="tight")
print("Saved figure (5 panels).")
