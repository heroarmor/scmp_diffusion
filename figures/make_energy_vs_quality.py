#!/usr/bin/env python3
"""
Energy-vs-quality figure for DiT-XL/2 SC (halve, cfg 1.5, w8a8, ImageNet-256, n=2000).

Reproduces the hand-authored figures/energy_vs_quality.html as a committable
matplotlib script and ADDS a pure-integer-quant baseline curve ("int").

  int-n = best of {symmetric, asymmetric} pure-quant (no SC) at W=A=n bits
          (per-metric envelope: min KID, max PSNR-vs-FP16).
  Two energy models are drawn because an int-n MAC's cost vs an int8 MAC is
  model-dependent:
      E = (n/8)^2   quadratic  (array-multiplier energy ~ bits^2)
      E = (n/8)     linear     (bit-serial, "linear in cycles")

KID uses a LOG y-axis so the int curve is continuous across all bit-widths
(the 6->7-bit "cliff" and the uniform L32 collapse are then on-scale, not
off-scale annotations). PSNR is already in dB (a log quantity) so its axis
stays linear.

Baselines are read from scmp_diffusion_fid_baseline_cfg15/_eval (already run);
values hardcoded below with provenance so the figure is self-contained.

Outputs: energy_vs_quality.{png,pdf,svg}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

# ---- palette (matches the HTML figure) --------------------------------------
C_UNI = "#2a78d6"   # uniform SC
C_ADA = "#128f66"   # adaptive MP
C_INT = "#7b3fbf"   # pure-int baseline (new)
C_REFW = "#8a8b81"  # W8A8 no-SC floor
C_FP  = "#b98900"   # FP w16a16
C_WARN = "#d03b3b"
INK, INK2, GRID = "#12130f", "#53554d", "#e4e5dd"

# ---- data -------------------------------------------------------------------
# SC uniform sweep (fixed stoc_len L), E = L/128.  From energy_vs_quality.html.
uni = [dict(L=32,  e=.25,  kid=8.48, psnr=20.49),   # collapses
       dict(L=48,  e=.375, kid=2.95, psnr=22.79),
       dict(L=64,  e=.5,   kid=3.05, psnr=23.22),
       dict(L=96,  e=.75,  kid=3.41, psnr=23.08),
       dict(L=128, e=1.,   kid=3.31, psnr=23.39)]
# SC adaptive MP (FLOP-weighted budget).
ada = [dict(L=48, e=.375, kid=4.21, psnr=22.71),
       dict(L=96, e=.75,  kid=3.12, psnr=25.02)]
refW = dict(e=1., kid=3.36, psnr=23.19)   # W8A8 no-SC int8 floor
refFPkid = 3.56                           # FP w16a16 KID (PSNR = inf)

# Pure-int baselines, best of sym/asym per metric.
# From scmp_diffusion_fid_baseline_cfg15/_eval/{kid,eval}_wNaN_{sym,asym}.txt
#   KID x1e3          PSNR-vs-FP16 mean (dB)
#   n  sym    asym    sym     asym
#   4  256.31 225.06  14.381  14.167
#   5  196.52 144.62  14.095  14.696
#   6  134.03  96.97  14.763  15.202
#   7   91.74   3.40  15.322  22.875
#   8    3.357  3.346 23.451  27.976
int_n   = np.array([4, 5, 6, 7, 8])
int_kid = np.array([225.06, 144.62, 96.97, 3.402, 3.346])     # min over {sym,asym}
int_ps  = np.array([14.381, 14.696, 15.202, 22.875, 27.976])  # max over {sym,asym}
E_quad = (int_n / 8.0) ** 2      # [0.25, 0.391, 0.563, 0.766, 1.0]
E_lin  = (int_n / 8.0)           # [0.5, 0.625, 0.75, 0.875, 1.0]

XMIN, XMAX = 0.18, 1.06
XT = [.25, .375, .5, .75, 1.]
XTL = {32: .25, 48: .375, 64: .5, 96: .75, 128: 1.}

# ---- figure scaffold: one axis per panel ------------------------------------
fig = plt.figure(figsize=(11.2, 5.5), dpi=200)
axK = fig.add_axes([0.075, 0.215, 0.385, 0.615])   # KID  (log y)
axP = fig.add_axes([0.585, 0.215, 0.385, 0.615])   # PSNR (linear y)

for ax in (axK, axP):
    ax.set_xlim(XMIN, XMAX)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)

# KID: log axis with clean, plain-number ticks
axK.set_yscale("log")
axK.set_ylim(2.8, 285)
KMAJ = [3, 4, 5, 10, 20, 50, 100, 200]
axK.yaxis.set_major_locator(FixedLocator(KMAJ))
axK.yaxis.set_minor_locator(NullLocator())
axK.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
axK.grid(True, which="major", color=GRID, lw=0.9, zorder=0)

# PSNR: linear
axP.set_ylim(13.5, 29)
axP.set_yticks([14, 16, 18, 20, 22, 24, 26, 28])
axP.grid(True, color=GRID, lw=0.9, zorder=0)

# ---- reference lines --------------------------------------------------------
axK.axhline(refW["kid"], color=C_REFW, ls=(0, (5, 4)), lw=1.6, zorder=1)
axK.axhline(refFPkid,    color=C_FP,   ls=(0, (5, 4)), lw=1.6, zorder=1)
axK.text(XMIN + 0.01, refFPkid * 1.012, "FP  3.56", color=C_FP, fontsize=8.5,
         ha="left", va="bottom", fontweight="bold")
axK.text(XMIN + 0.01, refW["kid"] * 0.988, "W8A8", color=C_REFW, fontsize=8.5,
         ha="left", va="top")
axP.axhline(refW["psnr"], color=C_REFW, ls=(0, (5, 4)), lw=1.6, zorder=1)
axP.text(XMAX - 0.01, refW["psnr"] - 0.15, "W8A8", color=C_REFW, fontsize=8.5,
         ha="right", va="top")

# ---- int baselines: two energy models (continuous curves) -------------------
istyle = dict(ms=6.5, mew=1.6, zorder=5)
axK.plot(E_quad, int_kid, "-s",  color=C_INT, mfc=C_INT,   mec=C_INT, lw=2.0, **istyle)
axK.plot(E_lin,  int_kid, "--s", color=C_INT, mfc="white", mec=C_INT, lw=1.8, **istyle)
axP.plot(E_quad, int_ps,  "-s",  color=C_INT, mfc=C_INT,   mec=C_INT, lw=2.0, **istyle)
axP.plot(E_lin,  int_ps,  "--s", color=C_INT, mfc="white", mec=C_INT, lw=1.8, **istyle)

# bit-width labels (placed along the quadratic curve)
kid_lab = {4: (6, 3), 5: (6, 3), 6: (-3, 8), 7: (-6, 9), 8: (7, 1)}
for n, x, y in zip(int_n, E_quad, int_kid):
    axK.annotate(f"int{n}", (x, y), textcoords="offset points",
                 xytext=kid_lab[int(n)], color=C_INT, fontsize=8, fontweight="bold")
ps_lab = {4: (4, -10), 5: (4, -10), 6: (5, 3), 7: (7, -4), 8: (-9, 2)}
for n, x, y in zip(int_n, E_quad, int_ps):
    ha = "right" if n == 8 else "left"
    axP.annotate(f"int{n}", (x, y), textcoords="offset points",
                 xytext=ps_lab[int(n)], color=C_INT, fontsize=8, fontweight="bold", ha=ha)

# small "cliff" note on the KID panel
axK.annotate("int cliff\n6→7 bit", (0.66, 20), color=C_INT, fontsize=8,
             ha="center", va="center", alpha=0.9)

# ---- SC uniform (full curve incl. L32 collapse) -----------------------------
axK.plot([d["e"] for d in uni], [d["kid"] for d in uni], "-o", color=C_UNI,
         mfc="white", mec=C_UNI, mew=2.0, lw=2.2, ms=6.5, zorder=6)
axP.plot([d["e"] for d in uni], [d["psnr"] for d in uni], "-o", color=C_UNI,
         mfc="white", mec=C_UNI, mew=2.0, lw=2.2, ms=6.5, zorder=6)
axK.annotate("L32 collapse\n8.48", (0.25, 8.48), textcoords="offset points",
             xytext=(8, 2), color=C_WARN, fontsize=8, fontweight="bold")

# ---- SC adaptive MP (diamonds) ----------------------------------------------
for d in ada:
    axK.plot(d["e"], d["kid"], "D", color=C_ADA, mec="white", mew=1.4, ms=8, zorder=7)
    axK.annotate(f"avg{d['L']}", (d["e"], d["kid"]), textcoords="offset points",
                 xytext=(9, -2), color=C_ADA, fontsize=8, fontweight="bold")
    axP.plot(d["e"], d["psnr"], "D", color=C_ADA, mec="white", mew=1.4, ms=8, zorder=7)
    dy = -9 if d["L"] == 48 else -2
    axP.annotate(f"avg{d['L']}", (d["e"], d["psnr"]), textcoords="offset points",
                 xytext=(9, dy), color=C_ADA, fontsize=8, fontweight="bold")

# ---- x tick labels (energy + L) ---------------------------------------------
for ax in (axK, axP):
    ax.set_xticks(XT)
    ax.set_xticklabels([f"{e:g}" for e in XT])
    for e in XT:
        L = [k for k, v in XTL.items() if abs(v - e) < 1e-9][0]
        ax.annotate(f"L{L}", (e, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -16), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color=C_REFW)

# ---- titles / labels --------------------------------------------------------
fig.text(0.5, 0.985,
         "SC energy vs quality  —  DiT-XL/2, halve, cfg 1.5, w8a8, ImageNet-256 (n=2000)"
         r"   |   $E=L/128$  ($E_{sc}=2E_{int}$ at int8)",
         ha="center", va="top", fontsize=12.5, fontweight="bold", color=INK)
fig.text(0.5, 0.945,
         r"int$n$ = best of {sym, asym} pure-quant (no SC);  drawn at $E=(n/8)^2$ "
         r"(quad, filled ■) and $E=n/8$ (lin, open □).  int collapses below 7-bit.",
         ha="center", va="top", fontsize=9.5, color=INK2)
fig.text(0.075, 0.888, "KID vs energy  (↓ better, ×10³, log)", fontsize=11,
         fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.585, 0.888, "PSNR vs energy  (↑ better, dB vs FP)", fontsize=11,
         fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.2675, 0.095, "relative energy  (× int8 MAC)", ha="center",
         fontsize=10, color=INK2)
fig.text(0.7775, 0.095, "relative energy  (× int8 MAC)", ha="center",
         fontsize=10, color=INK2)

# ---- legend -----------------------------------------------------------------
handles = [
    Line2D([], [], color=C_UNI, marker="o", mfc="white", mec=C_UNI, mew=2,
           lw=2.2, ms=7, label="Uniform SC (fixed stoc_len)"),
    Line2D([], [], color=C_ADA, marker="D", mec="white", lw=0, ms=8,
           label="Adaptive MP (FLOP-weighted)"),
    Line2D([], [], color=C_INT, marker="s", mfc=C_INT, mec=C_INT, lw=2,
           ms=7, label=r"int, best sym/asym  ($E\propto n^2$)"),
    Line2D([], [], color=C_INT, marker="s", mfc="white", mec=C_INT, ls="--",
           lw=1.8, ms=7, label=r"int, best sym/asym  ($E\propto n$)"),
    Line2D([], [], color=C_REFW, ls=(0, (5, 4)), lw=1.6, label="W8A8 no-SC (int8 floor)"),
    Line2D([], [], color=C_FP, ls=(0, (5, 4)), lw=1.6, label="FP w16a16"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           fontsize=9.5, bbox_to_anchor=(0.5, -0.035),
           columnspacing=1.8, handlelength=2.4)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "energy_vs_quality")
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", out + ".{png,pdf,svg}")
