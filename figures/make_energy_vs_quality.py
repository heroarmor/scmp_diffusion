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

Baselines are read from scmp_diffusion_fid_baseline_cfg15/_eval (already run);
values hardcoded below with provenance so the figure is self-contained.

Outputs: energy_vs_quality_int.{png,pdf,svg}
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
uni = [dict(L=32,  e=.25,  kid=8.48, psnr=20.49),   # collapses (KID off-scale)
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
int_kid = np.array([225.06, 144.62, 96.97, 3.402, 3.346])   # min over {sym,asym}
int_ps  = np.array([14.381, 14.696, 15.202, 22.875, 27.976])  # max over {sym,asym}
E_quad = (int_n / 8.0) ** 2      # [0.25, 0.391, 0.563, 0.766, 1.0]
E_lin  = (int_n / 8.0)           # [0.5, 0.625, 0.75, 0.875, 1.0]

# split masks: int7/int8 live on the "good" main band; int<=6 collapse off-scale
main = int_n >= 7
coll = int_n <= 6

XMIN, XMAX = 0.18, 1.06
XT = [.25, .375, .5, .75, 1.]
XTL = {32: .25, 48: .375, 64: .5, 96: .75, 128: 1.}

# ---- figure scaffold: broken y-axis per panel -------------------------------
# KID  : collapse band on TOP  (high KID = worse)
# PSNR : collapse band on BOTTOM (low PSNR = worse)
fig = plt.figure(figsize=(11.2, 5.6), dpi=200)
axKt = fig.add_axes([0.075, 0.680, 0.385, 0.150])  # KID collapse (top)
axKb = fig.add_axes([0.075, 0.205, 0.385, 0.430])  # KID main   (bottom)
axPt = fig.add_axes([0.585, 0.345, 0.385, 0.485])  # PSNR main   (top)
axPb = fig.add_axes([0.585, 0.205, 0.385, 0.100])  # PSNR collapse (bottom)

for ax in (axKt, axKb, axPt, axPb):
    ax.set_xlim(XMIN, XMAX)
    ax.grid(True, color=GRID, lw=0.9, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)

axKt.set_ylim(85, 245);  axKt.set_yticks([100, 150, 200])
axKb.set_ylim(2.85, 4.35); axKb.set_yticks([3.0, 3.4, 3.8, 4.2])
axPt.set_ylim(20, 28.6); axPt.set_yticks([20, 22, 24, 26, 28])
axPb.set_ylim(13.7, 15.7); axPb.set_yticks([14, 15])

# hide the shared inner spines / ticks across each break
for top, bot in ((axKt, axKb), (axPt, axPb)):
    top.spines["bottom"].set_visible(False)
    top.tick_params(labelbottom=False, bottom=False)
    bot.spines["top"].set_visible(False)

def break_marks(top, bot):
    d = 0.012
    for ax, ys in ((top, 0.0), (bot, 1.0)):
        kw = dict(transform=ax.transAxes, color=INK2, clip_on=False, lw=1.0)
        ax.plot((-d, +d), (ys - d*1.6, ys + d*1.6), **kw)
        ax.plot((1 - d, 1 + d), (ys - d*1.6, ys + d*1.6), **kw)
break_marks(axKt, axKb)
break_marks(axPt, axPb)

# ---- plotting helpers -------------------------------------------------------
def plot_series_split(ax_top, ax_bot, x, y, **kw):
    """draw the same series on both bands (points naturally fall in range)."""
    ax_top.plot(x, y, **kw)
    ax_bot.plot(x, y, **kw)

# reference lines (span full x on the main bands)
axKb.axhline(refW["kid"], color=C_REFW, ls=(0, (5, 4)), lw=1.6, zorder=1)
axKb.axhline(refFPkid,    color=C_FP,   ls=(0, (5, 4)), lw=1.6, zorder=1)
axKb.text(XMAX-0.01, refFPkid+0.015, "FP  3.56", color=C_FP, fontsize=8.5,
          ha="right", va="bottom", fontweight="bold")
axKb.text(XMIN+0.01, refW["kid"]+0.015, "W8A8", color=C_REFW, fontsize=8.5,
          ha="left", va="bottom")
axPt.axhline(refW["psnr"], color=C_REFW, ls=(0, (5, 4)), lw=1.6, zorder=1)
axPt.text(XMAX-0.01, refW["psnr"]-0.12, "W8A8", color=C_REFW, fontsize=8.5,
          ha="right", va="top")

# --- int baselines: two energy models ---
int_style = dict(ms=6.5, mew=1.6, zorder=5)
# quadratic: solid + filled squares
for msk, ax in ((main, axKb), (coll, axKt)):
    ax.plot(E_quad[msk], int_kid[msk], "-s", color=C_INT, mfc=C_INT,
            mec=C_INT, lw=2.0, **int_style)
for msk, ax in ((main, axPt), (coll, axPb)):
    ax.plot(E_quad[msk], int_ps[msk], "-s", color=C_INT, mfc=C_INT,
            mec=C_INT, lw=2.0, **int_style)
# linear: dashed + open squares
for msk, ax in ((main, axKb), (coll, axKt)):
    ax.plot(E_lin[msk], int_kid[msk], "--s", color=C_INT, mfc="white",
            mec=C_INT, lw=1.8, **int_style)
for msk, ax in ((main, axPt), (coll, axPb)):
    ax.plot(E_lin[msk], int_ps[msk], "--s", color=C_INT, mfc="white",
            mec=C_INT, lw=1.8, **int_style)

# bit-width labels on the int points (collapse band, quad positions)
for n, x, y in zip(int_n[coll], E_quad[coll], int_kid[coll]):
    axKt.annotate(f"int{n}", (x, y), textcoords="offset points",
                  xytext=(6, 2), color=C_INT, fontsize=8, fontweight="bold")
for n, x, y in zip(int_n[coll], E_quad[coll], int_ps[coll]):
    axPb.annotate(f"int{n}", (x, y), textcoords="offset points",
                  xytext=(5, -9), color=C_INT, fontsize=8, fontweight="bold")
# int7/int8 labels on the main bands
for n, x, y, dy in [(7, E_quad[3], int_kid[3], 9), (8, E_quad[4], int_kid[4], 9)]:
    axKb.annotate(f"int{n}", (x, y), textcoords="offset points",
                  xytext=(-2, dy), color=C_INT, fontsize=8, fontweight="bold",
                  ha="center")
axPt.annotate("int8", (E_quad[4], int_ps[4]), textcoords="offset points",
              xytext=(-9, 2), color=C_INT, fontsize=8, fontweight="bold", ha="right")
axPt.annotate("int7", (E_quad[3], int_ps[3]), textcoords="offset points",
              xytext=(6, -3), color=C_INT, fontsize=8, fontweight="bold")

# --- SC uniform (skip null; KID skips off-scale L32) ---
uk = [d for d in uni if d["L"] != 32]
axKb.plot([d["e"] for d in uk], [d["kid"] for d in uk], "-o", color=C_UNI,
          mfc="white", mec=C_UNI, mew=2.0, lw=2.2, ms=6.5, zorder=6)
axPt.plot([d["e"] for d in uni], [d["psnr"] for d in uni], "-o", color=C_UNI,
          mfc="white", mec=C_UNI, mew=2.0, lw=2.2, ms=6.5, zorder=6)

# uniform L32 collapse: off-scale KID, annotate like the original
axKb.annotate("L32 = 8.48\n(collapse)", (0.195, 3.98), color=C_WARN, fontsize=8.5,
              fontweight="bold", ha="left", va="top")
axKb.annotate("", xy=(0.235, 4.33), xytext=(0.235, 4.06),
              arrowprops=dict(arrowstyle="-|>", color=C_WARN, lw=1.6))

# --- SC adaptive MP (diamonds) ---
for d in ada:
    if d["kid"] is not None:
        axKb.plot(d["e"], d["kid"], "D", color=C_ADA, mec="white", mew=1.4,
                  ms=8, zorder=7)
        axKb.annotate(f"avg{d['L']}", (d["e"], d["kid"]), textcoords="offset points",
                      xytext=(9, -3), color=C_ADA, fontsize=8, fontweight="bold")
    axPt.plot(d["e"], d["psnr"], "D", color=C_ADA, mec="white", mew=1.4,
              ms=8, zorder=7)
    axPt.annotate(f"avg{d['L']}", (d["e"], d["psnr"]), textcoords="offset points",
                  xytext=(9, -2), color=C_ADA, fontsize=8, fontweight="bold")

# ---- x tick labels (energy + L) on the bottom axes --------------------------
for ax in (axKb, axPb):
    ax.set_xticks(XT)
    ax.set_xticklabels([f"{e:g}" for e in XT])
    for e in XT:
        L = [k for k, v in XTL.items() if abs(v - e) < 1e-9][0]
        ax.annotate(f"L{L}", (e, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -16), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color=C_REFW)
axKt.set_xticks(XT); axPt.set_xticks(XT)

# ---- titles / labels --------------------------------------------------------
fig.text(0.5, 0.985,
         "SC energy vs quality  —  DiT-XL/2, halve, cfg 1.5, w8a8, ImageNet-256 (n=2000)"
         r"   |   $E=L/128$  ($E_{sc}=2E_{int}$ at int8)",
         ha="center", va="top", fontsize=12.5, fontweight="bold", color=INK)
fig.text(0.5, 0.945,
         r"int$n$ = best of {sym, asym} pure-quant (no SC);  drawn at $E=(n/8)^2$ "
         r"(quad, filled ■) and $E=n/8$ (lin, open □).  int collapses below 7-bit.",
         ha="center", va="top", fontsize=9.5, color=INK2)
fig.text(0.075, 0.885, "KID vs energy  (↓ better, ×10³)", fontsize=11,
         fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.585, 0.885, "PSNR vs energy  (↑ better, dB vs FP)", fontsize=11,
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
           fontsize=9.5, bbox_to_anchor=(0.5, 0.005),
           columnspacing=1.8, handlelength=2.4)

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "energy_vs_quality")
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", out + ".{png,pdf,svg}")
