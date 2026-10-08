"""Accuracy on image questions vs TFLOP per image decision, from results/analysis.json -> results/frontier.png."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT

A = json.load(open(os.path.join(ROOT, "results", "analysis.json")))["rows"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(9.5, 6.6), dpi=160)
OFFSET = {"dense@70": (-6, -15), "dense@140": (-8, -15), "dense@280": (-50, -15), "neuron-50": (8, 5), "neuron-25": (8, 5),
          "random8-k4": (8, -4), "best8-k4": (8, 2), "coact8-k4": (8, -6), "26b@70": (-6, -15), "26b@140": (-48, 7),
          "26b@280": (-8, -15)}
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)


def pt(name):
    r = A.get(name)
    return (r["tflop_img"], 100 * r["acc_img"]) if r and r.get("tflop_img") else None


def series(names, color, label, labels, line=True, marker="o", hollow=False):
    pts = [(n, pt(n)) for n in names if pt(n)]
    if not pts:
        return
    xs, ys = [p[0] for _, p in pts], [p[1] for _, p in pts]
    if line:
        ax.plot(xs, ys, color=color, lw=2, zorder=2)
    ax.scatter(xs, ys, s=64, marker=marker, zorder=3, label=label,
               facecolors="none" if hollow else color, edgecolors=color if hollow else SURFACE, linewidths=2 if hollow else 1.5)
    for (n, (x, y)) in pts:
        ax.annotate(labels.get(n, n), (x, y), xytext=OFFSET.get(n, (7, 4)), textcoords="offset points", fontsize=8.5, color=INK2)


series(["dense@70", "dense@140", "dense@280"], BLUE, "Gemma 4 12B dense (image tokens 70 / 140 / 280)",
       {"dense@70": "12B @70", "dense@140": "12B @140", "dense@280": "12B @280"})
series(["26b@70", "26b@140", "26b@280"], ORANGE, "Gemma 4 26B-A4B + 550M vision tower (70 / 140 / 280)",
       {"26b@70": "26B @70", "26b@140": "26B @140", "26b@280": "26B @280"})
series(["coact8-k4", "best8-k4", "shared8-s1k3", "coact32-k16", "coact128-k64", "coact8-k2", "coact128-k32", "random8-k4"],
       AQUA, "12B MoEfied, oracle routing, no training (@280)",
       {"coact8-k4": "8x1920 top-4", "best8-k4": "8x1920 best-4", "shared8-s1k3": "shared+3 of 7",
        "coact32-k16": "32x480 top-16", "coact128-k64": "128x120 top-64", "coact8-k2": "8x1920 top-2",
        "coact128-k32": "128x120 top-32", "random8-k4": "random 8 top-4"}, line=False)
series(["neuron-50", "neuron-25", "neuron-12.5"], GRAY, "12B per-token top-p neurons (unstructured ceiling, not deployable)",
       {"neuron-50": "top 50% neurons", "neuron-25": "top 25%", "neuron-12.5": "top 12.5%"}, line=False, marker="D", hollow=True)

ax.set_xlabel("TFLOP per image decision (mean over image items, candidate-only readout)")
ax.set_ylabel("accuracy on image questions (%)")
ax.set_title("26B-A4B with its vision tower beats every 12B variant per FLOP; untrained MoE 12Bs collapse", loc="left", color=INK, fontsize=12, pad=12)
fig.legend(loc="lower left", bbox_to_anchor=(0.06, 0.035), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
fig.text(0.01, 0.01, "3,000 image MC items (ScienceQA-IMG, A-OKVQA, AI2D, MMStar); option-letter logits; L40S, bf16. Chance is about 27%.",
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(os.path.join(ROOT, "results", "frontier.png"), facecolor=SURFACE)
print("saved")
