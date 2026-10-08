"""Latency vs image-question accuracy: stock HF single pass (untuned) vs the Jevified, Jev-tuned fast path.
Numbers are the measured results from this study (results/jev/*.jsonl and fast-*.json on the L40S)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE = "#2a78d6", "#eb6834"

stock = [("E2B @70", 76.4, 53.7), ("E4B @70", 91.3, 65.5), ("12B @280", 100.5, 79.2)]
jev = [("E2B FP8+int4", 7.64, 69.6), ("E2B FP8", 8.83, 69.5), ("E2B bf16", 10.93, 70.0), ("E4B FP8+int4", 11.57, 74.9)]
off = {"E2B @70": (6, -12), "E4B @70": (6, -12), "12B @280": (-58, 6), "E2B FP8+int4": (-80, 8), "E2B FP8": (-40, -16),
       "E2B bf16": (6, 4), "E4B FP8+int4": (6, 4)}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(9, 5.8), dpi=160)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_xscale("log")
ax.axvline(10, color=INK2, lw=1.2, ls="--", zorder=1)
ax.text(10.3, 51.5, "10 ms", color=INK2, fontsize=9)
for pts, color, label in ((stock, BLUE, "Stock HF, one forward pass, untuned (GPU time only)"),
                          (jev, ORANGE, "Jevified + Jev-tuned, end to end (frame + question in, probabilities out)")):
    ax.scatter([p[1] for p in pts], [p[2] for p in pts], s=70, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3, label=label)
    for name, x, y in pts:
        ax.annotate(name, (x, y), xytext=off[name], textcoords="offset points", fontsize=8.5, color=INK2)
ax.set_xticks([5, 7, 10, 15, 20, 30, 50, 70, 100])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlim(5, 130); ax.set_ylim(50, 82)
ax.set_xlabel("latency per decision, batch 1 (ms, log scale)")
ax.set_ylabel("accuracy on image questions (%)")
ax.set_title("Jevified Gemma 4 E2B answers in 7.6 ms, 16 points more accurate than stock", loc="left", color=INK, fontsize=12, pad=12)
fig.legend(loc="lower left", bbox_to_anchor=(0.06, 0.03), ncol=1, frameon=False, fontsize=8.5, labelcolor=INK2)
fig.text(0.01, 0.005, "L40S. 3,000 image MC items (ScienceQA-IMG, A-OKVQA, AI2D, MMStar). 640x480 images, 70 image tokens except 12B (280).",
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig("/mnt/nvme/moefy/results/jev/latency_accuracy.png", facecolor=SURFACE)
print("saved")
