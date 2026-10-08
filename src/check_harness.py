"""Sanity checks before any experiment: loaders, batch-size invariance, and oracle k=8 == dense."""
import time

import numpy as np
import torch

import moe
from common import by_source, load, score
from data import eval_items

t0 = time.time()
items = eval_items(scale=0.03)
print("items", len(items), {s: sum(it["source"] == s for it in items) for s in sorted({it["source"] for it in items})},
      "load s", round(time.time() - t0, 1))
for it in items[:: max(1, len(items) // 6)]:
    print(it["source"], "|", it["question"][:80].replace("\n", " "), "|", it["options"], "->", it["answer"])

proc, model = load()
ctl = moe.install(model)

def run(bs):
    torch.cuda.synchronize(); t = time.time()
    p = score(model, proc, items, ctl, bs=bs)
    torch.cuda.synchronize()
    return p, time.time() - t

p1, t1 = run(1)
p8, t8 = run(8)
d = max(float(np.abs(np.array(p1[k]) - np.array(p8[k])).max()) for k in p1)
print(f"bs1 {t1:.1f}s  bs8 {t8:.1f}s  max |p1-p8| {d:.4f}")
print("dense", by_source(items, p1)["_all"])

g = torch.Generator().manual_seed(0)
ctl.set_partition([torch.randperm(15360, generator=g) for _ in range(48)], 8)
ctl.mode, ctl.k, ctl.collect = "oracle", 8, True
pk8, tk8 = run(8)
print(f"oracle k=8 {tk8:.1f}s", by_source(items, pk8, p8)["_all"])
print("rel err k=8 layers 0/24/47 (text, image):", ctl.rel_err()[[0, 24, 47]].tolist())
ctl.reset_stats(); ctl.k = 2
pk2, tk2 = run(8)
print(f"random oracle k=2 {tk2:.1f}s", by_source(items, pk2, p8)["_all"])
print("rel err k=2 layers 0/24/47 (text, image):", ctl.rel_err()[[0, 24, 47]].tolist())
print("usage layer 24", (ctl.usage[24] / ctl.usage[24].sum()).tolist())
ctl.mode, ctl.frac = "neuron", 0.25
pn, tn = run(8)
print(f"neuron 25% {tn:.1f}s", by_source(items, pn, p8)["_all"])
print("peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 1))
