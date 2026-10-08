"""Check the official-mask patch, the soft-token budget switch, and batch invariance under eager attention."""
import numpy as np
import torch

import common
from common import encode_batch, load, metrics, score
from data import eval_items

items = [it for it in eval_items(scale=0.05) if it["image"] is not None][:120]
proc, model = load()
for st in (70, 140, 280):
    enc = encode_batch(proc, items[:4], st)
    print("soft tokens", st, "image tokens per row", (enc["input_ids"] == model.config.image_token_id).sum(1).tolist())

common.MASK["mode"] = "official"
off1 = score(model, proc, items[:40], bs=1)
off8 = score(model, proc, items, bs=8)
print("official bs1 vs bs8 max diff", round(max(float(np.abs(np.array(off1[k]) - np.array(off8[k])).max()) for k in off1), 4))
common.MASK["mode"] = "hf"
hf8 = score(model, proc, items, bs=8)
print("official", metrics(items, off8))
print("hf-mask ", metrics(items, hf8, off8))
