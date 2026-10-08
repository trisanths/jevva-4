"""List items whose option probabilities change between batch size 1 and 8 inside score()."""
import numpy as np

from common import batches, encode_batch, load, score
from data import eval_items

items = eval_items(scale=0.03)
proc, model = load()
p1 = score(model, proc, items, bs=1)
p8 = score(model, proc, items, bs=8)
byid = {it["id"]: it for it in items}
bad = [(k, float(np.abs(np.array(p1[k]) - np.array(p8[k])).max())) for k in p1]
bad = [b for b in bad if b[1] > 0.02]
print("n bad", len(bad), "of", len(items))
for k, d in bad:
    it = byid[k]
    print(round(d, 3), k, it["source"], "n_opts", len(it["options"]), "p1", np.round(p1[k], 3), "p8", np.round(p8[k], 3))
for b in batches(items, 8):
    ids = [it["id"] for it in b]
    if any(k in ids for k, _ in bad):
        enc = encode_batch(proc, b)
        print("batch", ids, "len", tuple(enc["input_ids"].shape), "rows", enc["attention_mask"].sum(1).tolist(),
              "img tokens", (enc["input_ids"] == model.config.image_token_id).sum(1).tolist(),
              "pixel", tuple(enc["pixel_values"].shape) if "pixel_values" in enc else None)
