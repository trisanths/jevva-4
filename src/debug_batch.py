"""Find why left-padded batches change option probabilities."""
import numpy as np
import torch

from common import build, letter_ids, load
from data import eval_items

items = eval_items(scale=0.01)
img = [it for it in items if it["image"] is not None][:4]
txt = [it for it in items if it["image"] is None][:4]
proc, model = load()
cand = letter_ids(proc.tokenizer)


def probs(batch, **kw):
    enc = proc.apply_chat_template([build(proc, it) for it in batch], add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt", processor_kwargs=dict(padding=True)).to("cuda")
    with torch.no_grad():
        lg = model(**enc, logits_to_keep=1, **kw).logits[:, -1].float()
    return [torch.softmax(r[cand[:len(it["options"])]], -1).cpu().numpy() for it, r in zip(batch, lg)], enc


for name, group in (("text", txt), ("image", img), ("mixed", txt[:2] + img[:2])):
    single = [probs([it])[0][0] for it in group]
    batched, enc = probs(group)
    print(name, "keys", sorted(enc.keys()))
    print(" attn_mask row sums", enc["attention_mask"].sum(1).tolist(), "len", enc["input_ids"].shape)
    for k in ("mm_token_type_ids", "token_type_ids"):
        if k in enc:
            print(" ", k, [int(x) for x in enc[k].unique()])
    print(" max diff", [round(float(np.abs(a - b).max()), 4) for a, b in zip(single, batched)])
    # Same batch with explicit position ids that skip the padding.
    am = enc["attention_mask"]
    pos = (am.cumsum(1) - 1).clamp(min=0)
    batched2, _ = probs(group, position_ids=pos)
    print(" with position_ids", [round(float(np.abs(a - b).max()), 4) for a, b in zip(single, batched2)])
