"""Is score() deterministic per batch size, and which batch size matches an isolated forward?"""
import numpy as np
import torch

from common import build, letter_ids, load, score
from data import eval_items

items = eval_items(scale=0.03)
proc, model = load()
cand = letter_ids(proc.tokenizer)
watch = ["sqa-test-1281", "aok-validation-661", "mmlu-test-6416", "mmstar-1144", "mmlu-test-7621"]
sub = [it for it in items if it["id"] in watch]


def iso(it, attn_mask=True):
    enc = proc.apply_chat_template([build(proc, it)], add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors="pt").to("cuda")
    if not attn_mask:
        enc.pop("attention_mask")
    with torch.no_grad():
        lg = model(**enc, logits_to_keep=1).logits[0, -1].float()
    return np.round(torch.softmax(lg[cand[:len(it["options"])]], -1).cpu().numpy(), 3)


runs = {"bs1a": score(model, proc, items, bs=1), "bs1b": score(model, proc, items, bs=1),
        "bs8a": score(model, proc, items, bs=8), "bs8b": score(model, proc, items, bs=8),
        "sub_bs1": score(model, proc, sub, bs=1)}
for a, b in (("bs1a", "bs1b"), ("bs8a", "bs8b"), ("bs1a", "bs8a")):
    d = [float(np.abs(np.array(runs[a][k]) - np.array(runs[b][k])).max()) for k in runs[a]]
    print(a, "vs", b, "max", round(max(d), 4), "n>0.02", sum(x > 0.02 for x in d))
for it in sub:
    k = it["id"]
    print(k, "iso", iso(it), "iso-nomask", iso(it, False), "sub_bs1", np.round(runs["sub_bs1"][k], 3),
          "bs1", np.round(runs["bs1a"][k], 3), "bs8", np.round(runs["bs8a"][k], 3))
