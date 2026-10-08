"""Bisect the failing 8-row text batch."""
import numpy as np
import torch

from common import build, encode_batch, letter_ids, load
from data import eval_items

items = eval_items(scale=0.03)
byid = {it["id"]: it for it in items}
ids = ['mmlu-test-7621', 'arc-test-1095', 'mmlu-test-6416', 'arc-test-846', 'mmlu-test-12440', 'mmlu-test-2135', 'arc-test-546', 'mmlu-test-4030']
batch = [byid[i] for i in ids]


def run(model, proc, b, **kw):
    enc = encode_batch(proc, b).to("cuda")
    with torch.no_grad():
        lg = model(**enc, logits_to_keep=1, **kw).logits[:, -1].float()
    cand = letter_ids(proc.tokenizer)
    return [np.round(torch.softmax(r[cand[:len(it["options"])]], -1).cpu().numpy(), 3) for it, r in zip(b, lg)], enc


for attn in ("sdpa", "eager"):
    proc, model = load(attn)
    single = {it["id"]: run(model, proc, [it])[0][0] for it in batch}
    for name, b in (("all8", batch), ("first4", batch[:4]), ("last4", batch[4:]), ("pair", batch[1:3]), ("rev8", batch[::-1])):
        out, enc = run(model, proc, b)
        d = [float(np.abs(o - single[it["id"]]).max()) for it, o in zip(b, out)]
        print(attn, name, "rows", enc["attention_mask"].sum(1).tolist(), "diff", [round(x, 3) for x in d])
    out, enc = run(model, proc, batch)
    pos = (enc["attention_mask"].cumsum(1) - 1).clamp(min=0)
    out2, _ = run(model, proc, batch, position_ids=pos)
    print(attn, "all8+pos diff", [round(float(np.abs(o - single[it["id"]]).max()), 3) for it, o in zip(batch, out2)])
    print(attn, "mmlu-test-6416 single", single["mmlu-test-6416"], "all8", out[2])
    del model
    torch.cuda.empty_cache()
