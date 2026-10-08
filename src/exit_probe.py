"""Where does Gemma 4 12B make its decision? Capture the answer-position hidden state after every decoder
layer (one forward pass per item, no generation), then score two readouts per layer:

  lens : final RMSNorm + the option letters' (tied) embedding rows + softcap, i.e. the model's own head
         applied early, zero training
  head : a 5-way linear decision head (letters A..E, unavailable letters masked) fit on train-split items

Writes results/exit/hidden_<split>.pt (fp16 [N, L, D]) and results/exit/probe.json.
Usage: python exit_probe.py [soft_tokens]
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import ROOT, batches, encode_batch, letter_ids, load
from data import _aokvqa, _arc, _mmlu, _sqa, eval_items

OUT = os.path.join(ROOT, "results", "exit")
os.makedirs(OUT, exist_ok=True)
K = 5


def train_items(seed=3):
    return (_sqa("train", 1500, seed) + _aokvqa("train", 1500, seed) + _mmlu("auxiliary_train", 750, seed)
            + _arc("train", 750, seed))


@torch.no_grad()
def capture(model, proc, items, st):
    layers = model.model.language_model.layers
    buf, order = [], []
    cur = {}

    def mk(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cur[i] = h[:, -1].float().cpu()
        return hook

    hs = [l.register_forward_hook(mk(i)) for i, l in enumerate(layers)]
    t0 = time.time()
    for bi, batch in enumerate(batches(items, 4)):
        enc = encode_batch(proc, batch, st).to(model.device)
        model(**enc, logits_to_keep=1)
        buf.append(torch.stack([cur[i] for i in range(len(layers))], 1).half())   # [B, L, D]
        order += [it["id"] for it in batch]
        if bi % 50 == 0:
            print(f"  batch {bi} {time.time() - t0:.0f}s", flush=True)
    for h in hs:
        h.remove()
    H = torch.cat(buf)
    pos = {k: i for i, k in enumerate(order)}
    return H[[pos[it["id"]] for it in items]]


def labels(items):
    y = torch.tensor([it["answer"] for it in items])
    m = torch.zeros(len(items), K, dtype=torch.bool)
    for i, it in enumerate(items):
        m[i, : len(it["options"])] = True
    return y, m


def scores(logits, y, m, n_bins=15):
    logits = logits.masked_fill(~m, -1e9)
    p = logits.softmax(-1)
    conf, pred = p.max(-1)
    corr = (pred == y).float()
    nll = -torch.log(p[torch.arange(len(y)), y].clamp(min=1e-12)).mean()
    b = torch.clamp((conf * n_bins).long(), max=n_bins - 1)
    ece = sum(float((conf[b == i].mean() - corr[b == i].mean()).abs() * (b == i).float().mean())
              for i in range(n_bins) if (b == i).any())
    return dict(acc=float(corr.mean()), nll=float(nll), ece=ece)


def fit_head(X, y, m, Xv, yv, mv, wd=1e-3, steps=300):
    mu, sd = X.mean(0), X.std(0) + 1e-4
    X, Xv = (X - mu) / sd, (Xv - mu) / sd
    W = torch.zeros(X.shape[1], K, requires_grad=True, device=X.device)
    b = torch.zeros(K, requires_grad=True, device=X.device)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = (X @ W + b).masked_fill(~m, -1e9)
        loss = F.cross_entropy(z, y) + wd * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return (Xv @ W + b)


def main(st=280):
    proc, model = load(device_map="auto", max_memory={0: "15GiB", "cpu": "16GiB"})
    tr, ev = train_items(), eval_items()
    print("train", len(tr), "eval", len(ev), flush=True)
    paths = {s: os.path.join(OUT, f"hidden_{s}@{st}.pt") for s in ("train", "eval")}
    for s, its in (("train", tr), ("eval", ev)):
        if not os.path.exists(paths[s]):
            torch.save(capture(model, proc, its, st), paths[s])
    Htr, Hev = torch.load(paths["train"]).float(), torch.load(paths["eval"]).float()
    ytr, mtr = labels(tr)
    yev, mev = labels(ev)
    # Read the final norm and the letter rows of the tied embedding from the checkpoint, since CPU offload
    # can leave the in-memory copies as meta tensors.
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    f = safe_open(hf_hub_download("google/gemma-4-12B-it", "model.safetensors"), "pt")
    keys = list(f.keys())
    nk = [k for k in keys if k.endswith("language_model.norm.weight")][0]
    ek = [k for k in keys if k.endswith("language_model.embed_tokens.weight")][0]
    w_norm = f.get_tensor(nk).float().cuda()
    E = f.get_slice(ek)
    E = torch.stack([E[i:i + 1][0] for i in letter_ids(proc.tokenizer)[:K]]).float().cuda()   # [K, D]
    eps = model.config.get_text_config().rms_norm_eps
    cap = model.config.get_text_config().final_logit_softcapping
    img = torch.tensor([it["image"] is not None for it in ev])
    dev = "cuda"
    rows = []
    for l in range(Htr.shape[1]):
        hv = Hev[:, l].to(dev)
        with torch.no_grad():
            z = hv * (hv.pow(2).mean(-1, keepdim=True) + eps).pow(-0.5) * w_norm
            lens = z @ E.t()
            lens = cap * torch.tanh(lens / cap)
        head = fit_head(Htr[:, l].to(dev), ytr.to(dev), mtr.to(dev), hv, yev.to(dev), mev.to(dev))
        r = dict(layer=l + 1)
        for name, lg in (("lens", lens), ("head", head)):
            lg = lg.cpu()
            r[name] = scores(lg, yev, mev)
            r[name + "_img"] = scores(lg[img], yev[img], mev[img])["acc"]
            r[name + "_txt"] = scores(lg[~img], yev[~img], mev[~img])["acc"]
        rows.append(r)
        print(f"layer {l + 1:2d}  lens acc {r['lens']['acc']:.3f} (img {r['lens_img']:.3f} txt {r['lens_txt']:.3f})   "
              f"head acc {r['head']['acc']:.3f} (img {r['head_img']:.3f} txt {r['head_txt']:.3f}) nll {r['head']['nll']:.3f}", flush=True)
    json.dump(dict(soft_tokens=st, rows=rows), open(os.path.join(OUT, f"probe@{st}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 280)
