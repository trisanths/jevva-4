"""Stage 2: learned routers. For a partition, fit one linear router per layer that predicts each expert's
log output norm from the FFN input (closed-form ridge regression on train-split activations), then score
the eval set routing by the prediction instead of the oracle.

Usage: python stage2.py <partition> <k>[,<k>...] [shared]
Writes results/stage2/router-<partition>-s<shared>k<k>.json and adds rows to results/stage1/summary.json.
"""
import json
import os
import sys
import time

import torch
import torch.nn as nn

import moe
from common import ROOT, batches, by_source, encode_batch, load, score
from data import eval_items, profile_items

TOK_PER_ITEM = 96
OUT = os.path.join(ROOT, "results", "stage2")
os.makedirs(OUT, exist_ok=True)


def fit_routers(model, proc, ctl, part, E, lam=1e-2):
    ctl.set_partition(part, E)
    L, D = ctl.n_layers, model.config.text_config.hidden_size
    xtx = torch.zeros(L, D + 1, D + 1, dtype=torch.float64, device="cuda")
    xty = torch.zeros(L, D + 1, E, dtype=torch.float64, device="cuda")
    gen = torch.Generator(device="cpu").manual_seed(7)
    state = {}

    def hook(layer, x, h, c):
        if layer == 0:
            idx = []
            for b in range(x.shape[0]):
                pos = c.valid[b].nonzero().squeeze(1).cpu()
                idx.append(pos[torch.randperm(pos.numel(), generator=gen)[:TOK_PER_ITEM]])
            state["idx"] = idx
        g = c.gram(layer, model.model.language_model.layers[layer].mlp.mlp.down_proj.weight)
        B, T, _ = h.shape
        xs, ys = [], []
        for b, sel in enumerate(state["idx"]):
            sel = sel.to(h.device)
            hb = h[b, sel][..., c.perms[layer].to(h.device)].view(len(sel), E, -1)
            y = (torch.einsum("ten,enm->tem", hb, g).float() * hb.float()).sum(-1).clamp(min=1e-12).log()
            xs.append(x[b, sel].float())
            ys.append(y)
        X = torch.cat(xs).double()
        X = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], 1)
        Y = torch.cat(ys).double()
        xtx[layer] += X.t() @ X
        xty[layer] += X.t() @ Y

    ctl.mode = "dense"
    ctl.hook = hook
    items = profile_items()
    t0 = time.time()
    with torch.no_grad():
        for batch in batches(items, 8):
            enc = encode_batch(proc, batch).to("cuda")
            ctl.is_image = enc["input_ids"] == model.config.image_token_id
            ctl.valid = enc["attention_mask"].bool()
            model(**enc, logits_to_keep=1)
    ctl.hook = None
    routers = nn.ModuleList()
    for l in range(L):
        A = xtx[l]
        reg = lam * A.diagonal()[:-1].mean() * torch.eye(D + 1, dtype=A.dtype, device=A.device)
        reg[-1, -1] = 0
        W = torch.linalg.solve(A + reg, xty[l])          # [D+1, E]
        lin = nn.Linear(D, E).cuda().float()
        lin.weight.data = W[:-1].t().float().contiguous()
        lin.bias.data = W[-1].float().contiguous()
        routers.append(lin)
    print(f"fit {L} routers on {len(items)} items in {time.time() - t0:.0f}s", flush=True)
    return routers


def main(pname, ks, shared):
    E = int(pname.split("-")[1])
    proc, model = load()
    ctl = moe.install(model)
    parts = torch.load(os.path.join(ROOT, "partitions.pt"))
    routers = fit_routers(model, proc, ctl, parts[pname], E)
    torch.save(routers.state_dict(), os.path.join(OUT, f"routers-{pname}.pt"))
    ctl.routers = routers
    items = eval_items()
    ref = json.load(open(os.path.join(ROOT, "results", "stage1", "dense@280.json")))["probs"]
    summary_path = os.path.join(ROOT, "results", "stage1", "summary.json")
    for k in ks:
        name = f"router-{pname}-s{shared}k{k}"
        ctl.mode, ctl.k, ctl.shared, ctl.scope, ctl.collect = "router", k, shared, "all", True
        ctl.reset_stats()
        t0 = time.time()
        probs = score(model, proc, items, ctl, bs=8)
        dt = time.time() - t0
        res = by_source(items, probs, ref)
        u = ctl.usage / ctl.usage.sum(1, keepdim=True).clamp(min=1)
        rec = dict(name=name, mode="router", kw=dict(part=pname, k=k, shared=shared), soft_tokens=280, seconds=dt,
                   metrics=res, rel_err=ctl.rel_err().tolist(), usage_max_share=u.max(1).values.tolist())
        json.dump(dict(rec, probs=probs), open(os.path.join(OUT, f"{name}.json"), "w"))
        summary = json.load(open(summary_path))
        summary[name] = rec
        json.dump(summary, open(summary_path, "w"), indent=1)
        a = res["_all"]
        print(f"{name:28s} acc {a['acc']:.4f} nll {a['nll']:.3f} ece {a['ece']:.3f} kl {a['kl']:.3f} agree {a['agree']:.3f} "
              f"img {res['_image']['acc']:.4f} txt {res['_text']['acc']:.4f} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
