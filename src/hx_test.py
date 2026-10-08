"""Checks for the KV-handoff machinery (26B int4, one GPU):
  1. experts_torch (dequantized grouped bmm) against vLLM's fused int4 kernel, a few layers;
  2. gradient through the custom experts Function against a central finite difference of the kernel forward;
  3. oracle: the suffix runner fed the 26B's own header+middle KV reproduces the HF forward's letter probs;
  4. gradient of the suffix pass with respect to the mapped context K at two layers against finite differences.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx_core import (K, SUF26, WH, Runner26, experts_torch, letter_ids, letter_probs, load_26b, log_jsonl)
from common import encode_batch
from data import _aokvqa, _mmlu


def main():
    proc, model = load_26b("int4")
    run = Runner26(model, proc)
    layers = model.model.language_model.layers
    rec = dict(kind="test")
    torch.manual_seed(0)
    # 1. experts forward agreement
    errs = []
    for l in (0, 11, 29):
        lay = layers[l]
        x = torch.randn(48, 2816, device="cuda", dtype=torch.bfloat16) * 3
        _, w, idx = lay.router(x)
        h = lay.pre_feedforward_layernorm_2(x)
        a = lay.experts.inner(h, idx, w).float()
        b = experts_torch(lay.experts.inner, h, idx, w).float()
        errs.append(float((a - b).norm() / a.norm()))
    rec["experts_rel_err"] = [round(e, 5) for e in errs]
    # 2. gradient of experts (Function) vs finite difference of the kernel
    lay = layers[11]
    x = torch.randn(24, 2816, device="cuda", dtype=torch.bfloat16)
    _, w, idx = lay.router(x)
    h = lay.pre_feedforward_layernorm_2(x).float().requires_grad_(True)
    wq = w.detach().clone().requires_grad_(True)
    g = torch.randn(24, 2816, device="cuda")
    y = lay.experts(h.to(torch.bfloat16), idx, wq)
    (y.float() * g).sum().backward()
    v = torch.randn_like(h)
    vw = torch.randn_like(wq) * 0.05
    with torch.no_grad():
        dd = []
        for eps in (0.05, 0.1):
            fp = (lay.experts.inner((h + eps * v).to(torch.bfloat16), idx, wq + eps * vw).float() * g).sum()
            fm = (lay.experts.inner((h - eps * v).to(torch.bfloat16), idx, wq - eps * vw).float() * g).sum()
            dd.append(float((fp - fm) / (2 * eps)))
        an = float((h.grad * v).sum() + (wq.grad * vw).sum())
    rec["experts_grad"] = dict(autograd=round(an, 3), finite_diff=[round(d, 3) for d in dd])
    # 3. oracle: native KV through the runner vs HF forward
    cand = torch.tensor(letter_ids(proc.tokenizer)[:K], device="cuda")
    items = [dict(it, image=it["image"].resize(WH)) for it in _aokvqa("validation", 6, 5)] + _mmlu("test", 4, 5)
    kv = {}
    hooks = []
    for l, lay in enumerate(layers):
        a = lay.self_attn
        hooks.append(a.k_proj.register_forward_hook(lambda m, i, o, l=l: kv.__setitem__((l, "k"), o)))
        if a.v_proj is not None:
            hooks.append(a.v_proj.register_forward_hook(lambda m, i, o, l=l: kv.__setitem__((l, "v"), o)))
    diffs, agree, ctx_saved = [], 0, None
    for it in items:
        enc = encode_batch(proc, [it], 70).to("cuda")
        with torch.no_grad():
            zr = model(**enc, logits_to_keep=1).logits[:, -1].float()[:, cand]
        n = len(it["options"])
        pr = torch.softmax(zr[0, :n], -1)
        T = enc["input_ids"].shape[1]
        Tc = T - len(SUF26)
        img = it["image"] is not None
        h0 = 5 if img else 4
        # middle raw KV in mapper layout, header from the runner's own cached header
        parts = []
        for l in range(len(layers)):
            parts.append(kv[(l, "k")][:, h0:Tc])
            if (l, "v") in kv and layers[l].self_attn.v_proj is not None:
                parts.append(kv[(l, "v")][:, h0:Tc])
        Y = torch.cat(parts, -1)
        m = torch.tensor([Tc - h0], device="cuda")
        with torch.no_grad():
            ctx, valid, hh = run.context(Y, m, img)
            # header KV from the runner vs the HF forward's own header positions
            hk = run.header(img)[0][0]
            cos, sin = run.rope(torch.arange(h0, device="cuda")[None], run.types[0])
            nk, _ = run.kv_post(0, kv[(0, "k")][:, :h0], kv[(0, "v")][:, :h0], cos, sin)
            hdr_err = float((hk.float() - nk.float()).norm() / nk.float().norm())
            z = run.suffix(ctx, valid, hh, m)
        p = letter_probs(z, [n])[0, :n]
        diffs.append(float((p - pr).abs().max()))
        agree += int(p.argmax() == pr.argmax())
        if img and ctx_saved is None:
            ctx_saved = (Y.detach(), m, img, n)
    for hk_ in hooks:
        hk_.remove()
    rec["oracle"] = dict(max_dp=round(max(diffs), 4), mean_dp=round(sum(diffs) / len(diffs), 4), agree=f"{agree}/{len(items)}",
                         header_rel_err_layer0=round(hdr_err, 5))
    # 4. gradient of the suffix pass wrt mapped raw KV (directional derivative vs finite differences)
    Y, m, img, n = ctx_saved
    Yr = Y.float().clone().requires_grad_(True)
    ctx, valid, hh = run.context(Yr.to(torch.bfloat16), m, img)
    z = run.suffix(ctx, valid, hh, m)
    gvec = torch.randn(K, device="cuda")
    (z * gvec).sum().backward()
    out = []
    for l in (2, 17):
        off = sum(run.sizes[:l])
        v = torch.zeros_like(Yr)
        v[..., off: off + run.sizes[l]] = torch.randn_like(Yr[..., off: off + run.sizes[l]]) * Yr[..., off: off + run.sizes[l]].std()
        an = float((Yr.grad * v).sum())
        fd = []
        with torch.no_grad():
            for eps in (0.02, 0.05):
                c1, v1, _ = run.context((Yr + eps * v).to(torch.bfloat16), m, img)
                c2, v2, _ = run.context((Yr - eps * v).to(torch.bfloat16), m, img)
                fd.append(float(((run.suffix(c1, v1, hh, m) - run.suffix(c2, v2, hh, m)) * gvec).sum() / (2 * eps)))
        out.append(dict(layer=l, autograd=round(an, 4), finite_diff=[round(d, 4) for d in fd]))
    rec["suffix_grad"] = out
    rec["gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    print(json.dumps(rec), flush=True)
    log_jsonl("tests.jsonl", rec)


if __name__ == "__main__":
    main()
