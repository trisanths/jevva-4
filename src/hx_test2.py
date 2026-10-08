"""Localize the experts gradient check: input part and routing-weight part separately, against (a) finite
differences of the vLLM kernel, (b) finite differences and autograd of an fp32 torch reference on the same
dequantized weights. Also the suffix-pass gradient: autograd through the kernel Function against autograd through
the pure-torch experts (all-autograd reference), relative error over the whole gradient tensor."""
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hx_core
from hx_core import K, WH, Runner26, deq_int4, experts_torch, load_26b, log_jsonl
from common import encode_batch
from data import _aokvqa


def experts_fp32(mod, x, idx, w):
    out = torch.zeros(x.shape[0], x.shape[1], device=x.device)
    for e in idx.unique().tolist():
        t, kk = torch.where(idx == e)
        W13 = deq_int4(mod.w13[e:e + 1], mod.s13[e:e + 1])[0].float()
        W2 = deq_int4(mod.w2[e:e + 1], mod.s2[e:e + 1])[0].float()
        hh = x[t].float() @ W13.t()
        g, up = hh.chunk(2, -1)
        y = (F.gelu(g, approximate="tanh") * up) @ W2.t()
        out = out.index_add(0, t, y * w[t, kk][:, None].float())
    return out


def main():
    proc, model = load_26b("int4")
    layers = model.model.language_model.layers
    torch.manual_seed(0)
    rec = dict(kind="test2")
    lay = layers[11]
    mod = lay.experts.inner
    x = torch.randn(24, 2816, device="cuda", dtype=torch.bfloat16)
    _, w, idx = lay.router(x)
    h = lay.pre_feedforward_layernorm_2(x).float()
    g = torch.randn(24, 2816, device="cuda")
    v = torch.randn_like(h)
    vw = torch.randn_like(w) * 0.05
    res = {}
    for name, fn in (("kernel", lambda a, b: mod(a.to(torch.bfloat16), idx, b).float()),
                     ("torch_bf16", lambda a, b: experts_torch(mod, a.to(torch.bfloat16), idx, b).float()),
                     ("fp32", lambda a, b: experts_fp32(mod, a, idx, b))):
        with torch.no_grad():
            for part, (dv, dw) in (("x", (v, torch.zeros_like(vw))), ("w", (torch.zeros_like(v), vw))):
                fd = []
                for eps in (0.01, 0.05):
                    fp = (fn(h + eps * dv, w + eps * dw) * g).sum()
                    fm = (fn(h - eps * dv, w - eps * dw) * g).sum()
                    fd.append(round(float((fp - fm) / (2 * eps)), 2))
                res[f"fd_{name}_{part}"] = fd
    for name, fn in (("function", lambda a, b: lay.experts(a.to(torch.bfloat16), idx, b).float()),
                     ("fp32", lambda a, b: experts_fp32(mod, a, idx, b))):
        hr = h.clone().requires_grad_(True)
        wr = w.detach().clone().float().requires_grad_(True)
        (fn(hr, wr) * g).sum().backward()
        res[f"ad_{name}_x"] = round(float((hr.grad * v).sum()), 2)
        res[f"ad_{name}_w"] = round(float((wr.grad * vw).sum()), 2)
    rec["experts"] = res
    # suffix pass: Function backward vs all-autograd torch experts
    run = Runner26(model, proc)
    it = dict(_aokvqa("validation", 1, 7)[0])
    it["image"] = it["image"].resize(WH)
    enc = encode_batch(proc, [it], 70).to("cuda")
    kv = {}
    hooks = []
    for l, ly in enumerate(layers):
        a = ly.self_attn
        hooks.append(a.k_proj.register_forward_hook(lambda m, i, o, l=l: kv.__setitem__((l, "k"), o)))
        if a.v_proj is not None:
            hooks.append(a.v_proj.register_forward_hook(lambda m, i, o, l=l: kv.__setitem__((l, "v"), o)))
    with torch.no_grad():
        model(**enc, logits_to_keep=1)
    for hk in hooks:
        hk.remove()
    T = enc["input_ids"].shape[1]
    Tc = T - 9
    parts = []
    for l in range(len(layers)):
        parts.append(kv[(l, "k")][:, 5:Tc])
        if layers[l].self_attn.v_proj is not None:
            parts.append(kv[(l, "v")][:, 5:Tc])
    Y = torch.cat(parts, -1).float()
    m = torch.tensor([Tc - 5], device="cuda")
    gvec = torch.randn(K, device="cuda")
    grads = []
    for mode in ("function", "torch"):
        if mode == "torch":
            for ly in layers:
                ly.experts.forward = (lambda self_: (lambda hs, ti, tw: experts_torch(self_.inner, hs, ti, tw)))(ly.experts)
        Yr = Y.clone().requires_grad_(True)
        ctx, valid, hh = run.context(Yr.to(torch.bfloat16), m, True)
        z = run.suffix(ctx, valid, hh, m)
        (z * gvec).sum().backward()
        grads.append(Yr.grad.clone())
    a, b = grads
    rec["suffix_grad_function_vs_torch"] = dict(rel_err=round(float((a - b).norm() / b.norm()), 4),
                                                cos=round(float(F.cosine_similarity(a.flatten(), b.flatten(), 0)), 5),
                                                per_layer_cos=[round(float(F.cosine_similarity(x.flatten(), y.flatten(), 0)), 4)
                                                               for x, y in zip(torch.split(a, run.sizes, -1), torch.split(b, run.sizes, -1))][::5])
    print(json.dumps(rec), flush=True)
    log_jsonl("tests.jsonl", rec)


if __name__ == "__main__":
    main()
