"""A1: closed-form ridge map from E2B-jev residual features to the 26B's raw k_proj / v_proj outputs.

Fit on the calib split's "fit" items (400 image + 64 MMLU-aux text, middle positions only, header excluded),
features standardized, intercept, one ridge solve shared by every target column:
    (C + lam I) W = Xs^T (Y - mean) / n,   C = Xs^T Xs / n.
Selection on the "held" items (64 image + 16 text): for each candidate E2B layer set and lambda, R^2 of the raw
targets and the attention-output cosine at the 9 suffix positions (true suffix queries and suffix KV, header KV
true, middle KV true vs mapped), per 26B layer. Lambda is then chosen per 26B layer by that cosine. End-to-end
hybrid agreement with the 26B's own letter probs on the held items is reported for the best sets and for the
oracle (true middle KV).

Usage: python hx_ridge.py
Writes $ROOT/hx/b<budget>/ridge.pt and results/handoff/ridge.jsonl.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx_core import (HX, K, SUF26, D_E, Runner26, Store, letter_probs, layer_cols, load_26b, log_jsonl)

SETS = [[14], [9, 14], [4, 9, 14], [12, 13, 14], [8, 11, 14], [2, 6, 10, 14], [4, 9, 12, 14], [10, 12, 14]]
LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def main():
    t0 = time.time()
    st = Store("calib")
    assert st.layers == list(range(15))
    nl_total = None
    yraw = np.memmap(os.path.join(st.dir, "y.bin"), dtype=np.int16, mode="r")
    qraw = np.memmap(os.path.join(st.dir, "q.bin"), dtype=np.int16, mode="r")
    proc, model = load_26b("int4")
    run = Runner26(model, proc)
    T = run.total
    Y_all = yraw.reshape(-1, T)
    qdims = []
    for lay in run.layers:
        qdims.append(lay.self_attn.q_proj.out_features)
    Q_all = qraw.reshape(-1, sum(qdims))
    fit = [i for i, m in enumerate(st.meta) if m["tag"] == "fit"]
    held = [i for i, m in enumerate(st.meta) if m["tag"] == "held"]

    def y_mid(i):
        m = st.meta[i]
        a = Y_all[m["yoff"] + m["h"]: m["yoff"] + m["h"] + m["m"]]
        return torch.from_numpy(np.ascontiguousarray(a)).view(torch.bfloat16)

    def y_full(i):
        m = st.meta[i]
        return torch.from_numpy(np.ascontiguousarray(Y_all[m["yoff"]: m["yoff"] + m["n26"]])).view(torch.bfloat16)

    def q_suf(i):
        m = st.meta[i]
        return torch.from_numpy(np.ascontiguousarray(Q_all[m["qoff"]: m["qoff"] + len(SUF26)])).view(torch.bfloat16)

    # ---- sufficient statistics over the fit set (all 15 E2B layers) ----
    D = st.D
    n = 0
    Sx = torch.zeros(D, device="cuda", dtype=torch.float64)
    Sxx = torch.zeros(D, D, device="cuda")
    Sy = torch.zeros(T, device="cuda", dtype=torch.float64)
    Syy = torch.zeros(T, device="cuda", dtype=torch.float64)
    Sxy = torch.zeros(D, T, device="cuda")
    buf_x, buf_y = [], []

    def flush():
        nonlocal n
        X = torch.cat(buf_x).cuda().float()
        Yb = torch.cat(buf_y).cuda().float()
        n += X.shape[0]
        Sx.add_(X.sum(0).double())
        Sxx.addmm_(X.t(), X)
        Sy.add_(Yb.sum(0).double())
        Syy.add_((Yb * Yb).sum(0).double())
        Sxy.addmm_(X.t(), Yb)
        buf_x.clear(); buf_y.clear()

    rows = 0
    for i in fit:
        buf_x.append(st.x(i)); buf_y.append(y_mid(i))
        rows += buf_x[-1].shape[0]
        if rows >= 3000:
            flush(); rows = 0
    if buf_x:
        flush()
    print(f"stats over {n} rows in {time.time() - t0:.0f}s", flush=True)
    mu = (Sx / n).float()
    var = (torch.diagonal(Sxx).double() / n - (Sx / n) ** 2).clamp(min=1e-8)
    sd = var.sqrt().float()
    my = (Sy / n).float()
    vy = (Syy / n - (Sy / n) ** 2).float()

    # held-out data on GPU
    Xh = [st.x(i).cuda() for i in held]
    Yh = [y_mid(i).cuda() for i in held]
    Yf = [y_full(i).cuda() for i in held]
    Qh = [q_suf(i).cuda() for i in held]

    def attn_cos(Ymaps):
        """Mean attention-output cosine per 26B layer at the suffix positions, true vs mapped middle KV."""
        out = torch.zeros(len(run.layers), device="cuda")
        for j, i in enumerate(held):
            m = st.meta[i]
            n26, h, mm = m["n26"], m["h"], m["m"]
            pos = torch.arange(n26, device="cuda")[None]
            Yt = Yf[j][None]
            Ym = Yt.clone()
            Ym[0, h: h + mm] = Ymaps[j].to(torch.bfloat16)
            qs = torch.split(Qh[j], qdims, -1)
            for l, (pt, pm) in enumerate(zip(run.split(Yt), run.split(Ym))):
                kvh, d, hv = run.dims[l]
                cos, sin = run.rope(pos, run.types[l])
                kt, vt = run.kv_post(l, pt[0], pt[1], cos, sin)
                km, vm = run.kv_post(l, pm[0], pm[1], cos, sin)
                from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
                a = run.layers[l].self_attn
                q = a.q_norm(qs[l].view(1, len(SUF26), -1, d))
                q = apply_rotary_pos_emb(q, cos[:, -len(SUF26):], sin[:, -len(SUF26):], unsqueeze_dim=2).transpose(1, 2)
                rep = q.shape[1] // kvh
                mask = torch.ones(len(SUF26), n26, dtype=torch.bool, device="cuda").tril(n26 - len(SUF26))
                ot = F.scaled_dot_product_attention(q, kt.repeat_interleave(rep, 1), vt.repeat_interleave(rep, 1), attn_mask=mask, scale=1.0)
                om = F.scaled_dot_product_attention(q, km.repeat_interleave(rep, 1), vm.repeat_interleave(rep, 1), attn_mask=mask, scale=1.0)
                out[l] += F.cosine_similarity(ot.float(), om.float(), dim=-1).mean()
        return (out / len(held)).tolist()

    def r2(Ymaps):
        sse = torch.zeros(T, device="cuda")
        sst = torch.zeros(T, device="cuda")
        for Ym, Yt in zip(Ymaps, Yh):
            e = Ym.float() - Yt.float()
            sse += (e * e).sum(0)
            sst += ((Yt.float() - my) ** 2).sum(0)
        per = 1 - sse / sst.clamp(min=1e-6)
        return [float(p.mean()) for p in torch.split(per, run.sizes)]

    def solve(cols, lam):
        c = cols.cuda()
        Cxx = (Sxx[c][:, c] / n - torch.outer(mu[c], mu[c])) / torch.outer(sd[c], sd[c])
        Cxy = (Sxy[c] / n - torch.outer(mu[c], my)) / sd[c][:, None]
        A = Cxx + lam * torch.eye(len(c), device="cuda")
        L = torch.linalg.cholesky(A.double()).float()
        return torch.cholesky_solve(Cxy, L)                               # [d, T] on standardized inputs

    def apply(cols, W):
        c = cols.cuda()
        return [(((x[:, c].float() - mu[c]) / sd[c]) @ W + my).to(torch.bfloat16) for x in Xh]

    results = []
    skip = bool(int(os.environ.get("HX_SKIP_SEL", 0)))
    if skip:
        for line in open(os.path.join(os.environ.get("MOEFY_ROOT", "/mnt/nvme/moefy"), "results", "handoff", "ridge.jsonl")):
            r = json.loads(line)
            if r.get("kind") == "ridge_sel" and r["layers"] in SETS:
                results.append(r)
    for S in ([] if skip else SETS):
        cols = layer_cols(st.layers, S)
        for lam in LAMS:
            W = solve(cols, lam)
            Ym = apply(cols, W)
            rr, ac = r2(Ym), attn_cos(Ym)
            rec = dict(kind="ridge_sel", layers=S, lam=lam, r2_mean=round(float(np.mean(rr)), 4),
                       attn_cos_mean=round(float(np.mean(ac)), 4), attn_cos=[round(a, 4) for a in ac], r2=[round(r, 4) for r in rr])
            print(json.dumps({k: v for k, v in rec.items() if k not in ("attn_cos", "r2")}), flush=True)
            log_jsonl("ridge.jsonl", rec)
            results.append(rec)
            del W, Ym
            torch.cuda.empty_cache()

    def set_score(S):
        recs = [r for r in results if r["layers"] == S]
        return float(np.mean([max(r["attn_cos"][l] for r in recs) for l in range(len(run.layers))]))
    scores = {tuple(S): set_score(S) for S in SETS}
    force = [[int(x) for x in g.split(",")] for g in os.environ.get("HX_SETS", "").split(";") if g]
    final_sets = force or [list(max(scores, key=scores.get))]

    def per_layer_W(S):
        """Ridge map with lambda chosen per 26B layer by held-out attention cosine; solved one lambda at a time
        on just the columns that use it (keeps GPU memory flat)."""
        recs = [r for r in results if r["layers"] == S]
        lam_l = [max(recs, key=lambda r: r["attn_cos"][l])["lam"] for l in range(len(run.layers))]
        cols = layer_cols(st.layers, S)
        c = cols.cuda()
        Cxx = (Sxx[c][:, c] / n - torch.outer(mu[c], mu[c])) / torch.outer(sd[c], sd[c])
        W = torch.zeros(len(c), T, device="cuda")
        edges = np.cumsum([0] + run.sizes)
        for lam in sorted(set(lam_l)):
            L = torch.linalg.cholesky((Cxx + lam * torch.eye(len(c), device="cuda")).double()).float()
            for l in range(len(run.layers)):
                if lam_l[l] != lam:
                    continue
                a, b = int(edges[l]), int(edges[l + 1])
                rhs = (Sxy[c, a:b] / n - torch.outer(mu[c], my[a:b])) / sd[c][:, None]
                W[:, a:b] = torch.cholesky_solve(rhs, L)
        return W, lam_l, cols

    finals = {}
    for S in final_sets:
        W, lam_l, cols = per_layer_W(S)
        Ym = apply(cols, W)
        finals[tuple(S)] = dict(W=W.to(torch.bfloat16).cpu(), lam=lam_l, cols=cols, Ym=[y.cpu() for y in Ym],
                                r2=[round(r, 4) for r in r2(Ym)], attn_cos=[round(a, 4) for a in attn_cos(Ym)])
        del W, Ym
        torch.cuda.empty_cache()
    del Sxy, Sxx, Yf, Qh
    torch.cuda.empty_cache()
    best_set = final_sets[0]
    final = dict(layers=best_set, lam_per_layer=finals[tuple(best_set)]["lam"],
                 set_scores={str(list(k)): round(v, 4) for k, v in scores.items()},
                 r2=finals[tuple(best_set)]["r2"], attn_cos=finals[tuple(best_set)]["attn_cos"])

    # end-to-end on held items: chosen map, the default set at lambda 0.01, and the oracle (true middle KV)
    def e2e(Ymaps):
        agree, acc, kl = 0, 0, 0.0
        with torch.no_grad():
            for j, i in enumerate(held):
                m = st.meta[i]
                Y = Ymaps[j][None].to("cuda", torch.bfloat16)
                mm = torch.tensor([m["m"]], device="cuda")
                ctx, valid, h = run.context(Y, mm, m["image"])
                p = letter_probs(run.suffix(ctx, valid, h, mm), [m["n_opts"]])[0, : m["n_opts"]].double().cpu()
                t = torch.tensor(m["t26"], dtype=torch.float64)
                agree += int(p.argmax() == t.argmax()); acc += int(int(p.argmax()) == m["answer"])
                kl += float((t * (t.clamp(min=1e-12).log() - p.clamp(min=1e-12).log())).sum())
        N = len(held)
        return dict(agree_26b=round(agree / N, 4), acc=round(acc / N, 4), kl=round(kl / N, 4), n=N)

    for S, f in finals.items():
        final[f"e2e_{'_'.join(map(str, S))}"] = e2e(f["Ym"])
        final[f"attn_cos_mean_{'_'.join(map(str, S))}"] = round(float(np.mean(f["attn_cos"])), 4)
    final["e2e_oracle"] = e2e(Yh)
    final["e2e_intercept_only"] = e2e([my[None].expand(x.shape[0], -1).to(torch.bfloat16) for x in Xh])
    held_t26 = sum(int(np.argmax(st.meta[i]["t26"]) == st.meta[i]["answer"]) for i in held) / len(held)
    held_e2b = sum(int(np.argmax(st.meta[i]["e2b"]) == st.meta[i]["answer"]) for i in held) / len(held)
    final.update(kind="ridge_final", held_acc_26b_direct=round(held_t26, 4), held_acc_e2b_jev=round(held_e2b, 4),
                 fit_rows=n, seconds=round(time.time() - t0))
    print(json.dumps(final), flush=True)
    log_jsonl("ridge.jsonl", final)
    for j, (S, f) in enumerate(finals.items()):
        c = f["cols"].cuda()
        name = "ridge.pt" if j == 0 else f"ridge-{'_'.join(map(str, S))}.pt"
        torch.save(dict(W=f["W"], b=my.cpu(), mu=mu[c].cpu(), sd=sd[c].cpu(), layers=list(S), lam=f["lam"]), os.path.join(HX, name))
    print("saved", os.path.join(HX, "ridge.pt"), flush=True)


if __name__ == "__main__":
    main()
