"""Evaluate a hybrid (E2B features -> mapper -> 26B suffix pass) on the eval split, image and text items.

Usage: python hx_eval.py <tag> <mapper.pt> [lora.pt]
  mapper.pt: ridge.pt (A1) or a trained checkpoint from hx_train.py (mapper state + LoRA in one file).
Writes results/handoff/eval.jsonl (one record per run) and results/handoff/probs-<tag>.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx_core import (BUDGET, OUT, Mapper, Runner26, Store, acc_report, add_lora, layer_cols, letter_probs, load_26b,
                     log_jsonl)


def kl_agree(meta, probs, key):
    kl, ag = [], []
    for m in meta:
        p = np.clip(np.asarray(probs[m["id"]], dtype=np.float64), 1e-12, 1)
        t = np.clip(np.asarray(m[key], dtype=np.float64), 1e-12, 1)
        kl.append(float((t * (np.log(t) - np.log(p))).sum()))
        ag.append(int(p.argmax() == t.argmax()))
    return round(float(np.mean(kl)), 4), round(float(np.mean(ag)), 4)


def batches_for(meta, bs):
    order = sorted(range(len(meta)), key=lambda i: (not meta[i]["image"], meta[i]["m"]))
    out, cur = [], []
    for i in order:
        if cur and (len(cur) == bs or meta[cur[0]]["image"] != meta[i]["image"]):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def evaluate(run, mapper, st, cols, bs=16):
    probs = {}
    for b in batches_for(st.meta, bs):
        X, m = st.batch(b, cols)
        Y = mapper(X)
        img = st.meta[b[0]]["image"]
        ctx, valid, h = run.context(Y, m, img)
        z = run.suffix(ctx, valid, h, m)
        n = [st.meta[i]["n_opts"] for i in b]
        p = letter_probs(z, n)
        for j, i in enumerate(b):
            probs[st.meta[i]["id"]] = p[j, : n[j]].float().cpu().tolist()
    return probs


def report(tag, st, probs, extra=None):
    rec = dict(kind="eval", tag=tag, budget=BUDGET, acc=acc_report(st.meta, probs))
    img = [m for m in st.meta if m["image"]]
    txt = [m for m in st.meta if not m["image"]]
    rec["kl_to_26b_image"], rec["agree_26b_image"] = kl_agree(img, probs, "t26")
    rec["kl_to_26b_text"], rec["agree_26b_text"] = kl_agree(txt, probs, "t26")
    rec["agree_e2b_image"] = kl_agree(img, probs, "e2b")[1]
    rec["agree_e2b_text"] = kl_agree(txt, probs, "e2b")[1]
    if extra:
        rec.update(extra)
    return rec


def load_mapper(path, run):
    d = torch.load(path, map_location="cpu")
    if "W" in d:                                              # ridge.pt
        mp = Mapper(d["mu"], d["sd"], d["W"], d["b"], rank=0, train_bias=False)
        return mp, d["layers"], None, d
    cfg = d["cfg"]
    rd = torch.load(cfg["ridge"], map_location="cpu")
    mp = Mapper(rd["mu"], rd["sd"], rd["W"], rd["b"], rank=cfg["rank"], train_full=cfg.get("full", False))
    mp.load_state_dict(d["mapper"], strict=False)
    return mp, rd["layers"], d.get("lora"), cfg


def main(tag, path):
    t0 = time.time()
    st = Store("eval")
    proc, model = load_26b("int4")
    run = Runner26(model, proc)
    for img in (True, False):
        run.header(img)                                       # native header KV (base weights, before any LoRA)
    mp, layers, lora_sd, cfg = load_mapper(path, run)
    if lora_sd is not None:
        mods = add_lora(model, cfg.get("lora_r", 16), cfg.get("lora_alpha", 32))
        for mod, sd in zip(mods, lora_sd):
            mod.A.data.copy_(sd["A"]); mod.B.data.copy_(sd["B"])
    cols = layer_cols(st.layers, layers)
    probs = evaluate(run, mp.eval(), st, cols)
    rec = report(tag, st, probs, dict(mapper=path, seconds=round(time.time() - t0)))
    # references on the same items
    rec["ref_26b_direct"] = acc_report(st.meta, {m["id"]: m["t26"] for m in st.meta})
    rec["ref_e2b_jev"] = acc_report(st.meta, {m["id"]: m["e2b"] for m in st.meta})
    print(json.dumps(rec), flush=True)
    log_jsonl("eval.jsonl", rec)
    json.dump(probs, open(os.path.join(OUT, f"probs-{tag}.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
