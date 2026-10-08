"""Precompute for the KV-handoff test.

Per item (options permuted once at random for train-split items, untouched for eval):
  - E2B-jev residual stream (outputs of HX_LAYERS) at the middle positions -> feats.bin (bf16)
  - E2B-jev letter probs (its own template) and 26B direct letter probs (its own template, own vision tower)
  - calib split only: all 15 E2B layer outputs, and the 26B's raw k_proj / v_proj outputs at every position
    (y.bin) plus its raw q_proj outputs at the 9 suffix positions (q.bin)
Token ids from the two processors are checked per item: identical through the user text, then each template's
own suffix.

Usage: python hx_prep.py calib|eval|train
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx_core import (BOI, BUDGET, HDR, HX, K, LAYERS, SUF26, SUFE, WH, letter_ids, load_26b, load_e2b_jev)
from common import encode_batch
from data import LazyImageSet, _mmlu, eval_items, text_train_items

SPLIT = sys.argv[1]
BS = 1 if SPLIT == "calib" else 8
FEAT_LAYERS = list(range(15)) if SPLIT == "calib" else LAYERS


def permute(it, perm):
    return dict(it, options=[it["options"][p] for p in perm], answer=perm.index(it["answer"]))


def img_pool(seed):
    """LazyImageSet keys that have an image, with a cheap text-length proxy for batching."""
    from datasets import Image
    s = LazyImageSet(seed=seed)
    aok_q = s.aok.select_columns(["question", "choices"])
    sqa_meta = s.sqa.select_columns(["question", "hint", "choices", "image"]).cast_column("image", Image(decode=False))
    out = []
    for src, i in s.keys:
        if src == "aok":
            r = aok_q[i]
            out.append(((src, i), len(r["question"]) + sum(map(len, r["choices"]))))
        else:
            r = sqa_meta[i]
            if r["image"] is None:
                continue
            out.append(((src, i), len(r["question"]) + len(r["hint"] or "") + sum(map(len, r["choices"]))))
    return s, out


def build_jobs():
    rng = random.Random(1234)
    if SPLIT == "eval":
        items = eval_items()
        entries = [("item", dict(it, image=it["image"].resize(WH)) if it["image"] is not None else it, None, "eval")
                   for it in items]
        imgset = None
    elif SPLIT == "calib":
        imgset, pool = img_pool(21)
        pool = pool[:464]
        txt = _mmlu("auxiliary_train", 80, 31)
        entries = [("key", k, None, "fit" if j < 400 else "held") for j, (k, _) in enumerate(pool)]
        entries += [("item", it, None, "fit" if j < 64 else "held") for j, it in enumerate(txt)]
    else:
        imgset, pool = img_pool(11)
        txt = text_train_items()
        rng.shuffle(txt)
        entries = [("key", k, None, "train") for k, _ in sorted(pool, key=lambda kv: kv[1])]
        entries += [("item", it, None, "train") for it in sorted(txt[:10000], key=lambda it: len(it["question"]))]
    jobs, cur, cur_img = [], [], None
    for e in entries:
        is_img = e[0] == "key" or e[1]["image"] is not None
        if cur and (len(cur) == BS or is_img != cur_img):
            jobs.append(cur)
            cur = []
        cur.append(e)
        cur_img = is_img
    if cur:
        jobs.append(cur)
    return imgset, jobs


class Jobs(Dataset):
    def __init__(self, imgset, jobs, p26, pe2):
        self.imgset, self.jobs, self.p26, self.pe2 = imgset, jobs, p26, pe2

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, j):
        items, tags = [], []
        for kind, x, _, tag in self.jobs[j]:
            it = self.imgset.get(x) if kind == "key" else x
            if kind == "key":
                it = dict(it, image=it["image"].resize(WH))
            if SPLIT != "eval":
                r = random.Random(it["id"])
                perm = list(range(len(it["options"])))
                r.shuffle(perm)
                it = dict(permute(it, perm), perm=perm)
            items.append(it)
            tags.append(tag)
        a = encode_batch(self.p26, items, BUDGET)
        b = encode_batch(self.pe2, items, BUDGET)
        if "pixel_values" in a:
            assert torch.equal(a["pixel_values"], b["pixel_values"]), "pixel values differ"
            assert torch.equal(a["image_position_ids"], b["image_position_ids"])
        meta = []
        for r, it in enumerate(items):
            ra = a["input_ids"][r][a["attention_mask"][r].bool()].tolist()
            rb = b["input_ids"][r][b["attention_mask"][r].bool()].tolist()
            assert ra[:-len(SUF26)] == rb[:-len(SUFE)], f"prefix differs for {it['id']}"
            assert ra[-len(SUF26):] == SUF26 and rb[-len(SUFE):] == SUFE, it["id"]
            assert ra[:4] == HDR, it["id"]
            img = it["image"] is not None
            h = 5 if img else 4
            if img:
                assert ra[4] == BOI, it["id"]
            assert len(it["options"]) <= K
            meta.append(dict(id=it["id"], source=it["source"], image=img, n_opts=len(it["options"]), answer=it["answer"],
                             perm=it.get("perm"), h=h, n26=len(ra), m=len(ra) - len(SUF26) - h, tag=tags[r]))
        return dict(a), dict(b), meta


def main():
    from transformers import AutoProcessor
    out = os.path.join(HX, SPLIT)
    os.makedirs(out, exist_ok=True)
    for f in ("feats.bin", "meta.jsonl", "y.bin", "q.bin"):
        if os.path.exists(os.path.join(out, f)):
            os.remove(os.path.join(out, f))
    p26 = AutoProcessor.from_pretrained("google/gemma-4-26B-A4B-it", padding_side="left")
    pe2 = AutoProcessor.from_pretrained("google/gemma-4-E2B-it", padding_side="left")
    t0 = time.time()
    imgset, jobs = build_jobs()
    print(f"{SPLIT}: {len(jobs)} batches, {sum(map(len, jobs))} items, jobs built in {time.time() - t0:.0f}s", flush=True)
    _, me = load_e2b_jev()
    _, m26 = load_26b("int4")
    cand = torch.tensor(letter_ids(p26.tokenizer)[:K], device="cuda")
    # hooks
    e_out = {}
    for L in FEAT_LAYERS:
        me.model.language_model.layers[L].register_forward_hook(lambda mod, i, o, L=L: e_out.__setitem__(L, o))
    kv_out, q_out = {}, {}
    if SPLIT == "calib":
        for l, layer in enumerate(m26.model.language_model.layers):
            a = layer.self_attn
            a.k_proj.register_forward_hook(lambda mod, i, o, l=l: kv_out.__setitem__((l, "k"), o))
            if a.v_proj is not None:
                a.v_proj.register_forward_hook(lambda mod, i, o, l=l: kv_out.__setitem__((l, "v"), o))
            a.q_proj.register_forward_hook(lambda mod, i, o, l=l: q_out.__setitem__(l, o))
    nl = len(m26.model.language_model.layers)
    has_v = [m26.model.language_model.layers[l].self_attn.v_proj is not None for l in range(nl)]
    json.dump(dict(layers=FEAT_LAYERS, D=1536 * len(FEAT_LAYERS), budget=BUDGET, wh=WH), open(os.path.join(out, "config.json"), "w"))
    fe = open(os.path.join(out, "feats.bin"), "ab")
    fy = open(os.path.join(out, "y.bin"), "ab") if SPLIT == "calib" else None
    fq = open(os.path.join(out, "q.bin"), "ab") if SPLIT == "calib" else None
    fm = open(os.path.join(out, "meta.jsonl"), "w")
    off = yoff = qoff = 0
    dl = DataLoader(Jobs(imgset, jobs, p26, pe2), batch_size=None, shuffle=False, num_workers=12, prefetch_factor=4,
                    persistent_workers=False)
    n_done, t0 = 0, time.time()
    with torch.no_grad():
        for a, b, meta in dl:
            a = {k: v.cuda(non_blocking=True) for k, v in a.items()}
            b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
            ze = me(**b, logits_to_keep=1).logits[:, -1].float()[:, cand]
            z6 = m26(**a, logits_to_keep=1).logits[:, -1].float()[:, cand]
            Sa, Sb = a["input_ids"].shape[1], b["input_ids"].shape[1]
            feats = torch.cat([e_out[L] for L in FEAT_LAYERS], -1)
            for r, mt in enumerate(meta):
                n = mt["n_opts"]
                mt["e2b"] = torch.softmax(ze[r, :n], -1).tolist()
                mt["t26"] = torch.softmax(z6[r, :n], -1).tolist()
                pad_b = Sb - (mt["n26"] - len(SUF26) + len(SUFE))
                x = feats[r, pad_b + mt["h"]: pad_b + mt["h"] + mt["m"]].to(torch.bfloat16).contiguous()
                fe.write(x.view(torch.int16).cpu().numpy().tobytes())
                mt["off"] = off
                off += mt["m"]
                if fy is not None:
                    pad_a = Sa - mt["n26"]
                    parts = []
                    for l in range(nl):
                        parts.append(kv_out[(l, "k")][r, pad_a:])
                        if has_v[l]:
                            parts.append(kv_out[(l, "v")][r, pad_a:])
                    y = torch.cat(parts, -1).to(torch.bfloat16).contiguous()
                    fy.write(y.view(torch.int16).cpu().numpy().tobytes())
                    mt["yoff"] = yoff
                    yoff += y.shape[0]
                    q = torch.cat([q_out[l][r, -len(SUF26):] for l in range(nl)], -1).to(torch.bfloat16).contiguous()
                    fq.write(q.view(torch.int16).cpu().numpy().tobytes())
                    mt["qoff"] = qoff
                    qoff += q.shape[0]
                fm.write(json.dumps(mt) + "\n")
            n_done += len(meta)
            if n_done % 800 < len(meta):
                print(f"{n_done} items {time.time() - t0:.0f}s feats rows {off}", flush=True)
    for f in (fe, fy, fq, fm):
        if f is not None:
            f.close()
    meta = [json.loads(l) for l in open(os.path.join(out, "meta.jsonl"))]
    rec = dict(kind="prep", split=SPLIT, items=len(meta), rows=off, seconds=round(time.time() - t0),
               feat_layers=FEAT_LAYERS, budget=BUDGET, wh=WH)
    for key in ("t26", "e2b"):
        per = {}
        for m in meta:
            per.setdefault(m["source"], []).append(int(np.argmax(m[key]) == m["answer"]))
        rec[f"acc_{key}"] = {s: round(100 * sum(v) / len(v), 2) for s, v in sorted(per.items())}
        img = [v for s, l in per.items() if s in ("scienceqa", "aokvqa", "ai2d", "mmstar") for v in l]
        txt = [v for s, l in per.items() if s in ("mmlu", "arc", "csqa") for v in l]
        rec[f"acc_{key}"]["_image"] = round(100 * sum(img) / len(img), 2) if img else None
        rec[f"acc_{key}"]["_text"] = round(100 * sum(txt) / len(txt), 2) if txt else None
    print(json.dumps(rec), flush=True)
    from hx_core import log_jsonl
    log_jsonl("prep.jsonl", rec)


if __name__ == "__main__":
    main()
