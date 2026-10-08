"""Stage 1: training-free ceilings for MoEfying Gemma 4 12B, scored as Jev-style option decisions.

Runs every config on the same eval items with identical batches, saves per-item candidate logits
(results/stage1/<config>.json) and a summary table (results/stage1/summary.json). Configs:

  dense@280/140/70    no sparsity; image soft-token budget sweep (the no-training FLOP lever)
  blind@280           image replaced by a mid-grey image of the same size
  dense@280-rev       options in reversed order (selection-bias check)
  oracle-*            per-token best-k experts by output norm, for several partitions
  neuron-*            per-token top-fraction neurons (unstructured ceiling)
  depth-*             coact-8 top-2 applied to one third of the layers only
  image-only          coact-8 top-2 on image tokens, text tokens dense
"""
import json
import os
import sys
import time

import torch
from PIL import Image

import moe
from common import ROOT, by_source, load, score
from data import eval_items

OUT = os.path.join(ROOT, "results", "stage1")
os.makedirs(OUT, exist_ok=True)

# name: (mode, dict of controller settings, soft tokens, item transform)
CONFIGS = [
    ("dense@280", "dense", {}, 280, None),
    ("neuron-50", "neuron", dict(frac=0.5), 280, None),
    ("neuron-25", "neuron", dict(frac=0.25), 280, None),
    ("coact8-k4", "oracle", dict(part="coact-8", k=4), 280, None),
    ("coact8-k2", "oracle", dict(part="coact-8", k=2), 280, None),
    ("dense@140", "dense", {}, 140, None),
    ("dense@70", "dense", {}, 70, None),
    ("coact128-k64", "oracle", dict(part="coact-128", k=64), 280, None),
    ("coact32-k16", "oracle", dict(part="coact-32", k=16), 280, None),
    ("shared8-s1k3", "oracle", dict(part="shared-8", shared=1, k=3), 280, None),
    ("best8-k4", "best", dict(part="coact-8", k=4), 280, None),
    ("random8-k4", "oracle", dict(part="random-8", k=4), 280, None),
    ("depth-early-coact8-k4", "oracle", dict(part="coact-8", k=4, layers=(0, 16)), 280, None),
    ("depth-mid-coact8-k4", "oracle", dict(part="coact-8", k=4, layers=(16, 32)), 280, None),
    ("depth-late-coact8-k4", "oracle", dict(part="coact-8", k=4, layers=(32, 48)), 280, None),
    ("coact8-k4-image-only", "oracle", dict(part="coact-8", k=4, scope="image"), 280, None),
    ("coact128-k32", "oracle", dict(part="coact-128", k=32), 280, None),
    ("random128-k64", "oracle", dict(part="random-128", k=64), 280, None),
    ("neuron-12.5", "neuron", dict(frac=0.125), 280, None),
    ("blind@280", "dense", {}, 280, "grey"),
    ("dense@280-rev", "dense", {}, 280, "reverse"),
]


def transform(items, how):
    if how is None:
        return items
    out = []
    for it in items:
        it = dict(it)
        if how == "grey" and it["image"] is not None:
            it["image"] = Image.new("RGB", it["image"].size, (128, 128, 128))
        elif how == "reverse":
            it["options"] = it["options"][::-1]
            it["answer"] = len(it["options"]) - 1 - it["answer"]
        out.append(it)
    return out


def main(only=None, scale=1.0):
    items = eval_items(scale=scale)
    print("eval items", len(items), flush=True)
    proc, model = load()
    ctl = moe.install(model)
    parts = torch.load(os.path.join(ROOT, "partitions.pt"))
    summary_path = os.path.join(OUT, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    ref = None
    ref_path = os.path.join(OUT, "dense@280.json")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))["probs"]
    for name, mode, kw, st, how in CONFIGS:
        if only and name not in only:
            continue
        path = os.path.join(OUT, f"{name}.json")
        if os.path.exists(path):
            continue
        its = transform(items, how)
        ctl.mode, ctl.scope = mode, kw.get("scope", "all")
        ctl.k, ctl.shared, ctl.frac = kw.get("k", 8), kw.get("shared", 0), kw.get("frac", 1.0)
        if "part" in kw:
            E = int(kw["part"].split("-")[1])
            ctl.set_partition(parts[kw["part"]], E)
        ctl.collect = mode != "dense"
        ctl.reset_stats()
        lo, hi = kw.get("layers", (0, ctl.n_layers))
        for l, layer in enumerate(model.model.language_model.layers):
            layer.mlp.active = lo <= l < hi
        t0 = time.time()
        try:
            probs = score(model, proc, its, ctl, bs=8, soft_tokens=st)
        except Exception:
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue
        dt = time.time() - t0
        if name == "dense@280":
            ref = probs
        res = by_source(its, probs, ref if how != "reverse" else None)
        if how == "reverse":
            # Map reversed-option probabilities back to original order before comparing with dense.
            back = {k: v[::-1] for k, v in probs.items()}
            res["_vs_dense"] = by_source(items, back, ref)["_all"]
        rec = dict(name=name, mode=mode, kw={k: v for k, v in kw.items()}, soft_tokens=st, seconds=dt, metrics=res)
        if ctl.collect:
            rec["rel_err"] = ctl.rel_err().tolist()
            if ctl.usage is not None and mode in ("oracle", "best"):
                u = ctl.usage / ctl.usage.sum(1, keepdim=True).clamp(min=1)
                rec["usage_max_share"] = u.max(1).values.tolist()
        json.dump(dict(rec, probs=probs), open(path, "w"))
        summary[name] = rec
        json.dump(summary, open(summary_path, "w"), indent=1)
        a = res["_all"]
        print(f"{name:28s} acc {a['acc']:.4f} nll {a['nll']:.3f} ece {a['ece']:.3f} kl {a.get('kl', 0):.3f} "
              f"agree {a.get('agree', 1):.3f} img {res['_image']['acc']:.4f} txt {res['_text']['acc']:.4f} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main(only=set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else None,
         scale=float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
