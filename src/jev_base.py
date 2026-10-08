"""Untuned Jev-wrapper baselines for the small Gemma 4 models on the Stage 1 eval set.

Usage: python jev_base.py <model-id> [budgets]   e.g. google/gemma-4-E4B-it 280,140,70
Text-only items carry no image, so they are scored once (at the first budget) and reused.
Writes results/jev/<short>@<budget>.json and results/jev/summary.json.
"""
import json
import os
import sys
import time

from common import ROOT, by_source, load, score
from data import eval_items

OUT = os.path.join(ROOT, "results", "jev")
os.makedirs(OUT, exist_ok=True)


def run(mid, budgets, tag="base", adapter=None, bs=8):
    short = mid.split("/")[-1].replace("gemma-4-", "").replace("-it", "")
    items = eval_items()
    proc, model = load(mid=mid)
    if adapter:
        import torch
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
        nt = os.path.join(adapter, "norm.pt")
        if os.path.exists(nt):
            model.model.language_model.norm.weight.data = torch.load(nt)["norm"].to(model.dtype).cuda()
    summary_path = os.path.join(OUT, "summary.json")
    text_probs = None
    for st in budgets:
        name = f"{short}-{tag}@{st}"
        t0 = time.time()
        todo = items if text_probs is None else [it for it in items if it["image"] is not None]
        probs = score(model, proc, todo, None, bs=bs, soft_tokens=st)
        if text_probs is None:
            text_probs = {it["id"]: probs[it["id"]] for it in items if it["image"] is None}
        else:
            probs.update(text_probs)
        dt = time.time() - t0
        res = by_source(items, probs)
        rec = dict(name=name, model=mid, tag=tag, adapter=adapter, soft_tokens=st, seconds=dt, metrics=res)
        json.dump(dict(rec, probs=probs), open(os.path.join(OUT, f"{name}.json"), "w"))
        summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
        summary[name] = rec
        json.dump(summary, open(summary_path, "w"), indent=1)
        a = res["_all"]
        print(f"{name:22s} acc {a['acc']:.4f} nll {a['nll']:.3f} ece {a['ece']:.3f} img {res['_image']['acc']:.4f} "
              f"txt {res['_text']['acc']:.4f} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0], [int(x) for x in (a[1] if len(a) > 1 else "280,140,70").split(",")],
        tag=a[3] if len(a) > 3 else ("jev" if len(a) > 2 else "base"), adapter=a[2] if len(a) > 2 else None)
