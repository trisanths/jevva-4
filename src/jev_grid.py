"""Zero-shot accuracy of Jevified 12B variants (early exit L, image drop after M), no training.

Usage: python jev_grid.py L:M [L:M ...] [--st 280]   (M = L means no drop)
Writes results/jev/grid-L<L>-M<M>@<st>.json and prints one line per config.
"""
import json
import os
import sys
import time

import torch

from common import ROOT, by_source, score
from data import eval_items
from jevnet import State, install_drop, load_truncated

OUT = os.path.join(ROOT, "results", "jev")
os.makedirs(OUT, exist_ok=True)


def main(specs, st):
    items = eval_items()
    specs = sorted(specs, key=lambda s: (-s[0], s[1]))
    loaded, proc, model = None, None, None
    for L, M in specs:
        short = os.environ.get("JEV_MID", "google/gemma-4-12B-it").split("/")[-1].replace("gemma-4-", "").replace("-it", "")
        name = f"grid-{short}-L{L}-M{M}@{st}"
        path = os.path.join(OUT, f"{name}.json")
        if os.path.exists(path):
            continue
        if loaded != L:
            del model
            torch.cuda.empty_cache()
            proc, model = load_truncated(L, mid=os.environ.get("JEV_MID", "google/gemma-4-12B-it"))
            loaded = L
        state = State()
        hs = install_drop(model, M, state) if M < L else []
        t0 = time.time()
        probs = score(model, proc, items, state, bs=8, soft_tokens=st)
        for h in hs:
            h.remove()
        res = by_source(items, probs)
        json.dump(dict(name=name, L=L, M=M, soft_tokens=st, seconds=time.time() - t0, metrics=res, probs=probs), open(path, "w"))
        a = res["_all"]
        print(f"{name:22s} acc {a['acc']:.4f} img {res['_image']['acc']:.4f} txt {res['_text']['acc']:.4f} nll {a['nll']:.3f} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    st = 280
    if "--st" in args:
        i = args.index("--st"); st = int(args[i + 1]); args = args[:i] + args[i + 2:]
    main([tuple(int(x) for x in a.split(":")) for a in args], st)
