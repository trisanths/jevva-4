"""Stage 1 baseline: Gemma 4 26B-A4B (with its 550M vision tower) on the same items and prompts, at 280/140/70
image soft tokens. bf16 weights do not fit a 46 GB L40S, so accelerate keeps the overflow on the CPU; this
changes speed, not outputs. Writes results/stage1/26b@<budget>.json and adds rows to summary.json."""
import json
import os
import sys
import time

import torch

from common import ROOT, by_source, load, score
from data import eval_items

OUT = os.path.join(ROOT, "results", "stage1")


def main(budgets):
    items = eval_items()
    proc, model = load(mid="google/gemma-4-26B-A4B-it", device_map="auto",
                       max_memory={0: "34GiB", "cpu": "110GiB"}, experts_implementation="grouped_mm")
    print("device map spill:", sorted({str(v) for v in model.hf_device_map.values()}), flush=True)
    ref = json.load(open(os.path.join(OUT, "dense@280.json")))["probs"]
    summary_path = os.path.join(OUT, "summary.json")
    for st in budgets:
        name = f"26b@{st}"
        path = os.path.join(OUT, f"{name}.json")
        if os.path.exists(path):
            continue
        t0 = time.time()
        base = os.path.join(OUT, "26b@280.json")
        if st != 280 and os.path.exists(base):
            # Text-only items carry no image, so their answers cannot depend on the image budget: reuse them.
            prev = json.load(open(base))["probs"]
            probs = score(model, proc, [it for it in items if it["image"] is not None], None, bs=8, soft_tokens=st)
            probs.update({it["id"]: prev[it["id"]] for it in items if it["image"] is None})
        else:
            probs = score(model, proc, items, None, bs=8, soft_tokens=st)
        dt = time.time() - t0
        res = by_source(items, probs, ref)
        rec = dict(name=name, mode="26b", kw={}, soft_tokens=st, seconds=dt, metrics=res)
        json.dump(dict(rec, probs=probs), open(path, "w"))
        summary = json.load(open(summary_path))
        summary[name] = rec
        json.dump(summary, open(summary_path, "w"), indent=1)
        a = res["_all"]
        print(f"{name:12s} acc {a['acc']:.4f} nll {a['nll']:.3f} ece {a['ece']:.3f} kl-vs-12b {a['kl']:.3f} "
              f"img {res['_image']['acc']:.4f} txt {res['_text']['acc']:.4f} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main([int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "280,140,70").split(",")])
