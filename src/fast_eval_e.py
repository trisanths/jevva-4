"""Accuracy of the deployable fast path: images resized to 640x480, one compiled CUDA graph per prompt-length
bucket, optional FP8 (JEV_FP8=tensor) and optional Jev-tuned adapter. Scores the image eval items and compares
with the HF path on native-resolution images.

Usage: python fast_eval_e.py <model-id> <budget> [adapter] [tag]
Writes results/jev/fast-<short>-<tag>@<budget>.json.
"""
import json
import os
import sys
import time

import torch

from common import ROOT, by_source, encode_batch, load
from data import eval_items
from jev_fast_e import JevFastE

BUCKETS = (144, 192, 256, 320, 384, 448, 512, 640, 768)


def main(mid, budget, adapter=None, tag=None):
    short = mid.split("/")[-1].replace("gemma-4-", "").replace("-it", "")
    prec = ("fp8" + os.environ["JEV_FP8"]) if os.environ.get("JEV_FP8") else "bf16"
    comp = os.environ.get("JEV_COMPILE", "default")
    tag = tag or f"{prec}-{'c' if comp else 'nc'}" + ("-jev" if adapter else "")
    items = [dict(it, image=it["image"].resize((640, 480))) for it in eval_items() if it["image"] is not None]
    fp8 = os.environ.pop("JEV_FP8", None)            # merge the adapter in bf16, quantize afterwards
    proc, model = load(mid=mid, attn="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
        nt = os.path.join(adapter, "norm.pt")
        if os.path.exists(nt):
            model.model.language_model.norm.weight.data = torch.load(nt)["norm"].to(model.dtype).cuda()
    if fp8:
        from common import quantize_fp8
        from common import quantize_mixed
        if fp8 == "mixed":
            quantize_mixed(model)
        else:
            quantize_fp8(model, rowwise=fp8 == "row", weight_only=fp8 == "wo")
        os.environ["JEV_FP8"] = fp8
    for mod in model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = "sdpa"
    lens = {it["id"]: encode_batch(proc, [it], budget)["input_ids"].shape[1] for it in items}
    graphs, probs, t0 = {}, {}, time.time()
    for it in sorted(items, key=lambda x: lens[x["id"]]):
        S = next(b for b in BUCKETS if b >= lens[it["id"]])
        if S not in graphs:
            g = JevFastE(model, proc, budget, S)
            g.prepare(it)
            g.capture(compile_mode=comp or None)
            graphs[S] = g
            print(f"bucket S={S} captured at {time.time() - t0:.0f}s", flush=True)
        p = graphs[S](it)
        probs[it["id"]] = p[: len(it["options"])].float().cpu().tolist()
    res = by_source(items, probs)
    ref_path = os.path.join(ROOT, "results", "jev", f"{short}-base@{budget}.json")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))["probs"]
        res["_vs_hf_native"] = by_source(items, probs, ref)["_all"]
    name = f"fast-{short}-{tag}@{budget}"
    json.dump(dict(name=name, model=mid, budget=budget, adapter=adapter, precision=prec, compile=comp, metrics=res, probs=probs),
              open(os.path.join(ROOT, "results", "jev", f"{name}.json"), "w"))
    a = res["_all"]
    extra = res.get("_vs_hf_native", {})
    print(f"{name}: image acc {a['acc']:.4f} nll {a['nll']:.3f} ece {a['ece']:.3f} | vs HF native: agree {extra.get('agree', float('nan')):.3f} "
          f"kl {extra.get('kl', float('nan')):.3f} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), a[2] if len(a) > 2 and a[2] != "-" else None, a[3] if len(a) > 3 else None)
