"""Wall-clock per decision on this GPU, random weights (speed does not depend on values).

  12B dense                 the real architecture
  12B ffn-50 / ffn-25       the 12B with FFN width cut to 7680 / 3840: an ideal top-4 / top-2 MoE with zero
                            routing or dispatch overhead, i.e. the fastest a MoEfied 12B could run
  26B tower                 the 550M vision tower alone
  26B half LM               tower + the first 15 of 30 decoder layers; full LM time is extrapolated as
                            2 x (half - tower), since the 52 GB bf16 model does not fit on a 46 GB L40S

Inputs are real 280-token-budget image prompts. SDPA attention (fast path); numerics are irrelevant here.
Writes results/timing.json.
"""
import json
import os
import statistics
import time

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from common import ROOT, encode_batch
from data import eval_items

BATCHES = (1, 8)


def timed(fn, reps=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def build(mid, edit=None, experts=None):
    cfg = AutoConfig.from_pretrained(mid)
    if edit:
        edit(cfg)
    kw = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    if experts:
        kw["experts_implementation"] = experts
    with torch.device("cuda"):
        model = AutoModelForImageTextToText.from_config(cfg, **kw).eval()
    return model


def run_set(name, mid, items, edit=None, experts=None, tower_only=False):
    proc = AutoProcessor.from_pretrained(mid, padding_side="left")
    model = build(mid, edit, experts)
    out = {}
    for bs in BATCHES:
        enc = encode_batch(proc, (items * 8)[:bs]).to("cuda")
        with torch.no_grad():
            if tower_only:
                f = lambda: model.model.get_image_features(pixel_values=enc["pixel_values"],
                                                           image_position_ids=enc.get("image_position_ids"))
            else:
                f = lambda: model(**enc, logits_to_keep=1)
            ms = timed(f)
        out[bs] = dict(ms_batch=ms, ms_per_decision=ms / bs, seq=int(enc["input_ids"].shape[1]))
        print(f"{name:14s} bs {bs:2d}  {ms:8.1f} ms/batch  {ms / bs:7.1f} ms/decision  S={enc['input_ids'].shape[1]}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    items = [it for it in eval_items(scale=0.02) if it["image"] is not None][:8]
    res = {}
    m12, m26 = "google/gemma-4-12B-it", "google/gemma-4-26B-A4B-it"

    def width(n):
        def f(cfg):
            cfg.text_config.intermediate_size = n
        return f

    def half(cfg):
        t = cfg.text_config
        t.num_hidden_layers = 15
        t.layer_types = t.layer_types[:15]

    res["12B dense"] = run_set("12B dense", m12, items)
    res["12B ffn-50"] = run_set("12B ffn-50", m12, items, width(7680))
    res["12B ffn-25"] = run_set("12B ffn-25", m12, items, width(3840))
    for impl in ("grouped_mm", "batched_mm", None):
        try:
            res["26B tower"] = run_set("26B tower", m26, items, half, impl, tower_only=True)
            res["26B half"] = run_set(f"26B half {impl}", m26, items, half, impl)
            res["26B experts impl"] = impl
            break
        except Exception as e:
            print("experts impl", impl, "failed:", str(e).split("\n")[0][:160], flush=True)
            torch.cuda.empty_cache()
    if "26B half" in res:
        res["26B full (extrapolated)"] = {
            bs: dict(ms_per_decision=res["26B tower"][bs]["ms_per_decision"]
                     + 2 * (res["26B half"][bs]["ms_per_decision"] - res["26B tower"][bs]["ms_per_decision"]))
            for bs in BATCHES}
        for bs in BATCHES:
            print(f"26B full (extrap) bs {bs:2d}  {res['26B full (extrapolated)'][bs]['ms_per_decision']:7.1f} ms/decision", flush=True)
    res["gpu"] = torch.cuda.get_device_name()
    json.dump(res, open(os.path.join(ROOT, "results", "timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
