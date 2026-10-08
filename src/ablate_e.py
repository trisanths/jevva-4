"""How much does each Jev-specific saving buy? Compiled + graph-captured fast path with the KV-shared tail on the
answer token only (Jev) vs over every token (stock), and the letter rows (Jev) vs the 262K-row head.

Usage: python ablate_e.py <model-id> [budget]   -> appends to results/jev/ablation.jsonl
"""
import json
import os
import statistics
import sys

import torch

from common import ROOT, encode_batch, load
from data import _aokvqa
from jev_fast_e import JevFastE


def main(mid, budget=70):
    items = [dict(it, image=it["image"].resize((640, 480))) for it in _aokvqa("validation", 64, 5)]
    proc, model = load(mid=mid, attn="sdpa")
    for mod in model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = "sdpa"
    S = -(-max(encode_batch(proc, [it], budget)["input_ids"].shape[1] for it in items) // 16) * 16
    res = dict(model=mid, budget=budget, S=S, gpu=torch.cuda.get_device_name())
    ref = None
    for kv_skip, full_head in ((True, False), (False, False), (True, True), (False, True)):
        f = JevFastE(model, proc, budget, S)
        f.kv_skip, f.full_head = kv_skip, full_head
        f.prepare(items[0])
        f.capture(compile_mode="default")
        for _ in range(10):
            f.graph.replay()
        torch.cuda.synchronize()
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(100)]
        for a, b in ev:
            a.record(); f.graph.replay(); b.record()
        torch.cuda.synchronize()
        key = f"{'kvskip' if kv_skip else 'fulltail'}+{'fullhead' if full_head else 'cand'}"
        res[key + "_ms"] = round(statistics.median(a.elapsed_time(b) for a, b in ev), 2)
        out = f.out.clone()
        if ref is None:
            ref = out
        res[key + "_max_dp_vs_jev"] = round(float((out - ref).abs().max()), 4)
        del f
        torch.cuda.empty_cache()
    print(json.dumps(res), flush=True)
    with open(os.path.join(ROOT, "results", "jev", "ablation.jsonl"), "a") as fh:
        fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 70)
