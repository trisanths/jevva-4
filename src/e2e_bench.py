"""End-to-end decision latency: a 640x480 uint8 frame in pinned host memory + a question string in, option
probabilities on the host out. Steps timed together: upload, GPU image preprocessing, tokenization, masks,
CUDA-graph replay, readback. Also checks the raw path against the HF-processor path.

Usage: python e2e_bench.py <model-id> <budget> [adapter]      (JEV_FP8, JEV_COMPILE as elsewhere)
Appends to results/jev/e2e.jsonl.
"""
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

from common import ROOT, encode_batch, load, quantize_fp8
from data import _aokvqa
from jev_fast_e import JevFastE


def main(mid, budget, adapter=None):
    items = [dict(it, image=it["image"].resize((640, 480))) for it in _aokvqa("validation", 80, 7)]
    fp8 = os.environ.pop("JEV_FP8", None)
    proc, model = load(mid=mid, attn="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
        nt = os.path.join(adapter, "norm.pt")
        if os.path.exists(nt):
            model.model.language_model.norm.weight.data = torch.load(nt)["norm"].to(model.dtype).cuda()
    if fp8:
        from common import quantize_mixed
        if fp8 == "mixed":
            quantize_mixed(model)
        else:
            quantize_fp8(model, rowwise=fp8 == "row", weight_only=fp8 == "wo")
    for mod in model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = "sdpa"
    comp = os.environ.get("JEV_COMPILE", "default") or None
    S = -(-max(encode_batch(proc, [it], budget)["input_ids"].shape[1] for it in items) // 16) * 16
    f = JevFastE(model, proc, budget, S)
    f.prepare(items[0])
    f.setup_raw(items[0])
    f.capture(compile_mode=comp)
    frames = [torch.from_numpy(np.asarray(it["image"])).permute(2, 0, 1).contiguous().pin_memory() for it in items]
    # Equivalence: processor path vs raw path.
    same_ids, dps, agree = 0, [], 0
    for it, fr in zip(items[:32], frames[:32]):
        f.prepare(it); f.graph.replay(); a = f.out.clone(); ids_a = f.buf["ids"].clone()
        f.prepare_raw(fr.cuda(), it); f.graph.replay(); b = f.out.clone()
        same_ids += int(torch.equal(ids_a, f.buf["ids"]))
        dps.append(float((a - b).abs().max())); agree += int(a.argmax() == b.argmax())
    # End-to-end timing.
    t_e2e, t_prep = [], []
    for rep in range(3):
        for it, fr in zip(items, frames):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g = fr.cuda(non_blocking=True)
            f.prepare_raw(g, it)
            t1 = time.perf_counter()
            f.graph.replay()
            p = f.out[: len(it["options"])].float().cpu()
            t2 = time.perf_counter()
            if rep:
                t_e2e.append((t2 - t0) * 1e3); t_prep.append((t1 - t0) * 1e3)
    t_e2e.sort()
    res = dict(model=mid, budget=budget, precision=("fp8-" + fp8) if fp8 else "bf16", compile=comp, adapter=adapter, S=S,
               e2e_p50=round(t_e2e[len(t_e2e) // 2], 2), e2e_p90=round(t_e2e[int(len(t_e2e) * 0.9)], 2),
               e2e_p99=round(t_e2e[int(len(t_e2e) * 0.99)], 2), prep_p50=round(statistics.median(t_prep), 2),
               raw_vs_processor=dict(same_ids=f"{same_ids}/32", max_dp=round(max(dps), 4), argmax_agree=f"{agree}/32"),
               gpu=torch.cuda.get_device_name())
    print(json.dumps(res), flush=True)
    with open(os.path.join(ROOT, "results", "jev", "e2e.jsonl"), "a") as fh:
        fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), a[2] if len(a) > 2 and a[2] != "-" else None)
