"""Latency for the one-image, one-short-question case: A-OKVQA questions, 640x480 images, graph sized to those
prompts. Reports stock HF (one forward, SDPA, logits at one position) vs the Jev fast path, and a per-stage
split of the fast path (vision tower / LM layers over the sequence / KV-shared tail + readout).

Usage: python fast_bench_e.py <model-id> [budgets]
Appends to results/jev/fast_bench.jsonl.
"""
import json
import os
import statistics
import sys
import time

import torch

from common import ROOT, encode_batch, load
from data import _aokvqa
from jev_fast_e import JevFastE


def cuda_ms(fn, reps=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for a, b in ev:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in ev)


def stage_split(f):
    """Eager run of the fast path with events between stages (includes launch overhead)."""
    b = f.buf
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    with torch.no_grad():
        ev[0].record()
        h = f.patch(b["pix"], b["ppos"], torch.zeros(1, f.P, dtype=torch.bool, device="cuda"))
        rot = f.venc.rotary_emb(h, b["ppos"])
        for layer in f.venc.layers:
            h = layer(h, attention_mask=None, position_embeddings=rot, position_ids=b["ppos"])
        v = f.embed_vision(((b["pool"] @ h.float()) * (f.vcfg.hidden_size ** 0.5)).to(f.dtype))[0]
        ev[1].record()
        tok = f.lm.embed_tokens(b["ids"])
        ple_ids = f.lm.get_per_layer_inputs(b["ids"], tok)
        x = tok[0].index_copy(0, b["img_idx"], v.to(tok.dtype))[None]
        ple = f.lm.project_per_layer_inputs(x, ple_ids)
        rope = {t: f.lm.rotary_emb(x, b["pos"], t) for t in set(f.types)}
        shared = {}
        for i in range(f.first):
            t = f.types[i]
            x = f.layers[i](x, ple[:, :, i], shared_kv_states=shared, position_embeddings=rope[t],
                            attention_mask=b["mask"], position_ids=b["pos"])
        ev[2].record()
        xt, pos_t = x[:, -1:], b["pos"][:, -1:]
        for i in range(f.first, len(f.layers)):
            t = f.types[i]
            cos, sin = rope[t]
            xt = f.layers[i](xt, ple[:, -1:, i], shared_kv_states=shared, position_embeddings=(cos[:, -1:], sin[:, -1:]),
                             attention_mask=b["mask_last"], position_ids=pos_t)
        f.lm.norm(xt[:, 0]) @ f.E.t()
        ev[3].record()
    torch.cuda.synchronize()
    return [ev[i].elapsed_time(ev[i + 1]) for i in range(3)]


def main(mid, budgets):
    items = [dict(it, image=it["image"].resize((640, 480))) for it in _aokvqa("validation", 64, 5)]
    proc, model = load(mid=mid, attn="sdpa")
    for mod in model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = "sdpa"
    for st in budgets:
        lens = [encode_batch(proc, [it], st)["input_ids"].shape[1] for it in items]
        S = -(-max(lens) // 16) * 16
        enc = encode_batch(proc, items[:1], st).to("cuda")
        with torch.no_grad():
            stock = cuda_ms(lambda: model(**enc, logits_to_keep=1), reps=20)
        fast = JevFastE(model, proc, st, S)
        fast.prepare(items[0])
        mode = os.environ.get("JEV_COMPILE")
        fast.capture(compile_mode=mode)
        g = cuda_ms(fast.graph.replay, reps=100)
        # Agreement of the captured graph with the uncompiled eager fast path on a few items.
        agree = []
        for it in items[:16]:
            fast.prepare(it); fast.graph.replay(); pg = fast.out.clone()
            pe = fast._forward()
            agree.append(float((pg - pe).abs().max()))
        splits = [statistics.median(x) for x in zip(*[stage_split(fast) for _ in range(10)])]
        t = []
        for it in items[:20]:
            t0 = time.perf_counter(); fast.prepare(it); torch.cuda.synchronize(); t.append((time.perf_counter() - t0) * 1e3)
        res = dict(model=mid, precision=("fp8-" + os.environ["JEV_FP8"]) if os.environ.get("JEV_FP8") else "bf16", compile=mode, graph_vs_eager_max_dp=round(max(agree), 4), budget=st, S=S, median_len=int(statistics.median(lens)), P=fast.P, N=fast.N,
                   stock_ms=round(stock, 2), fast_ms=round(g, 2), eager_split_ms=dict(vision=round(splits[0], 2),
                   lm_prefill=round(splits[1], 2), tail_readout=round(splits[2], 2)), prep_ms=round(statistics.median(t), 2),
                   gpu=torch.cuda.get_device_name())
        print(json.dumps(res), flush=True)
        with open(os.path.join(ROOT, "results", "jev", "fast_bench.jsonl"), "a") as fh:
            fh.write(json.dumps(res) + "\n")
        del fast
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "70,140,280").split(",")])
