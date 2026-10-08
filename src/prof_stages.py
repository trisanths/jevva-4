"""Where do the milliseconds go in the compiled fast path? Compile and capture each stage as its own graph
(vision tower, LM layers over the sequence, KV-shared tail + readout) and time each replay.

Usage: python prof_stages.py <model-id> <budget>
"""
import json
import os
import statistics
import sys

import torch

from common import ROOT, encode_batch, load
from data import _aokvqa
from jev_fast_e import JevFastE


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            out = fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def ms(g, reps=100):
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for a, b in ev:
        a.record(); g.replay(); b.record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in ev)


def main(mid, budget):
    items = [dict(it, image=it["image"].resize((640, 480))) for it in _aokvqa("validation", 64, 5)]
    proc, model = load(mid=mid, attn="sdpa")
    for mod in model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = "sdpa"
    S = -(-max(encode_batch(proc, [it], budget)["input_ids"].shape[1] for it in items) // 16) * 16
    f = JevFastE(model, proc, budget, S)
    f.prepare(items[0])
    b = f.buf
    st = {}

    @torch.no_grad()
    def vision():
        h = f.patch(b["pix"], b["ppos"], torch.zeros(1, f.P, dtype=torch.bool, device="cuda"))
        rot = f.venc.rotary_emb(h, b["ppos"])
        for layer in f.venc.layers:
            h = layer(h, attention_mask=None, position_embeddings=rot, position_ids=b["ppos"])
        return f.embed_vision(((b["pool"] @ h.float()) * (f.vcfg.hidden_size ** 0.5)).to(f.dtype))[0]

    v_static = vision().clone()

    @torch.no_grad()
    def prefill():
        tok = f.lm.embed_tokens(b["ids"])
        ple_ids = f.lm.get_per_layer_inputs(b["ids"], tok)
        x = tok[0].index_copy(0, b["img_idx"], v_static.to(tok.dtype))[None]
        ple = f.lm.project_per_layer_inputs(x, ple_ids)
        rope = {t: f.lm.rotary_emb(x, b["pos"], t) for t in set(f.types)}
        shared = {}
        for i in range(f.first):
            t = f.types[i]
            x = f.layers[i](x, ple[:, :, i], shared_kv_states=shared, position_embeddings=rope[t],
                            attention_mask=b["mask"], position_ids=b["pos"])
        st["shared"], st["ple"], st["rope"], st["x"] = shared, ple, rope, x
        return x

    prefill()
    shared = {k: (kv[0].clone(), kv[1].clone()) for k, kv in st["shared"].items()}
    ple, rope, x_static = st["ple"].clone(), {k: (c.clone(), s.clone()) for k, (c, s) in st["rope"].items()}, st["x"].clone()

    @torch.no_grad()
    def tail():
        xt, pos_t = x_static[:, -1:], b["pos"][:, -1:]
        for i in range(f.first, len(f.layers)):
            t = f.types[i]
            cos, sin = rope[t]
            xt = f.layers[i](xt, ple[:, -1:, i], shared_kv_states=shared, position_embeddings=(cos[:, -1:], sin[:, -1:]),
                             attention_mask=b["mask_last"], position_ids=pos_t)
        return (f.lm.norm(xt[:, 0]) @ f.E.t()).float()

    res = dict(model=mid, budget=budget, S=S, P=f.P, N=f.N, precision=os.environ.get("JEV_FP8") and "fp8" or "bf16")
    for name, fn in (("vision", vision), ("prefill", prefill), ("tail", tail)):
        for mode in (None, "default"):
            g, _ = graph_of(torch.compile(fn, mode=mode, dynamic=False) if mode else fn)
            res[f"{name}_{'compiled' if mode else 'plain'}_ms"] = round(ms(g), 2)
            del g
    print(json.dumps(res), flush=True)
    with open(os.path.join(ROOT, "results", "jev", "stages.jsonl"), "a") as fh:
        fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 70)
