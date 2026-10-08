"""Deployment latency of the hybrid: uint8 frame + question in, letter probabilities out, batch 1, L40S.

Stages (each a CUDA graph over static buffers, optionally torch.compile'd first, plus one graph over all three):
  front   GPU-preprocessed patches -> E2B-jev vision tower -> 15 E2B layers over the left-padded prefix bucket
          -> outputs of the chosen layers (bf16, same as the precomputed features)
  mapper  one GEMM: features @ W' + b' (standardization folded into W', bf16), every bucket row
  suffix  26B k_norm / RoPE / v_norm on the mapped rows, native header KV in front, 9 suffix tokens through all
          30 layers (Marlin int4 experts), softcapped letter readout, softmax over the offered letters
Correctness: the fast path against the reference path (HF E2B features from the processor, unfolded mapper,
Runner26 on the same Marlin model) on 32 items.

Usage: python hx_bench.py <mapper.pt> [tag]     (JEV_COMPILE=default|"" ; HX_E2B_FP8=tensor to quantize the front)
"""
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx_core import (BUDGET, K, SUF26, WH, Mapper, Runner26, add_lora, letter_ids, load_26b, load_e2b_jev, log_jsonl)
from common import encode_batch
from data import _aokvqa, prompt_text
from jev_fast26 import cuda_ms
from jev_fast_e import JevFastE


def merged_map(path):
    """(W', b', layers): folded standardization, low-rank delta and LoRA (returned separately) merged."""
    d = torch.load(path, map_location="cpu")
    lora = None
    if "W" in d:
        mu, sd, W, b, layers = d["mu"], d["sd"], d["W"].float(), d["b"].float(), d["layers"]
    else:
        cfg = d["cfg"]
        rd = torch.load(cfg["ridge"], map_location="cpu")
        mu, sd, W, layers = rd["mu"], rd["sd"], rd["W"].float(), rd["layers"]
        st = d["mapper"]
        b = st["b"].float()
        if cfg["rank"]:
            W = W + st["U"].float() @ st["V"].float()
        lora = (d["lora"], cfg)
    Wf = W / sd[:, None]
    bf = b - (mu / sd) @ W
    return Wf.to(torch.bfloat16), bf.to(torch.bfloat16), layers, (mu, sd, W, b), lora


class Hybrid:
    def __init__(self, me, pe, run, W, b, layers, budget, S):
        self.f = JevFastE(me, pe, budget, S)
        self.me, self.pe, self.layers = me, pe, layers
        self.run = run
        self.hdr = self.run.header(True)
        self.h = self.hdr[0][0].shape[2]
        self.W, self.b = W.cuda(), b.cuda()
        self.S = S
        d = "cuda"
        self.F = torch.zeros(S, len(layers) * 1536, dtype=torch.bfloat16, device=d)
        self.Y = torch.zeros(S, self.run.total, dtype=torch.bfloat16, device=d)
        self.pos_mid = torch.zeros(1, S, dtype=torch.long, device=d)
        self.pos_suf = torch.zeros(1, len(SUF26), dtype=torch.long, device=d)
        self.ctx_valid = torch.zeros(1, self.h + S, dtype=torch.bool, device=d)
        self.suf_ids = torch.tensor([SUF26], device=d)
        self.cand = torch.zeros(K, dtype=torch.bool, device=d)
        self.out = None

    # ---- inputs
    def _hyb(self, n, n_opts):
        m = n - self.h
        r = torch.arange(self.S)
        valid = r >= self.S - m
        pos = torch.where(valid, self.h + r - (self.S - m), torch.zeros_like(r))
        self.pos_mid.copy_(pos[None])
        self.pos_suf.copy_((self.h + m + torch.arange(len(SUF26)))[None])
        self.ctx_valid.copy_(torch.cat([torch.ones(self.h, dtype=torch.bool), valid])[None])
        c = torch.zeros(K, dtype=torch.bool); c[:n_opts] = True
        self.cand.copy_(c)

    def prepare(self, item):
        enc = encode_batch(self.pe, [item], self.f.budget)
        ids = enc["input_ids"][0][: -5]
        pv, pp = enc["pixel_values"][0], enc["image_position_ids"][0]
        real = (pp != -1).all(-1)
        pv, pp = pv[real], pp[real]
        if self.f.buf is None:
            self.f._alloc(pv.shape[0], int((ids == self.f.img_tok).sum()))
        self.f._fill(ids, pv, pp, len(item["options"]))
        self._hyb(ids.numel(), len(item["options"]))
        return ids

    def setup_raw(self):
        marker = "QQQMARKERQQQ"
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": marker}]}]
        tmpl = self.pe.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        ph = self.pe.image_token if self.pe.image_token in tmpl else self.pe.boi_token
        expanded = f"{self.pe.boi_token}{self.pe.image_token * self.f.N}{self.pe.eoi_token}"
        self.raw_pre = tmpl.replace(ph, expanded, 1).split(marker)[0]

    def prepare_raw(self, frame_u8, item):
        import torchvision.transforms.v2.functional as tvF
        from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size
        H, Wd = frame_u8.shape[-2:]
        th, tw = get_aspect_ratio_preserving_size(H, Wd, 16, self.f.budget * 9, 3)
        img = tvF.resize(frame_u8, [th, tw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True).float() / 255.0
        ph, pw = th // 16, tw // 16
        patches = img.reshape(3, ph, 16, pw, 16).permute(1, 3, 2, 4, 0).reshape(ph * pw, -1)
        gx, gy = torch.meshgrid(torch.arange(pw, device="cuda"), torch.arange(ph, device="cuda"), indexing="xy")
        pp = torch.stack([gx, gy], -1).reshape(ph * pw, 2)
        ids = torch.tensor(self.pe.tokenizer(self.raw_pre + prompt_text(item), add_special_tokens=False)["input_ids"])
        self.f._fill(ids, patches, pp, len(item["options"]))
        self._hyb(ids.numel(), len(item["options"]))
        return ids

    # ---- stages
    @torch.no_grad()
    def front(self):
        f, b = self.f, self.f.buf
        h = f.patch(b["pix"], b["ppos"], torch.zeros(1, f.P, dtype=torch.bool, device="cuda"))
        rot = f.venc.rotary_emb(h, b["ppos"])
        for layer in f.venc.layers:
            h = layer(h, attention_mask=None, position_embeddings=rot, position_ids=b["ppos"])
        v = (b["pool"] @ h.float()) * (f.vcfg.hidden_size ** 0.5)
        v = f.embed_vision(v.to(f.dtype))[0]
        tok = f.lm.embed_tokens(b["ids"])
        ple_ids = f.lm.get_per_layer_inputs(b["ids"], tok)
        x = tok[0].index_copy(0, b["img_idx"], v.to(tok.dtype))[None]
        ple = f.lm.project_per_layer_inputs(x, ple_ids)
        rope = {t: f.lm.rotary_emb(x, b["pos"], t) for t in set(f.types)}
        shared, feats = {}, []
        for i in range(max(self.layers) + 1):
            t = f.types[i]
            x = f.layers[i](x, ple[:, :, i], shared_kv_states=shared, position_embeddings=rope[t],
                            attention_mask=b["mask"], position_ids=b["pos"])
            if i in self.layers:
                feats.append(x[0])
        self.F.copy_(torch.cat(feats, -1))

    @torch.no_grad()
    def mapper(self):
        torch.addmm(self.b, self.F, self.W, out=self.Y)

    @torch.no_grad()
    def suffix(self):
        run = self.run
        ropes = {t: run.rope(self.pos_mid, t) for t in set(run.types)}
        ctx = []
        for l, (kr, vr) in enumerate(run.split(self.Y[None])):
            cos, sin = ropes[run.types[l]]
            k, v = run.kv_post(l, kr, vr, cos, sin)
            hk, hv = self.hdr[l]
            ctx.append((torch.cat([hk, k], 2), torch.cat([hv, v], 2)))
        z = run.run(self.suf_ids, self.pos_suf, ctx, self.ctx_valid)[0]
        return torch.softmax(z.masked_fill(~self.cand[None], -1e9), -1)[0]

    def all(self):
        self.front(); self.mapper()
        return self.suffix()


def graph(fn, comp):
    g_fn = torch.compile(fn, mode=comp, dynamic=False) if comp else fn
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            g_fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = g_fn()
    return g, out


def main(path, tag=None):
    t0 = time.time()
    comp = os.environ.get("JEV_COMPILE", "default") or None
    items = [dict(it, image=it["image"].resize(WH)) for it in _aokvqa("validation", 80, 7)]
    pe, me = load_e2b_jev()
    if os.environ.get("HX_E2B_FP8"):
        from common import quantize_fp8
        quantize_fp8(me, rowwise=False)
    for mod in me.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = "sdpa"
    p26, m26 = load_26b("marlin")
    W, b, layers, raw, lora = merged_map(path)
    run = Runner26(m26, p26)
    run.header(True)                                         # native header KV with base weights, as in training
    if lora is not None:
        lsd, cfg = lora
        mods = add_lora(m26, cfg.get("lora_r", 16), cfg.get("lora_alpha", 32))
        for mod, sd in zip(mods, lsd):
            base = mod.base
            base.weight.data += (sd["B"].float() @ sd["A"].float() * mod.scale).to(base.weight.dtype).cuda()
            mod.on = False                                   # LoRA merged into the base weights
    S = -(-max(encode_batch(pe, [it], BUDGET)["input_ids"].shape[1] - 5 for it in items) // 16) * 16
    hy = Hybrid(me, pe, run, W, b, layers, BUDGET, S)
    hy.prepare(items[0])
    hy.setup_raw()
    # reference path: processor features through HF E2B hooks, unfolded mapper, Runner26 (same Marlin model)
    mu, sd, W0, b0 = raw
    ref_map = Mapper(mu, sd, W0, b0, rank=0, train_bias=False)
    feats = {}
    hooks = [me.model.language_model.layers[L].register_forward_hook(lambda m, i, o, L=L: feats.__setitem__(L, o)) for L in layers]
    ref = []
    with torch.no_grad():
        for it in items[:32]:
            enc = encode_batch(pe, [it], BUDGET).to("cuda")
            me(**enc, logits_to_keep=1)
            X = torch.cat([feats[L] for L in layers], -1)[:, hy.h: enc["input_ids"].shape[1] - 5]
            Y = ref_map(X)
            mm = torch.tensor([X.shape[1]], device="cuda")
            ctx, valid, h = hy.run.context(Y, mm, True)
            z = hy.run.suffix(ctx, valid, h, mm)
            ref.append(torch.softmax(z[0, : len(it["options"])], -1).float().cpu())
    for hk in hooks:
        hk.remove()
    gF, _ = graph(hy.front, comp)
    gM, _ = graph(hy.mapper, None)
    gS, outS = graph(hy.suffix, comp)
    gA, outA = graph(hy.all, None)
    frames = [torch.from_numpy(np.asarray(it["image"])).permute(2, 0, 1).contiguous().pin_memory() for it in items]
    same_ids, agree, dps = 0, 0, []
    for it, fr, r in zip(items[:32], frames[:32], ref):
        ids_a = hy.prepare(it)
        ids_b = hy.prepare_raw(fr.cuda(), it)
        same_ids += int(torch.equal(ids_a, ids_b))
        gA.replay()
        p = outA[: len(it["options"])].float().cpu()
        dps.append(float((p - r).abs().max())); agree += int(p.argmax() == r.argmax())
    hy.prepare(items[0])
    rec = dict(kind="bench", tag=tag, mapper=path, budget=BUDGET, S_bucket=S, ctx_len=hy.h + S, suffix_len=len(SUF26),
               compile=comp, e2b_fp8=os.environ.get("HX_E2B_FP8"), gpu=torch.cuda.get_device_name(),
               front_ms=round(cuda_ms(gF.replay, 50), 2), mapper_ms=round(cuda_ms(gM.replay, 50), 2),
               suffix_ms=round(cuda_ms(gS.replay, 50), 2), all_graph_ms=round(cuda_ms(gA.replay, 50), 2),
               fast_vs_ref=dict(argmax_agree=f"{agree}/32", max_dp=round(max(dps), 4), median_dp=round(statistics.median(dps), 4)),
               raw_vs_processor_ids=f"{same_ids}/32")
    ts, tp = [], []
    for rep in range(3):
        for it, fr in zip(items, frames):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            hy.prepare_raw(fr.cuda(non_blocking=True), it)
            t2 = time.perf_counter()
            gA.replay()
            p = outA[: len(it["options"])].float().cpu()
            if rep:
                ts.append((time.perf_counter() - t1) * 1e3); tp.append((t2 - t1) * 1e3)
    ts.sort()
    rec.update(e2e_p50=round(ts[len(ts) // 2], 2), e2e_p90=round(ts[int(len(ts) * 0.9)], 2),
               e2e_p99=round(ts[int(len(ts) * 0.99)], 2), prep_p50=round(statistics.median(tp), 2),
               gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1), seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True)
    log_jsonl("bench.jsonl", rec)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
