"""Jev fast path for Gemma 4 26B-A4B: one CUDA-graph replay per decision, experts through vLLM's fused MoE kernel.

Model: the bf16 checkpoint is loaded on the CPU (it does not fit a 48 GB card in bf16); every layer's 128 experts
are quantized on the GPU to FP8 per output channel (the same format as RedHatAI's FP8-dynamic checkpoint) and run
through vllm fused_experts (FP8 weights, dynamic per-token FP8 activations, gelu_tanh gating). Everything else stays
bf16 on the GPU.

Graph: raw patches -> vision tower (no padding, no mask, pooling as a precomputed matrix) -> projector -> image
features scattered into the prompt -> 30 decoder layers (sliding layers bidirectional inside the image block,
global layers causal) -> final norm at the answer position -> the offered letters' rows -> softcap -> softmax.

Usage (vLLM venv): python jev_fast26.py check|bench|acc [budget]
"""
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ROOT, encode_batch, letter_ids, score
from data import _aokvqa, eval_items

MID = "google/gemma-4-26B-A4B-it"
K = 5
WH = (640, 360)


class FusedFP8Experts(nn.Module):
    """Drop-in for Gemma4TextExperts: forward(hidden, top_k_index, top_k_weights) via vllm fused_experts."""

    def __init__(self, experts):
        super().__init__()
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config

        def q(w):  # [E, rows, cols] bf16 -> fp8 per output row, scale [E, rows, 1]
            w = w.cuda().float()
            s = (w.abs().amax(-1, keepdim=True) / 448.0).clamp(min=1e-12)
            return (w / s).clamp(-448, 448).to(torch.float8_e4m3fn), s

        self.w13, self.s13 = q(experts.gate_up_proj.data)
        self.w2, self.s2 = q(experts.down_proj.data)
        self.qc = fp8_w8a8_moe_quant_config(w1_scale=self.s13, w2_scale=self.s2, per_act_token_quant=True, per_out_ch_quant=True)
        self.act = MoEActivation.GELU_TANH

    def forward(self, hidden_states, top_k_index, top_k_weights):
        from vllm.model_executor.layers.fused_moe import fused_experts
        return fused_experts(hidden_states, self.w13, self.w2, top_k_weights.float(), top_k_index.to(torch.int32),
                             activation=self.act, quant_config=self.qc)


class FusedInt4Experts(nn.Module):
    """Same drop-in, 4-bit weights: symmetric round-to-nearest in groups along K, stored the way vLLM's Triton
    W4A16 MoE kernel reads them ([E, N, K/2] bytes, even k in the low nibble, implicit zero point 8)."""

    def __init__(self, experts, group=32):
        super().__init__()
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config

        def q(w):
            w = w.cuda().float()
            E, N, Kd = w.shape
            g = w.view(E, N, Kd // group, group)
            s = (g.abs().amax(-1, keepdim=True) / 7.0).clamp(min=1e-12)
            qv = (torch.round(g / s).clamp(-8, 7) + 8).to(torch.uint8).view(E, N, Kd)
            packed = (qv[..., 0::2] | (qv[..., 1::2] << 4)).contiguous()
            return packed, s.squeeze(-1).to(torch.bfloat16).contiguous()

        self.w13, self.s13 = q(experts.gate_up_proj.data)
        self.w2, self.s2 = q(experts.down_proj.data)
        self.qc = int4_w4a16_moe_quant_config(w1_scale=self.s13, w2_scale=self.s2, block_shape=[0, group])
        self.act = MoEActivation.GELU_TANH

    def forward(self, hidden_states, top_k_index, top_k_weights):
        from vllm.model_executor.layers.fused_moe import fused_experts
        return fused_experts(hidden_states, self.w13, self.w2, top_k_weights.float(), top_k_index.to(torch.int32),
                             activation=self.act, quant_config=self.qc)


def _gptq_pack(w, group):
    """[E, out, in] float -> GPTQ layout: qweight [E, in/8, out] int32 (uint4 with bias 8), scales [E, in/g, out]."""
    E, O, Kin = w.shape
    g = w.view(E, O, Kin // group, group)
    s = (g.abs().amax(-1, keepdim=True) / 7.0).clamp(min=1e-12)
    q = (torch.round(g / s).clamp(-8, 7) + 8).to(torch.int64).view(E, O, Kin)
    qt = q.transpose(1, 2).reshape(E, Kin // 8, 8, O)
    packed = torch.zeros(E, Kin // 8, O, dtype=torch.int64, device=w.device)
    for j in range(8):
        packed |= qt[:, :, j, :] << (4 * j)
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed).to(torch.int32)
    return packed.contiguous(), s.squeeze(-1).transpose(1, 2).to(torch.bfloat16).contiguous()


class FusedMarlinExperts(nn.Module):
    """Same drop-in, 4-bit weights through vLLM's Marlin MoE kernel. Quantization is identical to FusedInt4Experts
    (symmetric round-to-nearest, groups of 32); vLLM's own Marlin weight prep pads and repacks."""

    def __init__(self, experts, group=32):
        super().__init__()
        from types import SimpleNamespace
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _process_weights_marlin
        from vllm.scalar_type import scalar_types
        gu = experts.gate_up_proj.data.cuda().float()
        dn = experts.down_proj.data.cuda().float()
        inter = gu.shape[1] // 2
        w13q, w13s = _gptq_pack(gu, group)
        w2q, w2s = _gptq_pack(dn, group)
        del gu, dn
        out = _process_weights_marlin(SimpleNamespace(intermediate_size_per_partition=inter), None, 4, 8, group,
                                      w13q, w2q, w13s, w2s)
        self.w13, self.w2, self.s13, self.s2 = out[0], out[1], out[2], out[3]
        self.qtype = scalar_types.uint4b8.id
        self.act = MoEActivation.GELU_TANH

    def forward(self, hidden_states, top_k_index, top_k_weights):
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
        return fused_marlin_moe(hidden_states, self.w13, self.w2, None, None, self.s13, self.s2,
                                top_k_weights.float(), top_k_index.to(torch.int32), self.qtype, activation=self.act)


def load_fp8_model(attn="sdpa"):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(MID, padding_side="left")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(MID, dtype=torch.bfloat16, device_map="cpu",
                                                        attn_implementation=attn).eval()
    kind = os.environ.get("JEV_EXPERTS", "fp8")
    for layer in model.model.language_model.layers:
        layer.experts = {"int4": FusedInt4Experts, "marlin": FusedMarlinExperts}.get(kind, FusedFP8Experts)(layer.experts)
        torch.cuda.empty_cache()
    model.to("cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    if os.environ.get("JEV_VISION_FP8"):
        # FP8 (dynamic per-tensor activations, per-tensor weights) on the vision encoder's linear layers only.
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerTensor, quantize_
        quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
                  filter_fn=lambda m, fqn: isinstance(m, nn.Linear) and "vision_tower.encoder" in fqn)
    print(f"loaded + quantized experts ({kind}) in {time.time() - t0:.0f}s, GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    return proc, model


class JevFast26:
    def __init__(self, model, proc, budget, s_max):
        m = model.model
        self.proc, self.budget, self.S = proc, budget, s_max
        self.vt, self.embed_vision = m.vision_tower, m.embed_vision
        lm = m.language_model
        self.lm, self.layers = lm, lm.layers
        tc = model.config.get_text_config()
        self.types, self.cap = tc.layer_types, tc.final_logit_softcapping
        self.img_tok = model.config.image_token_id
        self.pad_id = tc.pad_token_id
        self.E = model.get_input_embeddings().weight[letter_ids(proc.tokenizer)[:K]].detach().clone()
        self.dtype = self.E.dtype
        self.neg = torch.finfo(self.dtype).min
        self.buf, self.graph, self.out = None, None, None

    def _alloc(self, P, N):
        d = "cuda"
        self.P, self.N = P, N
        self.buf = dict(
            pix=torch.zeros(1, P, self.vt.patch_embedder.input_proj.in_features, device=d),
            ppos=torch.zeros(1, P, 2, dtype=torch.long, device=d),
            pool=torch.zeros(1, N, P, device=d),
            ids=torch.zeros(1, self.S, dtype=torch.long, device=d),
            img_idx=torch.zeros(N, dtype=torch.long, device=d),
            pos=torch.zeros(1, self.S, dtype=torch.long, device=d),
            m_slide=torch.full((1, 1, self.S, self.S), self.neg, dtype=self.dtype, device=d),
            m_glob=torch.full((1, 1, self.S, self.S), self.neg, dtype=self.dtype, device=d),
            cand=torch.zeros(K, dtype=torch.bool, device=d),
        )

    def prepare(self, item):
        enc = encode_batch(self.proc, [item], self.budget)
        ids = enc["input_ids"][0]
        pv, pp = enc["pixel_values"][0], enc["image_position_ids"][0]
        real = (pp != -1).all(-1)
        self._fill(ids, pv[real], pp[real], len(item["options"]))

    def _fill(self, ids, pv, pp, n_opts):
        n = ids.numel()
        assert n <= self.S, (n, self.S)
        is_img = ids == self.img_tok
        P, N = pv.shape[0], int(is_img.sum())
        if self.buf is None:
            self._alloc(P, N)
        assert (P, N) == (self.P, self.N), ((P, N), (self.P, self.N))
        ppc = pp.cpu()
        max_x = int(ppc[:, 0].max()) + 1
        kid = torch.div(ppc, 3, rounding_mode="floor")
        kid = kid[:, 0] + (max_x // 3) * kid[:, 1]
        pool = F.one_hot(kid, N).float().t() / 9
        pad = self.S - n
        ids_full = torch.full((self.S,), self.pad_id, dtype=torch.long)
        ids_full[pad:] = torch.where(is_img, torch.full_like(ids, self.pad_id), ids)
        valid = torch.zeros(self.S, dtype=torch.bool); valid[pad:] = True
        img_full = torch.zeros(self.S, dtype=torch.bool); img_full[pad:] = is_img
        pos = torch.zeros(self.S, dtype=torch.long); pos[pad:] = torch.arange(n)
        causal = torch.ones(self.S, self.S, dtype=torch.bool).tril()
        eye = torch.eye(self.S, dtype=torch.bool)
        glob = (causal & valid[None]) | eye
        slide = ((causal | (img_full[:, None] & img_full[None])) & valid[None]) | eye
        z = torch.tensor(0.0, dtype=self.dtype)
        b = self.buf
        b["pix"].copy_(pv[None].float()); b["ppos"].copy_(pp[None]); b["pool"].copy_(pool[None])
        b["ids"].copy_(ids_full[None]); b["img_idx"].copy_(is_img.nonzero().squeeze(1) + pad); b["pos"].copy_(pos[None])
        b["m_slide"].copy_(torch.where(slide, z, self.neg)[None, None])
        b["m_glob"].copy_(torch.where(glob, z, self.neg)[None, None])
        c = torch.zeros(K, dtype=torch.bool); c[:n_opts] = True
        b["cand"].copy_(c)

    # raw-input path: GPU image preprocessing + pre-rendered template (same steps as the HF processor)
    def setup_raw(self):
        from data import prompt_text
        marker = "QQQMARKERQQQ"
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": marker}]}]
        tmpl = self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        ph = self.proc.image_token if self.proc.image_token in tmpl else self.proc.boi_token
        expanded = f"{self.proc.boi_token}{self.proc.image_token * self.N}{self.proc.eoi_token}"
        self.raw_pre, self.raw_post = tmpl.replace(ph, expanded, 1).split(marker)
        self.raw_text = prompt_text

    def prepare_raw(self, frame_u8, item):
        import torchvision.transforms.v2.functional as tvF
        from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size
        H, W = frame_u8.shape[-2:]
        th, tw = get_aspect_ratio_preserving_size(H, W, 16, self.budget * 9, 3)
        img = tvF.resize(frame_u8, [th, tw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True).float() / 255.0
        ph, pw = th // 16, tw // 16
        patches = img.reshape(3, ph, 16, pw, 16).permute(1, 3, 2, 4, 0).reshape(ph * pw, -1)
        gx, gy = torch.meshgrid(torch.arange(pw, device="cuda"), torch.arange(ph, device="cuda"), indexing="xy")
        pp = torch.stack([gx, gy], -1).reshape(ph * pw, 2)
        ids = torch.tensor(self.proc.tokenizer(self.raw_pre + self.raw_text(item) + self.raw_post, add_special_tokens=False)["input_ids"])
        self._fill(ids, patches, pp, len(item["options"]))

    @torch.no_grad()
    def vision(self):
        b, vt = self.buf, self.vt
        h = vt.patch_embedder(b["pix"], b["ppos"], torch.zeros(1, self.P, dtype=torch.bool, device="cuda"))
        rot = vt.encoder.rotary_emb(h, b["ppos"])
        for layer in vt.encoder.layers:
            h = layer(h, attention_mask=None, position_embeddings=rot, position_ids=b["ppos"])
        v = (b["pool"] @ h.float()) * (vt.config.hidden_size ** 0.5)
        if vt.config.standardize:
            v = (v - vt.std_bias.float()) * vt.std_scale.float()
        return self.embed_vision(v.to(self.dtype))[0]

    @torch.no_grad()
    def _forward(self):
        b = self.buf
        v = self.vision()
        x = self.lm.embed_tokens(b["ids"])[0].index_copy(0, b["img_idx"], v.to(self.dtype))[None]
        rope = {t: self.lm.rotary_emb(x, b["pos"], t) for t in set(self.types)}
        masks = {"sliding_attention": b["m_slide"], "full_attention": b["m_glob"]}
        for i, layer in enumerate(self.layers):
            t = self.types[i]
            x = layer(x, None, shared_kv_states={}, position_embeddings=rope[t], attention_mask=masks[t], position_ids=b["pos"])
        h = self.lm.norm(x[:, -1])
        z = (h @ self.E.t()).float()
        z = self.cap * torch.tanh(z / self.cap)
        return torch.softmax(z.masked_fill(~b["cand"], -1e9), -1)[0]

    def capture(self, compile_mode="default"):
        fn = torch.compile(self._forward, mode=compile_mode, dynamic=False) if compile_mode else self._forward
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = fn()

    def __call__(self, item):
        self.prepare(item)
        self.graph.replay()
        return self.out


def cuda_ms(fn, reps=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for a, b in ev:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in ev)


def main(mode, budget=280):
    proc, model = load_fp8_model()
    items = [dict(it, image=it["image"].resize(WH)) for it in _aokvqa("validation", 80, 21)]
    S = -(-max(encode_batch(proc, [it], budget)["input_ids"].shape[1] for it in items) // 16) * 16
    f = JevFast26(model, proc, budget, S)
    f.prepare(items[0])
    comp = os.environ.get("JEV_COMPILE", "default") or None
    if mode == "check":
        # The fast path (eager, then graph) against the HF forward of the same FP8-expert model.
        ref = score(model, proc, items[:24], None, bs=1, soft_tokens=budget)
        f.capture(compile_mode=comp)
        d, agree = [], 0
        for it in items[:24]:
            p = f(it).cpu()[: len(it["options"])]
            r = torch.tensor(ref[it["id"]])
            d.append(float((p - r).abs().max())); agree += int(p.argmax() == r.argmax())
        print(json.dumps(dict(kind="check", experts=os.environ.get("JEV_EXPERTS", "fp8"), budget=budget, S=S, max_dp=round(max(d), 4), median_dp=round(statistics.median(d), 4), agree=f"{agree}/24")), flush=True)
    f.capture(compile_mode=comp) if f.graph is None else None
    res = dict(kind="bench", vision_fp8=bool(os.environ.get("JEV_VISION_FP8")), experts=os.environ.get("JEV_EXPERTS", "fp8"), budget=budget, S=S, P=f.P, N=f.N, compile=comp, gpu=torch.cuda.get_device_name())
    res["graph_ms"] = round(cuda_ms(f.graph.replay, 50), 2)
    vf = torch.compile(f.vision, mode=comp, dynamic=False) if comp else f.vision
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            vf()
    torch.cuda.current_stream().wait_stream(s)
    gv = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gv):
        vf()
    res["vision_ms"] = round(cuda_ms(gv.replay, 50), 2)
    res["lm_ms_approx"] = round(res["graph_ms"] - res["vision_ms"], 2)
    t = []
    for it in items[:20]:
        t0 = time.perf_counter(); f.prepare(it); torch.cuda.synchronize(); t.append((time.perf_counter() - t0) * 1e3)
    res["prep_cpu_ms"] = round(statistics.median(t), 2)
    print(json.dumps(res), flush=True)
    with open(os.path.join(ROOT, "results", "jev", "fast26.jsonl"), "a") as fh:
        fh.write(json.dumps(res) + "\n")
    if mode == "e2e":
        import numpy as np
        f.setup_raw()
        frames = [torch.from_numpy(np.asarray(it["image"])).permute(2, 0, 1).contiguous().pin_memory() for it in items]
        same, agree = 0, 0
        for it, fr in zip(items[:24], frames[:24]):
            f.prepare(it); f.graph.replay(); a = f.out.clone(); ids_a = f.buf["ids"].clone()
            f.prepare_raw(fr.cuda(), it); f.graph.replay(); b = f.out.clone()
            same += int(torch.equal(ids_a, f.buf["ids"])); agree += int(a.argmax() == b.argmax())
        ts = []
        for rep in range(3):
            for it, fr in zip(items, frames):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                f.prepare_raw(fr.cuda(non_blocking=True), it); f.graph.replay()
                p = f.out[: len(it["options"])].float().cpu()
                if rep:
                    ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        rec = dict(kind="e2e", experts=os.environ.get("JEV_EXPERTS", "fp8"), budget=budget, S=S,
                   e2e_p50=round(ts[len(ts) // 2], 2), e2e_p90=round(ts[int(len(ts) * 0.9)], 2), e2e_p99=round(ts[int(len(ts) * 0.99)], 2),
                   raw_vs_processor=dict(same_ids=f"{same}/24", argmax_agree=f"{agree}/24"), gpu=torch.cuda.get_device_name())
        print(json.dumps(rec), flush=True)
        with open(os.path.join(ROOT, "results", "jev", "fast26.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    if mode == "acc":
        ev_items = [dict(it, image=it["image"].resize(WH)) for it in eval_items() if it["image"] is not None]
        graphs, probs, t0 = {}, {}, time.time()
        lens = {it["id"]: encode_batch(proc, [it], budget)["input_ids"].shape[1] for it in ev_items}
        for it in sorted(ev_items, key=lambda x: lens[x["id"]]):
            Sb = next(bb for bb in (320, 384, 448, 512, 640, 768, 1024) if bb >= lens[it["id"]])
            if Sb not in graphs:
                g = JevFast26(model, proc, budget, Sb); g.prepare(it); g.capture(compile_mode=comp); graphs[Sb] = g
            probs[it["id"]] = graphs[Sb](it)[: len(it["options"])].float().cpu().tolist()
        acc = {}
        for it in ev_items:
            p = probs[it["id"]]
            acc.setdefault(it["source"], []).append(max(range(len(p)), key=p.__getitem__) == it["answer"])
        rec = dict(kind="acc", vision_fp8=bool(os.environ.get("JEV_VISION_FP8")), experts=os.environ.get("JEV_EXPERTS", "fp8"), budget=budget, image_acc=round(100 * sum(sum(v) for v in acc.values()) / len(ev_items), 2),
                   per_source={k: round(100 * sum(v) / len(v), 2) for k, v in acc.items()}, seconds=round(time.time() - t0))
        print(json.dumps(rec), flush=True)
        with open(os.path.join(ROOT, "results", "jev", "fast26.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 280)
