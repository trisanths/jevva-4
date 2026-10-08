"""Fast path for a Jevified Gemma 4 E4B/E2B: one CUDA-graph replay per decision.

Jev-specific savings over the stock one-pass forward:
  - KV-shared tail: E4B's last 18 layers (E2B: 20) reuse keys/values stored by earlier layers, so for a
    decision only the answer position has to go through them. Exact.
  - candidate-only readout: the offered letters' rows of the tied embedding instead of the 262K-row head.
Plus serving mechanics: fixed shapes (images resized to one resolution, question left-padded into a fixed
bucket), the vision tower's pooling as a precomputed matrix, and the whole forward captured as a CUDA graph.

Usage: python jev_fast_e.py <model-id> [budget] [WxH] [adapter]
"""
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

from common import ROOT, encode_batch, letter_ids, load, score
from data import eval_items

K = 5


class JevFastE:
    def __init__(self, model, proc, budget, s_max):
        m = model.model
        self.proc, self.budget, self.S = proc, budget, s_max
        vt = m.vision_tower
        self.patch, self.venc, self.vcfg = vt.patch_embedder, vt.encoder, vt.config
        self.embed_vision = m.embed_vision
        lm = m.language_model
        self.lm, self.layers = lm, lm.layers
        tc = model.config.get_text_config()
        self.types = tc.layer_types
        self.first = tc.num_hidden_layers - tc.num_kv_shared_layers
        self.cap, self.pad_id = tc.final_logit_softcapping, tc.pad_token_id
        self.img_tok = model.config.image_token_id
        self.E = model.get_input_embeddings().weight[letter_ids(proc.tokenizer)[:K]].detach().clone()
        self.dtype = self.E.dtype
        self.neg = torch.finfo(self.dtype).min
        self.buf, self.graph, self.out, self.P = None, None, None, None
        self.kv_skip, self.full_head = True, False      # ablation switches (the Jev defaults are True, False)
        self.head_w = model.get_input_embeddings().weight
        self.cand_ids = torch.tensor(letter_ids(proc.tokenizer)[:K], device="cuda")

    def _alloc(self, P, N):
        dev = "cuda"
        self.P, self.N = P, N
        self.buf = dict(
            pix=torch.zeros(1, P, self.patch.input_proj.in_features, dtype=torch.float32, device=dev),
            ppos=torch.zeros(1, P, 2, dtype=torch.long, device=dev),
            pool=torch.zeros(1, N, P, dtype=torch.float32, device=dev),
            ids=torch.zeros(1, self.S, dtype=torch.long, device=dev),
            img_idx=torch.zeros(N, dtype=torch.long, device=dev),
            pos=torch.zeros(1, self.S, dtype=torch.long, device=dev),
            mask=torch.full((1, 1, self.S, self.S), self.neg, dtype=self.dtype, device=dev),
            mask_last=torch.full((1, 1, 1, self.S), self.neg, dtype=self.dtype, device=dev),
            cand=torch.zeros(K, dtype=torch.bool, device=dev),
        )

    def prepare(self, item):
        enc = encode_batch(self.proc, [item], self.budget)
        ids = enc["input_ids"][0]
        n = ids.numel()
        assert n <= self.S, (n, self.S)
        pv, pp = enc["pixel_values"][0], enc["image_position_ids"][0]
        real = (pp != -1).all(-1)
        pv, pp = pv[real], pp[real]
        P = pv.shape[0]
        is_img = ids == self.img_tok
        N = int(is_img.sum())
        if self.buf is None:
            self._alloc(P, N)
        assert (P, N) == (self.P, self.N), f"shape bucket changed: {(P, N)} vs {(self.P, self.N)}"
        k = int(round((P / N) ** 0.5))
        max_x = int(pp[:, 0].max()) + 1
        kid = torch.div(pp, k, rounding_mode="floor")
        kid = kid[:, 0] + (max_x // k) * kid[:, 1]
        pool = F.one_hot(kid, N).float().t() / (k * k)                      # [N, P], same arithmetic as HF's pooler
        pad = self.S - n
        ids_full = torch.full((self.S,), self.pad_id, dtype=torch.long)
        ids_full[pad:] = torch.where(is_img, torch.full_like(ids, self.pad_id), ids)
        valid = torch.zeros(self.S, dtype=torch.bool); valid[pad:] = True
        pos = torch.zeros(self.S, dtype=torch.long); pos[pad:] = torch.arange(n)
        img_idx = (is_img.nonzero().squeeze(1) + pad)
        causal = torch.ones(self.S, self.S, dtype=torch.bool).tril()
        allowed = (causal & valid[None, :]) | torch.eye(self.S, dtype=torch.bool)
        z = torch.tensor(0.0, dtype=self.dtype)
        b = self.buf
        b["pix"].copy_(pv[None]); b["ppos"].copy_(pp[None]); b["pool"].copy_(pool[None])
        b["ids"].copy_(ids_full[None]); b["img_idx"].copy_(img_idx); b["pos"].copy_(pos[None])
        b["mask"].copy_(torch.where(allowed, z, self.neg)[None, None])
        b["mask_last"].copy_(torch.where(valid, z, self.neg)[None, None, None])
        c = torch.zeros(K, dtype=torch.bool); c[: len(item["options"])] = True
        b["cand"].copy_(c)

    # ---------- raw-input path: GPU image preprocessing + pre-rendered chat template ----------
    def setup_raw(self, sample_item):
        """Render the chat template once with a marker where the question goes, and find the image placeholder."""
        from data import prompt_text
        marker = "QQQMARKERQQQ"
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": marker}]}]
        tmpl = self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        tok = self.proc.tokenizer
        img_ph = self.proc.image_token if self.proc.image_token in tmpl else self.proc.boi_token
        expanded = f"{self.proc.boi_token}{self.proc.image_token * self.N}{self.proc.eoi_token}"
        self.raw_pre, self.raw_post = tmpl.replace(img_ph, expanded, 1).split(marker)
        self.raw_prompt_text = prompt_text
        self.raw_tok = tok

    def prepare_raw(self, frame_u8, item):
        """frame_u8: uint8 CUDA tensor [3, H, W]. Same steps as Gemma4ImageProcessor, on the GPU."""
        import torchvision.transforms.v2.functional as tvF
        from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size
        H, W = frame_u8.shape[-2:]
        th, tw = get_aspect_ratio_preserving_size(H, W, 16, self.budget * 9, 3)
        img = tvF.resize(frame_u8, [th, tw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
        img = img.float() / 255.0
        ph, pw = th // 16, tw // 16
        patches = img.reshape(3, ph, 16, pw, 16).permute(1, 3, 2, 4, 0).reshape(ph * pw, -1)
        gx, gy = torch.meshgrid(torch.arange(pw, device="cuda"), torch.arange(ph, device="cuda"), indexing="xy")
        pp = torch.stack([gx, gy], -1).reshape(ph * pw, 2)
        ids = torch.tensor(self.raw_tok(self.raw_pre + self.raw_prompt_text(item) + self.raw_post,
                                        add_special_tokens=False)["input_ids"])
        self._fill(ids, patches, pp, len(item["options"]))

    def _fill(self, ids, pv, pp, n_opts):
        """Shared tail of prepare(): masks, positions and buffers from token ids and real patches."""
        n = ids.numel()
        assert n <= self.S, (n, self.S)
        is_img = ids == self.img_tok
        P, N = pv.shape[0], int(is_img.sum())
        assert (P, N) == (self.P, self.N), (P, N)
        k = int(round((P / N) ** 0.5))
        ppc = pp.cpu()
        max_x = int(ppc[:, 0].max()) + 1
        kid = torch.div(ppc, k, rounding_mode="floor")
        kid = kid[:, 0] + (max_x // k) * kid[:, 1]
        pool = F.one_hot(kid, N).float().t() / (k * k)
        pad = self.S - n
        ids_full = torch.full((self.S,), self.pad_id, dtype=torch.long)
        ids_full[pad:] = torch.where(is_img, torch.full_like(ids, self.pad_id), ids)
        valid = torch.zeros(self.S, dtype=torch.bool); valid[pad:] = True
        pos = torch.zeros(self.S, dtype=torch.long); pos[pad:] = torch.arange(n)
        allowed = (torch.ones(self.S, self.S, dtype=torch.bool).tril() & valid[None, :]) | torch.eye(self.S, dtype=torch.bool)
        z = torch.tensor(0.0, dtype=self.dtype)
        b = self.buf
        b["pix"].copy_(pv[None]); b["ppos"].copy_(pp[None]); b["pool"].copy_(pool[None])
        b["ids"].copy_(ids_full[None]); b["img_idx"].copy_(is_img.nonzero().squeeze(1) + pad); b["pos"].copy_(pos[None])
        b["mask"].copy_(torch.where(allowed, z, self.neg)[None, None])
        b["mask_last"].copy_(torch.where(valid, z, self.neg)[None, None, None])
        c = torch.zeros(K, dtype=torch.bool); c[:n_opts] = True
        b["cand"].copy_(c)

    @torch.no_grad()
    def _forward(self):
        b = self.buf
        # Vision tower over a full patch grid: no padding, so no mask and no data-dependent indexing.
        h = self.patch(b["pix"], b["ppos"], torch.zeros(1, self.P, dtype=torch.bool, device="cuda"))
        rot = self.venc.rotary_emb(h, b["ppos"])
        for layer in self.venc.layers:
            h = layer(h, attention_mask=None, position_embeddings=rot, position_ids=b["ppos"])
        v = (b["pool"] @ h.float()) * (self.vcfg.hidden_size ** 0.5)
        v = self.embed_vision(v.to(self.dtype))[0]                                     # [N, D]
        # Language model, layers before the KV-shared block, over the whole sequence.
        tok = self.lm.embed_tokens(b["ids"])                                             # image slots hold PAD
        ple_ids = self.lm.get_per_layer_inputs(b["ids"], tok)
        x = tok[0].index_copy(0, b["img_idx"], v.to(tok.dtype))[None]
        ple = self.lm.project_per_layer_inputs(x, ple_ids)                               # [1, S, L, d]
        rope = {t: self.lm.rotary_emb(x, b["pos"], t) for t in set(self.types)}
        shared = {}
        for i in range(self.first):
            t = self.types[i]
            x = self.layers[i](x, ple[:, :, i], shared_kv_states=shared, position_embeddings=rope[t],
                               attention_mask=b["mask"], position_ids=b["pos"])
        if self.kv_skip:
            # KV-shared tail: only the answer position.
            xt, pos_t = x[:, -1:], b["pos"][:, -1:]
            for i in range(self.first, len(self.layers)):
                t = self.types[i]
                cos, sin = rope[t]
                xt = self.layers[i](xt, ple[:, -1:, i], shared_kv_states=shared, position_embeddings=(cos[:, -1:], sin[:, -1:]),
                                    attention_mask=b["mask_last"], position_ids=pos_t)
        else:
            # Ablation: run the KV-shared layers over every position, as the stock forward does.
            xt = x
            for i in range(self.first, len(self.layers)):
                t = self.types[i]
                xt = self.layers[i](xt, ple[:, :, i], shared_kv_states=shared, position_embeddings=rope[t],
                                    attention_mask=b["mask"], position_ids=b["pos"])
            xt = xt[:, -1:]
        hN = self.lm.norm(xt[:, 0])
        if self.full_head:
            zl = (hN @ self.head_w.t())[:, self.cand_ids].float()        # ablation: 262K-row head, then pick letters
        else:
            zl = (hN @ self.E.t()).float()
        zl = self.cap * torch.tanh(zl / self.cap)
        return torch.softmax(zl.masked_fill(~b["cand"], -1e9), -1)[0]

    def capture(self, compile_mode=None):
        """Capture the forward as one CUDA graph. With compile_mode, Inductor first fuses the norms, RoPE and other
        elementwise work into fewer kernels (the graph is then captured around the compiled code)."""
        fn = self._forward
        if compile_mode:
            fn = torch.compile(self._forward, mode=compile_mode, fullgraph=False, dynamic=False)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = fn()

    def __call__(self, item):
        self.prepare(item)
        if self.graph is None:
            return self._forward()
        self.graph.replay()
        return self.out


def main(mid, budget=280, wh=(640, 480), adapter=None):
    items = [it for it in eval_items(scale=0.03) if it["image"] is not None]
    items = [dict(it, image=it["image"].resize(wh)) for it in items]         # one resolution = one graph
    proc, model = load(mid=mid, attn="eager")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
    ref = score(model, proc, items, None, bs=1, soft_tokens=budget)
    model.config._attn_implementation = "sdpa"
    for mod in model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = "sdpa"
    # Stock single forward (HF, SDPA, full-vocab head at one position) for the same input, for comparison.
    enc = encode_batch(proc, items[:1], budget).to("cuda")
    with torch.no_grad():
        for _ in range(3):
            model(**enc, logits_to_keep=1)
        torch.cuda.synchronize(); t = []
        for _ in range(10):
            t0 = time.perf_counter(); model(**enc, logits_to_keep=1); torch.cuda.synchronize(); t.append((time.perf_counter() - t0) * 1e3)
    stock_ms = statistics.median(t)
    n_max = max(encode_batch(proc, [it], budget)["input_ids"].shape[1] for it in items)
    fast = JevFastE(model, proc, budget, s_max=-(-n_max // 16) * 16)
    fast.prepare(items[0])
    fast.capture()
    diffs, agree = [], 0
    for it in items:
        p = fast(it).cpu()[: len(it["options"])]
        r = torch.tensor(ref[it["id"]])
        diffs.append(float((p - r).abs().max())); agree += int(p.argmax() == r.argmax())
    fast.prepare(items[0])
    for _ in range(10):
        fast.graph.replay()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(100)]
    for a, b in ev:
        a.record(); fast.graph.replay(); b.record()
    torch.cuda.synchronize()
    ms = sorted(a.elapsed_time(b) for a, b in ev)
    t = []
    for it in items[:20]:
        t0 = time.perf_counter(); fast.prepare(it); torch.cuda.synchronize(); t.append((time.perf_counter() - t0) * 1e3)
    res = dict(model=mid, budget=budget, wh=wh, S=fast.S, P=fast.P, N=fast.N, stock_ms=stock_ms,
               fast_p50=ms[50], fast_p99=ms[98], prep_ms=statistics.median(t),
               max_dp=max(diffs), median_dp=statistics.median(diffs), agree=f"{agree}/{len(items)}",
               gpu=torch.cuda.get_device_name(), adapter=adapter)
    print(json.dumps(res), flush=True)
    os.makedirs(os.path.join(ROOT, "results", "jev"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "jev", "fast_timing.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    wh = tuple(int(x) for x in a[2].split("x")) if len(a) > 2 else (640, 480)
    main(a[0], int(a[1]) if len(a) > 1 else 280, wh, a[3] if len(a) > 3 else None)
