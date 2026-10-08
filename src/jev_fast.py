"""Fast path for a Jevified Gemma 4 12B: one CUDA-graph replay per decision.

The graph runs: raw-patch embedder -> layers 1..M over text + image tokens -> gather the text positions ->
layers M+1..L over text only -> final norm at the answer position -> the offered letters' rows -> softcap
-> softmax. Shapes are static: S_MAX sequence slots (left padded), exactly `budget` image slots (padding
patches land in a dummy row), T = S_MAX - budget text slots above layer M. Per-request work outside the
graph (tokenize, patchify, masks) is timed separately.

Usage: python jev_fast.py L M [budget] [adapter_dir]  -> checks agreement with the HF path, then times it.
"""
import json
import os
import statistics
import sys
import time

import torch

from common import ROOT, encode_batch, letter_ids, score
from data import eval_items
from jevnet import State, install_drop, load_truncated

K = 5


class JevFast:
    def __init__(self, model, proc, drop_after, budget=280, s_max=None):
        m = model.model
        lm = m.language_model
        cfg = model.config.get_text_config()
        self.proc, self.budget = proc, budget
        self.embed_vision, self.embed_tokens = m.embed_vision, lm.embed_tokens
        self.layers, self.types = lm.layers, cfg.layer_types[: len(lm.layers)]
        self.rotary, self.norm = lm.rotary_emb, lm.norm
        self.M = min(drop_after, len(self.layers))
        self.cap = cfg.final_logit_softcapping
        self.img_tok = model.config.image_token_id
        self.E = lm.embed_tokens.weight[letter_ids(proc.tokenizer)[:K]].detach().clone()     # [K, D]
        self.S = s_max or budget + 200
        self.T = self.S - budget
        dev, D = "cuda", cfg.hidden_size
        self.dtype = lm.embed_tokens.weight.dtype
        neg = torch.finfo(self.dtype).min
        self.neg = neg
        self.buf = dict(
            ids=torch.zeros(1, self.S, dtype=torch.long, device=dev),
            pix=torch.zeros(1, budget, 6912, dtype=self.dtype, device=dev),
            ppos=torch.full((1, budget, 2), -1, dtype=torch.long, device=dev),
            img_idx=torch.full((budget,), self.S, dtype=torch.long, device=dev),
            pos=torch.zeros(1, self.S, dtype=torch.long, device=dev),
            m_slide=torch.full((1, 1, self.S, self.S), neg, dtype=self.dtype, device=dev),
            m_glob=torch.full((1, 1, self.S, self.S), neg, dtype=self.dtype, device=dev),
            txt_idx=torch.arange(self.T, device=dev),
            m_hi=torch.full((1, 1, self.T, self.T), neg, dtype=self.dtype, device=dev),
            cand=torch.zeros(K, dtype=torch.bool, device=dev),
        )
        self.graph, self.out = None, None

    # ---------- per request (outside the graph) ----------
    def prepare(self, item):
        enc = encode_batch(self.proc, [item], self.budget)
        ids = enc["input_ids"][0]
        n = ids.numel()
        assert n <= self.S, f"prompt {n} > S_MAX {self.S}"
        pad = self.S - n
        is_img = ids == self.img_tok
        n_img = int(is_img.sum())
        ids_full = torch.zeros(self.S, dtype=torch.long)
        ids_full[pad:] = torch.where(is_img, torch.zeros_like(ids), ids)
        valid = torch.zeros(self.S, dtype=torch.bool); valid[pad:] = True
        img_full = torch.zeros(self.S, dtype=torch.bool); img_full[pad:] = is_img
        pos = torch.zeros(self.S, dtype=torch.long); pos[pad:] = torch.arange(n)
        img_idx = torch.full((self.budget,), self.S, dtype=torch.long)
        img_idx[:n_img] = img_full.nonzero().squeeze(1)
        causal = torch.ones(self.S, self.S, dtype=torch.bool).tril()
        keys = valid[None, :]
        glob = causal & keys
        block = img_full[:, None] & img_full[None, :]
        slide = (causal | block) & keys
        eye = torch.eye(self.S, dtype=torch.bool)
        glob, slide = glob | eye, slide | eye                                   # pad rows attend to themselves
        # Exactly `budget` slots leave at layer M: the real image tokens plus, if the image produced fewer
        # tokens than the budget, that many padding positions (masked everywhere anyway).
        spare = self.budget - n_img
        assert spare <= pad, f"text part too long for S_MAX ({n - n_img} tokens)"
        leave = img_full.clone()
        leave[:spare] = True
        txt = (~leave).nonzero().squeeze(1)
        assert txt.numel() == self.T, (txt.numel(), self.T)
        hi = (torch.ones(self.T, self.T, dtype=torch.bool).tril() & valid[txt][None, :]) | torch.eye(self.T, dtype=torch.bool)
        b = self.buf
        pv = enc.get("pixel_values")
        b["ids"].copy_(ids_full[None])
        b["pos"].copy_(pos[None])
        if pv is not None:
            b["pix"].copy_(pv[:, : self.budget].to(self.dtype))
            b["ppos"].copy_(enc["image_position_ids"][:, : self.budget])
        b["img_idx"].copy_(img_idx)
        z = torch.tensor(0.0, dtype=self.dtype)
        b["m_slide"].copy_(torch.where(slide, z, self.neg)[None, None])
        b["m_glob"].copy_(torch.where(glob, z, self.neg)[None, None])
        b["txt_idx"].copy_(txt)
        b["m_hi"].copy_(torch.where(hi, z, self.neg)[None, None])
        c = torch.zeros(K, dtype=torch.bool); c[: len(item["options"])] = True
        b["cand"].copy_(c)

    # ---------- the captured computation ----------
    @torch.no_grad()
    def _forward(self):
        b = self.buf
        x = self.embed_tokens(b["ids"])[0]                                          # [S, D]
        v = self.embed_vision(b["pix"], b["ppos"], return_dict=True).pooler_output[0].to(x.dtype)       # [budget, D]
        x = torch.cat([x, x.new_zeros(1, x.shape[1])]).index_copy(0, b["img_idx"], v)[: self.S][None]
        pe = {t: self.rotary(x, b["pos"], t) for t in set(self.types)}
        masks = {"sliding_attention": b["m_slide"], "full_attention": b["m_glob"]}
        for i in range(self.M):
            t = self.types[i]
            x = self.layers[i](x, shared_kv_states={}, position_embeddings=pe[t], attention_mask=masks[t], position_ids=b["pos"])
        if self.M < len(self.layers):
            x = x[:, b["txt_idx"]]
            pos = b["pos"][:, b["txt_idx"]]
            pe = {t: self.rotary(x, pos, t) for t in set(self.types)}
            for i in range(self.M, len(self.layers)):
                t = self.types[i]
                x = self.layers[i](x, shared_kv_states={}, position_embeddings=pe[t], attention_mask=b["m_hi"], position_ids=pos)
        h = self.norm(x[:, -1])
        z = (h @ self.E.t()).float()
        z = self.cap * torch.tanh(z / self.cap)
        return torch.softmax(z.masked_fill(~b["cand"], -1e9), -1)[0]

    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._forward()

    def __call__(self, item, eager=False):
        self.prepare(item)
        if eager or self.graph is None:
            return self._forward()
        self.graph.replay()
        return self.out


def main(L, M, budget=280, adapter=None):
    items = [it for it in eval_items(scale=0.03)]
    img_items = [it for it in items if it["image"] is not None]
    # Reference: the HF path with the same truncation and image drop (eager attention).
    proc, model = load_truncated(L, attn="eager")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
        nt = os.path.join(adapter, "norm.pt")
        if os.path.exists(nt):
            model.model.language_model.norm.weight.data = torch.load(nt)["norm"].to(model.dtype).cuda()
    state = State()
    hs = install_drop(model, M, state) if M < L else []
    ref = score(model, proc, img_items, state, bs=1, soft_tokens=budget)
    for h in hs:
        h.remove()
    model.config._attn_implementation = "sdpa"
    for layer in model.model.language_model.layers:
        layer.self_attn.config._attn_implementation = "sdpa"
    n_max = max(encode_batch(proc, [it], budget)["input_ids"].shape[1] for it in img_items)
    # S_MAX must also leave room for `budget - n_img` spare slots taken from the padding.
    fast = JevFast(model, proc, M, budget, s_max=-(-(n_max + 32) // 16) * 16)
    fast.capture()
    diffs, agree = [], 0
    for it in img_items:
        p = fast(it).cpu()[: len(it["options"])]
        r = torch.tensor(ref[it["id"]])
        diffs.append(float((p - r).abs().max()))
        agree += int(p.argmax() == r.argmax())
    print(f"fast vs HF: max |dp| {max(diffs):.4f} median {statistics.median(diffs):.4f} argmax agree {agree}/{len(img_items)}", flush=True)
    # Timing: GPU replay only, and prepare (CPU) separately.
    it = img_items[0]
    fast.prepare(it)
    for _ in range(10):
        fast.graph.replay()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(50)]
    for a, b in ev:
        a.record(); fast.graph.replay(); b.record()
    torch.cuda.synchronize()
    gpu_ms = statistics.median(a.elapsed_time(b) for a, b in ev)
    t = []
    for it in img_items[:20]:
        t0 = time.perf_counter(); fast.prepare(it); torch.cuda.synchronize(); t.append((time.perf_counter() - t0) * 1e3)
    prep_ms = statistics.median(t)
    res = dict(L=L, M=M, budget=budget, gpu_ms=gpu_ms, prep_ms=prep_ms, max_dp=max(diffs), agree=agree, n=len(img_items),
               gpu=torch.cuda.get_device_name(), adapter=adapter)
    print(json.dumps(res), flush=True)
    os.makedirs(os.path.join(ROOT, "results", "jev"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "jev", "fast_timing.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]), int(a[2]) if len(a) > 2 else 280, a[3] if len(a) > 3 else None)
