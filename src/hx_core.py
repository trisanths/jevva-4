"""Shared pieces for the KV-handoff kill test.

E2B (Jev-tuned, frozen) prefills the prompt; a mapper turns its residual stream at a few layers into the raw
k_proj / v_proj outputs of every 26B layer; the 26B's own k_norm, RoPE (true positions) and v_norm are applied;
the 26B then runs only its chat-template suffix, attending to [native header KV; mapped KV] and causally to itself,
and the offered letters are read at the last position exactly as in JevFast26.

Prompt layout (both processors give identical ids up to the end of the user text; checked per item in hx_prep):
  header  <bos><|turn>user\n            (+ <|image> for image items; type 0, so outside the bidirectional block)
  middle  image tokens, <image|>, question text           (KV comes from the mapper)
  suffix  26B: <turn|>\n<|turn>model\n<|channel>thought\n<channel|>   (9 tokens; E2B's template stops after
          model\n, 5 tokens)
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: F401  (keeps the mask patch applied)
from common import ROOT, letter_ids

WH = (640, 480)
BUDGET = int(os.environ.get("HX_BUDGET", 70))
HX = os.path.join(ROOT, "hx", f"b{BUDGET}")
OUT = os.path.join(ROOT, "results", "handoff")
HDR = [2, 105, 2364, 107]
BOI = 255999
SUF26 = [106, 107, 105, 4368, 107, 100, 45518, 107, 101]
SUFE = [106, 107, 105, 4368, 107]
E2B_ID = "google/gemma-4-E2B-it"
M26_ID = "google/gemma-4-26B-A4B-it"
K = 5
LAYERS = [int(x) for x in os.environ.get("HX_LAYERS", "4,9,14").split(",")]
D_E = 1536


def log_jsonl(name, rec):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "a") as fh:
        fh.write(json.dumps(rec) + "\n")


# ----------------------------------------------------------------------------------------------- models
def load_e2b_jev(adapter=None):
    from peft import PeftModel
    from common import load
    adapter = adapter or os.path.join(ROOT, "adapters", "e2b-jev")
    proc, model = load(mid=E2B_ID, attn="sdpa")
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
    model.model.language_model.norm.weight.data = torch.load(os.path.join(adapter, "norm.pt"))["norm"].to(model.dtype).cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    return proc, model


def load_26b(kind="int4"):
    os.environ["JEV_EXPERTS"] = kind
    from jev_fast26 import load_fp8_model
    proc, model = load_fp8_model(attn="sdpa")
    if kind == "int4":
        for layer in model.model.language_model.layers:
            layer.experts = GradInt4Experts(layer.experts)
    return proc, model


# ------------------------------------------------------------- int4 experts with an input-gradient path
def deq_int4(packed, s, group=32):
    """FusedInt4Experts layout ([u, N, K/2] bytes, even k in the low nibble, zero point 8; scales [u, N, K/group])
    -> bf16 [u, N, K]."""
    q = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).to(torch.bfloat16) - 8
    u, n, kd = q.shape
    return (q.view(u, n, kd // group, group) * s[..., None]).view(u, n, kd)


def experts_torch(mod, x, idx, w):
    """Reference MoE forward in plain torch on dequantized weights of the touched experts only (grouped bmm)."""
    n, d = x.shape
    k = idx.shape[1]
    flat_e = idx.reshape(-1).long()
    flat_t = torch.arange(n, device=x.device).repeat_interleave(k)
    flat_w = w.reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    e_s, t_s, w_s = flat_e[order], flat_t[order], flat_w[order]
    U, counts = torch.unique_consecutive(e_s, return_counts=True)
    u, cmax = U.numel(), int(counts.max())
    starts = torch.cumsum(counts, 0) - counts
    gi = torch.repeat_interleave(torch.arange(u, device=x.device), counts)
    rank = torch.arange(e_s.numel(), device=x.device) - starts[gi]
    X = x.new_zeros(u, cmax, d).index_put((gi, rank), x[t_s])
    H = torch.bmm(X, deq_int4(mod.w13[U], mod.s13[U]).transpose(1, 2))
    g, up = H.chunk(2, -1)
    A = F.gelu(g, approximate="tanh") * up
    Y = torch.bmm(A, deq_int4(mod.w2[U], mod.s2[U]).transpose(1, 2))
    y = Y[gi, rank].float() * w_s[:, None].float()
    return x.new_zeros(n, d, dtype=torch.float32).index_add(0, t_s, y).to(x.dtype)


class _ExpertsFn(torch.autograd.Function):
    """Forward: vLLM's fused int4 kernel. Backward: recompute in torch on dequantized touched experts and return the
    gradient with respect to the expert input and the routing weights (no expert weight trains)."""

    @staticmethod
    def forward(ctx, x, w, idx, mod):
        ctx.mod = mod
        ctx.save_for_backward(x, w, idx)
        return mod(x, idx, w)

    @staticmethod
    def backward(ctx, gy):
        x, w, idx = ctx.saved_tensors
        with torch.enable_grad():
            xr = x.detach().requires_grad_(True)
            wr = w.detach().requires_grad_(True)
            y = experts_torch(ctx.mod, xr, idx, wr)
            gx, gw = torch.autograd.grad(y, (xr, wr), gy)
        return gx, gw, None, None


class GradInt4Experts(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if torch.is_grad_enabled() and (hidden_states.requires_grad or top_k_weights.requires_grad):
            return _ExpertsFn.apply(hidden_states, top_k_weights, top_k_index, self.inner)
        return self.inner(hidden_states, top_k_index, top_k_weights)


# ------------------------------------------------------------------------------------------------ LoRA
class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=32):
        super().__init__()
        self.base, self.scale, self.on = base, alpha / r, True
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def in_features(self):
        return self.base.in_features

    def forward(self, x):
        y = self.base(x)
        if self.on:
            y = y + ((x.to(self.A.dtype) @ self.A.t()) @ self.B.t() * self.scale).to(y.dtype)
        return y


def add_lora(model, r=16, alpha=32):
    mods = []
    for layer in model.model.language_model.layers:
        a = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            base = getattr(a, name)
            if base is None:
                continue
            m = LoRALinear(base, r, alpha)
            setattr(a, name, m)
            mods.append(m)
    return mods


def set_lora(mods, on):
    for m in mods:
        m.on = on


# ------------------------------------------------------------------------------------ 26B suffix runner
class Runner26:
    """26B decoder pass over a few new tokens that attend to a per-layer context KV (header + mapped middle)."""

    def __init__(self, model, proc):
        lm = model.model.language_model
        self.lm, self.layers = lm, lm.layers
        cfg = lm.config
        self.types, self.cap = cfg.layer_types, cfg.final_logit_softcapping
        self.E = model.get_input_embeddings().weight[letter_ids(proc.tokenizer)[:K]].detach().clone()
        self.dims = []
        for layer in self.layers:
            a = layer.self_attn
            d = a.head_dim
            self.dims.append((a.k_proj.out_features // d, d, a.v_proj is not None))
        self.sizes = [kvh * d * (2 if hv else 1) for kvh, d, hv in self.dims]
        self.total = sum(self.sizes)
        self.hdr = {}

    def rope(self, pos, t):
        return self.lm.rotary_emb(self.E, pos, t)

    def kv_post(self, l, k_raw, v_raw, cos, sin):
        """Raw k_proj / v_proj outputs [B, T, kvh*d] -> (K, V) [B, kvh, T, d] exactly as the layer stores them."""
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
        a = self.layers[l].self_attn
        kvh, d, hv = self.dims[l]
        B, T, _ = k_raw.shape
        k = k_raw.reshape(B, T, kvh, d)
        v = v_raw.reshape(B, T, kvh, d) if hv else k
        k = apply_rotary_pos_emb(a.k_norm(k), cos, sin, unsqueeze_dim=2).transpose(1, 2)
        return k, a.v_norm(v).transpose(1, 2)

    def split(self, Y):
        """Mapper output [..., total] -> per layer (k_raw, v_raw or None)."""
        out = []
        for (kvh, d, hv), part in zip(self.dims, torch.split(Y, self.sizes, -1)):
            out.append((part[..., : kvh * d], part[..., kvh * d:]) if hv else (part, None))
        return out

    def layer_fwd(self, l, x, cos, sin, ctx, mask):
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
        layer = self.layers[l]
        a = layer.self_attn
        kvh, d, hv = self.dims[l]
        B, S, D = x.shape
        h = layer.input_layernorm(x)
        q = apply_rotary_pos_emb(a.q_norm(a.q_proj(h).view(B, S, -1, d)), cos, sin, unsqueeze_dim=2).transpose(1, 2)
        kr = a.k_proj(h)
        vr = a.v_proj(h) if hv else None
        k, v = self.kv_post(l, kr, vr, cos, sin)
        if ctx is not None:
            k = torch.cat([ctx[0], k], 2)
            v = torch.cat([ctx[1], v], 2)
        rep = q.shape[1] // kvh
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        o = a.o_proj(o.transpose(1, 2).reshape(B, S, -1))
        x = x + layer.post_attention_layernorm(o)
        h1 = layer.post_feedforward_layernorm_1(layer.mlp(layer.pre_feedforward_layernorm(x)))
        flat = x.reshape(-1, D)
        _, w, idx = layer.router(flat)
        h2 = layer.experts(layer.pre_feedforward_layernorm_2(flat), idx, w).reshape(x.shape)
        h2 = layer.post_feedforward_layernorm_2(h2)
        x = x + layer.post_feedforward_layernorm(h1 + h2)
        return x * layer.layer_scalar, (kr, vr)

    def run(self, ids, pos, ctx=None, ctx_valid=None, collect=False):
        """ids, pos [B, S]; ctx: per layer (K, V) [B, kvh, Tc, d] or None; ctx_valid [B, Tc] bool.
        Returns softcapped letter logits [B, K] (all K letters) and, if collect, per-layer raw (k, v)."""
        B, S = ids.shape
        x = self.lm.embed_tokens(ids)
        causal = torch.ones(S, S, dtype=torch.bool, device=ids.device).tril()[None, None].expand(B, 1, S, S)
        mask = causal if ctx is None else torch.cat([ctx_valid[:, None, None, :].expand(B, 1, S, -1), causal], -1)
        ropes = {t: self.rope(pos, t) for t in set(self.types)}
        kvs = []
        for l in range(len(self.layers)):
            cos, sin = ropes[self.types[l]]
            x, kv = self.layer_fwd(l, x, cos, sin, None if ctx is None else ctx[l], mask)
            if collect:
                kvs.append(kv)
        h = self.lm.norm(x[:, -1])
        z = (h @ self.E.t()).float()
        return self.cap * torch.tanh(z / self.cap), kvs

    @torch.no_grad()
    def header(self, image):
        """Native 26B KV of the fixed header tokens (computed once per header kind)."""
        if image not in self.hdr:
            ids = torch.tensor([HDR + ([BOI] if image else [])], device="cuda")
            pos = torch.arange(ids.shape[1], device="cuda")[None]
            _, kvs = self.run(ids, pos, collect=True)
            out = []
            for l, (kr, vr) in enumerate(kvs):
                cos, sin = self.rope(pos, self.types[l])
                out.append(self.kv_post(l, kr, vr, cos, sin))
            self.hdr[image] = out
        return self.hdr[image]

    def context(self, Y, m, image):
        """Y: mapper output [B, M, total] for the middle positions (rows >= m[b] are padding); m [B] lengths.
        Returns per-layer (K, V) with the header in front, and the validity mask [B, h + M]."""
        hdr = self.header(image)
        h = hdr[0][0].shape[2]
        B, M, _ = Y.shape
        pos = (h + torch.arange(M, device=Y.device))[None].expand(B, M)
        ropes = {t: self.rope(pos, t) for t in set(self.types)}
        ctx = []
        for l, (kr, vr) in enumerate(self.split(Y)):
            cos, sin = ropes[self.types[l]]
            k, v = self.kv_post(l, kr, vr, cos, sin)
            hk, hv_ = hdr[l]
            ctx.append((torch.cat([hk.expand(B, -1, -1, -1), k], 2), torch.cat([hv_.expand(B, -1, -1, -1), v], 2)))
        valid = torch.arange(M, device=Y.device)[None] < m[:, None]
        valid = torch.cat([torch.ones(B, h, dtype=torch.bool, device=Y.device), valid], 1)
        return ctx, valid, h

    def suffix(self, ctx, valid, h, m):
        B = m.shape[0]
        ids = torch.tensor(SUF26, device=m.device)[None].expand(B, -1)
        pos = h + m[:, None] + torch.arange(len(SUF26), device=m.device)[None]
        return self.run(ids, pos, ctx, valid)[0]


def letter_probs(z, n_opts):
    """z [B, K] letter logits -> softmax over the offered letters (others -1e9)."""
    k = torch.as_tensor(n_opts, device=z.device)
    return torch.softmax(z.masked_fill(torch.arange(z.shape[1], device=z.device)[None] >= k[:, None], -1e9), -1)


# ---------------------------------------------------------------------------------------------- mapper
class Mapper(nn.Module):
    """Standardized E2B features -> raw 26B k_proj / v_proj outputs for every layer. W0 from ridge (frozen buffer
    or trainable), optional trainable low-rank delta (U V, V starts at zero)."""

    def __init__(self, mu, sd, W, b, rank=0, train_full=False, train_bias=True):
        super().__init__()
        self.register_buffer("mu", mu.float().cuda())
        self.register_buffer("sd", sd.float().cuda())
        if train_full:
            self.W = nn.Parameter(W.float().cuda())
        else:
            self.register_buffer("W", W.to(torch.bfloat16).cuda())
        if train_bias:
            self.b = nn.Parameter(b.float().cuda())
        else:
            self.register_buffer("b", b.float().cuda())
        self.rank = rank
        if rank:
            self.U = nn.Parameter(torch.randn(W.shape[0], rank, device="cuda") / math.sqrt(W.shape[0]))
            self.V = nn.Parameter(torch.zeros(rank, W.shape[1], device="cuda"))

    def forward(self, X):
        xs = ((X.float() - self.mu) / self.sd).to(torch.bfloat16)
        Y = xs @ self.W.to(torch.bfloat16) + self.b.to(torch.bfloat16)
        if self.rank:
            Y = Y + (xs @ self.U.to(torch.bfloat16)) @ self.V.to(torch.bfloat16)
        return Y


def load_ridge(path):
    d = torch.load(path, map_location="cpu")
    return d


# ------------------------------------------------------------------------------------- feature store
class Store:
    """Precomputed split: meta.jsonl (one row per item) + feats.bin (bf16 [rows, D], middle positions only)."""

    def __init__(self, split):
        self.dir = os.path.join(HX, split)
        cfg = json.load(open(os.path.join(self.dir, "config.json")))
        self.layers, self.D = cfg["layers"], cfg["D"]
        self.meta = [json.loads(l) for l in open(os.path.join(self.dir, "meta.jsonl"))]
        raw = np.memmap(os.path.join(self.dir, "feats.bin"), dtype=np.int16, mode="r")
        self.feats = raw.reshape(-1, self.D)
        self.by_id = {m["id"]: i for i, m in enumerate(self.meta)}

    def x(self, i, cols=None):
        m = self.meta[i]
        a = self.feats[m["off"]: m["off"] + m["m"]]
        t = torch.from_numpy(np.ascontiguousarray(a)).view(torch.bfloat16)
        return t if cols is None else t[:, cols]

    def batch(self, idxs, cols=None, device="cuda"):
        """Padded feature batch [B, M, D'] (bf16, zeros past each length) and lengths [B]."""
        xs = [self.x(i, cols) for i in idxs]
        M = max(x.shape[0] for x in xs)
        X = torch.zeros(len(xs), M, xs[0].shape[1], dtype=torch.bfloat16)
        for j, x in enumerate(xs):
            X[j, : x.shape[0]] = x
        m = torch.tensor([x.shape[0] for x in xs])
        return X.to(device, non_blocking=True), m.to(device)


def layer_cols(store_layers, want):
    """Column indices of the wanted E2B layers inside a store holding store_layers."""
    cols = []
    for L in want:
        j = store_layers.index(L)
        cols.extend(range(j * D_E, (j + 1) * D_E))
    return torch.tensor(cols)


def acc_report(meta, probs):
    """Accuracy overall/image/text and per source from {id: probs}."""
    per = {}
    for m in meta:
        p = probs[m["id"]]
        per.setdefault(m["source"], []).append(int(int(np.argmax(p)) == m["answer"]))
    img = [v for s, l in per.items() if s in ("scienceqa", "aokvqa", "ai2d", "mmstar") for v in l]
    txt = [v for s, l in per.items() if s in ("mmlu", "arc", "csqa") for v in l]
    rep = {s: round(100 * sum(l) / len(l), 2) for s, l in sorted(per.items())}
    rep["_image"] = round(100 * sum(img) / len(img), 2) if img else None
    rep["_text"] = round(100 * sum(txt) / len(txt), 2) if txt else None
    rep["_n"] = {s: len(l) for s, l in sorted(per.items())}
    return rep
