"""26B-A4B language model latency against sequence length, Jev readout, batch 1, no vision tower.

The question this answers: if a small model prefilled the input and handed the 26B only m projected states
plus a short suffix, how much faster would the 26B's pass be? At batch 1 the MoE is bound by the expert
weights it reads, so the saving depends on how many distinct experts a short sequence touches.
Inputs are the last S tokens of real prompts (image tokens replaced by their vision embeddings), so the
routing is realistic. Same kernels as jev_fast26 (Marlin int4 experts, compile + one CUDA graph)."""
import json, os, statistics, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from jev_fast26 import load_fp8_model, JevFast26, cuda_ms, WH, ROOT, K
from common import encode_batch
from data import _aokvqa


class Short:
    def __init__(self, f, S):
        self.f, self.S = f, S
        d, dt = "cuda", f.dtype
        self.x = torch.zeros(1, S, f.lm.config.hidden_size, dtype=dt, device=d)
        self.pos = torch.arange(S, device=d)[None]
        causal = torch.ones(S, S, dtype=torch.bool, device=d).tril()
        self.mask = torch.where(causal, torch.tensor(0.0, dtype=dt, device=d), f.neg)[None, None]
        self.cand = torch.zeros(K, dtype=torch.bool, device=d); self.cand[:4] = True

    @torch.no_grad()
    def forward(self):
        f, x = self.f, self.x
        rope = {t: f.lm.rotary_emb(x, self.pos, t) for t in set(f.types)}
        for i, layer in enumerate(f.layers):
            x = layer(x, None, shared_kv_states={}, position_embeddings=rope[f.types[i]],
                      attention_mask=self.mask, position_ids=self.pos)
        h = f.lm.norm(x[:, -1])
        z = (h @ f.E.t()).float()
        z = f.cap * torch.tanh(z / f.cap)
        return torch.softmax(z.masked_fill(~self.cand, -1e9), -1)[0]


def main():
    proc, model = load_fp8_model()
    items = [dict(it, image=it["image"].resize(WH)) for it in _aokvqa("validation", 40, 21)]
    budget = 70
    S0 = -(-max(encode_batch(proc, [it], budget)["input_ids"].shape[1] for it in items) // 16) * 16
    f = JevFast26(model, proc, budget, S0)
    # Full input embeddings (text + vision) for each item, so short windows route like real inputs.
    embs = []
    for it in items:
        f.prepare(it)
        with torch.no_grad():
            v = f.vision()
            x = f.lm.embed_tokens(f.buf["ids"])[0].index_copy(0, f.buf["img_idx"], v.to(f.dtype))
        n = int((f.buf["pos"][0] > 0).sum()) + 1
        embs.append(x[-n:].clone())
    # Distinct experts touched per layer, by sequence length (eager, real windows).
    seen = []
    def hook(mod, args, kw):
        idx = args[1] if len(args) > 1 else kw["top_k_index"]
        seen.append(int(idx.unique().numel()))
    hs = [layer.experts.register_forward_pre_hook(hook, with_kwargs=True) for layer in f.layers]
    comp = os.environ.get("JEV_COMPILE", "default") or None
    out = []
    for S in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 144, 288]:
        sh = Short(f, S)
        win = [e[-S:] for e in embs if e.shape[0] >= S]
        if not win:  # longer than any prompt: tile two prompts
            win = [torch.cat([embs[i], embs[i + 1]])[-S:] for i in range(0, len(embs) - 1, 2)]
        touched = []
        for w in win[:10]:
            seen.clear(); sh.x.copy_(w[None]); sh.forward()
            touched.append(statistics.mean(seen))
        for h in hs:
            h.remove()
        fn = torch.compile(sh.forward, mode=comp, dynamic=False) if comp else sh.forward
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        ms = []
        for w in win[:10]:
            sh.x.copy_(w[None]); ms.append(cuda_ms(g.replay, 30))
        rec = dict(kind="lm_short", experts=os.environ.get("JEV_EXPERTS", "fp8"), S=S, lm_ms=round(statistics.median(ms), 2),
                   lm_ms_min=round(min(ms), 2), lm_ms_max=round(max(ms), 2),
                   experts_touched_per_layer=round(statistics.mean(touched), 1), gpu=torch.cuda.get_device_name())
        print(json.dumps(rec), flush=True)
        out.append(rec)
        hs = [layer.experts.register_forward_pre_hook(hook, with_kwargs=True) for layer in f.layers]
        del g
        torch._dynamo.reset()
    with open(os.path.join(ROOT, "results", "jev", "lm_short.jsonl"), "a") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
