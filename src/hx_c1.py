"""C1 relay control: the same E2B-jev features, the same loss and data as A2, a parameter-matched trainable head,
and no 26B. Head: standardized features -> Linear to d -> + position and modality embeddings, a learned readout
token in front, n pre-norm transformer encoder layers (bidirectional over the middle tokens), LayerNorm, Linear
to the 5 letters (offered letters only in the softmax).

Usage: python hx_c1.py <tag> [epochs]   (HX_C1_LR, default 1e-4; HX_C1_LAYERS, default 5; HX_C1_D, default 1024)
"""
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx_core import HX, K, OUT, Store, acc_report, layer_cols, log_jsonl
from hx_eval import batches_for, kl_agree
from hx_train import BS, KLW, WARM, epoch_batches, masked_logp, teacher

LR = float(os.environ.get("HX_C1_LR", 1e-4))
NL = int(os.environ.get("HX_C1_LAYERS", 5))
DM = int(os.environ.get("HX_C1_D", 1024))


class Relay(nn.Module):
    def __init__(self, mu, sd, d_in, d=DM, n_layers=NL, heads=16, mlp_mult=4, max_len=1024):
        super().__init__()
        self.register_buffer("mu", mu.float())
        self.register_buffer("sd", sd.float())
        self.inp = nn.Linear(d_in, d)
        self.pos = nn.Embedding(max_len + 1, d)
        self.mod = nn.Embedding(2, d)
        self.cls = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, mlp_mult * d, dropout=0.0, activation="gelu",
                                                                batch_first=True, norm_first=True) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, K)

    def forward(self, X, m, img):
        B, M, _ = X.shape
        x = self.inp(((X.float() - self.mu) / self.sd))
        x = x + self.pos(torch.arange(1, M + 1, device=X.device))[None] + self.mod.weight[int(img)]
        x = torch.cat([self.cls.expand(B, 1, -1), x], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=X.device),
                         torch.arange(M, device=X.device)[None] >= m[:, None]], 1)
        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=pad)
        return self.head(self.norm(x[:, 0])).float()


@torch.no_grad()
def evaluate(net, st, cols):
    probs = {}
    net.eval()
    for b in batches_for(st.meta, 32):
        X, m = st.batch(b, cols)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(X, m, st.meta[b[0]]["image"])
        n = [st.meta[i]["n_opts"] for i in b]
        p = masked_logp(z, n).exp()
        for j, i in enumerate(b):
            probs[st.meta[i]["id"]] = p[j, : n[j]].float().cpu().tolist()
    net.train()
    return probs


def main(tag, epochs=1):
    t0 = time.time()
    tr, ev = Store("train"), Store("eval")
    rd = torch.load(os.path.join(HX, "ridge.pt"), map_location="cpu")
    cols_tr, cols_ev = layer_cols(tr.layers, rd["layers"]), layer_cols(ev.layers, rd["layers"])
    net = Relay(rd["mu"], rd["sd"], len(cols_tr)).cuda()
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"C1 trainable {n_params / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(0)
    plan = [epoch_batches(tr.meta, rng) for _ in range(epochs)]
    total = sum(map(len, plan))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    pool = ThreadPoolExecutor(1)
    step = 0
    stats = dict(loss=0.0, acc=0.0, n=0)
    for ep in range(epochs):
        batches = plan[ep]
        fut = pool.submit(tr.batch, batches[0], cols_tr, "cpu")
        for bi, b in enumerate(batches):
            X, m = fut.result()
            if bi + 1 < len(batches):
                fut = pool.submit(tr.batch, batches[bi + 1], cols_tr, "cpu")
            X, m = X.cuda(non_blocking=True), m.cuda()
            n = [tr.meta[i]["n_opts"] for i in b]
            y = torch.tensor([tr.meta[i]["answer"] for i in b], device="cuda")
            t = teacher(tr.meta, b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = net(X, m, tr.meta[b[0]]["image"])
            logp = masked_logp(z, n)
            loss = F.nll_loss(logp, y) + KLW * (t * (t.clamp(min=1e-12).log() - logp)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            stats["loss"] += float(loss); stats["acc"] += float((logp.argmax(-1) == y).float().mean()); stats["n"] += 1
            if step % 200 == 0:
                k = stats.pop("n")
                print(json.dumps({**{kk: round(v / k, 4) for kk, v in stats.items()}, "step": step, "total": total,
                                  "sec": round(time.time() - t0)}), flush=True)
                stats = dict(loss=0.0, acc=0.0, n=0)
        probs = evaluate(net, ev, cols_ev)
        img = [mm for mm in ev.meta if mm["image"]]
        txt = [mm for mm in ev.meta if not mm["image"]]
        rec = dict(kind="eval", tag=f"{tag}-ep{ep + 1}", arm="C1", acc=acc_report(ev.meta, probs), trainable=n_params,
                   lr=LR, layers=NL, d=DM, epoch=ep + 1, steps=step, seconds=round(time.time() - t0))
        rec["kl_to_26b_image"], rec["agree_26b_image"] = kl_agree(img, probs, "t26")
        rec["kl_to_26b_text"], rec["agree_26b_text"] = kl_agree(txt, probs, "t26")
        print(json.dumps(rec), flush=True)
        log_jsonl("eval.jsonl", rec)
        json.dump(probs, open(os.path.join(OUT, f"probs-{tag}-ep{ep + 1}.json"), "w"))
    torch.save(net.state_dict(), os.path.join(HX, f"{tag}.pt"))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
