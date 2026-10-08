"""A2: end-to-end training of the hybrid through the 26B suffix pass.

Trainable: the mapper's low-rank delta U V (rank HX_RANK, V starts at zero) and bias on top of the frozen ridge W
(or the full W with HX_FULL=1), plus LoRA r16/alpha 32 on the 26B's q/k/v/o projections, which only ever act on
the 9 suffix tokens (header KV is native and computed once with base weights; middle KV comes from the mapper).
Frozen: everything else; the int4 experts pass gradients to their input through the custom Function in hx_core.
Loss: cross-entropy over the offered letters (options were shuffled once per item at precompute time)
       + HX_KL * KL(26B direct letter probs || hybrid letter probs).
AdamW lr HX_LR (default 1e-4), betas 0.9/0.95, 100 warmup steps, cosine, batch 16, grad clip 1.0.

Usage: python hx_train.py <tag> [epochs]
Writes $ROOT/hx/b<budget>/<tag>.pt, logs to stdout, and evaluates on the eval split after each epoch.
"""
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx_core import (HX, K, OUT, Mapper, Runner26, Store, add_lora, layer_cols, load_26b, log_jsonl)
from hx_eval import evaluate, report

RANK = int(os.environ.get("HX_RANK", 512))
FULL = bool(int(os.environ.get("HX_FULL", 0)))
LR = float(os.environ.get("HX_LR", 1e-4))
MAP_LR = float(os.environ.get("HX_MAP_LR", LR))
KLW = float(os.environ.get("HX_KL", 1.0))
BS = 16
WARM = 100
MAX_STEPS = int(os.environ.get("HX_MAX_STEPS", 0))
# Training items with more middle tokens than this are skipped (704 of 33,274, all long text passages; the eval
# split has one). Batches run as micro-batches of at most TOK middle tokens, gradients accumulated.
MAXM = int(os.environ.get("HX_MAXM", 512))
TOK = int(os.environ.get("HX_TOK", 4096))


def epoch_batches(meta, rng, bs=BS):
    """Shuffle, then sort within chunks of 50 batches by (modality, length) to keep padding small; batches are
    single-modality; batch order shuffled."""
    idx = [i for i in range(len(meta)) if meta[i]["m"] <= MAXM]
    rng.shuffle(idx)
    out = []
    for s in range(0, len(idx), bs * 50):
        chunk = sorted(idx[s: s + bs * 50], key=lambda i: (meta[i]["image"], meta[i]["m"]))
        cur = []
        for i in chunk:
            if cur and (len(cur) == bs or meta[cur[0]]["image"] != meta[i]["image"]):
                out.append(cur)
                cur = []
            cur.append(i)
        if cur:
            out.append(cur)
    rng.shuffle(out)
    return out


def masked_logp(z, n_opts):
    k = torch.as_tensor(n_opts, device=z.device)
    z = z.masked_fill(torch.arange(z.shape[1], device=z.device)[None] >= k[:, None], -1e9)
    return F.log_softmax(z, -1)


def teacher(meta, b, device="cuda"):
    t = torch.zeros(len(b), K, device=device)
    for j, i in enumerate(b):
        p = meta[i]["t26"]
        t[j, : len(p)] = torch.tensor(p)
    return t


def main(tag, epochs=1):
    t0 = time.time()
    tr, ev = Store("train"), Store("eval")
    ridge_path = os.path.join(HX, "ridge.pt")
    rd = torch.load(ridge_path, map_location="cpu")
    cols_tr = layer_cols(tr.layers, rd["layers"])
    cols_ev = layer_cols(ev.layers, rd["layers"])
    proc, model = load_26b("int4")
    run = Runner26(model, proc)
    for img in (True, False):
        run.header(img)
    loras = add_lora(model)
    mapper = Mapper(rd["mu"], rd["sd"], rd["W"], rd["b"], rank=RANK, train_full=FULL)
    map_params = [p for p in mapper.parameters() if p.requires_grad]
    lora_params = [p for m in loras for p in (m.A, m.B)]
    n_train = sum(p.numel() for p in map_params + lora_params)
    print(f"trainable: mapper {sum(p.numel() for p in map_params) / 1e6:.1f}M, lora {sum(p.numel() for p in lora_params) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW([dict(params=map_params, lr=MAP_LR), dict(params=lora_params, lr=LR)],
                            betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(0)
    plan = [epoch_batches(tr.meta, rng) for _ in range(epochs)]
    total = sum(map(len, plan))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    cfg = dict(ridge=ridge_path, rank=RANK, full=FULL, lr=LR, map_lr=MAP_LR, kl=KLW, bs=BS, epochs=epochs,
               lora_r=16, lora_alpha=32, layers=rd["layers"], trainable=n_train)
    pool = ThreadPoolExecutor(1)
    step, run_stats = 0, dict(loss=0.0, ce=0.0, kl=0.0, acc=0.0, agree=0.0, n=0)
    for ep in range(epochs):
        batches = plan[ep]
        mapper.train()
        fut = pool.submit(tr.batch, batches[0], cols_tr, "cpu")
        for bi, b in enumerate(batches):
            X, m = fut.result()
            if bi + 1 < len(batches):
                fut = pool.submit(tr.batch, batches[bi + 1], cols_tr, "cpu")
            X, m = X.cuda(non_blocking=True), m.cuda()
            img = tr.meta[b[0]]["image"]
            n = [tr.meta[i]["n_opts"] for i in b]
            y = torch.tensor([tr.meta[i]["answer"] for i in b], device="cuda")
            t = teacher(tr.meta, b)
            opt.zero_grad(set_to_none=True)
            mb = max(1, TOK // max(1, int(m.max())))
            agg = dict(loss=0.0, ce=0.0, kl=0.0, acc=0.0, agree=0.0)
            for s0 in range(0, len(b), mb):
                sl = slice(s0, s0 + mb)
                w_ = (min(s0 + mb, len(b)) - s0) / len(b)
                Y = mapper(X[sl])
                ctx, valid, h = run.context(Y, m[sl], img)
                z = run.suffix(ctx, valid, h, m[sl])
                logp = masked_logp(z, n[sl])
                ce = F.nll_loss(logp, y[sl])
                kl = (t[sl] * (t[sl].clamp(min=1e-12).log() - logp)).sum(-1).mean()
                loss = ce + KLW * kl
                (loss * w_).backward()
                with torch.no_grad():
                    pred = logp.argmax(-1)
                    agg["loss"] += w_ * float(loss); agg["ce"] += w_ * float(ce); agg["kl"] += w_ * float(kl)
                    agg["acc"] += w_ * float((pred == y[sl]).float().mean())
                    agg["agree"] += w_ * float((pred == t[sl].argmax(-1)).float().mean())
                del Y, ctx, z, logp, loss
            torch.nn.utils.clip_grad_norm_(map_params + lora_params, 1.0)
            opt.step()
            sched.step()
            step += 1
            for kk, v in agg.items():
                run_stats[kk] += v
            run_stats["n"] += 1
            if step % (10 if MAX_STEPS else 50) == 0:
                k = run_stats.pop("n")
                msg = {kk: round(v / k, 4) for kk, v in run_stats.items()}
                msg.update(step=step, total=total, lr=f"{sched.get_last_lr()[0]:.2e}", sec=round(time.time() - t0),
                           gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
                print(json.dumps(msg), flush=True)
                run_stats = dict(loss=0.0, ce=0.0, kl=0.0, acc=0.0, agree=0.0, n=0)
            if MAX_STEPS and step >= MAX_STEPS:
                print(json.dumps(dict(smoke=True, steps=step, sec=round(time.time() - t0),
                                      gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))), flush=True)
                return
        ck = dict(cfg=cfg, mapper={k: v for k, v in mapper.state_dict().items() if k not in ("W", "mu", "sd")} if not FULL else mapper.state_dict(),
                  lora=[dict(A=m_.A.detach().cpu(), B=m_.B.detach().cpu()) for m_ in loras], epoch=ep + 1)
        path = os.path.join(HX, f"{tag}.pt")
        torch.save(ck, path)
        mapper.eval()
        te = time.time()
        probs = evaluate(run, mapper, ev, cols_ev)
        rec = report(f"{tag}-ep{ep + 1}", ev, probs, dict(cfg=cfg, epoch=ep + 1, train_seconds=round(te - t0),
                                                         eval_seconds=round(time.time() - te), steps=step))
        print(json.dumps(rec), flush=True)
        log_jsonl("eval.jsonl", rec)
        json.dump(probs, open(os.path.join(OUT, f"probs-{tag}-ep{ep + 1}.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
