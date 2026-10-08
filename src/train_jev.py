"""Jev-tune a Jevified Gemma 4 12B (exit at layer L, image tokens dropped after layer M).

LoRA (r=16, alpha=32, dropout 0.05; AdamW lr 1e-4, betas 0.9/0.95, 100 warmup steps, cosine, effective batch 16) on q/k/v/o/gate/up/down of the kept language layers, plus the final norm,
which is now the readout at the exit layer. The vision embedder and token embeddings stay frozen. Loss is
cross-entropy over the offered option letters at the answer position. Options are reshuffled for every
example; each batch draws its image budget from 70/140/280 so one adapter serves all budgets.

Usage: python train_jev.py L M [max_items] [tag]
Writes $MOEFY_ROOT/adapters/<tag>/ and logs/train_<tag>.log.
"""
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from common import ROOT, encode_batch, letter_ids
from data import LazyImageSet, text_train_items
from jevnet import State, install_drop, load_truncated

MICRO = int(os.environ.get("JEV_MICRO", 4))
ACCUM, LR, WARM_STEPS = 16 // MICRO, 1e-4, 100
BUDGETS = (70, 140, 280)


def shuffled(it, rng):
    perm = list(range(len(it["options"])))
    rng.shuffle(perm)
    return dict(it, options=[it["options"][p] for p in perm], answer=perm.index(it["answer"]))


def batches(img_set, text_items, max_items, rng):
    """Yield micro-batches of one modality; image and text batches interleave in proportion."""
    keys = img_set.keys[: int(max_items * 0.7)]
    texts = text_items[: max_items - len(keys)]
    plan = [("img", k) for k in keys] + [("txt", t) for t in texts]
    rng.shuffle(plan)
    img_buf, txt_buf = [], []
    for kind, x in plan:
        if kind == "img":
            it = img_set.get(x)
            if it is None:
                continue
            img_buf.append(shuffled(it, rng))
            if len(img_buf) == MICRO:
                yield img_buf, rng.choice(BUDGETS)
                img_buf = []
        else:
            txt_buf.append(shuffled(x, rng))
            if len(txt_buf) == MICRO:
                yield txt_buf, 280
                txt_buf = []


def main(L, M, max_items=24000, tag=None):
    tag = tag or f"L{L}-M{M}"
    out = os.path.join(ROOT, "adapters", tag)
    os.makedirs(out, exist_ok=True)
    proc, model = load_truncated(L, mid=os.environ.get("JEV_MID", "google/gemma-4-12B-it"))
    state = State()
    if M < L:
        install_drop(model, M, state)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                     target_modules=r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))")
    model = get_peft_model(model, cfg)
    norm = model.base_model.model.model.language_model.norm
    norm.weight.data = norm.weight.data.float()
    norm.weight.requires_grad_(True)
    for n, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()           # AdamW silently drops bf16 updates; keep trainables in fp32
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    cand = torch.tensor(letter_ids(proc.tokenizer), device="cuda")
    img_set, texts = LazyImageSet(), text_train_items()
    total_micro = max_items // MICRO
    total_steps = total_micro // ACCUM
    warm = WARM_STEPS
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    rng = random.Random(0)
    model.train()
    t0, step, micro, run_loss, run_acc = time.time(), 0, 0, 0.0, 0.0
    log = open(os.path.join(ROOT, "logs", f"train_{tag}.log"), "a")
    for items, st in batches(img_set, texts, max_items, rng):
        enc = encode_batch(proc, items, st).to("cuda")
        state.is_image = enc["input_ids"] == model.config.image_token_id
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1].float()
        z = logits[:, cand]
        k = torch.tensor([len(it["options"]) for it in items], device="cuda")
        z = z.masked_fill(torch.arange(len(cand), device="cuda")[None] >= k[:, None], -1e9)
        y = torch.tensor([it["answer"] for it in items], device="cuda")
        loss = F.cross_entropy(z, y)
        (loss / ACCUM).backward()
        run_loss += float(loss); run_acc += float((z.argmax(-1) == y).float().mean())
        micro += 1
        if micro % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                msg = (f"step {step}/{total_steps} loss {run_loss / (25 * ACCUM):.4f} acc {run_acc / (25 * ACCUM):.3f} "
                       f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
                run_loss = run_acc = 0.0
        if micro >= total_micro:
            break
    model.save_pretrained(out)
    torch.save({"norm": norm.weight.detach().cpu()}, os.path.join(out, "norm.pt"))
    print("saved", out, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]), int(a[2]) if len(a) > 2 else 24000, a[3] if len(a) > 3 else None)
