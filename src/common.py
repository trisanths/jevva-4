"""Model loading, input building and Jev-style option scoring shared by every stage."""
import math
import os

import numpy as np
import torch
import transformers.models.gemma4_unified.modeling_gemma4_unified as g4u
from transformers import AutoModelForImageTextToText, AutoProcessor

from data import LETTERS, prompt_text

MID = "google/gemma-4-12B-it"
ROOT = os.environ.get("MOEFY_ROOT", "/mnt/nvme/moefy")


_hf_masks = g4u.create_masks_for_generate
MASK = {"mode": "official"}


def _masks(**kw):
    """Plain forward calls in transformers 5.17 apply the image-block bidirectional overlay on every layer.
    Google's reference (and vLLM) keeps global layers causal and clips the overlay to the sliding window;
    transformers already has that as create_masks_for_vision_model but only calls it from generate()."""
    ids = kw.get("block_sequence_ids")
    if MASK["mode"] == "official" and ids is not None:
        kw.pop("block_sequence_ids")
        return g4u.create_masks_for_vision_model(block_sequence_ids=ids, **kw)
    return _hf_masks(**kw)


g4u.create_masks_for_generate = _masks


def quantize_fp8(model, rowwise=True, weight_only=False):
    """FP8 weights (and, unless weight_only, dynamic FP8 activations) via torchao on every Linear of the LM layers
    and vision encoder. Embeddings, the per-layer-embedding tables and the letter readout stay bf16."""
    from torchao.quantization import (Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig, PerRow,
                                      PerTensor, quantize_)
    if weight_only:
        cfg = Float8WeightOnlyConfig()
    else:
        cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow() if rowwise else PerTensor())

    def keep(mod, fqn):
        return (isinstance(mod, torch.nn.Linear) and ("language_model.layers" in fqn or "vision_tower.encoder" in fqn)
                and mod.in_features % 16 == 0 and mod.out_features % 16 == 0)

    quantize_(model, cfg, filter_fn=keep)
    return model


def quantize_mixed(model):
    """FP8 (dynamic activations, per-tensor) on the vision encoder and the layers before the KV-shared tail; int4
    weight-only (tinygemm tile packing, group 128) on the tail, which only ever sees the one answer token and is
    bound by reading weights."""
    import re
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Int4WeightOnlyConfig, PerTensor, quantize_
    tc = model.config.get_text_config()
    tail_from = tc.num_hidden_layers - (getattr(tc, "num_kv_shared_layers", 0) or 0)

    def idx(fqn):
        m = re.search(r"language_model\.layers\.(\d+)\.", fqn)
        return int(m.group(1)) if m else None

    def lin(mod):
        return isinstance(mod, torch.nn.Linear) and mod.in_features % 128 == 0 and mod.out_features % 16 == 0

    quantize_(model, Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d"),
              filter_fn=lambda m, f: lin(m) and idx(f) is not None and idx(f) >= tail_from)
    quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
              filter_fn=lambda m, f: lin(m) and ("vision_tower.encoder" in f or (idx(f) is not None and idx(f) < tail_from)))
    return model


def load(attn="eager", mid=MID, **kw):
    proc = AutoProcessor.from_pretrained(mid, padding_side="left")
    kw.setdefault("device_map", "cuda")
    model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16, attn_implementation=attn, **kw).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mode = os.environ.get("JEV_FP8")
    if mode == "mixed":
        quantize_mixed(model)
    elif mode:
        quantize_fp8(model, rowwise=mode == "row", weight_only=mode == "wo")
    return proc, model


def letter_ids(tok):
    return [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]


def build(proc, item):
    content = ([{"type": "image", "image": item["image"]}] if item["image"] is not None else [])
    content.append({"type": "text", "text": prompt_text(item)})
    return [{"role": "user", "content": content}]


def encode_batch(proc, items, soft_tokens=280):
    convs = [build(proc, it) for it in items]
    kw = dict(padding=True)
    if soft_tokens != 280:
        kw["images_kwargs"] = {"max_soft_tokens": soft_tokens}
    enc = proc.apply_chat_template(convs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors="pt", processor_kwargs=kw)
    # The answer is read at position -1, which is only valid for every row under left padding.
    assert bool(enc["attention_mask"][:, -1].all()), "batch is not left-padded"
    return enc


def batches(items, bs):
    """Group items by modality and approximate length so padding stays small."""
    order = sorted(range(len(items)), key=lambda i: (items[i]["image"] is None, len(prompt_text(items[i]))))
    for s in range(0, len(order), bs):
        yield [items[i] for i in order[s:s + bs]]


@torch.no_grad()
def score(model, proc, items, ctl=None, bs=8, soft_tokens=280, logits=False):
    """Return {id: option-probability list} (or raw candidate logits) using one forward pass per batch."""
    cand = letter_ids(proc.tokenizer)
    out = {}
    for batch in batches(items, bs):
        enc = encode_batch(proc, batch, soft_tokens).to(model.device)
        if ctl is not None:
            ctl.is_image = enc["input_ids"] == model.config.image_token_id
            ctl.valid = enc["attention_mask"].bool()
        lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
        for it, row in zip(batch, lg):
            z = row[cand[:len(it["options"])]]
            out[it["id"]] = (z if logits else torch.softmax(z, -1)).cpu().tolist()
    return out


def metrics(items, probs, ref=None, n_bins=15):
    """Accuracy, NLL, Brier, ECE, and (if ref given) KL(ref || probs) and argmax agreement."""
    acc, nll, brier, conf, corr, kl, agree = [], [], [], [], [], [], []
    for it in items:
        p = np.asarray(probs[it["id"]], dtype=np.float64)
        y = it["answer"]
        pred = int(p.argmax())
        acc.append(pred == y)
        nll.append(-math.log(max(p[y], 1e-12)))
        onehot = np.zeros_like(p); onehot[y] = 1
        brier.append(float(((p - onehot) ** 2).sum()))
        conf.append(p.max()); corr.append(pred == y)
        if ref is not None:
            q = np.asarray(ref[it["id"]], dtype=np.float64)
            kl.append(float((q * (np.log(np.clip(q, 1e-12, 1)) - np.log(np.clip(p, 1e-12, 1)))).sum()))
            agree.append(int(q.argmax()) == pred)
    conf, corr = np.asarray(conf), np.asarray(corr, dtype=np.float64)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = sum(abs(conf[bins == b].mean() - corr[bins == b].mean()) * (bins == b).mean()
              for b in range(n_bins) if (bins == b).any())
    m = dict(n=len(items), acc=float(np.mean(acc)), nll=float(np.mean(nll)), brier=float(np.mean(brier)), ece=float(ece))
    if ref is not None:
        m.update(kl=float(np.mean(kl)), agree=float(np.mean(agree)))
    return m


def by_source(items, probs, ref=None):
    srcs = sorted({it["source"] for it in items})
    res = {s: metrics([it for it in items if it["source"] == s], probs, ref) for s in srcs}
    img = [it for it in items if it["image"] is not None]
    txt = [it for it in items if it["image"] is None]
    res["_image"] = metrics(img, probs, ref) if img else None
    res["_text"] = metrics(txt, probs, ref) if txt else None
    res["_all"] = metrics(items, probs, ref)
    return res
