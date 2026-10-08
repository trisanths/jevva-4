"""The 26B-A4B vision tower alone, in the fast-path treatment (full patch grid, no mask, precomputed pooling
matrix, torch.compile, one CUDA graph), on this GPU. Loads only the vision tower weights from the checkpoint.

Usage: python tower26.py [budgets]   -> prints and appends to results/jev/tower26.jsonl
"""
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoProcessor
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

from common import ROOT, encode_batch
from data import _aokvqa

MID = "google/gemma-4-26B-A4B-it"


def load_tower():
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    cfg = AutoConfig.from_pretrained(MID)
    vcfg = cfg.vision_config
    vcfg._attn_implementation = "sdpa"
    with torch.device("cuda"):
        tower = Gemma4VisionModel(vcfg).to(torch.bfloat16).eval()
    idx = json.load(open(hf_hub_download(MID, "model.safetensors.index.json")))["weight_map"]
    sd = {}
    for name, shard in idx.items():
        if ".vision_tower." in name or name.startswith("vision_tower."):
            with safe_open(hf_hub_download(MID, shard), "pt") as f:
                sd[name.split("vision_tower.", 1)[1]] = f.get_tensor(name)
    missing, unexpected = tower.load_state_dict(sd, strict=False)
    print("missing", len(missing), missing[:4], "unexpected", len(unexpected), flush=True)
    return tower


def main(budgets):
    tower = load_tower()
    proc = AutoProcessor.from_pretrained(MID, padding_side="left")
    it = _aokvqa("validation", 4, 3)[0]
    it = dict(it, image=it["image"].resize((640, 360)))
    for st in budgets:
        enc = encode_batch(proc, [it], st)
        pv, pp = enc["pixel_values"][0], enc["image_position_ids"][0]
        real = (pp != -1).all(-1)
        pv, pp = pv[real].cuda()[None], pp[real].cuda()[None]
        P = pv.shape[1]; k = 3; N = P // 9
        ppc = pp[0].cpu()
        max_x = int(ppc[:, 0].max()) + 1
        kid = torch.div(ppc, k, rounding_mode="floor"); kid = kid[:, 0] + (max_x // k) * kid[:, 1]
        pool = (F.one_hot(kid, N).float().t() / 9).cuda()[None]
        root, std = tower.config.hidden_size ** 0.5, tower.config.standardize

        @torch.no_grad()
        def fwd():
            h = tower.patch_embedder(pv, pp, torch.zeros(1, P, dtype=torch.bool, device="cuda"))
            rot = tower.encoder.rotary_emb(h, pp)
            for layer in tower.encoder.layers:
                h = layer(h, attention_mask=None, position_embeddings=rot, position_ids=pp)
            v = (pool @ h.float()) * root
            if std:
                v = (v - tower.std_bias.float()) * tower.std_scale.float()
            return v

        ref = fwd()
        with torch.no_grad():
            hf = tower(pixel_values=enc["pixel_values"].cuda(), pixel_position_ids=enc["image_position_ids"].cuda()).last_hidden_state
        dp = float((ref[0].to(hf.dtype) - hf.reshape(-1, hf.shape[-1])[: ref.shape[1]]).abs().max() / hf.abs().max())
        res = dict(budget=st, P=P, N=N, rel_maxdiff_vs_hf=round(dp, 4), gpu=torch.cuda.get_device_name())
        for mode in (None, "default", "max-autotune-no-cudagraphs"):
            fn = torch.compile(fwd, mode=mode, dynamic=False) if mode else fwd
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = fn()
            for _ in range(10):
                g.replay()
            torch.cuda.synchronize()
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(50)]
            for a, b in ev:
                a.record(); g.replay(); b.record()
            torch.cuda.synchronize()
            res[f"{mode or 'graph'}_ms"] = round(statistics.median(a.elapsed_time(b) for a, b in ev), 2)
            del g
        print(json.dumps(res), flush=True)
        with open(os.path.join(ROOT, "results", "jev", "tower26.jsonl"), "a") as fh:
            fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main([int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "280,140,70").split(",")])
