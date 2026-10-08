"""Profile FFN activations on train-split prompts and build balanced neuron partitions per layer.

For each layer we record a_i(t) = |h_i(t)| * ||W_down[:, i]||, the size of neuron i's contribution to
the FFN output at token t, on a random sample of tokens from image+text and text-only prompts (train
splits only, disjoint from the eval set). Partitions:

  coact-E   : balanced spherical k-means on each neuron's L2-normalised a_i(.) profile, E groups
              (neurons that fire on the same tokens share an expert)
  shared-8  : group 0 = the 1920 neurons with the largest mean a_i (always on); the other 13440 are
              split into 7 groups by coact k-means
  random-E  : a random permutation

Output: $MOEFY_ROOT/partitions.pt = {name: [LongTensor(15360)] * 48} plus per-layer diagnostics.
"""
import json
import os
import time

import torch
import torch.nn.functional as F

import moe
from common import ROOT, encode_batch, batches, load
from data import profile_items

TOK_PER_ITEM = 32
N = 15360


def balanced_assign(S, cap):
    """Assign rows of S [N, E] to columns, at most `cap` per column, greedily by proposal rounds."""
    n, E = S.shape
    assign = torch.full((n,), -1, dtype=torch.long, device=S.device)
    load = torch.zeros(E, dtype=torch.long, device=S.device)
    while True:
        un = (assign < 0).nonzero().squeeze(1)
        if un.numel() == 0:
            return assign
        Su = S[un].clone()
        Su[:, load >= cap] = float("-inf")
        best_s, best = Su.max(1)
        for e in best.unique().tolist():
            room = cap - int(load[e])
            if room <= 0:
                continue
            cand = un[best == e]
            if cand.numel() > room:
                cand = cand[torch.topk(best_s[best == e], room).indices]
            assign[cand] = e
            load[e] += cand.numel()


def balanced_kmeans(V, E, iters=25, seed=0):
    """V [n, T] with unit rows. Returns group id per row, every group exactly n // E rows."""
    n = V.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = V[torch.randperm(n, generator=g)[:E].to(V.device)].clone()
    assign = None
    for _ in range(iters):
        new = balanced_assign(V @ C.t(), n // E)
        if assign is not None and bool((new == assign).all()):
            break
        assign = new
        C = torch.zeros(E, V.shape[1], device=V.device).index_add_(0, assign, V)
        C = F.normalize(C, dim=1)
    return assign


def to_perm(assign, E):
    """Group ids -> permutation listing group 0's neurons first, then group 1's, and so on."""
    return torch.cat([(assign == e).nonzero().squeeze(1) for e in range(E)]).cpu()


def main():
    proc, model = load()
    ctl = moe.install(model)
    items = profile_items()
    L = ctl.n_layers
    buf = [[] for _ in range(L)]
    kind = []
    gen = torch.Generator(device="cpu").manual_seed(0)
    state = {}

    def hook(layer, x, h, c):
        if layer == 0:
            # Choose the sampled positions once per batch, from valid (non-padding) tokens.
            picks = []
            for b in range(h.shape[0]):
                pos = c.valid[b].nonzero().squeeze(1).cpu()
                sel = pos[torch.randperm(pos.numel(), generator=gen)[:TOK_PER_ITEM]]
                picks.append((b, sel))
                kind.append(c.is_image[b, sel.to(c.is_image.device)].cpu())
            state["picks"] = picks
        w = model.model.language_model.layers[layer].mlp.wnorm
        for b, sel in state["picks"]:
            buf[layer].append((h[b, sel.to(h.device)].abs() * w).to(torch.float16).cpu())

    ctl.hook = hook
    t0 = time.time()
    with torch.no_grad():
        for batch in batches(items, 8):
            enc = encode_batch(proc, batch).to("cuda")
            ctl.is_image = enc["input_ids"] == model.config.image_token_id
            ctl.valid = enc["attention_mask"].bool()
            model(**enc, logits_to_keep=1)
    ctl.hook = None
    is_img = torch.cat(kind)
    print(f"profiled {len(items)} items, {is_img.numel()} tokens ({int(is_img.sum())} image) in {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()

    parts = {k: [] for k in ("coact-8", "coact-32", "coact-128", "shared-8", "random-8", "random-32", "random-128")}
    diag = []
    for l in range(L):
        A = torch.cat(buf[l]).float().cuda()          # [T, N]
        buf[l] = None
        mean_all = A.mean(0)
        # Concentration: share of total contribution mass held by the top 12.5/25/50% neurons by mean.
        srt = mean_all.sort(descending=True).values
        cum = srt.cumsum(0) / srt.sum()
        # Per-token: fraction of neurons needed for 90% of that token's contribution mass (L1 proxy).
        tok_sorted = A.sort(dim=1, descending=True).values
        need90 = ((tok_sorted.cumsum(1) / tok_sorted.sum(1, keepdim=True)) < 0.9).sum(1).float() / N
        V = F.normalize(A.t().contiguous(), dim=1)     # [N, T]
        for E in (8, 32, 128):
            parts[f"coact-{E}"].append(to_perm(balanced_kmeans(V, E, seed=l), E))
        shared = torch.topk(mean_all, N // 8).indices
        rest = torch.ones(N, dtype=torch.bool, device=A.device)
        rest[shared] = False
        rest_idx = rest.nonzero().squeeze(1)
        sub = balanced_kmeans(V[rest_idx], 7, seed=l)
        parts["shared-8"].append(torch.cat([shared.cpu()] + [rest_idx[sub == e].cpu() for e in range(7)]))
        g = torch.Generator(device="cpu").manual_seed(1000 + l)
        for E in (8, 32, 128):
            parts[f"random-{E}"].append(torch.randperm(N, generator=g))
        diag.append(dict(layer=l, top12=float(cum[N // 8 - 1]), top25=float(cum[N // 4 - 1]), top50=float(cum[N // 2 - 1]),
                         need90_text=float(need90[~is_img.cuda()].mean()), need90_image=float(need90[is_img.cuda()].mean())))
        print(json.dumps(diag[-1]), flush=True)
        del A, V, tok_sorted
        torch.cuda.empty_cache()
    for k, v in parts.items():
        assert all(sorted(p.tolist()) == list(range(N)) for p in v[:2]), k
    torch.save(parts, os.path.join(ROOT, "partitions.pt"))
    json.dump(diag, open(os.path.join(ROOT, "results", "profile_diag.json"), "w"), indent=1)
    print("saved", list(parts))


if __name__ == "__main__":
    main()
