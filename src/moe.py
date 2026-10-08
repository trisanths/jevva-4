"""Run Gemma 4 12B's dense GeGLU FFNs as partitioned experts, without changing any weight.

Each layer's 15360 intermediate neurons are split into E equal groups by a permutation. An expert's
output is h[:, group] @ W_down[:, group].T, so the sum over all E experts is the dense FFN. A
controller picks which experts (or which neurons) each token keeps:

  dense   : unmodified FFN
  oracle  : per token, keep the `shared` always-on groups plus the top-k other groups ranked by the
            L2 norm of their output (computed as h_e^T W_e^T W_e h_e) (the best a router could do if it saw the FFN's own activations)
  neuron  : per token, keep the top `frac` of neurons ranked by |h_i| * ||W_down[:, i]|| (an upper
            bound for any structured selection at that density)
  router  : per token, keep the shared groups plus the top-k groups scored by a learned linear router
            on the FFN input
  best    : per token, try every k-subset of the non-shared groups and keep the one whose output after
            the post-FFN RMSNorm is closest to the dense output after that norm (that norm rescales a
            partial sum, so the largest-norm experts are not always the best subset)

The chosen experts become a neuron mask on h ahead of the unmodified down projection, so keeping every
expert reproduces the dense layer exactly and bf16 rounding cannot pose as sparsity damage. This path
measures quality only; it computes the full FFN and is not fast.

`scope` limits sparsity to image tokens or text tokens; the other tokens run dense. The controller also
accumulates per-layer relative output error split by token type, and expert-usage counts.
"""
import torch
import torch.nn as nn


class Controller:
    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.mode = "dense"
        self.k = 8
        self.shared = 0
        self.frac = 1.0
        self.scope = "all"
        self.n_groups = 8
        self.perms = None      # list of LongTensor [N] per layer; group g = perm[g*n:(g+1)*n]
        self.gids = None       # list of LongTensor [N] per layer; neuron -> group
        self.routers = None    # nn.ModuleList of Linear(D, E) or None
        self.is_image = None   # BoolTensor [B, T] for the current forward
        self.valid = None      # BoolTensor [B, T], False on padding
        self.collect = False
        self.stats = torch.zeros(n_layers, 2, 2, dtype=torch.float64)  # layer, (text, image), (err2, ref2)
        self.usage = None      # [L, E] selection counts over valid tokens
        self.hook = None       # optional callable(layer, x, h, ctl) for profiling
        self._wg = {}
        self._wd = {}
        self._subsets = {}

    def subsets(self, k, device):
        """Bool [S, E] rows = every choice of `shared` fixed groups plus k of the rest."""
        key = (self.n_groups, self.shared, k)
        m = self._subsets.get(key)
        if m is None:
            import itertools
            rest = range(self.shared, self.n_groups)
            rows = []
            for c in itertools.combinations(rest, k):
                r = torch.zeros(self.n_groups, dtype=torch.bool)
                r[: self.shared] = True
                r[list(c)] = True
                rows.append(r)
            m = torch.stack(rows).to(device)
            self._subsets[key] = m
        return m

    def grouped_down(self, layer, w_down):
        """W_down columns in partition order, [E, n, D]."""
        w = self._wd.get(layer)
        if w is None:
            perm = self.perms[layer].to(w_down.device)
            w = w_down[:, perm].t().contiguous().view(self.n_groups, -1, w_down.shape[0])
            self._wd[layer] = w
        return w

    def set_partition(self, perms, n_groups):
        self.perms = [p.long() for p in perms]
        self.n_groups = n_groups
        n = self.perms[0].numel() // n_groups
        self.gids = []
        for p in self.perms:
            g = torch.empty_like(p)
            g[p] = torch.arange(n_groups).repeat_interleave(n)
            self.gids.append(g)
        self.usage = torch.zeros(self.n_layers, n_groups, dtype=torch.float64)
        self._wg = {}
        self._wd = {}

    def reset_stats(self):
        self.stats.zero_()
        if self.usage is not None:
            self.usage.zero_()

    def rel_err(self):
        """Per-layer relative L2 error of the FFN output, [L, 2] for (text, image)."""
        return (self.stats[..., 0] / self.stats[..., 1].clamp(min=1e-30)).sqrt()

    def gram(self, layer, w_down):
        """Per-expert Gram matrices W_e^T W_e, [E, n, n], so that ||y_e||^2 = h_e^T G_e h_e."""
        g = self._wg.get(layer)
        if g is None:
            perm = self.perms[layer].to(w_down.device)
            w = w_down[:, perm].float().t().contiguous().view(self.n_groups, -1, w_down.shape[0])  # [E, n, D]
            g = torch.bmm(w, w.transpose(1, 2)).to(w_down.dtype)
            self._wg[layer] = g
        return g


class SparseMLP(nn.Module):
    def __init__(self, mlp, layer, ctl):
        super().__init__()
        self.mlp = mlp
        self.layer = layer
        self.ctl = ctl
        self.active = True
        w = mlp.down_proj.weight
        self.register_buffer("wnorm", w.float().norm(dim=0).to(w.dtype), persistent=False)

    def expert_scores(self, x, h):
        ctl, m = self.ctl, self.mlp
        if ctl.mode == "oracle":
            B, T, _ = h.shape
            g = ctl.gram(self.layer, m.down_proj.weight)                                # [E, n, n]
            hp = h[..., ctl.perms[self.layer].to(h.device)].view(B, T, ctl.n_groups, -1)
            return (torch.einsum("bten,enm->btem", hp, g).float() * hp.float()).sum(-1)  # ||y_e||^2, [B, T, E]
        r = ctl.routers[self.layer]
        return r(x.to(r.weight.dtype)).float()

    def best_subset(self, h):
        ctl, m = self.ctl, self.mlp
        B, T, _ = h.shape
        wd = ctl.grouped_down(self.layer, m.down_proj.weight)                         # [E, n, D]
        hp = h[..., ctl.perms[self.layer].to(h.device)].view(B, T, ctl.n_groups, -1)
        ye = torch.einsum("bten,end->bted", hp, wd).float()                           # [B, T, E, D]
        M = ctl.subsets(ctl.k, h.device)                                              # [S, E]
        target = self.post_ln(ye.sum(2))                                             # [B, T, D]
        best_d = torch.full((B, T), float("inf"), device=h.device)
        best_i = torch.zeros((B, T), dtype=torch.long, device=h.device)
        for s0 in range(0, M.shape[0], 8):
            ys = torch.einsum("bted,se->btsd", ye, M[s0:s0 + 8].float())
            d = (self.post_ln(ys) - target.unsqueeze(2)).pow(2).sum(-1)              # [B, T, s]
            v, i = d.min(-1)
            upd = v < best_d
            best_d = torch.where(upd, v, best_d)
            best_i = torch.where(upd, i + s0, best_i)
        return M[best_i]                                                              # [B, T, E]

    def forward(self, x):
        ctl, m = self.ctl, self.mlp
        h = m.act_fn(m.gate_proj(x)) * m.up_proj(x)
        if ctl.hook is not None:
            ctl.hook(self.layer, x, h, ctl)
        if ctl.mode == "dense" or not self.active:
            return m.down_proj(h)
        if ctl.mode == "neuron":
            score = h.abs() * self.wnorm
            keep = max(1, int(round(ctl.frac * h.shape[-1])))
            thr = torch.topk(score, keep, dim=-1).values[..., -1:]
            mask = score >= thr
        elif ctl.mode == "best":
            sel = self.best_subset(h)
            if ctl.usage is not None and ctl.valid is not None:
                ctl.usage[self.layer] += sel[ctl.valid].double().sum(0).cpu()
            mask = sel[..., ctl.gids[self.layer].to(h.device)]
        else:
            score = self.expert_scores(x, h)
            if ctl.shared:
                score[..., : ctl.shared] = float("inf")
            top = torch.topk(score, ctl.shared + ctl.k, dim=-1).indices
            sel = torch.zeros_like(score, dtype=torch.bool).scatter_(-1, top, True)       # [B, T, E]
            if ctl.usage is not None and ctl.valid is not None:
                ctl.usage[self.layer] += sel[ctl.valid].double().sum(0).cpu()
            mask = sel[..., ctl.gids[self.layer].to(h.device)]                            # [B, T, N]
        if ctl.scope != "all":
            img = ctl.is_image.unsqueeze(-1)
            mask = mask | (~img if ctl.scope == "image" else img)
        y = m.down_proj(h * mask)
        if ctl.collect:
            dense_y = m.down_proj(h)
            err = (y.float() - dense_y.float()).pow(2).sum(-1)
            ref = dense_y.float().pow(2).sum(-1)
            valid = ctl.valid if ctl.valid is not None else torch.ones_like(ctl.is_image)
            for t, msk in ((0, ~ctl.is_image & valid), (1, ctl.is_image & valid)):
                ctl.stats[self.layer, t, 0] += float(err[msk].sum())
                ctl.stats[self.layer, t, 1] += float(ref[msk].sum())
        return y


def install(model):
    lm = model.model.language_model
    ctl = Controller(len(lm.layers))
    for i, layer in enumerate(lm.layers):
        if not isinstance(layer.mlp, SparseMLP):
            layer.mlp = SparseMLP(layer.mlp, i, ctl)
            layer.mlp.post_ln = layer.post_feedforward_layernorm
    return ctl
