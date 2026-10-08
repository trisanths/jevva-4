"""Marlin 4-bit experts vs the Triton 4-bit experts on the same quantized weights (one layer of the 26B)."""
import json
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from types import SimpleNamespace

from jev_fast26 import FusedInt4Experts, FusedMarlinExperts

MID = "google/gemma-4-26B-A4B-it"
idx = json.load(open(hf_hub_download(MID, "model.safetensors.index.json")))["weight_map"]
def get(name):
    with safe_open(hf_hub_download(MID, idx[name]), "pt") as f:
        return f.get_tensor(name)
L = 3
ex = SimpleNamespace(gate_up_proj=torch.nn.Parameter(get(f"model.language_model.layers.{L}.experts.gate_up_proj"), requires_grad=False),
                     down_proj=torch.nn.Parameter(get(f"model.language_model.layers.{L}.experts.down_proj"), requires_grad=False))
tri, mar = FusedInt4Experts(ex), FusedMarlinExperts(ex)
torch.manual_seed(0)
for T in (1, 64, 336):
    h = (torch.randn(T, 2816, device="cuda") * 0.5).to(torch.bfloat16)
    idx_ = torch.stack([torch.randperm(128, device="cuda")[:8] for _ in range(T)])
    w = torch.softmax(torch.randn(T, 8, device="cuda"), -1)
    with torch.no_grad():
        a, b = tri(h, idx_, w).float(), mar(h, idx_, w).float()
    print(T, "rel err", float((a - b).norm() / a.norm()), "max abs", float((a - b).abs().max()), "ref norm", float(a.norm()))
