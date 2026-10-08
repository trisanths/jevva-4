"""Does PyTorch's memory-efficient SDPA kernel disagree with the math backend at Gemma 4 12B's attention shapes?
Compares each backend with an fp32 reference; prints the max absolute error."""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(0)


def run(hd, hq, hkv, S, causal, scale, expand):
    q = torch.randn(1, hq, S, hd, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, hkv, S, hd, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, hkv, S, hd, device="cuda", dtype=torch.bfloat16)
    kf, vf = k.float().repeat_interleave(hq // hkv, 1), v.float().repeat_interleave(hq // hkv, 1)
    mask = torch.full((S, S), float("-inf"), device="cuda").triu(1) if causal else 0
    ref = torch.softmax(q.float() @ kf.transpose(-1, -2) * scale + mask, -1) @ vf
    if expand:
        k, v = k.repeat_interleave(hq // hkv, 1), v.repeat_interleave(hq // hkv, 1)
    kw = dict(is_causal=causal, scale=scale, enable_gqa=not expand)
    out = {}
    for name, b in (("math", SDPBackend.MATH), ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
        try:
            with sdpa_kernel([b]):
                out[name] = round(float((F.scaled_dot_product_attention(q, k, v, **kw).float() - ref).abs().max()), 4)
        except Exception as e:
            out[name] = str(e).split("\n")[0][:60]
    return out


print(torch.__version__, torch.cuda.get_device_name())
for hd, hkv in ((256, 8), (512, 1)):
    for causal in (True, False):
        for expand in (True, False):
            print(f"head_dim {hd} kv_heads {hkv} causal {causal} kv expanded {expand}:", run(hd, 16, hkv, 320, causal, 1.0, expand))
