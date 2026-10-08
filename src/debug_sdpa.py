"""Which SDPA backend disagrees with eager on an unpadded single sequence?"""
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from common import encode_batch, letter_ids, load
from data import eval_items

items = eval_items(scale=0.03)
watch = ["mmlu-test-6416", "sqa-test-1281", "aok-validation-661", "mmstar-1144", "mmlu-test-2135"]
sub = [it for it in items if it["id"] in watch]
proc, model = load("sdpa")
cand = letter_ids(proc.tokenizer)


def one(it):
    enc = encode_batch(proc, [it]).to("cuda")
    with torch.no_grad():
        lg = model(**enc, logits_to_keep=1).logits[0, -1].float()
    return np.round(torch.softmax(lg[cand[:len(it["options"])]], -1).cpu().numpy(), 3)


print("torch", torch.__version__, "cudnn sdp", torch.backends.cuda.cudnn_sdp_enabled())
for name, backs in (("default", None), ("flash", [SDPBackend.FLASH_ATTENTION]), ("efficient", [SDPBackend.EFFICIENT_ATTENTION]),
                    ("cudnn", [SDPBackend.CUDNN_ATTENTION]), ("math", [SDPBackend.MATH]),
                    ("no-cudnn", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])):
    try:
        if backs is None:
            res = {it["id"]: one(it) for it in sub}
        else:
            with sdpa_kernel(backs):
                res = {it["id"]: one(it) for it in sub}
        print(name, {k: v.tolist() for k, v in res.items()})
    except Exception as e:
        print(name, "ERROR", str(e).split("\n")[0][:150])
