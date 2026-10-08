"""Check that the stored eval features (left-padded batches of 8) match a batch-1 recomputation (no padding)."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx_core import LAYERS, WH, Store, load_e2b_jev, log_jsonl
from common import encode_batch
from data import _aokvqa, _mmlu, _sqa


def main():
    st = Store("eval")
    items = [dict(it, image=it["image"].resize(WH)) for it in _sqa("test", 12, 0) + _aokvqa("validation", 12, 0)] + _mmlu("test", 12, 0)
    pe, me = load_e2b_jev()
    feats = {}
    for L in LAYERS:
        me.model.language_model.layers[L].register_forward_hook(lambda m, i, o, L=L: feats.__setitem__(L, o))
    errs = []
    with torch.no_grad():
        for it in items:
            j = st.by_id[it["id"]]
            mt = st.meta[j]
            enc = encode_batch(pe, [it], 70).to("cuda")
            me(**enc, logits_to_keep=1)
            X = torch.cat([feats[L] for L in LAYERS], -1)[0, mt["h"]: mt["h"] + mt["m"]].float()
            S = st.x(j).float().cuda()
            errs.append(float((X - S).norm() / X.norm()))
    rec = dict(kind="feat_check", n=len(errs), max_rel_err=round(max(errs), 5), mean_rel_err=round(sum(errs) / len(errs), 5))
    print(json.dumps(rec), flush=True)
    log_jsonl("tests.jsonl", rec)


if __name__ == "__main__":
    main()
