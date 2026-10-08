"""Inspect the prompt layout shared by E2B and 26B: header, image block, user text, suffix (CPU only)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: F401  (mask patch)
from transformers import AutoProcessor

from common import build, encode_batch
from data import _aokvqa, _mmlu, _sqa

p26 = AutoProcessor.from_pretrained("google/gemma-4-26B-A4B-it", padding_side="left")
pe2 = AutoProcessor.from_pretrained("google/gemma-4-E2B-it", padding_side="left")
tok = p26.tokenizer
items = _aokvqa("validation", 4, 0) + _sqa("test", 4, 0)
items = [dict(it, image=it["image"].resize((640, 480))) for it in items] + _mmlu("test", 2, 0)
print(repr(p26.apply_chat_template(build(p26, items[0]), add_generation_prompt=True, tokenize=False)))
print(repr(p26.apply_chat_template(build(p26, items[-1]), add_generation_prompt=True, tokenize=False)))
print("same template:", p26.chat_template == pe2.chat_template)
for it in items:
    for budget in (70, 280):
        a = encode_batch(p26, [it], budget)
        b = encode_batch(pe2, [it], budget)
        same = a["input_ids"].shape == b["input_ids"].shape and bool((a["input_ids"] == b["input_ids"]).all())
        ids = a["input_ids"][0].tolist()
        n_img = sum(1 for t in ids if t == 258880)
        print(it["id"], budget, "len", len(ids), "img", n_img, "same", same, "keys", sorted(a.keys()))
    if it["image"] is not None:
        a = encode_batch(p26, [it], 70)
        ids = a["input_ids"][0].tolist()
        mm = a.get("mm_token_type_ids")
        print("head:", [(t, tok.decode([t])) for t in ids[:6]])
        first_img = ids.index(258880)
        print("around img start:", [(t, tok.decode([t])) for t in ids[first_img - 2:first_img + 1]])
        last_img = len(ids) - 1 - ids[::-1].index(258880)
        print("around img end:", [(t, tok.decode([t])) for t in ids[last_img:last_img + 3]])
        print("tail:", [(t, tok.decode([t])) for t in ids[-12:]])
        if mm is not None:
            m = mm[0].tolist()
            print("mm types at boi/img/eoi:", m[first_img - 1], m[first_img], m[last_img], m[last_img + 1], "set", sorted(set(m)))
    else:
        ids = encode_batch(p26, [it], 70)["input_ids"][0].tolist()
        print("text head:", [(t, tok.decode([t])) for t in ids[:6]], "tail:", [(t, tok.decode([t])) for t in ids[-12:]])

for it in items[:1] + items[-1:]:
    a = encode_batch(p26, [it], 70)["input_ids"][0].tolist()
    b = encode_batch(pe2, [it], 70)["input_ids"][0].tolist()
    print(len(a), len(b))
    print("26B tail:", [(t, tok.decode([t])) for t in a[-14:]])
    print("E2B tail:", [(t, tok.decode([t])) for t in b[-14:]])
    print("E2B head:", [(t, tok.decode([t])) for t in b[:6]])
print(repr(pe2.apply_chat_template(build(pe2, items[-1]), add_generation_prompt=True, tokenize=False)))
