"""Load Gemma 4 12B, score one image MCQ by option-letter logits, print the prompt tail and timings."""
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

MID = "google/gemma-4-12B-it"
proc = AutoProcessor.from_pretrained(MID)
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda").eval()
print("load s", round(time.time() - t0, 1), "gpu GB", round(torch.cuda.memory_allocated() / 1e9, 2))
print(type(model).__name__)
print(model.model.language_model.layers[0].mlp)

tok = proc.tokenizer
for s in ["A", "B", "C", "D", " A", "(A)"]:
    print(repr(s), tok.encode(s, add_special_tokens=False))

ds = load_dataset("derek-thomas/ScienceQA", split="validation", streaming=True)
ex = next(x for x in ds if x["image"] is not None)
letters = "ABCDE"[: len(ex["choices"])]
q = ex["question"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(letters, ex["choices"]))
q += "\nAnswer with the option's letter from the given choices directly."
msgs = [{"role": "user", "content": [{"type": "image", "image": ex["image"]}, {"type": "text", "text": q}]}]
inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
ids = inputs["input_ids"][0]
print("seq len", ids.shape[0], "image tokens", int((ids == model.config.image_token_id).sum()))
print("tail:", repr(tok.decode(ids[-40:])))
cand = [tok.encode(l, add_special_tokens=False)[0] for l in letters]
with torch.no_grad():
    for _ in range(2):
        torch.cuda.synchronize(); t0 = time.time()
        out = model(**inputs, logits_to_keep=1)
        torch.cuda.synchronize(); dt = time.time() - t0
    logits = out.logits[0, -1].float()
    p = torch.softmax(logits[cand], -1)
    top = torch.topk(torch.softmax(logits, -1), 5)
print("forward s", round(dt, 3))
print("answer", ex["answer"], "probs", [round(x, 3) for x in p.tolist()])
print("top-5 vocab", [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(top.values, top.indices)])
