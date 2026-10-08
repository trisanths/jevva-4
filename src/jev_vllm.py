"""Jev-style decisions on Gemma 4 26B-A4B through vLLM (fused MoE kernels, CUDA graphs).

One request = image + lettered question; the engine emits exactly one token, restricted to the offered letters,
with logprobs_mode=processed_logprobs so the returned logprobs are the softmax over those letters only.

  python jev_vllm.py acc <model> [budget]            accuracy on the 3,000 image eval items
  python jev_vllm.py lat <model> [budget] [n]        batch-1 latency, base64 640x360 JPEG per request (like
                                                     Overshoot's audit): Jev (1 token) vs 8 generated tokens
Results append to results/jev/vllm.jsonl.
"""
import base64
import io
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from data import LETTERS, _aokvqa, eval_items, prompt_text  # noqa: E402

ROOT = os.environ.get("MOEFY_ROOT", "/mnt/nvme/moefy")
OUT = os.path.join(ROOT, "results", "jev", "vllm.jsonl")


def engine(model, budget):
    """JEV_ENC=1 turns on vLLM's compiled vision encoder captured as a CUDA graph (both off by default)."""
    from vllm import LLM
    kw = {}
    if os.environ.get("JEV_ENC"):
        kw["compilation_config"] = {"compile_mm_encoder": True, "cudagraph_mm_encoder": True,
                                    "encoder_cudagraph_max_vision_items_per_batch": 1}
    return LLM(model=model, max_model_len=1024, limit_mm_per_prompt={"image": 1, "video": 0},
               mm_processor_kwargs={"max_soft_tokens": budget}, gpu_memory_utilization=0.88,
               enable_prefix_caching=False, logprobs_mode="processed_logprobs", max_num_seqs=16, seed=0, **kw)


def letter_token_ids(llm):
    tok = llm.get_tokenizer()
    return [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]


def jpeg_uri(img, wh=None, q=90):
    if wh:
        img = img.resize(wh)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def messages(item, uri):
    content = [{"type": "image_url", "image_url": {"url": uri}}] if uri else []
    content.append({"type": "text", "text": prompt_text(item)})
    return [{"role": "user", "content": content}]


def jev_params(cand, k):
    from vllm import SamplingParams
    return SamplingParams(max_tokens=1, temperature=0.0, allowed_token_ids=cand[:k], logprobs=k)


def probs_from(out, cand, k):
    lp = out.outputs[0].logprobs[0]            # {token_id: Logprob}
    z = [lp[t].logprob if t in lp else -1e9 for t in cand[:k]]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def acc(model, budget):
    llm = engine(model, budget)
    cand = letter_token_ids(llm)
    items = [it for it in eval_items() if it["image"] is not None]
    t0 = time.time()
    probs = {}
    for s in range(0, len(items), 64):
        chunk = items[s:s + 64]
        convs = [messages(it, jpeg_uri(it["image"])) for it in chunk]
        params = [jev_params(cand, len(it["options"])) for it in chunk]
        outs = llm.chat(convs, params, use_tqdm=False)
        for it, o in zip(chunk, outs):
            probs[it["id"]] = probs_from(o, cand, len(it["options"]))
    by = {}
    for it in items:
        p = probs[it["id"]]
        by.setdefault(it["source"], []).append((max(range(len(p)), key=p.__getitem__) == it["answer"], -math.log(max(p[it["answer"]], 1e-12))))
    res = {k: round(100 * sum(a for a, _ in v) / len(v), 2) for k, v in by.items()}
    allv = [x for v in by.values() for x in v]
    rec = dict(kind="acc", model=model, budget=budget, enc_graph=bool(os.environ.get("JEV_ENC")), image_acc=round(100 * sum(a for a, _ in allv) / len(allv), 2),
               nll=round(sum(n for _, n in allv) / len(allv), 3), per_source=res, n=len(allv), seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    json.dump(dict(rec, probs=probs), open(os.path.join(ROOT, "results", "jev", f"vllm-{model.split('/')[-1]}@{budget}.json"), "w"))


def lat_text(model, budget, n=60):
    """Same Jev request without the image: LM prefill over the text + engine overhead only."""
    llm = engine(model, budget)
    cand = letter_token_ids(llm)
    items = _aokvqa("validation", n + 10, 21)
    for it in items[:5]:
        llm.chat([messages(it, None)], [jev_params(cand, len(it["options"]))], use_tqdm=False)
    ts = []
    for it in items[10:10 + n]:
        t0 = time.perf_counter()
        llm.chat([messages(it, None)], [jev_params(cand, len(it["options"]))], use_tqdm=False)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    res = dict(kind="lat_text", model=model, n=n, jev_1tok=dict(p50=round(ts[len(ts) // 2], 2), p90=round(ts[int(len(ts) * 0.9)], 2)))
    print(json.dumps(res), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(res) + "\n")


def lat(model, budget, n=60):
    from vllm import SamplingParams
    llm = engine(model, budget)
    cand = letter_token_ids(llm)
    items = _aokvqa("validation", n + 10, 21)
    uris = [jpeg_uri(it["image"], (640, 360)) for it in items]
    gen8 = SamplingParams(max_tokens=8, min_tokens=8, temperature=0.0, ignore_eos=True)
    res = dict(kind="lat", model=model, budget=budget, n=n, enc_graph=bool(os.environ.get("JEV_ENC")))
    for name, mk in (("jev_1tok", lambda it: jev_params(cand, len(it["options"]))), ("gen_8tok", lambda it: gen8)):
        for it, u in zip(items[:5], uris[:5]):                      # warmup
            llm.chat([messages(it, u)], [mk(it)], use_tqdm=False)
        ts = []
        for it, u in zip(items[10:10 + n], uris[10:10 + n]):
            t0 = time.perf_counter()
            llm.chat([messages(it, u)], [mk(it)], use_tqdm=False)
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        res[name] = dict(p50=round(ts[len(ts) // 2], 2), p90=round(ts[int(len(ts) * 0.9)], 2), mean=round(statistics.mean(ts), 2))
        print(name, res[name], flush=True)
    print(json.dumps(res), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "acc":
        acc(a[1], int(a[2]) if len(a) > 2 else 280)
    elif a[0] == "lat_text":
        lat_text(a[1], int(a[2]) if len(a) > 2 else 280)
    else:
        lat(a[1], int(a[2]) if len(a) > 2 else 280, int(a[3]) if len(a) > 3 else 60)
