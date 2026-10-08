"""Summarise Stage 1/2: FLOPs per decision from measured token counts, calibration after a temperature refit,
paired bootstrap intervals against dense@280 and the FLOP-matched no-training references, and the kill checks.

Runs on CPU. Writes results/analysis.json and prints a table.
"""
import glob
import hashlib
import json
import math
import os

import numpy as np
from transformers import AutoProcessor

from common import ROOT, encode_batch
from data import eval_items

R = os.path.join(ROOT, "results")

# Parameter counts from the safetensors headers (see research notes).
ATTN12, FFN12, EMB12_MM = 2_406_481_920, 8_493_465_600, 41_287_680
ACT26, TOWER26_LIN = 3_083_632_640, 545_070_384 + 884_736
L12 = [("s", 16 * 256)] * 40 + [("g", 16 * 512)] * 8      # (type, H*head_dim) for the 12B, 5 sliding : 1 global
L26 = [("s", 16 * 256)] * 25 + [("g", 16 * 512)] * 5
VH, VL = 1152, 27


def attn_pairs(S, N):
    """(query, key) pairs under the official mask with S < 1024: causal, plus the image block both ways on
    sliding layers."""
    causal = S * (S + 1) / 2
    return {"g": causal, "s": causal + N * (N - 1) / 2}


def flops_12b(S, N, ffn_frac=1.0, router=0):
    lin = 2 * (ATTN12 + ffn_frac * FFN12 + router) * S + 2 * EMB12_MM * N
    p = attn_pairs(S, N)
    att = sum(4 * hd * p[t] for t, hd in L12)
    return lin + att


def flops_26b(S, N, budget):
    P = 9 * budget                           # HF pads the tower input to 9 x budget patches
    tower = 2 * TOWER26_LIN * P + VL * 4 * VH * P * P + 2 * P * N * VH + 2 * VH * 2816 * N
    p = attn_pairs(S, N)
    return tower + 2 * ACT26 * S + sum(4 * hd * p[t] for t, hd in L26)


def ffn_fraction(rec):
    kw = rec.get("kw", {})
    if rec["mode"] == "neuron":
        return kw["frac"]
    if rec["mode"] in ("oracle", "best", "router"):
        E = int(kw["part"].split("-")[1])
        f = (kw.get("shared", 0) + kw["k"]) / E
        lo, hi = kw.get("layers", (0, 48))
        f = (f * (hi - lo) + (48 - (hi - lo))) / 48
        if kw.get("scope") == "image":
            return None                        # mixed; reported without FLOPs
        return f
    return 1.0


def token_counts(items):
    """Measured S and N per item for each budget (same processor logic for both models)."""
    path = os.path.join(R, "token_counts.json")
    if os.path.exists(path):
        return json.load(open(path))
    proc = AutoProcessor.from_pretrained("google/gemma-4-12B-it", padding_side="left")
    out = {}
    for st in (70, 140, 280):
        d = {}
        for it in items:
            enc = encode_batch(proc, [it], st)
            ids = enc["input_ids"][0]
            d[it["id"]] = (int(ids.numel()), int((ids == 258880).sum()))
        out[str(st)] = d
    json.dump(out, open(path, "w"))
    return out


def split_ids(items):
    cal = {it["id"] for it in items if int(hashlib.md5(it["id"].encode()).hexdigest(), 16) % 10 < 3}
    return cal


def logp(p):
    return np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1))


def fit_T(rows):
    best = (1e9, 1.0)
    for T in np.exp(np.linspace(math.log(0.3), math.log(8), 120)):
        nll = 0.0
        for lp, y in rows:
            z = lp / T
            z = z - z.max()
            nll -= z[y] - math.log(np.exp(z).sum())
        best = min(best, (nll / len(rows), T))
    return best[1]


def calib(items, probs, T, n_bins=15):
    nll, brier, conf, corr = [], [], [], []
    for it in items:
        z = logp(probs[it["id"]]) / T
        p = np.exp(z - z.max()); p /= p.sum()
        y = it["answer"]
        nll.append(-math.log(max(p[y], 1e-12)))
        oh = np.zeros_like(p); oh[y] = 1
        brier.append(((p - oh) ** 2).sum())
        conf.append(p.max()); corr.append(float(p.argmax() == y))
    conf, corr = np.array(conf), np.array(corr)
    b = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = sum(abs(conf[b == i].mean() - corr[b == i].mean()) * (b == i).mean() for i in range(n_bins) if (b == i).any())
    return dict(nll=float(np.mean(nll)), brier=float(np.mean(brier)), ece15=float(ece))


def correct_vec(items, probs):
    return np.array([float(int(np.argmax(probs[it["id"]])) == it["answer"]) for it in items])


def boot_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    items = eval_items()
    byid = {it["id"]: it for it in items}
    tc = token_counts(items)
    cal = split_ids(items)
    test = [it for it in items if it["id"] not in cal]
    calr = [it for it in items if it["id"] in cal]
    img = [it for it in items if it["image"] is not None]
    txt = [it for it in items if it["image"] is None]
    summary = json.load(open(os.path.join(R, "stage1", "summary.json")))
    files = {os.path.basename(p)[:-5]: p for p in glob.glob(os.path.join(R, "stage*", "*.json")) if not p.endswith("summary.json")}
    probs = {n: json.load(open(p))["probs"] for n, p in files.items() if n in summary}
    dense = probs["dense@280"]
    rows = {}
    for name, rec in summary.items():
        if name not in probs or name == "dense@280-rev":
            continue
        P = probs[name]
        st = str(rec.get("soft_tokens", 280))
        img_ids = [it["id"] for it in img]
        if name.startswith("blind"):
            fl = fl_img = None
        elif rec["mode"] == "26b":
            per = {i: flops_26b(*tc[st][i], int(st)) for i in byid}
            fl, fl_img = np.mean(list(per.values())) / 1e12, np.mean([per[i] for i in img_ids]) / 1e12
        else:
            f = ffn_fraction(rec)
            router = 0 if rec["mode"] in ("dense", "neuron") else 48 * 3840 * int(rec["kw"]["part"].split("-")[1])
            if f is None:
                fl = fl_img = None
            else:
                per = {i: flops_12b(*tc[st][i], f, router) for i in byid}
                fl, fl_img = np.mean(list(per.values())) / 1e12, np.mean([per[i] for i in img_ids]) / 1e12
        T = fit_T([(logp(P[it["id"]]), it["answer"]) for it in calr])
        a_all, a_img, a_txt = (correct_vec(s, P).mean() for s in (items, img, txt))
        d_all = boot_diff(correct_vec(items, P), correct_vec(items, dense))
        rows[name] = dict(acc=a_all, acc_img=a_img, acc_txt=a_txt, d_vs_dense=d_all, tflop=fl, tflop_img=fl_img, T=T,
                          raw=calib(test, P, 1.0), refit=calib(test, P, T),
                          kl=rec["metrics"]["_all"].get("kl"), agree=rec["metrics"]["_all"].get("agree"))
    # FLOP-matched no-training references (the better of dense 12B@70 and 26B@140 for top-2-class configs).
    def best_of(names):
        c = [n for n in names if n in rows]
        return max(c, key=lambda n: rows[n]["acc"]) if c else None
    R2, R4 = best_of(["dense@70", "26b@140"]), best_of(["dense@140"])
    for name, r in rows.items():
        for ref_name, key in ((R2, "d_vs_R2"), (R4, "d_vs_R4")):
            if ref_name:
                r[key] = boot_diff(correct_vec(items, probs[name]), correct_vec(items, probs[ref_name]))
    # Kill checks from the red-team spec, for the best structured arm at each sparsity class.
    checks = {}
    kl70 = rows.get("dense@70", {}).get("kl")
    for cls, ref, arms in (("25%", R2, ["coact8-k2", "shared8-s1k1", "coact128-k32", "neuron-25"]),
                           ("50%", R4, ["coact8-k4", "best8-k4", "shared8-s1k3", "coact32-k16", "coact128-k64", "neuron-50"])):
        for a in arms:
            if a not in rows or ref is None:
                continue
            r = rows[a]
            key = "d_vs_R2" if cls == "25%" else "d_vs_R4"
            checks[a] = dict(
                K1=bool(r[key][0] >= 0.02 and r[key][1] > 0), d_vs_ref=r[key],
                K2=bool(kl70 is not None and r["kl"] < kl70),
                K4=bool(r["refit"]["nll"] <= rows[ref]["refit"]["nll"] and r["refit"]["ece15"] <= rows["dense@280"]["refit"]["ece15"] + 0.02),
                K5=bool(rows["dense@280"]["acc_txt"] - r["acc_txt"] <= 0.02))
    out = dict(R2=R2, R4=R4, rows=rows, checks=checks, n_items=len(items), n_test=len(test))
    json.dump(out, open(os.path.join(R, "analysis.json"), "w"), indent=1)
    print(f"R2 = {R2}   R4 = {R4}   items {len(items)} (calibration split {len(calr)}, test {len(test)})")
    hdr = f"{'config':28s} {'TFimg':>6s} {'TFLOP':>6s} {'acc':>6s} {'img':>6s} {'txt':>6s} {'d_dense [95% CI]':>22s} {'d_R2':>7s} {'KL':>6s} {'agree':>6s} {'T':>5s} {'NLL':>6s} {'NLLr':>6s} {'ECEr':>6s}"
    print(hdr)
    for name, r in sorted(rows.items(), key=lambda kv: -(kv[1]["tflop"] or 0)):
        d = r["d_vs_dense"]
        print(f"{name:28s} {r['tflop_img'] if r['tflop_img'] is not None else float('nan'):6.2f} {r['tflop'] if r['tflop'] is not None else float('nan'):6.2f} {r['acc']:6.3f} {r['acc_img']:6.3f} {r['acc_txt']:6.3f} "
              f"{d[0]*100:+6.1f} [{d[1]*100:+5.1f},{d[2]*100:+5.1f}] {r.get('d_vs_R2', (float('nan'),))[0]*100:+7.1f} "
              f"{(r['kl'] or 0):6.3f} {(r['agree'] or 1):6.3f} {r['T']:5.2f} {r['raw']['nll']:6.3f} {r['refit']['nll']:6.3f} {r['refit']['ece15']:6.3f}")
    print("kill checks (K1 acc >= ref + 2pp with CI > 0; K2 KL < KL(dense@70); K4 NLL/ECE after refit; K5 text guard):")
    for a, c in checks.items():
        print(f"  {a:16s} K1 {c['K1']!s:5s} (d_ref {c['d_vs_ref'][0]*100:+.1f} [{c['d_vs_ref'][1]*100:+.1f},{c['d_vs_ref'][2]*100:+.1f}])  K2 {c['K2']!s:5s} K4 {c['K4']!s:5s} K5 {c['K5']!s:5s}")


if __name__ == "__main__":
    main()
