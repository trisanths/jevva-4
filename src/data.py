"""Multiple-choice items for Jev-style scoring, plus a disjoint profiling pool.

Every item is a dict: id, source, image (PIL or None), question, options (list of str), answer (int).
Eval items come from test/validation splits; profiling items from train splits only.
"""
import random
import re

from datasets import load_dataset

LETTERS = "ABCDEFGH"


def _sqa(split, n, seed):
    ds = load_dataset("derek-thomas/ScienceQA", split=split)
    items = []
    rows = list(range(len(ds)))
    random.Random(seed).shuffle(rows)
    for i in rows:
        ex = ds[i]
        if ex["image"] is None:
            continue
        q = ex["question"]
        if ex.get("hint"):
            q = f"Context: {ex['hint']}\n{q}"
        items.append(dict(id=f"sqa-{split}-{i}", source="scienceqa", image=ex["image"].convert("RGB"),
                          question=q, options=list(ex["choices"]), answer=int(ex["answer"])))
        if len(items) >= n:
            break
    return items


def _aokvqa(split, n, seed):
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split=split)
    rows = list(range(len(ds)))
    random.Random(seed).shuffle(rows)
    items = []
    for i in rows[:n]:
        ex = ds[i]
        items.append(dict(id=f"aok-{split}-{i}", source="aokvqa", image=ex["image"].convert("RGB"),
                          question=ex["question"], options=list(ex["choices"]), answer=int(ex["correct_choice_idx"])))
    return items


def _ai2d(n, seed):
    ds = load_dataset("lmms-lab/ai2d", split="test")
    rows = list(range(len(ds)))
    random.Random(seed).shuffle(rows)
    items = []
    for i in rows[:n]:
        ex = ds[i]
        items.append(dict(id=f"ai2d-{i}", source="ai2d", image=ex["image"].convert("RGB"),
                          question=ex["question"], options=list(ex["options"]), answer=int(ex["answer"])))
    return items


def _mmstar(n, seed):
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    rows = list(range(len(ds)))
    random.Random(seed).shuffle(rows)
    items = []
    for i in rows:
        ex = ds[i]
        # MMStar folds the options into the question text: "...\nOptions: A: x, B: y, C: z, D: w"
        m = re.split(r"\n?Options:\s*", ex["question"], maxsplit=1)
        if len(m) != 2:
            continue
        stem, opts = m
        parts = re.split(r"(?:^|,\s*)([A-D])[:.]\s*", opts.strip())
        found = {parts[j]: parts[j + 1].strip().rstrip(",") for j in range(1, len(parts) - 1, 2)}
        if sorted(found) != ["A", "B", "C", "D"] or ex["answer"] not in found:
            continue
        items.append(dict(id=f"mmstar-{i}", source="mmstar", image=ex["image"].convert("RGB"),
                          question=stem.strip(), options=[found[l] for l in "ABCD"], answer="ABCD".index(ex["answer"])))
        if len(items) >= n:
            break
    return items


def _mmlu(split, n, seed):
    ds = load_dataset("cais/mmlu", "all", split=split)
    rows = list(range(len(ds)))
    random.Random(seed).shuffle(rows)
    return [dict(id=f"mmlu-{split}-{i}", source="mmlu", image=None, question=ds[i]["question"],
                 options=list(ds[i]["choices"]), answer=int(ds[i]["answer"])) for i in rows[:n]]


def _arc(split, n, seed):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    rows = list(range(len(ds)))
    random.Random(seed).shuffle(rows)
    items = []
    for i in rows:
        ex = ds[i]
        labels = ex["choices"]["label"]
        if ex["answerKey"] not in labels:
            continue
        items.append(dict(id=f"arc-{split}-{i}", source="arc", image=None, question=ex["question"],
                          options=list(ex["choices"]["text"]), answer=labels.index(ex["answerKey"])))
        if len(items) >= n:
            break
    return items


def eval_items(scale=1.0, seed=0):
    s = lambda k: max(1, int(k * scale))
    return (_sqa("test", s(800), seed) + _aokvqa("validation", s(800), seed) + _ai2d(s(800), seed)
            + _mmstar(s(600), seed) + _mmlu("test", s(800), seed) + _arc("test", s(600), seed))


def profile_items(n_image=600, n_text=300, seed=1):
    """Train-split items for activation profiling and router training; disjoint from eval_items."""
    half = n_image // 2
    return (_sqa("train", half, seed) + _aokvqa("train", n_image - half, seed)
            + _mmlu("auxiliary_train", n_text // 2, seed) + _arc("train", n_text - n_text // 2, seed))


def prompt_text(item):
    letters = LETTERS[: len(item["options"])]
    body = item["question"].strip() + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(letters, item["options"]))
    return body + "\nAnswer with the option's letter from the given choices directly."


def _csqa(split, n, seed):
    ds = load_dataset("tau/commonsense_qa", split=split)
    rows = list(range(len(ds)))
    random.Random(seed).shuffle(rows)
    items = []
    for i in rows:
        ex = ds[i]
        labels = ex["choices"]["label"]
        if ex["answerKey"] not in labels:
            continue
        items.append(dict(id=f"csqa-{split}-{i}", source="csqa", image=None, question=ex["question"],
                          options=list(ex["choices"]["text"]), answer=labels.index(ex["answerKey"])))
        if len(items) >= n:
            break
    return items


class LazyImageSet:
    """Train-split image MC items that decode their image only when fetched (keeps RAM low)."""

    def __init__(self, seed=11, n_aok=17056, n_sqa=6218):
        self.aok = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
        self.sqa = load_dataset("derek-thomas/ScienceQA", split="train")
        rng = random.Random(seed)
        a = list(range(len(self.aok)))[:n_aok]
        # ScienceQA rows without images are skipped cheaply by checking the image column lazily at fetch time.
        s = list(range(len(self.sqa)))
        self.keys = [("aok", i) for i in a] + [("sqa", i) for i in s]
        rng.shuffle(self.keys)
        self.n_sqa = n_sqa

    def __len__(self):
        return len(self.keys)

    def get(self, k):
        src, i = k
        if src == "aok":
            ex = self.aok[i]
            return dict(id=f"aok-train-{i}", source="aokvqa", image=ex["image"].convert("RGB"), question=ex["question"],
                        options=list(ex["choices"]), answer=int(ex["correct_choice_idx"]))
        ex = self.sqa[i]
        if ex["image"] is None:
            return None
        q = ex["question"] if not ex.get("hint") else f"Context: {ex['hint']}\n{ex['question']}"
        return dict(id=f"sqa-train-{i}", source="scienceqa", image=ex["image"].convert("RGB"), question=q,
                    options=list(ex["choices"]), answer=int(ex["answer"]))


def text_train_items(seed=12):
    return (_mmlu("auxiliary_train", 8000, seed) + _arc("train", 1119, seed)
            + [dict(it, id=it["id"].replace("arc-", "arce-")) for it in _arc_easy("train", 2251, seed)]
            + _csqa("train", 4000, seed))


def _arc_easy(split, n, seed):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    rows = list(range(len(ds)))
    random.Random(seed).shuffle(rows)
    items = []
    for i in rows:
        ex = ds[i]
        labels = ex["choices"]["label"]
        if ex["answerKey"] not in labels:
            continue
        items.append(dict(id=f"arc-{split}-{i}", source="arc", image=None, question=ex["question"],
                          options=list(ex["choices"]["text"]), answer=labels.index(ex["answerKey"])))
        if len(items) >= n:
            break
    return items
