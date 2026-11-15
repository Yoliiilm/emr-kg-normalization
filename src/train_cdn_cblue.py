"""Supervised CHIP-CDN baseline: port of the official CBLUE pipeline (third_party/CBLUE).

Same algorithm as cblue/data/data_process.py::CDNDataProcessor + CDNForCLSTrainer/CDNForNUMTrainer
+ metrics/cblue_commit.py::cdn_commit_prediction:
  1. recall top-200 ICD names by jieba TF-IDF cosine + character overlap (|x∩y| / min(|x|,|y|));
  2. rank: RoBERTa sentence-pair classifier, 5 hard + 5 random negatives per mention;
  3. num: RoBERTa 3-way classifier for the number of answers (1 / 2 / ≥3);
  4. keep candidates with p≥0.5, sort by (recall/2 + p)/2, take num (raised by 及/伴/; counts).

Differences from the official script, for this environment and a leak-free dev evaluation:
  - sklearn TF-IDF instead of gensim; ICD names from data/dicts/icd10_beijing_v601.json;
  - candidate labels = ICD + train labels only (official also adds dev labels);
  - checkpoints are selected on a 500-mention split of train, not on dev.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import jieba
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "dicts" / "raw"
ICD = ROOT / "data" / "dicts" / "icd10_beijing_v601.json"
MODEL_DIR = ROOT / "outputs" / "models" / "cdn_cblue"
OUT = ROOT / "outputs" / "metrics"
ENCODER = "hfl/chinese-roberta-wwm-ext"

jieba.setLogLevel(60)


def str_q2b(s: str) -> str:
    out = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000:
            code = 0x20
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        out.append(chr(code))
    return "".join(out)


def process_text(s: str) -> str:
    s = str_q2b(s).strip('"')
    for a in ("\\", ",", "、", "?", ":", ".", "/"):
        s = s.replace(a, ";")
    return s.replace("~", "-")


def process_label(s: str) -> list[str]:
    return [x for x in str_q2b(s).strip('"').split("##") if x]


def load_split(name: str) -> list[tuple[str, str, list[str]]]:
    rows = json.loads((RAW / f"CHIP-CDN_{name}.json").read_text(encoding="utf-8"))
    out = []
    for r in rows:
        raw = str(r.get("text", "")).lstrip("\ufeff").strip()
        if raw:
            out.append((raw, process_text(raw), process_label(str(r.get("normalized_result", "")))))
    return out


class Recaller:
    def __init__(self, labels: list[str]):
        self.labels = labels
        self.tfidf = TfidfVectorizer(tokenizer=jieba.lcut, lowercase=False, token_pattern=None, smooth_idf=False)
        self.label_tfidf = self.tfidf.fit_transform(labels)
        self.chars = CountVectorizer(analyzer="char", lowercase=False, binary=True)
        self.label_chars = self.chars.fit_transform(labels).T.tocsr()
        self.label_len = np.asarray(self.label_chars.sum(axis=0)).ravel()

    def recall(self, texts: list[str], k: int = 200, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
        idx_out = np.zeros((len(texts), k), dtype=np.int64)
        score_out = np.zeros((len(texts), k))
        for s in range(0, len(texts), chunk):
            part = texts[s : s + chunk]
            sim = (self.tfidf.transform(part) @ self.label_tfidf.T).toarray()
            xc = self.chars.transform(part)
            inter = (xc @ self.label_chars).toarray()
            xlen = np.asarray(xc.sum(axis=1)).ravel()
            denom = np.minimum(xlen[:, None], self.label_len[None, :])
            dice = np.divide(inter, denom, out=np.zeros_like(inter, dtype=float), where=denom > 0)
            scores = sim + dice
            top = np.argpartition(-scores, k, axis=1)[:, :k]
            top_scores = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_scores, axis=1)
            idx_out[s : s + chunk] = np.take_along_axis(top, order, axis=1)
            score_out[s : s + chunk] = np.take_along_axis(top_scores, order, axis=1)
        return idx_out, score_out


def cls_pairs(data, recalled, label2id, id2label, n_neg: int, rng: random.Random):
    pairs = []
    for (_, text, golds), cands in zip(data, recalled):
        gold_ids = {label2id[g] for g in golds if g in label2id}
        for g in golds:
            pairs.append((text, g, 1))
        hard = [c for c in cands if c not in gold_ids]
        for c in hard[:n_neg]:
            pairs.append((text, id2label[c], 0))
        rest = hard[n_neg:]
        rng.shuffle(rest)
        for c in rest[:n_neg]:
            pairs.append((text, id2label[c], 0))
    return pairs


def num_label(golds: list[str]) -> int:
    return min(len(golds), 3) - 1


def train_model(
    name: str,
    examples: list[tuple],
    val_fn,
    num_labels: int,
    epochs: int,
    lr: float,
    batch_size: int,
    max_len: int,
    tok,
    device,
    eval_every: int | None = None,
):
    model = AutoModelForSequenceClassification.from_pretrained(ENCODER, num_labels=num_labels).to(device)
    loader = DataLoader(examples, batch_size=batch_size, shuffle=True, collate_fn=lambda b: b)
    steps = epochs * len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, best_state, step = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for batch in loader:
            if len(batch[0]) == 3:
                enc = tok([b[0] for b in batch], [b[1] for b in batch], truncation=True, max_length=max_len,
                          padding=True, return_tensors="pt")
            else:
                enc = tok([b[0] for b in batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            y = torch.tensor([b[-1] for b in batch], device=device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(**enc, labels=y).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if eval_every and step % eval_every == 0:
                score = val_fn(model)
                if score > best:
                    best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        score = val_fn(model)
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{name}] epoch {ep + 1}/{epochs} val={score:.4f} best={best:.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model, best


@torch.no_grad()
def score_pairs(model, tok, texts: list[str], cands: np.ndarray, id2label, device, max_len: int, bs: int = 1024):
    """Positive-class probability for every (text, recalled candidate)."""
    model.eval()
    flat_t = [t for t, row in zip(texts, cands) for _ in row]
    flat_c = [id2label[c] for row in cands for c in row]
    probs = np.zeros(len(flat_t), dtype=np.float32)
    for s in range(0, len(flat_t), bs):
        enc = tok(flat_t[s : s + bs], flat_c[s : s + bs], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**enc).logits.float()
        probs[s : s + bs] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probs.reshape(cands.shape)


@torch.no_grad()
def predict_num(model, tok, texts: list[str], device, max_len: int, bs: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s : s + bs], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(**enc).logits.argmax(-1).cpu().numpy())
    return np.concatenate(out)


def commit(text: str, probs: np.ndarray, num: int, cands: np.ndarray, rscores: np.ndarray, id2label) -> list[str]:
    """cblue_commit.cdn_commit_prediction, unchanged logic."""
    active = probs >= 0.5
    if not active.any():
        return [text]
    p, r, c = probs[active], rscores[active], cands[active]
    c = c[np.argsort((r / 2 + p) / 2)[::-1]]
    num = num + 1
    extra = text.count("及") + text.count("伴") + text.count(";") + 1
    num = max(num, extra)
    if num == 1:
        chosen = c[:1]
    elif num == 2:
        chosen = c[:2]
    else:
        chosen = c[: max(extra, num, 3)]
    return [id2label[i] for i in chosen]


def set_prf(preds: list[list[str]], golds: list[list[str]]) -> tuple[float, float, float]:
    tp = n_p = n_g = 0
    for p, g in zip(preds, golds):
        ps = {re.sub(r"\s+", "", x) for x in p}
        gs = {re.sub(r"\s+", "", x) for x in g}
        tp += len(ps & gs)
        n_p += len(ps)
        n_g += len(gs)
    pr = tp / n_p if n_p else 0.0
    rc = tp / n_g if n_g else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recall-k", type=int, default=200)
    ap.add_argument("--num-neg", type=int, default=5)
    ap.add_argument("--cls-epochs", type=int, default=3)
    ap.add_argument("--num-epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2021)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_all = load_split("train")
    dev = load_split("dev")
    rng.shuffle(train_all)
    val, train = train_all[: args.n_val], train_all[args.n_val :]

    icd_names = list(json.loads(ICD.read_text(encoding="utf-8"))["name_to_codes"].keys())
    labels = list(dict.fromkeys([str_q2b(n) for n in icd_names] + [g for _, _, gs in train_all for g in gs]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = labels
    print(f"labels={len(labels)} train={len(train)} val={len(val)} dev={len(dev)} device={device}", flush=True)

    t0 = time.time()
    rec = Recaller(labels)
    r_train, _ = rec.recall([t for _, t, _ in train], args.recall_k)
    r_val, s_val = rec.recall([t for _, t, _ in val], args.recall_k)
    r_dev, s_dev = rec.recall([t for _, t, _ in dev], args.recall_k)

    def recall_hit(data, cands):
        hit = sum(all(label2id.get(g, -1) in set(row) for g in gs) for (_, _, gs), row in zip(data, cands))
        return hit / len(data)

    print(f"recall@{args.recall_k}: val={recall_hit(val, r_val):.4f} dev={recall_hit(dev, r_dev):.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    tok = AutoTokenizer.from_pretrained(ENCODER)
    pairs = cls_pairs(train, r_train, label2id, id2label, args.num_neg, rng)
    print(f"cls pairs={len(pairs)}", flush=True)

    val_texts = [t for _, t, _ in val]
    val_golds = [gs for _, _, gs in val]

    def val_cls(model):
        probs = score_pairs(model, tok, val_texts, r_val, id2label, device, args.max_len)
        preds = [commit(t, p, len(g) - 1 if len(g) < 3 else 2, c, s, id2label)
                 for t, p, g, c, s in zip(val_texts, probs, val_golds, r_val, s_val)]
        return set_prf(preds, val_golds)[2]

    cls_model, cls_best = train_model("cls", pairs, val_cls, 2, args.cls_epochs, args.lr, args.batch_size,
                                      args.max_len, tok, device, eval_every=max(1, len(pairs) // args.batch_size // 2))

    num_examples = [(t, num_label(gs)) for _, t, gs in train]
    val_num_y = np.array([num_label(gs) for gs in val_golds])

    def val_num(model):
        return float((predict_num(model, tok, val_texts, device, args.max_len) == val_num_y).mean())

    num_model, num_best = train_model("num", num_examples, val_num, 3, args.num_epochs, args.lr, args.batch_size,
                                      args.max_len, tok, device)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(cls_model.state_dict(), MODEL_DIR / "cls.pt")
    torch.save(num_model.state_dict(), MODEL_DIR / "num.pt")

    dev_texts = [t for _, t, _ in dev]
    probs = score_pairs(cls_model, tok, dev_texts, r_dev, id2label, device, args.max_len)
    nums = predict_num(num_model, tok, dev_texts, device, args.max_len)
    preds = [commit(t, p, int(n), c, s, id2label) for t, p, n, c, s in zip(dev_texts, probs, nums, r_dev, s_dev)]

    golds_all = [gs for _, _, gs in dev]
    golds_no_ca = [[g for g in gs if g != "癌"] for gs in golds_all]
    p_all, r_all, f_all = set_prf(preds, golds_all)
    p, r, f = set_prf([[x for x in pr if x != "癌"] for pr in preds], golds_no_ca)
    strict1 = sum(bool(pr) and pr[0] in set(g) for pr, g in zip(preds, golds_no_ca)) / len(dev)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cdn_supervised_dev_preds.json").write_text(
        json.dumps([{"text": raw, "pred": pr} for (raw, _, _), pr in zip(dev, preds)], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    report = {
        "method": "CBLUE CDN baseline (recall + RoBERTa-wwm-ext rank + num), ported",
        "encoder": ENCODER,
        "args": vars(args),
        "n_labels": len(labels),
        "recall_at_k_dev": recall_hit(dev, r_dev),
        "val_cls_set_f1": cls_best,
        "val_num_acc": num_best,
        "dev_set_p": p, "dev_set_r": r, "dev_set_f1": f,
        "dev_set_f1_with_ca_label": f_all,
        "dev_acc_strict_top1": strict1,
    }
    (OUT / "cdn_supervised.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
