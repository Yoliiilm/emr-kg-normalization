"""SapBERT-style contrastive fine-tuning of the ICD fallback retriever (Liu et al. 2021).

Positive pairs are CHIP-CDN train mentions and their ICD standard names. Each query is contrasted
with the in-batch positives of other queries plus hard negatives: the current model's top ICD hits
that are not gold, re-mined every epoch. Other gold names of the same query are masked, not
treated as negatives.

10% of CHIP-CDN train is held out: it picks the epoch and calibrates the acceptance threshold so the
fine-tuned retriever is accepted at the same held-out precision as the off-the-shelf one at 0.94.
CHIP-CDN dev is never used here.

Outputs: data/models/sapbert_cdn/ (encoder), data/dicts/icd_sapbert.pt (ICD embeddings + min_score),
outputs/metrics/sapbert_train.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from bert_linker import CACHE, MIN_SCORE, MODEL_NAME, SAPBERT_CACHE, SAPBERT_DIR, accept, mean_pool
from normalize import clean_surface, icd_variants, strip_grade

TRAIN = ROOT / "data" / "dicts" / "raw" / "CHIP-CDN_train.json"
ICD_JSON = ROOT / "data" / "dicts" / "icd10_beijing_v601.json"
MAX_LEN = 32


def gold_norms(row: dict, icd_names: set[str]) -> list[str]:
    raw = str(row.get("normalized_result", "")).strip()
    return [x.strip() for x in raw.split("##") if x.strip() in icd_names]


def encode(model, tokenizer, texts, device, batch_size=256, grad=False):
    vecs = []
    for i in range(0, len(texts), batch_size):
        tok = tokenizer(texts[i : i + batch_size], padding=True, truncation=True, max_length=MAX_LEN,
                        return_tensors="pt").to(device)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            hidden = model(**tok).last_hidden_state
        vecs.append(F.normalize(mean_pool(hidden.float(), tok["attention_mask"]), p=2, dim=1))
    return torch.cat(vecs, 0)


@torch.no_grad()
def topk_names(model, tokenizer, queries, icd_emb, names, device, k):
    model.eval()
    q = encode(model, tokenizer, queries, device)
    scores, idx = (q @ icd_emb.T).topk(k, dim=1)
    return [[(names[j], float(s)) for s, j in zip(srow, irow)] for srow, irow in zip(scores.tolist(), idx.tolist())]


def holdout_queries(rows):
    """Mirror the pipeline: the retriever sees ICD variants of the cleaned, grade-stripped surface."""
    out = []
    for text, golds in rows:
        base = strip_grade(clean_surface(text)) or text
        cands = [c for c in dict.fromkeys(icd_variants(base)) if len(c) >= 4]
        out.append((text, golds, cands))
    return out


def holdout_eval(model, tokenizer, rows, icd_emb, names, device) -> dict:
    """Unfiltered top-1 accuracy plus per-candidate top-3 hits for threshold calibration."""
    top1 = topk_names(model, tokenizer, [t for t, _ in rows], icd_emb, names, device, 1)
    acc = sum(top1[i][0][0] in set(g) for i, (_, g) in enumerate(rows)) / len(rows)
    hq = holdout_queries(rows)
    flat = [c for _, _, cands in hq for c in cands]
    hits = topk_names(model, tokenizer, flat, icd_emb, names, device, 3) if flat else []
    per_row, pos = [], 0
    for text, golds, cands in hq:
        cand_hits = []
        for c in cands:
            cand_hits.append((c, hits[pos]))
            pos += 1
        per_row.append((set(golds), cand_hits))
    return {"top1_acc": acc, "per_row": per_row}


def filtered_at(per_row, threshold: float) -> tuple[float, float, int]:
    """Coverage and answered precision of retrieve()+accept() at a given score threshold."""
    answered = correct = 0
    for golds, cand_hits in per_row:
        pred = None
        for c, hits in cand_hits:
            for hit, score in hits:
                if accept(c, hit, score, threshold):
                    pred = hit
                    break
            if pred:
                break
        if pred:
            answered += 1
            correct += pred in golds
    n = len(per_row)
    return answered / n, (correct / answered if answered else 0.0), answered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--hard", type=int, default=4, help="hard negatives per query")
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    names = sorted(json.loads(ICD_JSON.read_text(encoding="utf-8"))["name_to_codes"].keys())
    name_set = set(names)
    rows = []
    for r in json.loads(TRAIN.read_text(encoding="utf-8")):
        text = str(r.get("text", "")).lstrip("\ufeff").strip()
        golds = gold_norms(r, name_set)
        if text and golds:
            rows.append((text, golds))
    random.shuffle(rows)
    n_hold = int(len(rows) * args.holdout)
    hold, train = rows[:n_hold], rows[n_hold:]

    pairs = []
    gold_of = {}
    for text, golds in train:
        variants = {text}
        cleaned = clean_surface(text)
        if cleaned and len(cleaned) >= 2:
            variants.add(cleaned)
        for v in variants:
            gold_of.setdefault(v, set()).update(golds)
            pairs.extend((v, g) for g in golds)
    queries = sorted(gold_of)
    print(f"train rows {len(train)}  holdout {len(hold)}  pairs {len(pairs)}  ICD names {len(names)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    # Off-the-shelf reference on the held-out split: precision of the current 0.94 rule.
    base_emb = torch.load(CACHE, map_location="cpu", weights_only=False)["emb"].to(device)
    base = holdout_eval(model, tokenizer, hold, base_emb, names, device)
    base_cov, base_prec, _ = filtered_at(base["per_row"], MIN_SCORE)
    print(f"[roberta] holdout top1 {base['top1_acc']:.4f}  filtered@{MIN_SCORE}: coverage {base_cov:.4f} precision {base_prec:.4f}")
    del base_emb

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(pairs) + args.batch - 1) // args.batch)
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.eval()
        icd_emb = encode(model, tokenizer, names, device).detach()
        mined = topk_names(model, tokenizer, queries, icd_emb, names, device, 30)
        hard_of = {q: [n for n, _ in hits if n not in gold_of[q]] for q, hits in zip(queries, mined)}
        del icd_emb

        model.train()
        random.shuffle(pairs)
        total = 0.0
        for i in range(0, len(pairs), args.batch):
            batch = pairs[i : i + args.batch]
            qs = [q for q, _ in batch]
            negs = []
            for q in qs:
                pool = hard_of[q][:15]
                negs.extend(random.sample(pool, min(args.hard, len(pool))))
            cand = list(dict.fromkeys([p for _, p in batch] + negs))
            col = {c: j for j, c in enumerate(cand)}
            target = torch.tensor([col[p] for _, p in batch], device=device)
            mask = torch.zeros(len(batch), len(cand), dtype=torch.bool, device=device)
            for r, (q, p) in enumerate(batch):
                for g in gold_of[q]:
                    if g != p and g in col:
                        mask[r, col[g]] = True
            qv = encode(model, tokenizer, qs, device, grad=True)
            cv = encode(model, tokenizer, cand, device, grad=True)
            logits = (qv @ cv.T / args.temp).masked_fill(mask, -1e4)
            loss = F.cross_entropy(logits, target)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            total += loss.item() * len(batch)

        icd_emb = encode(model, tokenizer, names, device).detach()
        ev = holdout_eval(model, tokenizer, hold, icd_emb, names, device)
        rec = {"epoch": epoch, "loss": total / len(pairs), "holdout_top1": ev["top1_acc"]}
        history.append(rec)
        print(f"epoch {epoch} loss {rec['loss']:.4f} holdout top1 {ev['top1_acc']:.4f}")
        if best is None or ev["top1_acc"] > best["holdout_top1"]:
            best = {**rec, "per_row": ev["per_row"]}
            SAPBERT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(SAPBERT_DIR)
            tokenizer.save_pretrained(SAPBERT_DIR)
            best_emb = icd_emb.cpu()

    # Lowest threshold whose held-out precision is at least the off-the-shelf precision at 0.94.
    grid = [round(0.30 + 0.01 * k, 2) for k in range(70)]
    curve = [(t, *filtered_at(best["per_row"], t)) for t in grid]
    ok = [c for c in curve if c[2] >= base_prec and c[3] >= 5]
    min_score = ok[0][0] if ok else 0.99
    cov, prec, _ = filtered_at(best["per_row"], min_score)
    print(f"[sapbert] best epoch {best['epoch']}  min_score {min_score}  holdout filtered coverage {cov:.4f} precision {prec:.4f}")

    torch.save({"model": str(SAPBERT_DIR), "names": names, "emb": best_emb, "min_score": min_score}, SAPBERT_CACHE)
    out = ROOT / "outputs" / "metrics" / "sapbert_train.json"
    out.write_text(json.dumps({
        "args": vars(args),
        "n_train_rows": len(train), "n_holdout_rows": len(hold), "n_pairs": len(pairs),
        "roberta_holdout": {"top1": base["top1_acc"], "min_score": MIN_SCORE, "coverage": base_cov, "precision": base_prec},
        "sapbert_holdout": {"best_epoch": best["epoch"], "top1": best["holdout_top1"], "min_score": min_score,
                            "coverage": cov, "precision": prec},
        "history": history,
        "threshold_curve": [{"t": t, "coverage": c, "precision": p, "answered": a} for t, c, p, a in curve],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {SAPBERT_DIR}, {SAPBERT_CACHE}, {out}")


if __name__ == "__main__":
    main()
