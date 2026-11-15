"""Train and evaluate RoBERTa-BiLSTM-CRF on CCKS2019 / Yidu-S4K."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ner_data import NERCharDataset, load_bio_docs, load_label_list
from ner_model import BertBiLstmCrf

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_bio(path: Path, docs: list[tuple[list[str], list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, (chars, tags) in enumerate(docs):
            for ch, tag in zip(chars, tags):
                f.write(f"{ch} {tag}\n")
            if i < len(docs) - 1:
                f.write("\n")


@torch.no_grad()
def predict_docs(model, tokenizer, docs, id2label, max_len, device, batch_size: int = 32) -> list[list[str]]:
    model.eval()
    label2id = {lab: i for i, lab in id2label.items()}
    ds = NERCharDataset(docs, tokenizer, label2id, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    pred_docs = [["O"] * len(chars) for chars, _ in docs]
    offset = 0
    for batch in loader:
        bsz = batch["input_ids"].size(0)
        decoded_batch = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            crf_mask=batch["crf_mask"].to(device),
        )
        for i in range(bsz):
            doc_idx, start, n_chars = ds.origins[offset + i]
            char_ids = decoded_batch[i][1 : 1 + n_chars]
            aligned = [id2label.get(x, "O") for x in char_ids]
            if len(aligned) < n_chars:
                aligned += ["O"] * (n_chars - len(aligned))
            pred_docs[doc_idx][start : start + n_chars] = aligned[:n_chars]
        offset += bsz
    return pred_docs


def evaluate(model, tokenizer, docs, id2label, max_len, device) -> dict:
    pred_docs = predict_docs(model, tokenizer, docs, id2label, max_len, device)
    y_true = [tags for _, tags in docs]
    y_pred = pred_docs
    metrics = {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "report": classification_report(y_true, y_pred, digits=4),
    }
    return metrics, pred_docs


def build_optimizer(model, lr_bert: float, lr_head: float, weight_decay: float):
    bert_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("bert."):
            bert_params.append(param)
        else:
            head_params.append(param)
    return AdamW(
        [
            {"params": bert_params, "lr": lr_bert},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="hfl/chinese-roberta-wwm-ext")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--lr_bert", type=float, default=2e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lstm_hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    set_seed(args.seed)
    processed = ROOT / "data" / "processed"
    model_dir = ROOT / "outputs" / "models"
    metric_dir = ROOT / "outputs" / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    labels = load_label_list(processed / "label_list.txt")
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    train_docs = load_bio_docs(processed / "train.bio")
    dev_docs = load_bio_docs(processed / "dev.bio")
    test_docs = load_bio_docs(processed / "test.bio")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={args.model_name} train={len(train_docs)} dev={len(dev_docs)} test={len(test_docs)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = NERCharDataset(train_docs, tokenizer, label2id, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = BertBiLstmCrf(
        args.model_name,
        num_labels=len(labels),
        lstm_hidden=args.lstm_hidden,
        dropout=args.dropout,
    ).to(device)

    optimizer = build_optimizer(model, args.lr_bert, args.lr_head, args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_path = model_dir / "best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(
                batch["input_ids"],
                batch["attention_mask"],
                labels=batch["labels"],
                crf_mask=batch["crf_mask"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{running / n_batches:.4f}")

        dev_metrics, _ = evaluate(model, tokenizer, dev_docs, id2label, args.max_len, device)
        row = {
            "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "dev_precision": dev_metrics["precision"],
            "dev_recall": dev_metrics["recall"],
            "dev_f1": dev_metrics["f1"],
        }
        history.append(row)
        print(
            f"epoch {epoch}: loss={row['train_loss']:.4f} "
            f"dev P={row['dev_precision']:.4f} R={row['dev_recall']:.4f} F1={row['dev_f1']:.4f}"
        )
        if row["dev_f1"] > best_f1:
            best_f1 = row["dev_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label2id": label2id,
                    "args": vars(args),
                    "dev_f1": best_f1,
                },
                best_path,
            )
            (metric_dir / "dev_report.txt").write_text(dev_metrics["report"], encoding="utf-8")
            print(f"  saved best model F1={best_f1:.4f} -> {best_path}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_metrics, test_pred = evaluate(model, tokenizer, test_docs, id2label, args.max_len, device)
    pred_docs = [(chars, pred) for (chars, _gold), pred in zip(test_docs, test_pred)]
    write_bio(processed / "test_pred.bio", pred_docs)
    (metric_dir / "test_report.txt").write_text(test_metrics["report"], encoding="utf-8")

    summary = {
        "model_name": args.model_name,
        "device": str(device),
        "epochs": args.epochs,
        "best_dev_f1": best_f1,
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "history": history,
        "n_train_chunks": len(train_ds),
        "n_test_docs": len(test_docs),
    }
    (metric_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("TEST P={:.4f} R={:.4f} F1={:.4f}".format(
        test_metrics["precision"], test_metrics["recall"], test_metrics["f1"]
    ))
    print(test_metrics["report"])
    return summary


if __name__ == "__main__":
    main()
