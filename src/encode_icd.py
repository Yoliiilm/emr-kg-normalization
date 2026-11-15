"""Encode ICD-10 names with Chinese RoBERTa for unsupervised linking."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
ICD_JSON = ROOT / "data" / "dicts" / "icd10_beijing_v601.json"
CACHE = ROOT / "data" / "dicts" / "icd_roberta.pt"


@torch.no_grad()
def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(1)
    denom = mask.sum(1).clamp(min=1e-6)
    return summed / denom


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, batch_size: int = 64, max_len: int = 32):
    model.eval()
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tok = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        hidden = model(**tok).last_hidden_state
        pooled = mean_pool(hidden, tok["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vecs.append(pooled.cpu())
        if i % 2048 == 0:
            print(f"encoded {min(i + batch_size, len(texts))}/{len(texts)}")
    return torch.cat(vecs, dim=0)


def main() -> None:
    icd = json.loads(ICD_JSON.read_text(encoding="utf-8"))
    names = sorted(icd["name_to_codes"].keys())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"encoding {len(names)} ICD names on {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    emb = encode_texts(names, tokenizer, model, device)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": MODEL_NAME, "names": names, "emb": emb}, CACHE)
    print(f"wrote {CACHE} shape={tuple(emb.shape)}")


if __name__ == "__main__":
    main()
