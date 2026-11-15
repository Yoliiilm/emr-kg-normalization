"""Unsupervised RoBERTa retrieval against the ICD-10 name list."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from clinical_gates import ORGANS, KEY_MORPHS, morph_ok, site_ok

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "dicts" / "icd_roberta.pt"
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
SAPBERT_DIR = ROOT / "data" / "models" / "sapbert_cdn"
SAPBERT_CACHE = ROOT / "data" / "dicts" / "icd_sapbert.pt"
RETRIEVERS = {
    "roberta": (MODEL_NAME, CACHE),
    "sapbert": (str(SAPBERT_DIR), SAPBERT_CACHE),
}

MIN_LEN = 4
MIN_SCORE = 0.94
MAX_EXTRA = 3
MIN_JACCARD = 0.72

_LINKERS: dict[str, "BertIcdLinker"] = {}


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(1)
    denom = mask.sum(1).clamp(min=1e-6)
    return summed / denom


def char_jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def accept(query: str, hit: str, score: float, min_score: float = MIN_SCORE) -> bool:
    if len(query) < MIN_LEN or score < min_score:
        return False
    if len(hit) - len(query) > MAX_EXTRA:
        return False
    qc, hc = set(query), set(hit)
    if not (qc <= hc or char_jaccard(query, hit) >= MIN_JACCARD):
        return False
    if not morph_ok(query, hit):
        return False
    if not site_ok(query, hit):
        return False
    # reject invented laterality (肺癌 → 右肺…)
    q_has = any(s in query for s in ("左", "右"))
    if not q_has and (("左" in hit) or ("右" in hit)) and not any(s in hit for s in ("两", "双")):
        if len(hit) >= len(query):
            return False
    if "左" in query and "右" in hit and "左" not in hit:
        return False
    if "右" in query and "左" in hit and "右" not in hit:
        return False
    return True


class BertIcdLinker:
    def __init__(self, kind: str = "roberta"):
        model_name, cache = RETRIEVERS[kind]
        if not cache.exists():
            hint = "src/encode_icd.py" if kind == "roberta" else "src/train_sapbert.py"
            raise FileNotFoundError(f"Missing {cache}; run python {hint}")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        pack = torch.load(cache, map_location="cpu", weights_only=False)
        self.kind = kind
        self.names = pack["names"]
        self.emb = pack["emb"]
        self.min_score = float(pack.get("min_score", MIN_SCORE))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._encoded_queries: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        cached = self._encoded_queries.get(text)
        if cached is not None:
            return cached
        tok = self.tokenizer(text, truncation=True, max_length=32, return_tensors="pt")
        tok = {k: v.to(self.device) for k, v in tok.items()}
        hidden = self.model(**tok).last_hidden_state
        pooled = mean_pool(hidden, tok["attention_mask"])
        encoded = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu()
        self._encoded_queries[text] = encoded
        return encoded

    def top1(self, query: str) -> tuple[str, float] | None:
        if len(query) < 2:
            return None
        candidates = self.retrieve_candidates(query, 1)
        return candidates[0] if candidates else None

    def retrieve_candidates(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k raw cosine candidates before acceptance filters.

        The calibrated normalizer needs the rejected candidates as hard negatives;
        ``retrieve`` intentionally hides those candidates for the legacy pipeline.
        """
        if len(query) < 2 or k <= 0:
            return []
        scores = (self.emb @ self.encode(query).T).squeeze(1)
        topv, topi = torch.topk(scores, min(k, len(self.names)))
        return [(self.names[int(i)], float(v)) for v, i in zip(topv.tolist(), topi.tolist())]

    def retrieve(self, query: str) -> tuple[str, float] | None:
        if len(query) < MIN_LEN:
            return None
        vec = self.encode(query)
        scores = (self.emb @ vec.T).squeeze(1)
        topv, topi = torch.topk(scores, 3)
        for score, i in zip(topv.tolist(), topi.tolist()):
            hit = self.names[i]
            if accept(query, hit, float(score), self.min_score):
                return hit, float(score)
        return None


def get_linker(kind: str = "roberta") -> BertIcdLinker:
    if kind not in _LINKERS:
        _LINKERS[kind] = BertIcdLinker(kind)
    return _LINKERS[kind]
