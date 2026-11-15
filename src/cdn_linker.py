"""Supervised CHIP-CDN linker (models trained by src/train_cdn_cblue.py) for graph normalization."""

from __future__ import annotations

import json
import threading

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_cdn_cblue import (
    ENCODER,
    ICD,
    MODEL_DIR,
    Recaller,
    commit,
    load_split,
    predict_num,
    process_text,
    score_pairs,
    str_q2b,
)

CACHE = MODEL_DIR / "link_cache_scored.json"
MAX_LEN = 64
RECALL_K = 200


class SupervisedCDN:
    def __init__(self) -> None:
        if not (MODEL_DIR / "cls.pt").exists():
            raise FileNotFoundError(f"Missing {MODEL_DIR / 'cls.pt'}; run python src/train_cdn_cblue.py")
        icd_names = json.loads(ICD.read_text(encoding="utf-8"))["name_to_codes"].keys()
        train_labels = [g for _, _, gs in load_split("train") for g in gs]
        self.labels = list(dict.fromkeys([str_q2b(n) for n in icd_names] + train_labels))
        self.label_set = set(self.labels)
        self.recaller = Recaller(self.labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(ENCODER)
        self.cls = self._load("cls.pt", 2)
        self.num = self._load("num.pt", 3)
        self.cache: dict[str, list[list]] = (
            json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
        )
        self._dirty = 0

    def _load(self, name: str, n: int):
        model = AutoModelForSequenceClassification.from_pretrained(ENCODER, num_labels=n)
        model.load_state_dict(torch.load(MODEL_DIR / name, map_location="cpu"))
        return model.to(self.device).eval()

    def is_label(self, name: str) -> bool:
        return name in self.label_set

    def link_scored(self, text: str) -> list[tuple[str, float]]:
        """(standard name, ranker probability) in commit order; [] when no candidate passes p≥0.5."""
        if text not in self.cache:
            self.link_many([text])
        return [(n, float(p)) for n, p in self.cache[text]]

    def link(self, text: str, min_p: float = 0.5) -> list[str]:
        return [n for n, p in self.link_scored(text) if p >= min_p]

    def link_many(self, texts: list[str]) -> None:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if not todo:
            return
        proc = [process_text(t) for t in todo]
        cands, rscores = self.recaller.recall(proc, RECALL_K)
        probs = score_pairs(self.cls, self.tok, proc, cands, self.labels, self.device, MAX_LEN)
        nums = predict_num(self.num, self.tok, proc, self.device, MAX_LEN)
        for raw, t, p, n, c, s in zip(todo, proc, probs, nums, cands, rscores):
            out = commit(t, p, int(n), c, s, self.labels)
            prob = {self.labels[i]: float(q) for i, q in zip(c, p)}
            self.cache[raw] = [] if out == [t] else [[x, round(prob[x], 4)] for x in out if x != "癌"]
        self._dirty += len(todo)
        if self._dirty >= 200:
            self.save()

    def save(self) -> None:
        if self._dirty:
            CACHE.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
            self._dirty = 0


_LINKER: SupervisedCDN | None = None
_LOCK = threading.Lock()


def get_cdn_linker() -> SupervisedCDN:
    global _LINKER
    with _LOCK:
        if _LINKER is None:
            _LINKER = SupervisedCDN()
    return _LINKER


def warm(texts) -> None:
    """Batch-link many mentions up front (much faster than one GPU call per mention)."""
    linker = get_cdn_linker()
    linker.link_many(list(texts))
    linker.save()
