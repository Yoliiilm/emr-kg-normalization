"""Cue-based clinical relation evidence (beyond bag-of-entity co-occurrence)."""

from __future__ import annotations

import re

# Treatment / order cues commonly seen in Chinese EMR narrative.
DRUG_CUES = (
    "予以",
    "给予",
    "口服",
    "静滴",
    "静注",
    "静脉",
    "肌注",
    "化疗",
    "靶向",
    "方案",
    "用药",
    "服用",
    "雾化",
    "泵入",
    "灌注",
)
SURG_CUES = (
    "行",
    "予行",
    "在全麻下",
    "在腰麻下",
    "手术",
    "切除",
    "造瘘",
    "吻合",
)
DRUG_CUE_RE = re.compile("|".join(map(re.escape, DRUG_CUES)))
SURG_CUE_RE = re.compile("|".join(map(re.escape, SURG_CUES)))


def _window(text: str, a_start: int, a_end: int, b_start: int, b_end: int, pad: int = 24) -> str:
    lo = max(0, min(a_start, b_start) - pad)
    hi = min(len(text), max(a_end, b_end) + pad)
    return text[lo:hi]


def pair_has_drug_cue(text: str, disease: dict, drug: dict, pad: int = 28) -> bool:
    win = _window(text, disease["start"], disease["end"], drug["start"], drug["end"], pad=pad)
    if DRUG_CUE_RE.search(win):
        return True
    # drug itself in imperative neighborhood
    local = text[max(0, drug["start"] - 12) : min(len(text), drug["end"] + 8)]
    return bool(DRUG_CUE_RE.search(local))


def pair_has_surg_cue(text: str, disease: dict, surg: dict, pad: int = 32) -> bool:
    win = _window(text, disease["start"], disease["end"], surg["start"], surg["end"], pad=pad)
    if SURG_CUE_RE.search(win):
        return True
    local = text[max(0, surg["start"] - 16) : min(len(text), surg["end"] + 6)]
    return bool(SURG_CUE_RE.search(local))


def score_pair(text: str, src: dict, dst: dict, rel: str) -> tuple[int, str]:
    """Return (weight_bonus, evidence) for one entity pair in a document."""
    if rel == "使用药物" and src["label"] == "疾病和诊断" and dst["label"] == "药物":
        if pair_has_drug_cue(text, src, dst):
            return 1, "pattern"
        return 0, "cooccur"
    if rel == "实施手术" and src["label"] == "疾病和诊断" and dst["label"] == "手术":
        if pair_has_surg_cue(text, src, dst):
            return 1, "pattern"
        return 0, "cooccur"
    return 0, "cooccur"
