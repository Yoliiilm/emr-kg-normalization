"""Frozen precision-first gate over the top-five calibrated candidates."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import joblib

import normalize
from calibrated_normalize import (
    CandidateGenerator,
    ConfidenceCalibrator,
    _compact,
    install_alias_rows,
    load_chip_rows,
    split_calibration_rows,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "outputs/metrics/experiments/reach85/meta_gate_model.joblib"
DEFAULT_MANIFEST = ROOT / "outputs/metrics/experiments/reach85/freeze_manifest.json"
DEFAULT_TRAIN = ROOT / "data/dicts/raw/CHIP-CDN_train.json"
TOP_K = 5
EXCLUDED_SOURCES = {"chip"}


@dataclass(frozen=True)
class GateDecision:
    surface: str
    name: str
    accepted: bool
    score: float
    source: str
    base_confidence: float
    reason: str
    candidates: tuple[dict, ...] = ()


def surface_features(surface: str) -> list[float]:
    compact = _compact(surface)
    return [
        len(compact) / 20.0,
        len(re.findall(r"[,，、;；()（）]", surface)) / 4.0,
        len(re.findall(r"\d", surface)) / 8.0,
        float(re.search(r"左|右|双侧|单侧", surface) is not None),
        float(re.search(r"上叶|下叶|中叶|段|叶|部|区|侧|壁", surface) is not None),
        float(re.search(r"转移|继发|浸润|原位|早期|晚期|复发", surface) is not None),
        float(re.search(r"待查|疑似|不除外|考虑|排除|未见|阴性|可疑", surface) is not None),
        float(re.search(r"增生|息肉|腺瘤|囊肿|结节|占位|病变|术后|切除", surface) is not None),
        float(re.search(r"癌|恶性|肿瘤|肉瘤|白血病|淋巴瘤", surface) is not None),
        float(len(compact) >= 12),
    ]


def same_side(left: str, right: str) -> float:
    sides = ("左", "右", "双侧", "单侧")
    return float(all((cue in left) == (cue in right) for cue in sides))


def candidate_features(
    surface: str,
    candidate: dict,
    rank: int,
    candidates: list[dict],
    pool_size: int,
) -> list[float]:
    compact_surface = _compact(surface)
    compact_name = _compact(candidate["name"])
    top_confidence = float(candidates[0]["confidence"])
    confidence = float(candidate["confidence"])
    common = len(set(compact_surface) & set(compact_name))
    union = len(set(compact_surface) | set(compact_name))
    source = candidate["source"]
    return [
        confidence,
        confidence - top_confidence,
        confidence / top_confidence if top_confidence else 0.0,
        (rank - 1) / 4.0,
        len(candidates) / 5.0,
        1.0 / rank,
        math.log1p(max(pool_size, 0)) / 10.0,
        abs(len(compact_surface) - len(compact_name)) / 20.0,
        SequenceMatcher(None, compact_surface, compact_name).ratio(),
        common / union if union else 0.0,
        float(compact_name.startswith(compact_surface[:2])) if compact_surface else 0.0,
        float(compact_name.endswith(compact_surface[-2:])) if compact_surface else 0.0,
        same_side(surface, candidate["name"]),
        float("癌" in compact_name),
        float("术" in compact_name),
        float(re.search(r"炎|病|综合征", compact_name) is not None),
        float(re.search(r"肿瘤|恶性|肉瘤|白血病|淋巴瘤", compact_name) is not None),
        *[float(source == value) for value in (
            "exact", "variant", "rule", "curated", "chip", "fuzzy", "bert", "supervised"
        )],
    ]


class PrecisionGate:
    """Frozen candidate reranker used as a precision-first normalization mode."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        manifest_path: Path = DEFAULT_MANIFEST,
        train_path: Path = DEFAULT_TRAIN,
        retriever: str = "sapbert",
    ):
        self.model = joblib.load(model_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.threshold = float(manifest["threshold"])
        train_rows = load_chip_rows(train_path)
        fit_rows, _ = split_calibration_rows(train_rows)
        install_alias_rows(fit_rows)
        self.calibrator = ConfidenceCalibrator.load(
            ROOT / "outputs/models/experiments/source_shrinkage/"
            "normalization_calibrator_sapbert_source_chip_reject.json"
        )
        self.generator = CandidateGenerator(include_neural=True, retriever=retriever)
        self._cache: dict[str, GateDecision] = {}

    def decide(self, surface: str) -> GateDecision:
        cached = self._cache.get(surface)
        if cached is not None:
            return cached
        pool = self.generator.generate(surface)
        if not pool:
            decision = GateDecision(
                surface=surface,
                name=normalize.clean_surface(surface),
                accepted=False,
                score=0.0,
                source="",
                base_confidence=0.0,
                reason="no_candidate",
            )
            self._cache[surface] = decision
            return decision

        scored = sorted(
            ((self.calibrator.score(candidate), candidate) for candidate in pool),
            key=lambda pair: (-pair[0], pair[1].name),
        )
        top = [
            {
                "name": candidate.name,
                "source": candidate.source,
                "confidence": float(score),
            }
            for score, candidate in scored[:TOP_K]
        ]
        allowed = [
            (rank, candidate)
            for rank, candidate in enumerate(top, start=1)
            if candidate["source"] not in EXCLUDED_SOURCES
        ]
        if not allowed:
            decision = GateDecision(
                surface=surface,
                name=normalize.clean_surface(surface),
                accepted=False,
                score=0.0,
                source="",
                base_confidence=0.0,
                reason="no_eligible_candidate",
                candidates=tuple(top),
            )
            self._cache[surface] = decision
            return decision

        vectors = [
            surface_features(surface)
            + candidate_features(surface, candidate, rank, top, len(pool))
            for rank, candidate in allowed
        ]
        gate_scores = self.model.predict_proba(vectors)[:, 1]
        best_index = max(
            range(len(allowed)),
            key=lambda index: (
                float(gate_scores[index]),
                allowed[index][1]["confidence"],
                -allowed[index][0],
            ),
        )
        rank, best = allowed[best_index]
        score = float(gate_scores[best_index])
        accepted = score >= self.threshold
        decision = GateDecision(
            surface=surface,
            name=best["name"] if accepted else normalize.clean_surface(surface),
            accepted=accepted,
            score=score,
            source=best["source"],
            base_confidence=float(best["confidence"]),
            reason="accepted" if accepted else "below_gate_threshold",
            candidates=tuple(
                {
                    **candidate,
                    "rank": candidate_rank,
                    "gate_score": float(gate_scores[index]),
                    "selected": index == best_index,
                }
                for index, (candidate_rank, candidate) in enumerate(allowed)
            ),
        )
        self._cache[surface] = decision
        return decision


def configure_precision_gate() -> dict:
    old = normalize.normalization_settings()
    normalize.configure_normalization(use_chip_cdn=False, retriever="sapbert")
    return old
