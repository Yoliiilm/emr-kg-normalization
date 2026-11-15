"""Confidence-calibrated, abstaining disease normalization.

The legacy normalizer returns the first candidate accepted by its cascade.  This
module keeps several candidates, applies the existing clinical safety gates, and
uses a small logistic calibrator trained on CHIP-CDN to decide whether the best
candidate is reliable enough to merge.  It is deliberately optional: the legacy
pipeline and its reported numbers are unchanged unless calibration is enabled.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import numpy as np

import normalize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "outputs" / "models" / "normalization_calibrator.json"
SOURCES = ("exact", "variant", "rule", "curated", "chip", "fuzzy", "bert", "supervised")
NEURAL_TOP_K = 50
SOURCE_PRIOR = {
    "exact": 1.0,
    "variant": 0.97,
    "rule": 0.92,
    "curated": 0.88,
    "chip": 0.82,
    "fuzzy": 0.75,
    "bert": 0.65,
    "supervised": 0.70,
}


@dataclass(frozen=True)
class Candidate:
    name: str
    source: str
    raw_score: float
    lexical: float
    length_delta: int
    safe: bool = True


@dataclass(frozen=True)
class Decision:
    surface: str
    name: str
    source: str
    confidence: float
    accepted: bool
    margin: float
    reason: str
    candidates: tuple[dict, ...] = ()


class BinaryLogisticModel:
    """Small dependency-free logistic regression for candidate calibration."""

    def __init__(self, l2: float = 1e-2, steps: int = 1200, learning_rate: float = 0.08):
        self.l2 = l2
        self.steps = steps
        self.learning_rate = learning_rate
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, x: list[list[float]], y: list[int]) -> "BinaryLogisticModel":
        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float)
        self.coef_ = np.zeros(xx.shape[1], dtype=float)
        self.intercept_ = 0.0
        for _ in range(self.steps):
            logits = np.clip(xx @ self.coef_ + self.intercept_, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-logits))
            error = p - yy
            grad_w = (xx.T @ error) / len(xx) + self.l2 * self.coef_
            grad_b = float(error.mean())
            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b
        return self

    def predict_proba(self, x: list[list[float]]) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("calibrator has not been fitted")
        xx = np.asarray(x, dtype=float)
        logits = np.clip(xx @ self.coef_ + self.intercept_, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack((1.0 - p, p))


def _compact(text: str) -> str:
    return "".join(str(text).split())


def _lexical(a: str, b: str) -> float:
    a, b = _compact(a), _compact(b)
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    jaccard = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    return 0.5 * jaccard + 0.5 * SequenceMatcher(None, a, b).ratio()


def _normalise_gold(values: Iterable[str]) -> set[str]:
    return {_compact(x) for x in values if _compact(x) and _compact(x) != "癌"}


class CandidateGenerator:
    """Generate candidates from the existing dictionary and retrieval stack."""

    def __init__(
        self,
        *,
        include_neural: bool = True,
        include_supervised: bool = False,
        retriever: str | None = None,
    ):
        self.include_neural = include_neural
        self.include_supervised = include_supervised
        self.retriever = retriever
        self._cache: dict[str, tuple[Candidate, ...]] = {}

    def _safe(self, surface: str, name: str, source: str) -> bool:
        if not name:
            return False
        if source in {"exact", "variant"} and _compact(surface) == _compact(name):
            return True
        # These candidates are produced only after normalize's dedicated
        # morphology/site rules have fired; applying the alias gate again can
        # incorrectly reject valid head-noun rewrites such as 肝曲横结肠腺瘤.
        if source == "rule":
            return True
        try:
            return normalize.chip_alias_ok(surface, name)
        except ModuleNotFoundError:
            # Keep lexical smoke tests usable in the lightweight workspace
            # runtime; the full project environment uses bert_linker gates.
            if "左" in surface and "右" in name and "左" not in name:
                return False
            if "右" in surface and "左" in name and "右" not in name:
                return False
            if "术后" in name and "术后" not in surface:
                return False
            if source in {"rule", "curated", "chip"}:
                return True
            return _lexical(surface, name) >= 0.45

    def generate(self, surface: str) -> list[Candidate]:
        cached = self._cache.get(surface)
        if cached is not None:
            return list(cached)
        idx = normalize.get_index()
        raw = normalize.light_clean(surface)
        cleaned = normalize.clean_surface(raw)
        found: dict[str, Candidate] = {}

        def add(name: str | None, source: str, score: float) -> None:
            if not name:
                return
            name = idx.resolve_name(name) or name
            if name not in idx.names:
                return
            safe = self._safe(raw, name, source)
            if not safe:
                return
            cand = Candidate(
                name=name,
                source=source,
                raw_score=float(score),
                lexical=_lexical(raw, name),
                length_delta=abs(len(_compact(name)) - len(_compact(raw))),
                safe=True,
            )
            old = found.get(name)
            if old is None or cand.raw_score > old.raw_score:
                found[name] = cand

        add(idx.resolve_name(raw), "exact", SOURCE_PRIOR["exact"])
        if cleaned != raw:
            add(idx.resolve_name(cleaned), "variant", SOURCE_PRIOR["variant"])
        for base in (raw, cleaned, normalize.strip_grade(cleaned)):
            for variant in normalize.icd_variants(base):
                if variant != base:
                    add(idx.resolve_name(variant), "variant", SOURCE_PRIOR["variant"])

        if normalize._SETTINGS.get("use_new_rules", True):
            add(idx.metastasis_hit(cleaned), "rule", SOURCE_PRIOR["rule"])
            add(idx.site_morph_hit(cleaned), "rule", SOURCE_PRIOR["rule"])
        rule_name = normalize.rule_disease(raw)
        if rule_name != raw:
            add(rule_name, "rule", SOURCE_PRIOR["rule"])

        curated = idx.lookup_curated(raw, cleaned)
        add(curated, "curated", SOURCE_PRIOR["curated"])

        for key in (raw, cleaned):
            alias = idx.alias.get(key) or idx.alias_clean.get(normalize.clean_surface(key))
            add(alias, "chip", SOURCE_PRIOR["chip"])

        # A small lexical candidate pool provides hard negatives without scanning
        # the full ICD table for every mention.
        for name in idx.by_head.get(cleaned[:2], ()):
            if abs(len(name) - len(cleaned)) <= 6:
                ratio = SequenceMatcher(None, cleaned, name).ratio()
                if ratio >= 0.70:
                    add(name, "fuzzy", ratio)

        if self.include_neural:
            try:
                from bert_linker import get_linker

                linker = get_linker(self.retriever or normalize._SETTINGS.get("retriever", "roberta"))
                # Retrieve both forms: cleaning removes grades and uncertainty
                # tails, while the raw form can preserve site/morphology cues.
                seen_queries: set[str] = set()
                for query in (cleaned, raw):
                    if not query or query in seen_queries:
                        continue
                    seen_queries.add(query)
                    for name, score in linker.retrieve_candidates(query, NEURAL_TOP_K):
                        add(name, "bert", score)
            except (FileNotFoundError, OSError, RuntimeError):
                # Neural retrieval is optional; lexical candidates remain usable.
                pass

        if self.include_supervised:
            try:
                from cdn_linker import get_cdn_linker

                for name, score in get_cdn_linker().link_scored(raw):
                    add(name, "supervised", float(score))
            except (FileNotFoundError, OSError, RuntimeError):
                pass

        result = tuple(sorted(found.values(), key=lambda x: (-x.raw_score, -x.lexical, x.name)))
        self._cache[surface] = result
        return list(result)


class ConfidenceCalibrator:
    """Logistic calibration over candidate provenance and safety features."""

    # Fixed operating floors keep held-out evaluation independent from the
    # dev labels. A paper run can learn these on a separate calibration split.
    SOURCE_FLOORS = {
        "exact": 0.80,
        "variant": 0.80,
        "rule": 0.75,
        "curated": 0.70,
        "chip": 0.80,
        "fuzzy": 0.95,
        "bert": 0.85,
        "supervised": 0.85,
    }

    def __init__(
        self,
        model: BinaryLogisticModel,
        threshold: float,
        min_margin: float,
        calibration_info: dict | None = None,
        source_models: dict[str, BinaryLogisticModel] | None = None,
        rejected_sources: set[str] | None = None,
    ):
        self.model = model
        self.threshold = float(threshold)
        self.min_margin = float(min_margin)
        self.calibration_info = dict(calibration_info or {})
        self.source_models = dict(source_models or {})
        self.rejected_sources = set(rejected_sources or ())

    @staticmethod
    def vector(candidate: Candidate) -> list[float]:
        return [
            candidate.raw_score,
            candidate.lexical,
            min(candidate.length_delta, 12) / 12.0,
            SOURCE_PRIOR.get(candidate.source, 0.5),
            1.0 if candidate.source in {"exact", "variant", "rule", "curated"} else 0.0,
            *[1.0 if candidate.source == source else 0.0 for source in SOURCES],
        ]

    @staticmethod
    def _shrink_source_model(
        global_model: BinaryLogisticModel,
        source_model: BinaryLogisticModel,
        positives: int,
        negatives: int,
        strength: float,
    ) -> tuple[BinaryLogisticModel, float]:
        """Shrink a source fit toward the global fit using its weaker class support."""
        if strength < 0:
            raise ValueError("source_shrinkage_strength cannot be negative")
        if strength == 0:
            return source_model, 1.0
        support_weight = min(
            positives / (positives + strength),
            negatives / (negatives + strength),
        )
        shrunk = BinaryLogisticModel()
        shrunk.coef_ = (
            support_weight * source_model.coef_
            + (1.0 - support_weight) * global_model.coef_
        )
        shrunk.intercept_ = (
            support_weight * source_model.intercept_
            + (1.0 - support_weight) * global_model.intercept_
        )
        return shrunk, float(support_weight)

    @classmethod
    def fit(
        cls,
        train_rows: list[tuple[str, list[str]]],
        calibration_rows: list[tuple[str, list[str]]],
        *,
        generator: CandidateGenerator,
        target_precision: float = 0.85,
        min_accepted: int = 50,
        min_coverage: float = 0.03,
        min_source_positives: int = 20,
        min_source_negatives: int = 50,
        source_shrinkage_strength: float = 0.0,
        source_model_mode: str = "auto",
        rejected_sources: tuple[str, ...] = (),
    ) -> "ConfidenceCalibrator":
        if source_shrinkage_strength < 0:
            raise ValueError("source_shrinkage_strength cannot be negative")
        if min_source_positives < 20 or min_source_negatives < 50:
            raise ValueError("source support gates cannot be lower than 20 positives and 50 negatives")
        if source_model_mode not in {"auto", "global", "source", "shrink"}:
            raise ValueError("source_model_mode must be auto|global|source|shrink")
        if source_model_mode == "auto":
            source_model_mode = "shrink" if source_shrinkage_strength > 0 else "source"
        if source_model_mode == "shrink" and source_shrinkage_strength <= 0:
            raise ValueError("shrink mode requires source_shrinkage_strength > 0")
        if source_model_mode != "shrink":
            source_shrinkage_strength = 0.0
        x, y = [], []
        source_rows: dict[str, list[tuple[list[float], int]]] = {source: [] for source in SOURCES}
        for surface, gold in train_rows:
            gold_set = _normalise_gold(gold)
            for candidate in generator.generate(surface):
                vector = cls.vector(candidate)
                label = int(_compact(candidate.name) in gold_set)
                x.append(vector)
                y.append(label)
                source_rows.setdefault(candidate.source, []).append((vector, label))
        if not x or len(set(y)) < 2:
            raise ValueError("Calibration data must contain positive and negative candidates")
        # Balance the highly skewed candidate labels by duplicating positives in
        # the small calibration set. This keeps the implementation dependency-free.
        positives = [row for row, label in zip(x, y) if label]
        negatives = [row for row, label in zip(x, y) if not label]
        if positives and len(positives) < len(negatives):
            repeat = max(1, min(20, len(negatives) // len(positives)))
            x_fit = negatives + positives * repeat
            y_fit = [0] * len(negatives) + [1] * (len(positives) * repeat)
        else:
            x_fit, y_fit = x, y
        model = BinaryLogisticModel().fit(x_fit, y_fit)

        source_models: dict[str, BinaryLogisticModel] = {}
        source_support: dict[str, dict[str, int | bool]] = {}
        source_shrinkage: dict[str, float] = {}
        for source, rows in source_rows.items():
            positives_source = [row for row, label in rows if label]
            negatives_source = [row for row, label in rows if not label]
            enough = (
                len(positives_source) >= min_source_positives
                and len(negatives_source) >= min_source_negatives
            )
            source_support[source] = {
                "positive": len(positives_source),
                "negative": len(negatives_source),
                "eligible": enough,
                "fitted": enough and source_model_mode != "global",
            }
            if not enough or source_model_mode == "global":
                continue
            if len(positives_source) < len(negatives_source):
                repeat_source = max(1, min(20, len(negatives_source) // len(positives_source)))
                source_x = negatives_source + positives_source * repeat_source
                source_y = [0] * len(negatives_source) + [1] * (len(positives_source) * repeat_source)
            else:
                source_x, source_y = [row for row, _ in rows], [label for _, label in rows]
            fitted_source_model = BinaryLogisticModel().fit(source_x, source_y)
            source_model, shrinkage_weight = cls._shrink_source_model(
                model,
                fitted_source_model,
                len(positives_source),
                len(negatives_source),
                source_shrinkage_strength,
            )
            source_models[source] = source_model
            source_shrinkage[source] = shrinkage_weight

        if min_accepted < 1:
            raise ValueError("min_accepted must be positive")
        if not 0.0 < min_coverage <= 1.0:
            raise ValueError("min_coverage must be in (0, 1]")

        # Select operating points on complete mentions, matching decide().
        # Candidate-level thresholding would optimize a different task because
        # inference only accepts the best candidate and also applies source
        # floors and a margin.
        rows_sources: list[tuple[float, float, str, int]] = []

        def predict_candidate(candidate: Candidate) -> float:
            scorer = source_models.get(candidate.source, model)
            return float(scorer.predict_proba([cls.vector(candidate)])[0, 1])

        for surface, gold in calibration_rows:
            gold_set = _normalise_gold(gold)
            candidates = generator.generate(surface)
            if not candidates:
                continue
            probs = sorted(
                [(predict_candidate(c), c) for c in candidates],
                key=lambda x: -x[0],
            )
            best_p, best = probs[0]
            second_p = probs[1][0] if len(probs) > 1 else 0.0
            margin = best_p - second_p
            label = int(_compact(best.name) in gold_set)
            rows_sources.append((best_p, margin, best.source, label))
        calibration_denominator = max(1, len(calibration_rows))

        def metrics(threshold: float, margin: float) -> dict:
            selected = [label for p, m, source, label in rows_sources
                        if p >= max(threshold, cls.SOURCE_FLOORS.get(source, threshold))
                        and m >= margin]
            accepted = len(selected)
            precision = sum(selected) / accepted if accepted else 0.0
            coverage = accepted / calibration_denominator
            f1 = 2 * precision * coverage / (precision + coverage) if precision + coverage else 0.0
            return {
                "threshold": threshold,
                "min_margin": margin,
                "accepted": accepted,
                "precision": precision,
                "coverage": coverage,
                "f1": f1,
                "target_met": precision >= target_precision,
                "support_met": accepted >= min_accepted and coverage >= min_coverage,
            }

        thresholds = sorted({round(x / 100, 2) for x in range(20, 97, 2)})
        margins = sorted({0.0, 0.01, 0.03, 0.05, 0.08, 0.12})
        points = [metrics(t, m) for t in thresholds for m in margins]
        supported = [p for p in points if p["support_met"]]
        if not supported:
            raise ValueError(
                "no operating point satisfies min_accepted/min_coverage; "
                "reduce the support constraints or increase calibration data"
            )
        target_supported = [p for p in supported if p["target_met"]]
        conservative = max(
            target_supported or supported,
            key=lambda p: (p["precision"], p["coverage"], p["threshold"], -p["min_margin"]),
        )
        balanced = max(
            supported,
            key=lambda p: (p["f1"], p["coverage"], p["precision"], -p["threshold"]),
        )
        info = {
            "target_precision": target_precision,
            "min_accepted": min_accepted,
            "min_coverage": min_coverage,
            "calibration_mentions_scored": len(rows_sources),
            "conservative": conservative,
            "balanced": balanced,
            "target_met_with_support": bool(target_supported),
            "support_met_for_selection": bool([p for p in points if p["support_met"]]),
            "selection_records_include_source_floors": True,
            "source_support": source_support,
            "source_model_thresholds": {
                "min_source_positives": min_source_positives,
                "min_source_negatives": min_source_negatives,
            },
            "source_shrinkage": {
                "method": "support_weighted_parameter_interpolation",
                "strength": source_shrinkage_strength,
                "source_weights": source_shrinkage,
            },
            "source_model_mode": source_model_mode,
            "rejected_sources": sorted(set(rejected_sources)),
        }
        chosen = conservative
        return cls(
            model,
            chosen["threshold"],
            min_margin=chosen["min_margin"],
            calibration_info=info,
            source_models=source_models,
            rejected_sources=set(rejected_sources),
        )

    def score(self, candidate: Candidate) -> float:
        model = self.source_models.get(candidate.source, self.model)
        return float(model.predict_proba([self.vector(candidate)])[0, 1])

    def decide(self, surface: str, candidates: list[Candidate]) -> Decision:
        if not candidates:
            return Decision(surface, normalize.clean_surface(surface), "calibrated_reject", 0.0, False, 0.0, "no_candidate")
        scored = sorted(((self.score(c), c) for c in candidates), key=lambda x: (-x[0], x[1].name))
        best_p, best = scored[0]
        second_p = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_p - second_p
        source_floor = self.SOURCE_FLOORS.get(best.source, self.threshold)
        blocked = best.source in self.rejected_sources
        accepted = not blocked and best_p >= max(self.threshold, source_floor) and margin >= self.min_margin
        if accepted:
            reason = "accepted"
        elif blocked:
            reason = "source_blocked"
        elif best_p < self.threshold:
            reason = "low_confidence"
        elif best_p < source_floor:
            reason = "source_guard"
        else:
            reason = "small_margin"
        return Decision(
            surface=surface,
            name=best.name if accepted else normalize.clean_surface(surface),
            source=f"calibrated_{best.source}" if accepted else "calibrated_reject",
            confidence=best_p,
            accepted=accepted,
            margin=margin,
            reason=reason,
            candidates=tuple({"name": c.name, "source": c.source, "confidence": round(p, 6)} for p, c in scored[:5]),
        )

    def save(self, path: Path = DEFAULT_MODEL) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "coef": self.model.coef_.tolist() if self.model.coef_ is not None else [],
            "intercept": self.model.intercept_,
            "threshold": self.threshold,
            "min_margin": self.min_margin,
            "sources": list(SOURCES),
            "feature_version": 3,
            "neural_top_k": NEURAL_TOP_K,
            "calibration": self.calibration_info,
            "rejected_sources": sorted(self.rejected_sources),
            "source_models": {
                source: {
                    "coef": model.coef_.tolist() if model.coef_ is not None else [],
                    "intercept": model.intercept_,
                }
                for source, model in self.source_models.items()
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL) -> "ConfidenceCalibrator":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = BinaryLogisticModel()
        model.coef_ = np.asarray(payload["coef"], dtype=float)
        model.intercept_ = float(payload["intercept"])
        source_models = {}
        for source, source_payload in payload.get("source_models", {}).items():
            source_model = BinaryLogisticModel()
            source_model.coef_ = np.asarray(source_payload["coef"], dtype=float)
            source_model.intercept_ = float(source_payload["intercept"])
            source_models[source] = source_model
        return cls(
            model,
            payload["threshold"],
            payload.get("min_margin", 0.03),
            payload.get("calibration"),
            source_models=source_models,
            rejected_sources=set(payload.get("rejected_sources", ())),
        )


def load_chip_rows(path: Path) -> list[tuple[str, list[str]]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in rows:
        surface = str(row.get("text", "")).lstrip("\ufeff").strip()
        gold = [x.strip() for x in str(row.get("normalized_result", "")).split("##") if x.strip()]
        if surface and gold:
            out.append((surface, gold))
    return out


def split_calibration_rows(
    rows: list[tuple[str, list[str]]], fraction: float = 0.2
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    """Deterministically split train rows into model-fit and calibration subsets."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration fraction must be between 0 and 1")
    cutoff = int(round(fraction * 1000))
    calibration, fit = [], []
    for surface, gold in rows:
        digest = hashlib.sha1(surface.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:2], "big") % 1000
        (calibration if bucket < cutoff else fit).append((surface, gold))
    if not fit or not calibration:
        raise ValueError("calibration split produced an empty subset")
    return fit, calibration


def install_alias_rows(rows: list[tuple[str, list[str]]]) -> int:
    """Install a split-local alias table on the loaded dictionary index.

    The checked-in CHIP table contains train and dev rows.  Calibration must
    use train aliases only, otherwise a held-out mention can become an exact
    candidate before the calibrator is evaluated.
    """
    idx = normalize.get_index()
    aliases: dict[str, str] = {}
    for surface, gold in rows:
        for target in gold:
            resolved = idx.resolve_name(target)
            if resolved:
                aliases[surface] = resolved
                break
    idx.alias = aliases
    idx.alias_clean = {normalize.clean_surface(k): v for k, v in aliases.items() if normalize.clean_surface(k)}
    return len(aliases)


def fit_and_save(
    train_path: Path,
    dev_path: Path,
    out_path: Path = DEFAULT_MODEL,
    *,
    include_neural: bool = False,
    retriever: str | None = None,
    target_precision: float = 0.85,
    calibration_fraction: float = 0.2,
    min_accepted: int = 50,
    min_coverage: float = 0.03,
    min_source_positives: int = 20,
    min_source_negatives: int = 50,
    source_shrinkage_strength: float = 0.0,
    source_model_mode: str = "auto",
    rejected_sources: tuple[str, ...] = (),
) -> ConfidenceCalibrator:
    generator = CandidateGenerator(include_neural=include_neural, retriever=retriever)
    train_rows = load_chip_rows(train_path)
    fit_rows, calibration_rows = split_calibration_rows(train_rows, calibration_fraction)
    old = normalize.normalization_settings()
    normalize.configure_normalization(use_chip_cdn=False)
    try:
        # Only model-fit aliases are visible while fitting and calibrating.
        install_alias_rows(fit_rows)
        calibrator = ConfidenceCalibrator.fit(
            fit_rows,
            calibration_rows,
            generator=generator,
            target_precision=target_precision,
            min_accepted=min_accepted,
            min_coverage=min_coverage,
            min_source_positives=min_source_positives,
            min_source_negatives=min_source_negatives,
            source_shrinkage_strength=source_shrinkage_strength,
            source_model_mode=source_model_mode,
            rejected_sources=rejected_sources,
        )
    finally:
        normalize.configure_normalization(**old)
    calibrator.calibration_info = {
        "strategy": "train_internal_split",
        "train_path": str(train_path),
        "calibration_fraction": calibration_fraction,
        "fit_rows": len(fit_rows),
        "calibration_rows": len(calibration_rows),
        "dev_path_reserved_for_evaluation": str(dev_path),
        "alias_source": "fit_rows_only",
        "target_precision": target_precision,
        "min_accepted": min_accepted,
        "min_coverage": min_coverage,
        "min_source_positives": min_source_positives,
        "min_source_negatives": min_source_negatives,
        "source_shrinkage_strength": calibrator.calibration_info.get("source_shrinkage", {}).get("strength", source_shrinkage_strength),
        "source_model_mode": calibrator.calibration_info.get("source_model_mode", source_model_mode),
        "rejected_sources": sorted(rejected_sources),
        "operating_points": calibrator.calibration_info.get("conservative"),
        "balanced_operating_point": calibrator.calibration_info.get("balanced"),
        "target_met_with_support": calibrator.calibration_info.get("target_met_with_support", False),
        "source_support": calibrator.calibration_info.get("source_support", {}),
        "source_models_fitted": sorted(calibrator.source_models),
        "source_shrinkage": calibrator.calibration_info.get("source_shrinkage", {}),
    }
    calibrator.save(out_path)
    return calibrator


def calibrated_lookup(
    surface: str,
    *,
    model_path: Path = DEFAULT_MODEL,
    include_neural: bool = True,
) -> Decision:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing calibrator {model_path}; run src/fit_calibrator.py first")
    calibrator = ConfidenceCalibrator.load(model_path)
    generator = CandidateGenerator(include_neural=include_neural)
    return calibrator.decide(surface, generator.generate(surface))
