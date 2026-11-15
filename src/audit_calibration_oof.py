"""Write row-level out-of-fold audit data for the neural source calibrator.

All predictions are made on train mentions excluded from both coefficient fitting
and operating-point selection. This command is descriptive and never reads dev.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from calibrated_normalize import (
    CandidateGenerator,
    ConfidenceCalibrator,
    install_alias_rows,
    load_chip_rows,
    split_calibration_rows,
    _compact,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "dicts" / "raw"
OUT_DIR = ROOT / "outputs" / "metrics" / "experiments" / "source_shrinkage"
DEFAULT_OUT = OUT_DIR / "train_oof_audit_neural_source_chip_reject.json"


class CachedGenerator:
    def __init__(self, generator: CandidateGenerator):
        self.generator = generator
        self.cache = {}

    def generate(self, surface: str):
        if surface not in self.cache:
            self.cache[surface] = self.generator.generate(surface)
        return list(self.cache[surface])


def fold_rows(rows, fold: int, folds: int):
    fit, held_out = [], []
    for row in rows:
        bucket = int.from_bytes(hashlib.sha1(row[0].encode("utf-8")).digest()[:4], "big") % folds
        (held_out if bucket == fold else fit).append(row)
    return fit, held_out


def cue_flags(surface: str) -> dict[str, bool]:
    """Overlapping lexical strata for error review, not clinical labels."""
    return {
        "laterality": bool(re.search(r"左|右|双侧|单侧", surface)),
        "anatomical_detail": bool(re.search(r"上叶|下叶|中叶|段|叶|部|区|侧|壁", surface)),
        "metastasis_or_stage": bool(re.search(r"转移|继发|浸润|原位|早期|晚期|复发", surface)),
        "uncertainty_or_negation": bool(re.search(r"待查|疑似|不除外|考虑|排除|未见|阴性|可疑", surface)),
        "pathology_or_finding": bool(re.search(r"增生|息肉|腺瘤|囊肿|结节|占位|病变|术后|切除", surface)),
        "malignancy": bool(re.search(r"癌|恶性|肿瘤|肉瘤|白血病|淋巴瘤", surface)),
        "long_surface": len(_compact(surface)) >= 12,
    }


def row_record(fold: int, surface: str, gold: list[str], generator, calibrator):
    pool = generator.generate(surface)
    decision = calibrator.decide(surface, pool)
    gold_set = {_compact(item) for item in gold if _compact(item) != "癌"}
    accepted_correct = decision.accepted and _compact(decision.name) in gold_set
    pool_gold = [candidate for candidate in pool if _compact(candidate.name) in gold_set]
    scored_gold = [candidate for candidate in decision.candidates if _compact(candidate["name"]) in gold_set]
    best = decision.candidates[0] if decision.candidates else None
    return {
        "fold": fold,
        "surface": surface,
        "gold": gold,
        "accepted": decision.accepted,
        "predicted": decision.name if decision.accepted else None,
        "correct": bool(accepted_correct),
        "reason": decision.reason,
        "confidence": round(decision.confidence, 6),
        "margin": round(decision.margin, 6),
        "best_source": best["source"] if best else "no_candidate",
        "pool_size": len(pool),
        "gold_in_pool": bool(pool_gold),
        "pool_gold_rank": next((i + 1 for i, candidate in enumerate(pool)
                                 if _compact(candidate.name) in gold_set), None),
        "gold_in_scored_top5": bool(scored_gold),
        "top5": list(decision.candidates),
        "cues": cue_flags(surface),
    }


def summarize(records: list[dict], folds: int) -> dict:
    reasons = defaultdict(list)
    sources = defaultdict(list)
    confidence = defaultdict(list)
    cues = defaultdict(list)
    fold_records = defaultdict(list)
    for row in records:
        reasons[row["reason"]].append(row)
        sources[row["best_source"]].append(row)
        confidence[(int(row["confidence"] * 10) / 10, row["accepted"])].append(row)
        fold_records[row["fold"]].append(row)
        for cue, present in row["cues"].items():
            if present:
                cues[cue].append(row)

    def group_metrics(rows):
        accepted = [row for row in rows if row["accepted"]]
        correct = sum(row["correct"] for row in accepted)
        return {
            "mentions": len(rows),
            "accepted": len(accepted),
            "correct": correct,
            "coverage": len(accepted) / len(rows) if rows else 0.0,
            "accepted_precision": correct / len(accepted) if accepted else None,
            "candidate_pool_recall": sum(row["gold_in_pool"] for row in rows) / len(rows) if rows else 0.0,
            "scored_top5_recall": sum(row["gold_in_scored_top5"] for row in rows) / len(rows) if rows else 0.0,
        }

    fold_summary = {}
    for fold in range(folds):
        fold_rows_ = fold_records[fold]
        fold_summary[str(fold)] = {
            **group_metrics(fold_rows_),
            "accepted_false_positives": sum(row["accepted"] and not row["correct"] for row in fold_rows_),
            "rejected_with_gold_in_top5": sum(not row["accepted"] and row["gold_in_scored_top5"] for row in fold_rows_),
            "rejected_without_any_gold_candidate": sum(not row["accepted"] and not row["gold_in_pool"] for row in fold_rows_),
        }

    reason_summary = {key: group_metrics(value) for key, value in sorted(reasons.items())}
    source_summary = {key: group_metrics(value) for key, value in sorted(sources.items())}
    cue_summary = {key: group_metrics(value) for key, value in sorted(cues.items())}
    confidence_summary = {
        f"{score:.1f}|{'accepted' if accepted else 'rejected'}": group_metrics(value)
        for (score, accepted), value in sorted(confidence.items())
    }
    accepted_rows = [row for row in records if row["accepted"]]
    return {
        "overall": group_metrics(records),
        "accepted_false_positives": len(accepted_rows) - sum(row["correct"] for row in accepted_rows),
        "rejected_with_gold_in_top5": sum(not row["accepted"] and row["gold_in_scored_top5"] for row in records),
        "rejected_without_any_gold_candidate": sum(not row["accepted"] and not row["gold_in_pool"] for row in records),
        "by_fold": fold_summary,
        "by_reason": reason_summary,
        "by_best_candidate_source": source_summary,
        "by_confidence_decile_and_decision": confidence_summary,
        "overlapping_surface_cues": cue_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=RAW / "CHIP-CDN_train.json")
    parser.add_argument("--retriever", choices=("roberta", "sapbert"), default="roberta")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--limit", type=int, help="optional deterministic smoke-test cap")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")

    from normalize import configure_normalization, normalization_settings

    rows = load_chip_rows(args.train)
    if args.limit:
        rows = rows[:args.limit]
    old = normalization_settings()
    configure_normalization(use_chip_cdn=False, retriever=args.retriever)
    records = []
    fold_meta = []
    try:
        for fold in range(args.folds):
            outer_train, held_out = fold_rows(rows, fold, args.folds)
            fit_rows, calibration_rows = split_calibration_rows(outer_train)
            install_alias_rows(fit_rows)
            generator = CachedGenerator(CandidateGenerator(include_neural=True, retriever=args.retriever))
            calibrator = ConfidenceCalibrator.fit(
                fit_rows,
                calibration_rows,
                generator=generator,
                target_precision=0.85,
                min_accepted=50,
                min_coverage=0.03,
                min_source_positives=20,
                min_source_negatives=50,
                source_model_mode="source",
                rejected_sources=("chip",),
            )
            fold_meta.append({
                "fold": fold,
                "fit_rows": len(fit_rows),
                "calibration_rows": len(calibration_rows),
                "held_out_rows": len(held_out),
                "threshold": calibrator.threshold,
                "min_margin": calibrator.min_margin,
                "source_support": calibrator.calibration_info["source_support"],
            })
            records.extend(row_record(fold, surface, gold, generator, calibrator)
                           for surface, gold in held_out)
    finally:
        configure_normalization(**old)

    result = {
        "protocol": "5-fold train OOF; each row excluded from model fit and threshold selection",
        "train_path": str(args.train),
        "rows": len(records),
        "retriever": args.retriever,
        "strategy": "source-specific + reject chip, support gates 20 positive / 50 negative",
        "folds": fold_meta,
        "summary": summarize(records, args.folds),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
