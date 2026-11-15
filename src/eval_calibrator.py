"""Evaluate the fitted abstaining normalizer on a held-out CHIP-CDN split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from calibrated_normalize import (
    DEFAULT_MODEL,
    CandidateGenerator,
    ConfidenceCalibrator,
    install_alias_rows,
    load_chip_rows,
    split_calibration_rows,
    _compact,
    NEURAL_TOP_K,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "dicts" / "raw"
OUT = ROOT / "outputs" / "metrics"


def evaluate(rows, calibrator, generator) -> dict:
    accepted = correct = gold_recalled = pool_recalled = 0
    source_counts = Counter()
    source_correct = Counter()
    source_offered = Counter()
    source_accepted = Counter()
    source_accepted_correct = Counter()
    reason_counts = Counter()
    for surface, gold in rows:
        pool = generator.generate(surface)
        decision = calibrator.decide(surface, pool)
        gold_set = {_compact(x) for x in gold if _compact(x) != "癌"}
        if any(_compact(c.name) in gold_set for c in pool):
            pool_recalled += 1
        reason_counts[decision.reason] += 1
        source = decision.candidates[0]["source"] if decision.candidates else None
        if source:
            source_offered[source] += 1
        if decision.accepted:
            accepted += 1
            source_counts[decision.source] += 1
            if source:
                source_accepted[source] += 1
            if _compact(decision.name) in gold_set:
                correct += 1
                source_correct[decision.source] += 1
                if source:
                    source_accepted_correct[source] += 1
        if any(_compact(c["name"]) in gold_set for c in decision.candidates):
            gold_recalled += 1
    n = len(rows)
    return {
        "n": n,
        "threshold": calibrator.threshold,
        "accepted": accepted,
        "coverage": accepted / n if n else 0.0,
        "accepted_precision": correct / accepted if accepted else 0.0,
        "overall_accuracy": correct / n if n else 0.0,
        "candidate_recall_at_5": gold_recalled / n if n else 0.0,
        "candidate_pool_recall": pool_recalled / n if n else 0.0,
        "correct": correct,
        "source_counts": dict(source_counts),
        "source_precision": {
            source: source_correct[source] / count
            for source, count in source_counts.items()
        },
        "per_source": {
            source: {
                "offered": source_offered[source],
                "offered_rate": source_offered[source] / n if n else 0.0,
                "accepted": source_accepted[source],
                "coverage": source_accepted[source] / n if n else 0.0,
                "conditional_coverage": source_accepted[source] / source_offered[source] if source_offered[source] else 0.0,
                "correct": source_accepted_correct[source],
                "precision": source_accepted_correct[source] / source_accepted[source] if source_accepted[source] else 0.0,
            }
            for source in sorted(set(source_offered) | set(source_accepted))
        },
        "reason_counts": dict(reason_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT / "calibrated_normalization.json",
        help="metrics output path; use a separate path when comparing calibrators",
    )
    parser.add_argument("--neural-candidates", action="store_true")
    parser.add_argument("--retriever", choices=("roberta", "sapbert"), default="roberta")
    parser.add_argument(
        "--include-chip-cdn",
        action="store_true",
        help="include the train/dev-derived alias table; disabled by default for strict held-out evaluation",
    )
    args = parser.parse_args()
    data = args.data or (RAW / f"CHIP-CDN_{args.split}.json")
    calibrator = ConfidenceCalibrator.load(args.model)
    from normalize import configure_normalization, normalization_settings

    old = normalization_settings()
    configure_normalization(use_chip_cdn=args.include_chip_cdn, retriever=args.retriever)
    try:
        alias_source = "full_chip_cdn" if args.include_chip_cdn else "train_only"
        if not args.include_chip_cdn:
            train_rows = load_chip_rows(RAW / "CHIP-CDN_train.json")
            fit_rows, _calibration_rows = split_calibration_rows(train_rows)
            install_alias_rows(fit_rows)
            alias_source = "train_fit_only"
        generator = CandidateGenerator(include_neural=args.neural_candidates, retriever=args.retriever)
        result = {
            "split": args.split,
            "model": str(args.model),
            "neural_candidates": args.neural_candidates,
            "retriever": args.retriever,
            "include_chip_cdn": args.include_chip_cdn,
            "alias_source": alias_source,
            "neural_top_k": NEURAL_TOP_K,
            "metrics": evaluate(load_chip_rows(data), calibrator, generator),
        }
    finally:
        configure_normalization(**old)
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
