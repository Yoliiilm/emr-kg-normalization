"""Diagnose calibration-to-held-out distribution shift without fitting anything."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from calibrated_normalize import (
    DEFAULT_MODEL,
    NEURAL_TOP_K,
    CandidateGenerator,
    ConfidenceCalibrator,
    _compact,
    install_alias_rows,
    load_chip_rows,
    split_calibration_rows,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "dicts" / "raw"
OUT = ROOT / "outputs" / "metrics"


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        str(q): round(ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))], 6)
        for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    }


def summarize(rows, calibrator: ConfidenceCalibrator, generator: CandidateGenerator) -> dict:
    best_scores, margins, pool_sizes = [], [], []
    records = []
    source_counts = Counter()
    pool_recall = accepted = correct = 0
    for surface, gold in rows:
        pool = generator.generate(surface)
        gold_set = {_compact(x) for x in gold if _compact(x) != "癌"}
        if any(_compact(c.name) in gold_set for c in pool):
            pool_recall += 1
        pool_sizes.append(len(pool))
        if pool:
            scored = sorted(((calibrator.score(c), c) for c in pool), key=lambda x: -x[0])
            best_scores.append(scored[0][0])
            margin = scored[0][0] - (scored[1][0] if len(scored) > 1 else 0.0)
            margins.append(margin)
            records.append((scored[0][0], margin, scored[0][1].source,
                            int(_compact(scored[0][1].name) in gold_set)))
        decision = calibrator.decide(surface, pool)
        if decision.accepted:
            accepted += 1
            source_counts[decision.source] += 1
            correct += int(_compact(decision.name) in gold_set)
    n = len(rows)
    curve = []
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.94):
        selected = [label for p, margin, source, label in records
                    if p >= threshold
                    and p >= calibrator.SOURCE_FLOORS.get(source, threshold)
                    and margin >= calibrator.min_margin]
        curve.append({
            "threshold": threshold,
            "accepted": len(selected),
            "coverage": len(selected) / n if n else 0.0,
            "precision": sum(selected) / len(selected) if selected else 0.0,
        })
    source_strata = {}
    for source in sorted({record[2] for record in records}):
        available = [record for record in records if record[2] == source]
        accepted_source = [
            record for record in available
            if record[0] >= max(calibrator.threshold, calibrator.SOURCE_FLOORS.get(source, calibrator.threshold))
            and record[1] >= calibrator.min_margin
        ]
        source_strata[source] = {
            "available": len(available),
            "best_candidate_precision": sum(record[3] for record in available) / len(available),
            "accepted": len(accepted_source),
            "accepted_coverage_of_source": len(accepted_source) / len(available),
            "accepted_precision": (
                sum(record[3] for record in accepted_source) / len(accepted_source)
                if accepted_source else 0.0
            ),
            "mean_confidence": statistics.mean(record[0] for record in available),
            "mean_margin": statistics.mean(record[1] for record in available),
        }
    return {
        "n": n,
        "candidate_pool_recall": pool_recall / n if n else 0.0,
        "candidate_count": {
            "mean": round(statistics.mean(pool_sizes), 4) if pool_sizes else 0.0,
            "median": statistics.median(pool_sizes) if pool_sizes else 0.0,
            "p90": quantiles([float(x) for x in pool_sizes]).get("0.9", 0.0),
        },
        "best_confidence": quantiles(best_scores),
        "margin": quantiles(margins),
        "accepted": accepted,
        "coverage": accepted / n if n else 0.0,
        "accepted_precision": correct / accepted if accepted else 0.0,
        "source_counts": dict(source_counts),
        "source_strata": source_strata,
        "threshold_curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train", type=Path, default=RAW / "CHIP-CDN_train.json")
    parser.add_argument("--dev", type=Path, default=RAW / "CHIP-CDN_dev.json")
    parser.add_argument("--retriever", choices=("roberta", "sapbert"), default="roberta")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--neural-candidates", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT / "calibration_shift.json")
    args = parser.parse_args()

    from normalize import configure_normalization, normalization_settings

    train_rows = load_chip_rows(args.train)
    fit_rows, calibration_rows = split_calibration_rows(train_rows, args.calibration_fraction)
    calibrator = ConfidenceCalibrator.load(args.model)
    old = normalization_settings()
    configure_normalization(use_chip_cdn=False, retriever=args.retriever)
    try:
        install_alias_rows(fit_rows)
        generator = CandidateGenerator(
            include_neural=args.neural_candidates,
            retriever=args.retriever,
        )
        result = {
            "model": str(args.model),
            "retriever": args.retriever,
            "neural_candidates": args.neural_candidates,
            "neural_top_k": NEURAL_TOP_K,
            "calibration": summarize(calibration_rows, calibrator, generator),
            "dev": summarize(load_chip_rows(args.dev), calibrator, generator),
        }
    finally:
        configure_normalization(**old)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
