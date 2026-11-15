"""Fit the optional confidence calibrator with a train-internal calibration split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrated_normalize import DEFAULT_MODEL, fit_and_save

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "dicts" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=RAW / "CHIP-CDN_train.json")
    parser.add_argument(
        "--dev",
        type=Path,
        default=RAW / "CHIP-CDN_dev.json",
        help="held-out report split; never used for threshold selection",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument(
        "--min-accepted",
        type=int,
        default=50,
        help="minimum calibration mentions supporting an operating point",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.03,
        help="minimum calibration coverage supporting an operating point",
    )
    parser.add_argument(
        "--min-source-positives",
        type=int,
        default=20,
        help="minimum fit positives required for a source-specific calibrator",
    )
    parser.add_argument(
        "--min-source-negatives",
        type=int,
        default=50,
        help="minimum fit negatives required for a source-specific calibrator",
    )
    parser.add_argument(
        "--source-shrinkage-strength",
        type=float,
        default=0.0,
        help="shrink source-specific coefficients toward the global fit; 0 disables shrinkage",
    )
    parser.add_argument(
        "--source-model-mode",
        choices=("auto", "global", "source", "shrink"),
        default="auto",
        help="auto preserves legacy behavior: source models at strength 0, shrinkage at positive strength",
    )
    parser.add_argument(
        "--reject-source",
        action="append",
        choices=("exact", "variant", "rule", "curated", "chip", "fuzzy", "bert", "supervised"),
        default=[],
        help="reject all mentions whose top candidate comes from this source (repeatable)",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument(
        "--neural-candidates",
        action="store_true",
        help="include BERT/SapBERT top-k candidates; requires the local encoder cache",
    )
    parser.add_argument("--retriever", choices=("roberta", "sapbert"), default="roberta")
    args = parser.parse_args()
    calibrator = fit_and_save(
        args.train,
        args.dev,
        args.out,
        include_neural=args.neural_candidates,
        retriever=args.retriever,
        target_precision=args.target_precision,
        calibration_fraction=args.calibration_fraction,
        min_accepted=args.min_accepted,
        min_coverage=args.min_coverage,
        min_source_positives=args.min_source_positives,
        min_source_negatives=args.min_source_negatives,
        source_shrinkage_strength=args.source_shrinkage_strength,
        source_model_mode=args.source_model_mode,
        rejected_sources=tuple(args.reject_source),
    )
    print(
        json.dumps(
            {
                "model": str(args.out),
                "threshold": calibrator.threshold,
                "min_margin": calibrator.min_margin,
                "target_precision": args.target_precision,
                "min_accepted": args.min_accepted,
                "min_coverage": args.min_coverage,
                "min_source_positives": args.min_source_positives,
                "min_source_negatives": args.min_source_negatives,
                "source_shrinkage_strength": args.source_shrinkage_strength,
                "effective_source_shrinkage_strength": calibrator.calibration_info.get("source_shrinkage_strength"),
                "source_model_mode": calibrator.calibration_info.get("source_model_mode"),
                "rejected_sources": sorted(calibrator.rejected_sources),
                "neural_candidates": args.neural_candidates,
                "retriever": args.retriever,
                "calibration": calibrator.calibration_info,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
