"""Compare calibration strategies with nested, train-internal held-out folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from calibrated_normalize import (
    CandidateGenerator,
    ConfidenceCalibrator,
    Decision,
    SOURCE_PRIOR,
    install_alias_rows,
    load_chip_rows,
    split_calibration_rows,
    _compact,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "dicts" / "raw"
DEFAULT_OUT = ROOT / "outputs" / "metrics" / "experiments" / "source_shrinkage" / "train_internal_folds.json"
MIN_SOURCE_POSITIVES = 20
MIN_SOURCE_NEGATIVES = 50


def fold_rows(rows, fold: int, folds: int):
    fit, held_out = [], []
    for row in rows:
        bucket = int.from_bytes(hashlib.sha1(row[0].encode("utf-8")).digest()[:4], "big") % folds
        (held_out if bucket == fold else fit).append(row)
    return fit, held_out


class CachedGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.cache = {}

    def generate(self, surface):
        if surface not in self.cache:
            self.cache[surface] = self.generator.generate(surface)
        return self.cache[surface]


class ScoreThresholdBaseline:
    """Train-calibrated threshold over a conventional candidate confidence score."""

    def __init__(self, method: str, threshold: float, min_margin: float, *, always_accept: bool = False):
        self.method = method
        self.threshold = threshold
        self.min_margin = min_margin
        self.always_accept = always_accept

    def score(self, candidate) -> float:
        if self.method == "raw_score":
            return candidate.raw_score
        if self.method == "lexical":
            return candidate.lexical
        if self.method == "source_prior":
            return SOURCE_PRIOR.get(candidate.source, 0.5)
        raise ValueError(f"unknown baseline method: {self.method}")

    def decide(self, surface, candidates):
        if not candidates:
            from normalize import clean_surface

            return Decision(surface, clean_surface(surface), "calibrated_reject", 0.0, False, 0.0, "no_candidate")
        ranked = sorted(
            ((self.score(candidate), candidate) for candidate in candidates),
            key=lambda pair: (-pair[0], -pair[1].lexical, pair[1].name),
        )
        best_score, best = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        margin = best_score - second_score
        accepted = self.always_accept or (best_score >= self.threshold and margin >= self.min_margin)
        return Decision(
            surface,
            best.name if accepted else surface,
            f"baseline_{best.source}" if accepted else "calibrated_reject",
            best_score,
            accepted,
            margin,
            "accepted" if accepted else "low_confidence",
            tuple({"name": candidate.name, "source": candidate.source, "confidence": score}
                  for score, candidate in ranked[:5]),
        )


def ranked_baseline_candidates(candidates, method):
    scorer = ScoreThresholdBaseline(method, 0.0, 0.0)
    return sorted(
        ((scorer.score(candidate), candidate) for candidate in candidates),
        key=lambda pair: (-pair[0], -pair[1].lexical, pair[1].name),
    )


def fit_score_baseline(rows, generator, method, *, target_precision, min_accepted, min_coverage):
    scored_rows = []
    score_values = set()
    for surface, gold in rows:
        ranked = ranked_baseline_candidates(generator.generate(surface), method)
        if not ranked:
            continue
        best_score, best = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        correct = int(_compact(best.name) in {_compact(x) for x in gold if _compact(x) != "癌"})
        scored_rows.append((best_score, best_score - second_score, correct))
        score_values.add(round(best_score, 6))
    if not scored_rows:
        raise ValueError(f"no candidates for {method} baseline calibration")
    thresholds = sorted(score_values | {0.0})
    margins = (0.0, 0.01, 0.03, 0.05, 0.08, 0.12)
    points = []
    denominator = max(1, len(rows))
    for threshold in thresholds:
        for margin in margins:
            labels = [correct for score, delta, correct in scored_rows if score >= threshold and delta >= margin]
            accepted = len(labels)
            precision = sum(labels) / accepted if accepted else 0.0
            coverage = accepted / denominator
            if accepted >= min_accepted and coverage >= min_coverage:
                points.append({"threshold": threshold, "min_margin": margin, "accepted": accepted,
                               "precision": precision, "coverage": coverage})
    if not points:
        raise ValueError(f"no supported operating point for {method} baseline")
    target_points = [point for point in points if point["precision"] >= target_precision]
    chosen = max(target_points or points,
                 key=lambda point: (point["precision"], point["coverage"], point["threshold"], -point["min_margin"]))
    return ScoreThresholdBaseline(method, chosen["threshold"], chosen["min_margin"]), chosen


def collect_confidence_rows(rows, strategy, generator):
    predictions = []
    for surface, gold in rows:
        decision = strategy.decide(surface, generator.generate(surface))
        candidate = decision.candidates[0] if decision.candidates else None
        correct = bool(candidate and _compact(candidate["name"]) in {
            _compact(value) for value in gold if _compact(value) != "癌"
        })
        predictions.append({
            "surface": surface,
            "confidence": float(decision.confidence) if candidate else -1.0,
            "correct": int(correct),
            "accepted": int(decision.accepted),
        })
    return predictions


def risk_coverage_summary(predictions, target_precision: float) -> dict:
    ordered = sorted(predictions, key=lambda row: (-row["confidence"], row["surface"]))
    total = len(ordered)
    points = {}
    for coverage in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0):
        accepted = min(total, max(1, round(total * coverage))) if total else 0
        correct = sum(row["correct"] for row in ordered[:accepted])
        points[f"{coverage:.2f}"] = {
            "accepted": accepted,
            "actual_coverage": accepted / total if total else 0.0,
            "precision": correct / accepted if accepted else 0.0,
        }
    best_count = best_correct = 0
    correct = 0
    for count, row in enumerate(ordered, start=1):
        correct += row["correct"]
        if correct / count >= target_precision:
            best_count, best_correct = count, correct
    return {
        "n": total,
        "max_coverage_at_target_precision_oracle": best_count / total if total else 0.0,
        "accepted_at_target_precision_oracle": best_count,
        "precision_at_target_precision_oracle": best_correct / best_count if best_count else 0.0,
        "precision_at_fixed_coverage": points,
        "interpretation": "Outer-fold oracle frontier uses held-out labels to describe ranking quality only; it is not a deployable threshold or an unbiased selected operating point.",
    }


def _risk_coverage_metrics(rows: list[dict]) -> tuple[float, float]:
    accepted = [row for row in rows if row["accepted"]]
    coverage = len(accepted) / len(rows) if rows else 0.0
    precision = sum(row["correct"] for row in accepted) / len(accepted) if accepted else 0.0
    return coverage, precision


def paired_strategy_bootstrap(predictions: dict[str, list[dict]], reference: str = "global", *, samples: int = 2000, seed: int = 23) -> dict:
    """Cluster-bootstrap paired outer-fold outcomes by disease surface."""
    if reference not in predictions:
        return {}
    reference_rows = predictions[reference]
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(reference_rows):
        groups.setdefault(row["surface"], []).append(index)
    surfaces = sorted(groups)
    rng = random.Random(seed)
    result = {}
    for name, rows in predictions.items():
        if name == reference or len(rows) != len(reference_rows):
            continue
        if [row["surface"] for row in rows] != [row["surface"] for row in reference_rows]:
            raise ValueError(f"outer predictions are not aligned for {name} vs {reference}")
        point_cov = _risk_coverage_metrics(rows)[0] - _risk_coverage_metrics(reference_rows)[0]
        point_prec = _risk_coverage_metrics(rows)[1] - _risk_coverage_metrics(reference_rows)[1]
        coverage_deltas, precision_deltas = [], []
        for _ in range(samples):
            sampled_surfaces = [rng.choice(surfaces) for _ in surfaces]
            indices = [index for surface in sampled_surfaces for index in groups[surface]]
            sampled_rows = [rows[index] for index in indices]
            sampled_ref = [reference_rows[index] for index in indices]
            coverage_deltas.append(_risk_coverage_metrics(sampled_rows)[0] - _risk_coverage_metrics(sampled_ref)[0])
            precision_deltas.append(_risk_coverage_metrics(sampled_rows)[1] - _risk_coverage_metrics(sampled_ref)[1])

        def interval(values):
            values.sort()
            return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]

        result[name] = {
            "reference": reference,
            "coverage_delta": point_cov,
            "coverage_delta_cluster_bootstrap_95ci": interval(coverage_deltas),
            "precision_delta": point_prec,
            "precision_delta_cluster_bootstrap_95ci": interval(precision_deltas),
            "bootstrap_samples": samples,
            "cluster_unit": "disease_surface",
        }
    return result


def render_report(result: dict) -> str:
    lines = [
        "Nested train-only comparison of normalization decision methods",
        "=" * 76,
        f"retriever={result['retriever']}; neural_candidates={result['neural_candidates']}; folds={result['folds']};",
        f"evaluated outer-fold mentions per strategy={next(iter(result['summary'].values()))['n']}",
        f"target precision={result['target_precision']:.0%}; source fit gates={result['min_source_positives']} positives / {result['min_source_negatives']} negatives",
        "Thresholds and coefficients are fitted on outer-training data only; outer folds are evaluation-only.",
        "",
        f"{'strategy':30s} {'accepted':>9s} {'coverage':>10s} {'precision':>11s} {'F1':>8s}",
    ]
    preferred = ["retrieval_top1", "raw_score_threshold", "lexical_threshold", "source_prior_threshold",
                 "global", "source", "shrink", "chip_reject", "shrink_chip_reject"]
    preferred += sorted(name for name in result["summary"] if name.startswith("shrink_strength_"))
    for name in preferred:
        row = result["summary"].get(name)
        if row:
            lines.append(f"{name:30s} {row['accepted']:9d} {row['coverage']:10.3%} {row['precision']:11.3%} {row['f1']:8.4f}")
    lines += ["", "Per-source accepted coverage / precision"]
    for mode in ("global", "source", "shrink", "chip_reject", "shrink_chip_reject"):
        if mode not in result["summary"]:
            continue
        lines.append(f"{mode}:")
        for source, row in sorted(result["summary"][mode]["sources"].items()):
            if row["offered"]:
                lines.append(f"  {source:14s} offered={row['offered']:4d} accepted={row['accepted']:4d} coverage={row['coverage']:.3%} precision={row['precision']:.3%}")
    lines += ["", "Shrinkage sensitivity (same folds; zero means source-specific fit)"]
    for name in sorted((n for n in result["summary"] if n.startswith("shrink_strength_")), key=lambda n: float(n.removeprefix("shrink_strength_"))):
        row = result["summary"][name]
        lines.append(f"{name}: coverage={row['coverage']:.3%}, precision={row['precision']:.3%}, accepted={row['accepted']}")
    lines += ["", "Outer-fold confidence ranking frontier (oracle diagnostic; not a deployable policy)"]
    for name, row in result["outer_fold_risk_coverage"].items():
        lines.append(f"{name}: best prefix at >= {result['target_precision']:.0%} precision = {row['accepted_at_target_precision_oracle']} / {row['n']} ({row['max_coverage_at_target_precision_oracle']:.3%})")
    lines += ["", "Paired outer-fold deltas vs global (surface-cluster bootstrap 95% CI)"]
    lines.append(f"{'strategy':30s} {'coverage delta':>16s} {'precision delta':>17s}")
    for name in preferred:
        row = result["paired_outer_strategy_deltas_vs_global"].get(name)
        if row:
            cov_ci = row["coverage_delta_cluster_bootstrap_95ci"]
            prec_ci = row["precision_delta_cluster_bootstrap_95ci"]
            lines.append(
                f"{name:30s} {row['coverage_delta']:+.3%} [{cov_ci[0]:+.3%}, {cov_ci[1]:+.3%}] "
                f"{row['precision_delta']:+.3%} [{prec_ci[0]:+.3%}, {prec_ci[1]:+.3%}]"
            )
    lines += ["", "Caution: CHIP-CDN train is reused for internal development/evaluation and is not an independent benchmark.",
              "No method met the target merely by assumption; report the held-out results and do not treat the outer-fold oracle as a selected threshold."]
    return "\n".join(lines) + "\n"


def evaluate_rows(rows, calibrator, generator) -> dict:
    total = accepted = correct = 0
    offered = Counter()
    source_accepted = Counter()
    source_correct = Counter()
    reason_counts = Counter()
    for surface, gold in rows:
        total += 1
        pool = generator.generate(surface)
        decision = calibrator.decide(surface, pool)
        reason_counts[decision.reason] += 1
        source = decision.candidates[0]["source"] if decision.candidates else "no_candidate"
        offered[source] += 1
        is_correct = bool({_compact(x) for x in gold if _compact(x) != "癌"} & {
            _compact(decision.name)
        })
        if decision.accepted:
            accepted += 1
            source_accepted[source] += 1
            if is_correct:
                correct += 1
                source_correct[source] += 1
    sources = {}
    for source in sorted(set(offered) | set(source_accepted)):
        sources[source] = {
            "offered": offered[source],
            "offered_rate": offered[source] / total if total else 0.0,
            "accepted": source_accepted[source],
            "coverage": source_accepted[source] / total if total else 0.0,
            "conditional_coverage": source_accepted[source] / offered[source] if offered[source] else 0.0,
            "precision": source_correct[source] / source_accepted[source] if source_accepted[source] else 0.0,
            "correct": source_correct[source],
        }
    return {
        "n": total,
        "accepted": accepted,
        "correct": correct,
        "coverage": accepted / total if total else 0.0,
        "precision": correct / accepted if accepted else 0.0,
        "f1": 2 * (correct / total) * (correct / accepted) / ((correct / total) + (correct / accepted))
        if accepted and correct else 0.0,
        "sources": sources,
        "reason_counts": dict(reason_counts),
    }


def aggregate(folds: list[dict], modes: list[str]) -> dict:
    summary = {}
    for mode in modes:
        rows = [fold["strategies"][mode] for fold in folds if fold["strategies"].get(mode, {}).get("status") == "ok"]
        total = sum(row["evaluation"]["n"] for row in rows)
        accepted = sum(row["evaluation"]["accepted"] for row in rows)
        correct = sum(row["evaluation"]["correct"] for row in rows)
        sources = {}
        source_names = sorted({source for row in rows for source in row["evaluation"]["sources"]})
        for source in source_names:
            offered = sum(row["evaluation"]["sources"].get(source, {}).get("offered", 0) for row in rows)
            source_accepted = sum(row["evaluation"]["sources"].get(source, {}).get("accepted", 0) for row in rows)
            source_correct = sum(row["evaluation"]["sources"].get(source, {}).get("correct", 0) for row in rows)
            sources[source] = {
                "offered": offered,
                "offered_rate": offered / total if total else 0.0,
                "accepted": source_accepted,
                "coverage": source_accepted / total if total else 0.0,
                "conditional_coverage": source_accepted / offered if offered else 0.0,
                "precision": source_correct / source_accepted if source_accepted else 0.0,
                "correct": source_correct,
            }
        support_names = sorted({
            source
            for row in rows
            for source in row.get("fit_support", {})
        })
        source_fit_support = {
            source: {
                "eligible_folds": sum(bool(row.get("fit_support", {}).get(source, {}).get("eligible")) for row in rows),
                "fitted_folds": sum(bool(row.get("fit_support", {}).get(source, {}).get("fitted")) for row in rows),
                "positive_min": min((row.get("fit_support", {}).get(source, {}).get("positive", 0) for row in rows), default=0),
                "negative_min": min((row.get("fit_support", {}).get(source, {}).get("negative", 0) for row in rows), default=0),
            }
            for source in support_names
        }
        summary[mode] = {
            "successful_folds": len(rows),
            "n": total,
            "accepted": accepted,
            "correct": correct,
            "coverage": accepted / total if total else 0.0,
            "precision": correct / accepted if accepted else 0.0,
            "f1": 2 * (correct / total) * (correct / accepted) / ((correct / total) + (correct / accepted))
            if accepted and correct else 0.0,
            "sources": sources,
            "source_fit_support": source_fit_support,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=RAW / "CHIP-CDN_train.json")
    parser.add_argument("--retriever", choices=("roberta", "sapbert"), default="roberta")
    parser.add_argument("--neural-candidates", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument("--min-accepted", type=int, default=50)
    parser.add_argument("--min-coverage", type=float, default=0.03)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--source-shrinkage-strength", type=float, default=100.0)
    parser.add_argument(
        "--shrinkage-strengths",
        default="0,10,25,50,100,250,1000",
        help="comma-separated predeclared strengths for shrinkage sensitivity analysis; 0 is source-specific",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.source_shrinkage_strength <= 0:
        parser.error("--source-shrinkage-strength must be positive for the shrinkage comparison")

    from normalize import configure_normalization, normalization_settings

    try:
        shrinkage_strengths = sorted({float(value) for value in args.shrinkage_strengths.split(",")})
    except ValueError:
        parser.error("--shrinkage-strengths must be a comma-separated list of numbers")
    if not shrinkage_strengths or any(value < 0 for value in shrinkage_strengths):
        parser.error("--shrinkage-strengths must contain non-negative values")
    strength_modes = {f"shrink_strength_{strength:g}": strength for strength in shrinkage_strengths}
    baseline_modes = ["retrieval_top1", "raw_score_threshold", "lexical_threshold", "source_prior_threshold"]
    modes = ["global", "source", "shrink", "chip_reject", "shrink_chip_reject", *strength_modes, *baseline_modes]
    rows = load_chip_rows(args.train)
    old = normalization_settings()
    configure_normalization(use_chip_cdn=False, retriever=args.retriever)
    fold_results = []
    outer_predictions = {mode: [] for mode in modes}
    try:
        for fold in range(args.folds):
            outer_train, evaluation_rows = fold_rows(rows, fold, args.folds)
            try:
                fit_rows, calibration_rows = split_calibration_rows(outer_train, args.calibration_fraction)
            except ValueError as exc:
                fold_results.append({"fold": fold, "status": "failed", "error": str(exc), "strategies": {}})
                continue
            install_alias_rows(fit_rows)
            generator = CachedGenerator(CandidateGenerator(
                include_neural=args.neural_candidates,
                retriever=args.retriever,
            ))
            strategy_rows = {}
            specifications = {
                "global": ("global", 0.0, ()),
                "source": ("source", 0.0, ()),
                "shrink": ("shrink", args.source_shrinkage_strength, ()),
                "chip_reject": ("source", 0.0, ("chip",)),
                "shrink_chip_reject": ("shrink", args.source_shrinkage_strength, ("chip",)),
            }
            for mode, strength in strength_modes.items():
                specifications[mode] = ("source" if strength == 0 else "shrink", strength, ())
            for mode, (source_model_mode, strength, rejected_sources) in specifications.items():
                try:
                    calibrator = ConfidenceCalibrator.fit(
                        fit_rows,
                        calibration_rows,
                        generator=generator,
                        target_precision=args.target_precision,
                        min_accepted=args.min_accepted,
                        min_coverage=args.min_coverage,
                        min_source_positives=MIN_SOURCE_POSITIVES,
                        min_source_negatives=MIN_SOURCE_NEGATIVES,
                        source_shrinkage_strength=strength,
                        source_model_mode=source_model_mode,
                        rejected_sources=rejected_sources,
                    )
                    strategy_rows[mode] = {
                        "status": "ok",
                        "threshold": calibrator.threshold,
                        "min_margin": calibrator.min_margin,
                        "fit_support": calibrator.calibration_info["source_support"],
                        "evaluation": evaluate_rows(evaluation_rows, calibrator, generator),
                    }
                    outer_predictions[mode].extend(collect_confidence_rows(evaluation_rows, calibrator, generator))
                except ValueError as exc:
                    strategy_rows[mode] = {"status": "failed", "error": str(exc)}

            top1 = ScoreThresholdBaseline("raw_score", float("-inf"), 0.0, always_accept=True)
            strategy_rows["retrieval_top1"] = {
                "status": "ok",
                "threshold": None,
                "min_margin": None,
                "fit_support": {},
                "evaluation": evaluate_rows(evaluation_rows, top1, generator),
            }
            outer_predictions["retrieval_top1"].extend(collect_confidence_rows(evaluation_rows, top1, generator))
            for baseline_name, score_method in (
                ("raw_score_threshold", "raw_score"),
                ("lexical_threshold", "lexical"),
                ("source_prior_threshold", "source_prior"),
            ):
                try:
                    baseline, operating_point = fit_score_baseline(
                        calibration_rows,
                        generator,
                        score_method,
                        target_precision=args.target_precision,
                        min_accepted=args.min_accepted,
                        min_coverage=args.min_coverage,
                    )
                    strategy_rows[baseline_name] = {
                        "status": "ok",
                        "threshold": operating_point["threshold"],
                        "min_margin": operating_point["min_margin"],
                        "calibration_operating_point": operating_point,
                        "fit_support": {},
                        "evaluation": evaluate_rows(evaluation_rows, baseline, generator),
                    }
                    outer_predictions[baseline_name].extend(collect_confidence_rows(evaluation_rows, baseline, generator))
                except ValueError as exc:
                    strategy_rows[baseline_name] = {"status": "failed", "error": str(exc)}
            fold_results.append({
                "fold": fold,
                "status": "ok",
                "fit_rows": len(fit_rows),
                "calibration_rows": len(calibration_rows),
                "evaluation_rows": len(evaluation_rows),
                "strategies": strategy_rows,
            })
    finally:
        configure_normalization(**old)

    summary = aggregate(fold_results, modes)
    risk_coverage = {
        mode: risk_coverage_summary(predictions, args.target_precision)
        for mode, predictions in outer_predictions.items()
        if predictions
    }
    chip_shrink = summary["shrink"]["sources"].get("chip", {})
    chip_fit = summary["shrink"]["source_fit_support"].get("chip", {})
    enough_chip_support = chip_shrink.get("accepted", 0) >= args.min_accepted
    chip_precision = chip_shrink.get("precision", 0.0)
    chip_policy = "shrink" if chip_fit.get("fitted_folds", 0) > 0 and enough_chip_support and chip_precision >= args.target_precision else "strict_reject"
    result = {
        "protocol": "nested train-only CV: outer held-out fold is not used to fit coefficients or select thresholds",
        "train_path": str(args.train),
        "folds": args.folds,
        "successful_folds": sum(row["status"] == "ok" for row in fold_results),
        "neural_candidates": args.neural_candidates,
        "retriever": args.retriever,
        "min_source_positives": MIN_SOURCE_POSITIVES,
        "min_source_negatives": MIN_SOURCE_NEGATIVES,
        "target_precision": args.target_precision,
        "source_shrinkage_strength": args.source_shrinkage_strength,
        "shrinkage_strengths_sensitivity_grid": shrinkage_strengths,
        "summary": summary,
        "outer_fold_risk_coverage": risk_coverage,
        "paired_outer_strategy_deltas_vs_global": paired_strategy_bootstrap(outer_predictions),
        "chip_policy_recommendation": {
            "policy": chip_policy,
            "reason": (
                f"chip source model fitted in {chip_fit.get('fitted_folds', 0)}/{args.folds} folds; "
                f"shrunk precision {chip_precision:.3f} on {chip_shrink.get('accepted', 0)} accepted outer-fold mentions; "
                f"requires >= {args.target_precision:.3f} precision and >= {args.min_accepted} accepted"
            ),
            "note": "chip is shrink-eligible only in folds meeting both 20-positive and 50-negative fit gates",
        },
        "folds_detail": fold_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    text_out = args.out.with_suffix(".txt")
    text_out.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {text_out}")


if __name__ == "__main__":
    main()
