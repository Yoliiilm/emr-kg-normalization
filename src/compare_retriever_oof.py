"""Paired comparison of two row-aligned train OOF calibrator audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "metrics" / "experiments" / "source_shrinkage" / "retriever_oof_comparison.json"


def metric_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    accepted = np.asarray([row["accepted"] for row in records], dtype=bool)
    correct = np.asarray([row["correct"] for row in records], dtype=bool)
    return {
        "candidate_pool_recall": np.asarray([row["gold_in_pool"] for row in records], dtype=float),
        "scored_top5_recall": np.asarray([row["gold_in_scored_top5"] for row in records], dtype=float),
        "coverage": accepted.astype(float),
        "accepted_correct": correct.astype(float),
        "accepted": accepted.astype(float),
    }


def summarize_pair(left: dict, right: dict, *, bootstrap_samples: int = 2000, seed: int = 42) -> dict:
    left_records, right_records = left["records"], right["records"]
    if len(left_records) != len(right_records):
        raise ValueError("OOF audits have different row counts")
    keys = ("fold", "surface", "gold")
    for index, (a, b) in enumerate(zip(left_records, right_records)):
        if any(a[key] != b[key] for key in keys):
            raise ValueError(f"OOF records are not aligned at row {index}")
    n = len(left_records)
    if n == 0:
        raise ValueError("OOF audits are empty")
    l_arr, r_arr = metric_arrays(left_records), metric_arrays(right_records)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, n, size=(bootstrap_samples, n), endpoint=False)
    result = {}
    for metric in ("candidate_pool_recall", "scored_top5_recall", "coverage", "accepted_precision"):
        if metric == "accepted_precision":
            l_num = l_arr["accepted_correct"][samples].sum(axis=1)
            l_den = l_arr["accepted"][samples].sum(axis=1)
            r_num = r_arr["accepted_correct"][samples].sum(axis=1)
            r_den = r_arr["accepted"][samples].sum(axis=1)
            l_boot = np.divide(l_num, l_den, out=np.zeros_like(l_num), where=l_den > 0)
            r_boot = np.divide(r_num, r_den, out=np.zeros_like(r_num), where=r_den > 0)
            l_point = float(l_arr["accepted_correct"].sum() / max(1, l_arr["accepted"].sum()))
            r_point = float(r_arr["accepted_correct"].sum() / max(1, r_arr["accepted"].sum()))
        else:
            l_values, r_values = l_arr[metric], r_arr[metric]
            l_boot = l_values[samples].mean(axis=1)
            r_boot = r_values[samples].mean(axis=1)
            l_point, r_point = float(l_values.mean()), float(r_values.mean())
        delta_boot = r_boot - l_boot
        result[metric] = {
            "roberta": l_point,
            "sapbert": r_point,
            "delta_sapbert_minus_roberta": r_point - l_point,
            "paired_bootstrap_95ci_delta": [float(x) for x in np.quantile(delta_boot, [0.025, 0.975])],
        }
    result["counts"] = {
        "rows": n,
        "roberta_accepted": int(l_arr["accepted"].sum()),
        "sapbert_accepted": int(r_arr["accepted"].sum()),
        "roberta_correct_accepted": int(l_arr["accepted_correct"].sum()),
        "sapbert_correct_accepted": int(r_arr["accepted_correct"].sum()),
    }
    result["protocol"] = "Mention-paired bootstrap on the same 5-fold CHIP-CDN train OOF rows; strict string correctness; 20/50 source support gates; chip top-candidate path rejected in both runs."
    result["bootstrap_samples"] = bootstrap_samples
    result["seed"] = seed
    result["interpretation"] = "Candidate recall and accepted precision are distinct; ICD proximity is not counted as correct."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roberta", type=Path, default=ROOT / "outputs" / "metrics" / "experiments" / "source_shrinkage" / "train_oof_audit_neural_source_chip_reject.json")
    parser.add_argument("--sapbert", type=Path, default=ROOT / "outputs" / "metrics" / "experiments" / "source_shrinkage" / "train_oof_audit_neural_sapbert_source_chip_reject.json")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    left = json.loads(args.roberta.read_text(encoding="utf-8"))
    right = json.loads(args.sapbert.read_text(encoding="utf-8"))
    result = summarize_pair(left, right, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    result["inputs"] = {"roberta": str(args.roberta), "sapbert": str(args.sapbert)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
