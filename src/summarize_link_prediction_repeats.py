"""Aggregate repeated record-held-out TransE raw-vs-normalized comparisons."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "outputs" / "metrics" / "experiments" / "link_prediction" / "paired_train_only"
DEFAULT_OUT = DEFAULT_DIR / "transe_repeated_seeds_summary.json"
METRICS = ("mrr", "hits_at_1", "hits_at_3", "hits_at_10", "sampled_auc")
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def aggregate(results: list[dict]) -> dict:
    if not results:
        raise ValueError("at least one result file is required")
    rows = []
    for result in results:
        raw_n = result["g_raw"]["metrics"]["n_test_edges"]
        norm_n = result["g_norm"]["metrics"]["n_test_edges"]
        if raw_n != norm_n:
            raise ValueError(f"raw/norm test edge counts are not paired for seed {result['seed']}: {raw_n} != {norm_n}")
        row = {"seed": result["seed"]}
        for side in ("g_raw", "g_norm"):
            row[side] = {metric: result[side]["metrics"][metric] for metric in METRICS}
            row[f"{side}_degree_baseline"] = {
                metric: result[side].get("degree_baseline", {}).get(metric)
                for metric in METRICS
            }
        row["n_test_edges"] = {
            "g_raw": result["g_raw"]["metrics"]["n_test_edges"],
            "g_norm": result["g_norm"]["metrics"]["n_test_edges"],
        }
        row["paired_test_edges_sha256"] = result.get("paired_test_edges_sha256")
        row["delta_norm_minus_raw"] = result["delta_norm_minus_raw"]
        rows.append(row)
    summary = {}
    degree_summary = {}
    for metric in METRICS:
        raw = [row["g_raw"][metric] for row in rows]
        norm = [row["g_norm"][metric] for row in rows]
        deltas = [row["delta_norm_minus_raw"][metric] for row in rows]
        mean_delta = statistics.mean(deltas)
        critical = T_975.get(len(deltas) - 1, 1.96)
        half = critical * statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else None
        summary[metric] = {
            "raw_mean": statistics.mean(raw),
            "normalized_mean": statistics.mean(norm),
            "per_seed_delta_norm_minus_raw": deltas,
            "mean_delta": mean_delta,
            "delta_95ci_t_approx": [mean_delta - half, mean_delta + half] if half is not None else None,
            "seeds_improved": sum(value > 0 for value in deltas),
        }
        baseline_rows = [row for row in rows if row["g_raw_degree_baseline"][metric] is not None and row["g_norm_degree_baseline"][metric] is not None]
        if baseline_rows:
            raw_base = [row["g_raw_degree_baseline"][metric] for row in baseline_rows]
            norm_base = [row["g_norm_degree_baseline"][metric] for row in baseline_rows]
            base_deltas = [norm_value - raw_value for raw_value, norm_value in zip(raw_base, norm_base)]
            mean_base_delta = statistics.mean(base_deltas)
            base_critical = T_975.get(len(base_deltas) - 1, 1.96)
            base_half = base_critical * statistics.stdev(base_deltas) / math.sqrt(len(base_deltas)) if len(base_deltas) > 1 else None
            degree_summary[metric] = {
                "raw_mean": statistics.mean(raw_base),
                "normalized_mean": statistics.mean(norm_base),
                "per_seed_delta_norm_minus_raw": base_deltas,
                "mean_delta": mean_base_delta,
                "delta_95ci_t_approx": [mean_base_delta - base_half, mean_base_delta + base_half] if base_half is not None else None,
                "seeds_improved": sum(value > 0 for value in base_deltas),
            }
    return {
        "protocol": "Repeated 20% record-held-out evaluations with train-only canonical maps; within each seed, raw and normalized models use a one-to-one identical test edge sample eligible in both training vocabularies; report paired TransE and training-drug-degree baseline deltas.",
        "n_seeds": len(rows),
        "all_raw_norm_test_counts_paired": True,
        "seeds": [row["seed"] for row in rows],
        "by_metric": summary,
        "degree_baseline_by_metric": degree_summary,
        "per_seed": rows,
        "caution": "Each seed uses the same one-to-one held-out surface-edge sample for raw/norm; candidate entity universes still differ after canonicalization. Repeated-seed t intervals describe split/initialization variability and are not a clinical significance test.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, default=[DEFAULT_DIR / f"transe_seed{seed}.json" for seed in range(42, 47)])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    summary = aggregate(results)
    summary["inputs"] = [str(path) for path in args.inputs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
