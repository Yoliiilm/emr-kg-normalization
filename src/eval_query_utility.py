"""Evaluate graph-query utility with ranking metrics.

This complements the existing set-level query F1 with Recall@K, Precision@K,
F1@K, MRR, and binary nDCG.  The gold graph is kept on the legacy normalizer;
the optional calibrated configuration is evaluated against that fixed reference.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

QUERY_RELS = ("使用药物", "实施手术")

OUT = ROOT / "outputs" / "metrics"


def semantic_id(nid: str) -> str:
    label, name = nid.split("::", 1)
    return f"{label}::{name}"


def neighbors_by_rel(rel_counts, rel_type: str) -> dict[str, set[str]]:
    out = defaultdict(set)
    for (source, relation, target), _weight in rel_counts.items():
        if relation == rel_type:
            out[source].add(target)
    return out


def ranking_metrics(ranked: list[str], gold: set[str], k: int) -> dict[str, float]:
    """Compute binary relevance metrics for one ranked answer list."""
    top = ranked[:k]
    hits = [1 if item in gold else 0 for item in top]
    tp = sum(hits)
    precision = tp / k if k else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    ndcg = dcg / ideal if ideal else 0.0
    return {f"precision@{k}": precision, f"recall@{k}": recall, f"f1@{k}": f1, f"ndcg@{k}": ndcg}


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"n_queries": 0}
    keys = rows[0]
    return {"n_queries": len(rows), **{key: sum(row[key] for row in rows) / len(rows) for key in keys}}


def _ranked_targets(rels: dict, source: str, rel_type: str) -> list[str]:
    scored = defaultdict(int)
    for (s, r, t), weight in rels.items():
        if s == source and r == rel_type:
            scored[semantic_id(t)] += weight
    return [target for target, _ in sorted(scored.items(), key=lambda x: (-x[1], x[0]))]


def _norm_node(g: dict, alias: str, canonical: str) -> str | None:
    names = g["pred_alias_names"].get(alias, ())
    candidates = [f"疾病和诊断::{name}" for name in names if f"疾病和诊断::{name}" in g["pred_norm_nodes"]]
    semantic = f"疾病和诊断::{canonical}"
    if semantic in candidates:
        return semantic
    if candidates:
        return candidates[0]
    if g.get("pred_context"):
        return semantic if semantic in g["pred_norm_nodes"] else None
    return None


def evaluate(g: dict, rel_type: str, ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    gold_nb = neighbors_by_rel(g["gold_norm_rels"], rel_type)
    rows = {"g_raw": [], "g_norm": []}
    for canon_nid, aliases in g["gold_surfaces"].items():
        if not canon_nid.startswith("疾病和诊断::"):
            continue
        canonical = canon_nid.split("::", 1)[1]
        gold = {semantic_id(t) for t in gold_nb.get(canon_nid, set())}
        if not gold:
            continue
        for alias in sorted(aliases):
            raw_nid = f"疾病和诊断::{alias}"
            norm_nid = _norm_node(g, alias, canonical)
            ranked_raw = _ranked_targets(g["pred_raw_rels"], raw_nid, rel_type)
            ranked_norm = _ranked_targets(g["pred_norm_rels"], norm_nid, rel_type) if norm_nid else []
            for graph_name, ranked in (("g_raw", ranked_raw), ("g_norm", ranked_norm)):
                row = {}
                for k in ks:
                    row.update(ranking_metrics(ranked, gold, k))
                first = next((i + 1 for i, item in enumerate(ranked) if item in gold), None)
                row["mrr"] = 1.0 / first if first else 0.0
                rows[graph_name].append(row)
    return {name: mean_rows(values) for name, values in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated", action="store_true", help="evaluate calibrated abstaining normalization")
    parser.add_argument("--model", type=Path, default=ROOT / "outputs" / "models" / "normalization_calibrator.json")
    parser.add_argument("--no-neural", action="store_true", help="disable BERT/SapBERT candidates in calibrated mode")
    parser.add_argument("--retriever", choices=("roberta", "sapbert"), default="roberta")
    parser.add_argument(
        "--out",
        type=Path,
        help="metrics output path; defaults to the standard legacy/calibrated path",
    )
    args = parser.parse_args()

    from eval_g_raw_vs_g_norm import build_graphs
    from ner_data import load_bio_docs
    from normalize import configure_normalization, normalization_settings

    processed = ROOT / "data" / "processed"
    gold_docs = load_bio_docs(processed / "test.bio")
    pred_docs = load_bio_docs(processed / "test_pred.bio")
    old = normalization_settings()
    configure_normalization()
    pred_settings = None
    if args.calibrated:
        pred_settings = normalization_settings()
        pred_settings.update(
            calibration_mode="abstain",
            calibration_model=str(args.model),
            calibration_neural=not args.no_neural,
            retriever=args.retriever,
        )
    graphs = build_graphs(gold_docs, pred_docs, pred_settings=pred_settings)
    result = {
        "normalization": "calibrated" if args.calibrated else "legacy",
        "model": str(args.model) if args.calibrated else None,
        "metrics": {rel: evaluate(graphs, rel) for rel in QUERY_RELS},
    }
    configure_normalization(**old)
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = args.out or OUT / ("query_utility_calibrated.json" if args.calibrated else "query_utility_legacy.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
