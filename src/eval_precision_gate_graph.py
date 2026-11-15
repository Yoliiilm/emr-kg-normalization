"""Build and evaluate a precision-first normalization graph.

The fixed reference graph, drug/procedure normalization, document split, and
query procedure are unchanged. Only disease-name acceptance for the predicted
graph is replaced by the frozen top-five precision gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_g_raw_vs_g_norm import (  # noqa: E402
    QUERY_RELS,
    build_canonical_map,
    collect_freq,
    graph_counts,
    mean_prf,
    query_rows,
)
from eval_bootstrap import cluster_bootstrap, sign_flip_test  # noqa: E402
from eval_query_utility import evaluate as evaluate_ranking  # noqa: E402
from export_graph import write_csv, write_cypher  # noqa: E402
from ner_data import load_bio_docs  # noqa: E402
from normalize import (  # noqa: E402
    canonical_names,
    configure_normalization,
    coverage_report,
    get_index,
    normalization_settings,
)
from precision_gate import PrecisionGate, configure_precision_gate  # noqa: E402


OUT = ROOT / "outputs/metrics/experiments/reach85/precision_gate_graph.json"
GRAPH_OUT = ROOT / "outputs/neo4j_precision_gate"


def gate_canonical_map(gate: PrecisionGate, freq: dict) -> tuple[dict, dict, dict]:
    mapping = {}
    decisions = {}
    for label, names in freq.items():
        for name in names:
            if label != "疾病和诊断":
                canonical = canonical_names(name, label)
                mapping[(label, name)] = canonical[0] if len(canonical) == 1 else canonical
                continue
            decision = gate.decide(name)
            decisions[name] = {
                "accepted": decision.accepted,
                "name": decision.name,
                "source": decision.source,
                "score": decision.score,
                "base_confidence": decision.base_confidence,
                "reason": decision.reason,
            }
            mapping[(label, name)] = decision.name
    stats = {
        "n_surfaces": len(decisions),
        "accepted": sum(item["accepted"] for item in decisions.values()),
        "coverage": (
            sum(item["accepted"] for item in decisions.values()) / len(decisions)
            if decisions else 0.0
        ),
        "accepted_sources": dict(
            Counter(
                item["source"]
                for item in decisions.values()
                if item["accepted"]
            )
        ),
    }
    return mapping, decisions, stats


def build_fixed_reference(gold_docs) -> dict:
    gold_freq = collect_freq(gold_docs)
    gold_canon = build_canonical_map(gold_freq)
    gold_raw_nodes, gold_raw_rels, _, _ = graph_counts(gold_docs, None)
    gold_norm_nodes, gold_norm_rels, gold_surfaces, _ = graph_counts(gold_docs, gold_canon)
    return {
        "gold_raw_nodes": gold_raw_nodes,
        "gold_raw_rels": gold_raw_rels,
        "gold_norm_nodes": gold_norm_nodes,
        "gold_norm_rels": gold_norm_rels,
        "gold_surfaces": gold_surfaces,
    }


def make_graph_bundle(
    gold_reference: dict,
    pred_docs,
    pred_canon,
    alias_resolver=None,
) -> dict:
    pred_freq = collect_freq(pred_docs)
    pred_raw_nodes, pred_raw_rels, _, _ = graph_counts(pred_docs, None)
    pred_norm_nodes, pred_norm_rels, pred_surfaces, pred_evidence = graph_counts(
        pred_docs, pred_canon
    )
    gold_aliases = {
        alias
        for nid, aliases in gold_reference["gold_surfaces"].items()
        if nid.startswith("疾病和诊断::")
        for alias in aliases
    }
    pred_alias_names = {}
    for alias in gold_aliases:
        resolved = alias_resolver(alias) if alias_resolver else pred_canon.get(("疾病和诊断", alias))
        pred_alias_names[alias] = (
            (resolved,) if isinstance(resolved, str)
            else (resolved or (alias,))
        )
    return {
        **gold_reference,
        "pred_raw_nodes": pred_raw_nodes,
        "pred_raw_rels": pred_raw_rels,
        "pred_norm_nodes": pred_norm_nodes,
        "pred_norm_rels": pred_norm_rels,
        "pred_surfaces": pred_surfaces,
        "pred_evidence": pred_evidence,
        "pred_alias_names": pred_alias_names,
        "pred_context": bool(normalization_settings().get("use_context")),
        "pred_canon": pred_canon,
    }


def summarize_graph(g: dict, label: str) -> dict:
    result = {"label": label, "graph": {}, "set_query": {}, "ranking_query": {}}
    result["graph"] = {
        "nodes": len(g["pred_norm_nodes"]),
        "relations": len(g["pred_norm_rels"]),
        "type_counts": dict(
            Counter(node["type"] for node in g["pred_norm_nodes"].values())
        ),
        "n_surfaces_merged": sum(
            1
            for (entity_type, source), target in g["pred_canon"].items()
            if entity_type == "疾病和诊断" and source != target
        ),
    }
    for rel in QUERY_RELS:
        rows = query_rows(g, rel)
        result["set_query"][rel] = {
            "g_raw": mean_prf([row["p_raw"] for row in rows]),
            "g_norm": mean_prf([row["p_norm"] for row in rows]),
            "n_queries": len(rows),
        }
        result["ranking_query"][rel] = evaluate_ranking(g, rel)
    return result


def paired_gate_vs_legacy(legacy: dict, gate: dict, rel: str) -> dict:
    legacy_rows = {
        (row["query"], row["canonical"]): row
        for row in query_rows(legacy, rel)
    }
    gate_rows = {
        (row["query"], row["canonical"]): row
        for row in query_rows(gate, rel)
    }
    keys = sorted(set(legacy_rows) & set(gate_rows))
    paired = [
        {
            "query": key[0],
            "canonical": key[1],
            "f_raw": legacy_rows[key]["f_norm"],
            "f_norm": gate_rows[key]["f_norm"],
            "delta": gate_rows[key]["f_norm"] - legacy_rows[key]["f_norm"],
        }
        for key in keys
    ]
    deltas = [row["delta"] for row in paired]
    return {
        "n_queries": len(paired),
        "legacy_f1": sum(row["f_raw"] for row in paired) / len(paired),
        "gate_f1": sum(row["f_norm"] for row in paired) / len(paired),
        "delta_gate_minus_legacy": sum(deltas) / len(deltas),
        "n_improved": sum(delta > 0 for delta in deltas),
        "n_worse": sum(delta < 0 for delta in deltas),
        "n_same": sum(abs(delta) < 1e-12 for delta in deltas),
        "bootstrap_by_disease": cluster_bootstrap(paired, "canonical"),
        "p_sign_flip_two_sided": sign_flip_test(deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--graph-dir", type=Path, default=GRAPH_OUT)
    args = parser.parse_args()
    processed = ROOT / "data/processed"
    gold_docs = load_bio_docs(processed / "test.bio")
    pred_docs = load_bio_docs(processed / "test_pred.bio")
    old = normalization_settings()
    configure_normalization()
    pred_freq = collect_freq(pred_docs)
    legacy_canon = build_canonical_map(pred_freq)
    gold_reference = build_fixed_reference(gold_docs)
    legacy_graph = make_graph_bundle(
        gold_reference,
        pred_docs,
        legacy_canon,
        alias_resolver=lambda surface: canonical_names(surface, "疾病和诊断"),
    )
    legacy_coverage = coverage_report(pred_freq)

    configure_precision_gate()
    gate = PrecisionGate()
    gate_canon, decisions, gate_stats = gate_canonical_map(gate, pred_freq)
    gate_graph = make_graph_bundle(
        gold_reference,
        pred_docs,
        gate_canon,
        alias_resolver=lambda surface: gate.decide(surface).name,
    )
    configure_normalization(**old)

    exported_graph = {
        "source": "pred_test_precision_gate",
        "nodes": list(gate_graph["pred_norm_nodes"].values()),
        "relations": [
            {
                "source": source,
                "type": relation,
                "target": target,
                "weight": weight,
            }
            for (source, relation, target), weight in sorted(
                gate_graph["pred_norm_rels"].items(),
                key=lambda item: -item[1],
            )
        ],
        "n_nodes": len(gate_graph["pred_norm_nodes"]),
        "n_relations": len(gate_graph["pred_norm_rels"]),
    }
    write_csv(exported_graph, args.graph_dir)
    write_cypher(exported_graph, args.graph_dir / "precision_gate.cypher")
    (args.graph_dir / "normalization_decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = {
        "protocol": {
            "test_docs": len(pred_docs),
            "gold_docs": len(gold_docs),
            "reference": "fixed full-system normalized graph from gold NER entities",
            "prediction_graph": "predicted test NER entities",
            "gate_model": "outputs/metrics/experiments/reach85/meta_gate_model.joblib",
            "gate_threshold": gate.threshold,
            "gate_features": (
                "calibrated confidence, score margin, rank, source, lexical/"
                "source consistency, and site/laterality/malignancy/metastasis/"
                "postoperative consistency"
            ),
            "development_or_test_tuning": False,
        },
        "gate_decisions": gate_stats,
        "legacy_coverage": legacy_coverage,
        "legacy": summarize_graph(legacy_graph, "full-system normalization"),
        "precision_gate": summarize_graph(gate_graph, "precision-first gate"),
        "comparisons": {
            rel: paired_gate_vs_legacy(legacy_graph, gate_graph, rel)
            for rel in QUERY_RELS
        },
        "decisions": decisions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "gate": gate_stats,
        "legacy": result["legacy"],
        "precision_gate": result["precision_gate"],
        "comparisons": result["comparisons"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
