"""Summarize available CCKS2017 transfer evidence without treating proxies as gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
EXPERIMENTS = METRICS / "experiments" / "cross_dataset_calibration"


def summarize(calibration: dict, graph: dict, external: dict) -> dict:
    acceptance = calibration["acceptance"]
    graph_relations = graph["results"]["off"]["rels"]
    graph_utility = {
        relation: {
            "queries": result["n"],
            "raw_f1": result["f1_raw"],
            "normalized_f1": result["f1_norm"],
            "delta": result["f1_norm"] - result["f1_raw"],
            "delta_ci95": result["raw_vs_norm_ci95"],
            "better_queries": result["better"],
            "worse_queries": result["worse"],
            "sign_flip_p": result["p_sign_flip"],
        }
        for relation, result in graph_relations.items()
    }

    external_support = {}
    for source, result in calibration["external_kg_edge_support"].items():
        paired = result["calibrated_abstain"]["paired"]
        external_support[source] = {
            "raw_edges": paired["g_raw"]["edges"],
            "raw_supported": paired["g_raw"]["supported"],
            "raw_support_rate": paired["g_raw"]["rate"],
            "normalized_edges": paired["g_norm"]["edges"],
            "normalized_supported": paired["g_norm"]["supported"],
            "normalized_support_rate": paired["g_norm"]["rate"],
            "delta_normalized_minus_raw": paired["delta"],
            "delta_ci95": paired["delta_ci95"],
            "common_external_diseases": paired["n_common_ext_diseases"],
        }

    return {
        "protocol": "Summary of separate existing CCKS2017 evaluations; the calibrated coverage/external-support results and the legacy full-normalizer graph-query results are from different normalizer configurations.",
        "dataset": {
            "records": calibration["records"],
            "disease_mentions": acceptance["disease_mentions"],
            "unique_disease_surfaces": acceptance["unique_disease_surfaces"],
            "text_overlap_with_ccks2019_mean_15gram": external["overlap"]["mean_share"],
            "text_overlap_records_over_50pct": external["overlap"]["n_docs_over_50pct"],
        },
        "transfer_coverage": {
            "accepted_mentions": acceptance["accepted_mentions"],
            "mention_coverage": acceptance["mention_coverage"],
            "accepted_unique_surfaces": acceptance["accepted_unique_surfaces"],
            "surface_coverage": acceptance["surface_coverage"],
            "accepted_names_with_icd": acceptance["accepted_names_with_icd"],
            "icd_hit_rate_among_accepted": acceptance["accepted_icd_hit_rate"],
            "decision_source_counts": acceptance["mention_weighted_decision_sources"],
            "strict_precision": None,
        },
        "graph_query_utility": graph_utility,
        "graph_query_utility_model": "Legacy full normalization system evaluated by eval_ccks2017_graph.py; this is not the SapBERT abstaining calibrator.",
        "external_kg_support": external_support,
        "interpretation": {
            "supported": [
                "The normalizer transfers operationally to a second hospital corpus, with mention and surface coverage reported separately.",
                "On the existing CCKS2017 graph-query protocol, the legacy full normalizer improves medication and surgery query F1.",
                "External-KG support does not show a positive paired change for the calibrated transfer.",
            ],
            "not_supported": [
                "The accepted-name ICD hit rate is not normalization accuracy.",
                "The graph-query reference graph is normalized with this project's full normalizer, so it is not independent gold for canonical-name correctness.",
                "CCKS2017 has no canonical-name labels; strict precision and recall cannot be computed.",
                "External-KG support is corroboration only, and its coverage is incomplete.",
            ],
        },
        "inputs": {
            "calibrated_transfer": "outputs/metrics/experiments/cross_dataset_calibration/ccks2017_sapbert_source_chip_reject.json",
            "graph_query": "outputs/metrics/ccks2017_graph.json",
            "external_support_and_overlap": "outputs/metrics/cross_dataset_ccks2017.json",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=EXPERIMENTS / "ccks2017_sapbert_source_chip_reject.json",
    )
    parser.add_argument("--graph", type=Path, default=METRICS / "ccks2017_graph.json")
    parser.add_argument("--external", type=Path, default=METRICS / "cross_dataset_ccks2017.json")
    parser.add_argument("--out", type=Path, default=EXPERIMENTS / "ccks2017_transfer_summary.json")
    args = parser.parse_args()

    result = summarize(
        json.loads(args.calibration.read_text(encoding="utf-8")),
        json.loads(args.graph.read_text(encoding="utf-8")),
        json.loads(args.external.read_text(encoding="utf-8")),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    text_path = args.out.with_suffix(".txt")
    lines = [
        "CCKS2017 跨医院迁移：可验证结果汇总",
        "=" * 52,
        f"病历 {result['dataset']['records']} 份；疾病提及 {result['dataset']['disease_mentions']}；不同写法 {result['dataset']['unique_disease_surfaces']}。",
        f"SapBERT 校准器接受 {result['transfer_coverage']['accepted_mentions']}/{result['dataset']['disease_mentions']} 次提及，覆盖率 {result['transfer_coverage']['mention_coverage']:.1%}；"
        f"接受项 ICD 命中 {result['transfer_coverage']['accepted_names_with_icd']}/{result['transfer_coverage']['accepted_mentions']}，此项不是准确率。",
        "以下图谱查询 F1 属于既有完整规则系统，不是上面的 SapBERT 拒识校准器：",
    ]
    for relation, label in (("使用药物", "用药"), ("实施手术", "手术")):
        item = result["graph_query_utility"][relation]
        lines.append(
            f"{label}查询 F1：{item['raw_f1']:.3f} → {item['normalized_f1']:.3f}，"
            f"差 {item['delta']:+.3f} [{item['delta_ci95'][0]:+.3f}, {item['delta_ci95'][1]:+.3f}]（{item['queries']} 个查询）。"
        )
    for source, item in result["external_kg_support"].items():
        lines.append(
            f"{source} 外部支持率：{item['raw_support_rate']:.3%} → {item['normalized_support_rate']:.3%}，"
            f"差 {item['delta_normalized_minus_raw']:+.3%} [{item['delta_ci95'][0]:+.3%}, {item['delta_ci95'][1]:+.3%}]。"
        )
    lines += [
        "限制：CCKS2017 没有标准化名称金标；查询参考图使用本项目完整标准化规则。覆盖、外部支持和查询 F1 来自不同配置，不能合并解释为同一模型的效果，也不能证明标准名准确率。",
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {text_path}")


if __name__ == "__main__":
    main()
