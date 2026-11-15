"""Ablation: cumulative normalization components, all scored against one fixed standard answer.

The gold graph and the neighbor-name judge always use the full system; only G_norm is
rebuilt with each row's settings, so rows are directly comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_g_raw_vs_g_norm import compute_stats
from ner_data import load_bio_docs
from normalize import configure_normalization

FULL = {
    "use_chip_cdn": True,
    "use_drug_alias": True,
    "use_surgery_alias": True,
    "disease_rules_only": False,
    "use_split": True,
    "use_context": True,
}
OFF = {**FULL, "use_chip_cdn": False, "use_drug_alias": False, "use_surgery_alias": False,
       "use_split": False, "use_context": False}

MODES = [
    ("rules_only", {**OFF, "disease_rules_only": True}, "仅规则（去修饰/分期 + 癌→恶性肿瘤改写，不查任何词表）"),
    ("icd", OFF, "ICD 词表（精确/变体/规则/前缀/curated/BERT）"),
    ("+chip", {**OFF, "use_chip_cdn": True}, "+ 过滤 CHIP 兜底"),
    ("+alias", {**OFF, "use_chip_cdn": True, "use_drug_alias": True, "use_surgery_alias": True}, "+ 药物/手术别名"),
    ("+split", {**FULL, "use_context": False}, "+ 多诊断切分"),
    ("full", FULL, "+ 无部位组织学词借上下文部位（完整系统）"),
]


def summarize(stats: dict) -> dict:
    cov = stats["dict_coverage"]
    vg = stats["vs_gold_graph"]
    return {
        "icd_hit_rate": cov["icd_hit_rate"],
        "source_counts": cov["source_counts"],
        "n_split": cov.get("n_split_compound", 0),
        "n_disease_merges": stats["n_disease_merges"],
        "g_norm_nodes": stats["g_norm"]["n_nodes"],
        "drug_f1_raw": vg["使用药物"]["g_raw"]["f1"],
        "drug_f1_norm": vg["使用药物"]["g_norm"]["f1"],
        "surg_f1_raw": vg["实施手术"]["g_raw"]["f1"],
        "surg_f1_norm": vg["实施手术"]["g_norm"]["f1"],
    }


def main() -> None:
    processed = ROOT / "data" / "processed"
    gold_docs = load_bio_docs(processed / "test.bio")
    pred_docs = load_bio_docs(processed / "test_pred.bio")
    metric_dir = ROOT / "outputs" / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)

    configure_normalization(**FULL)
    rows = []
    lines = [
        "消融：归一化组件逐步累加（统一标准答案：官方标注图用完整系统归一化）",
        "=" * 64,
        "g_raw 各行相同（不归一化）；只有 G_norm 按该行设置重建。",
        "",
    ]
    for key, cfg, title in MODES:
        print(f"=== {key}: {title} ===")
        stats = compute_stats(gold_docs, pred_docs, pred_settings=cfg)
        row = summarize(stats) | {"mode": key, "title": title}
        rows.append(row)
        lines.append(f"【{key}】{title}")
        lines.append(f"  用药  g_raw F1={row['drug_f1_raw']:.4f}  g_norm F1={row['drug_f1_norm']:.4f}")
        lines.append(f"  手术  g_raw F1={row['surg_f1_raw']:.4f}  g_norm F1={row['surg_f1_norm']:.4f}")
        lines.append(
            f"  ICD命中率={row['icd_hit_rate']:.4f}  疾病合并数={row['n_disease_merges']}  "
            f"切分复合诊断={row['n_split']}  norm节点={row['g_norm_nodes']}"
        )
        lines.append(f"  来源={row['source_counts']}")
        lines.append("")

    lines.append("简表（汇报用）")
    lines.append("mode\t用药F1_norm\t手术F1_norm\tICD命中\t疾病合并")
    for r in rows:
        lines.append(
            f"{r['mode']}\t{r['drug_f1_norm']:.4f}\t{r['surg_f1_norm']:.4f}\t"
            f"{r['icd_hit_rate']:.4f}\t{r['n_disease_merges']}"
        )
    lines.append("")
    lines.append("标准答案敏感性：官方标注图改用「不切分、不借上下文」归一化后重算")
    lines.append("mode\t用药 raw→norm\t手术 raw→norm")
    configure_normalization(**{**FULL, "use_split": False, "use_context": False})
    for key, cfg in (("+alias", MODES[3][1]), ("full", FULL)):
        stats = compute_stats(gold_docs, pred_docs, pred_settings=cfg)
        r = summarize(stats)
        lines.append(
            f"{key}\t{r['drug_f1_raw']:.4f}→{r['drug_f1_norm']:.4f}\t{r['surg_f1_raw']:.4f}→{r['surg_f1_norm']:.4f}"
        )
    configure_normalization(**FULL)

    lines.append("")
    lines.append("注: 药物/手术别名主要作用于邻居命名；统一标准答案下邻居名已由完整系统判定，")
    lines.append("    因此该行对查询 F1 影响很小，其价值体现在图谱节点合并与 Neo4j 可读性上。")

    text = "\n".join(lines) + "\n"
    (metric_dir / "ablation.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
