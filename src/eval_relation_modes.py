"""Compare co-occurrence vs pattern / hybrid clinical relation graphs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_g_raw_vs_g_norm import QUERY_RELS, build_graphs, eval_vs_gold
from ner_data import load_bio_docs
from normalize import configure_normalization


def main() -> None:
    configure_normalization()
    processed = ROOT / "data" / "processed"
    gold_docs = load_bio_docs(processed / "test.bio")
    pred_docs = load_bio_docs(processed / "test_pred.bio")

    lines = [
        "关系抽取模式对比（同一归一化；改的是边是否需要医嘱/手术线索）",
        "=" * 60,
        "",
    ]
    for mode in ("cooccur", "hybrid", "pattern"):
        # gold stays cooccur-normalized for a stable reference
        g = build_graphs(gold_docs, pred_docs, relation_mode=mode)
        pattern_edges = sum(
            1
            for key, bag in g["pred_evidence"].items()
            if bag.get("pattern", 0) > 0 and key[1] in QUERY_RELS
        )
        drug = eval_vs_gold(g, "使用药物")["g_norm"]["f1"]
        surg = eval_vs_gold(g, "实施手术")["g_norm"]["f1"]
        lines.append(
            f"【{mode}】用药F1={drug:.4f} 手术F1={surg:.4f}  "
            f"norm边={len(g['pred_norm_rels'])}  pattern支持查询边={pattern_edges}"
        )

    lines.append("")
    lines.append("解读: pattern 提高边的可解释性；相对 gold 共现图时 recall 可能下降。")
    lines.append("hybrid = 共现 + 线索加权，兼顾覆盖与医嘱语义。")

    out = ROOT / "outputs" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (out / "relation_modes.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
