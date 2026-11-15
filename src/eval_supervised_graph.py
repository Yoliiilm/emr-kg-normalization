"""Graph query F1 with the supervised CHIP-CDN linker plugged into disease normalization.

Two standard answers, because every G_norm variant can also re-normalize the gold graph:
  fixed  - gold normalized by the current full system (supervised_cdn=off); rows comparable;
  self   - gold normalized with the same settings as G_norm (the main-table convention).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_g_raw_vs_g_norm import build_graphs, collect_freq, eval_vs_gold, query_rows
from ner_data import load_bio_docs
from normalize import configure_normalization, lookup_with_source

OUT = ROOT / "outputs" / "metrics"
RELS = ("使用药物", "实施手术")
MODES = [
    ("off", "现有完整系统（不用监督模型）"),
    ("fallback", "+ 监督 CDN 兜底（仅原本未链接的表面词）"),
    ("primary", "监督 CDN 优先（精确 ICD 之后立即使用）"),
]


def scores(g) -> dict:
    out = {}
    for rel in RELS:
        r = eval_vs_gold(g, rel)
        out[rel] = (r["g_raw"]["f1"], r["g_norm"]["f1"], r["g_norm"]["n"])
    return out


def main() -> None:
    processed = ROOT / "data" / "processed"
    gold_docs = load_bio_docs(processed / "test.bio")
    pred_docs = load_bio_docs(processed / "test_pred.bio")

    lines = ["监督 CHIP-CDN 模型接入建图（疾病归一化）", "=" * 64, ""]
    report: dict = {"modes": {}}
    base_rows = {}

    for mode, title in MODES:
        configure_normalization()
        g_fixed = build_graphs(gold_docs, pred_docs, pred_settings={"supervised_cdn": mode})
        fixed = scores(g_fixed)
        configure_normalization(supervised_cdn=mode)
        g_self = build_graphs(gold_docs, pred_docs)
        self_ = scores(g_self)
        cov = g_self["coverage"]

        diff = {}
        for rel in RELS:
            rows = {(r["query"], r["canonical"]): r["f_norm"] for r in query_rows(g_fixed, rel)}
            if mode == "off":
                base_rows[rel] = rows
            common = rows.keys() & base_rows[rel].keys()
            diff[rel] = (
                sum(rows[k] > base_rows[rel][k] + 1e-9 for k in common),
                sum(rows[k] < base_rows[rel][k] - 1e-9 for k in common),
            )

        report["modes"][mode] = {
            "fixed_gold": fixed,
            "self_gold": self_,
            "icd_hit_rate": cov["icd_hit_rate"],
            "source_counts": cov["source_counts"],
            "vs_off_better_worse": diff,
        }
        lines.append(f"【{mode}】{title}")
        lines.append(f"  链接率={cov['icd_hit_rate']:.4f}  监督模型链接={cov['source_counts'].get('cdn_sup', 0)}  "
                     f"原样保留={cov['source_counts'].get('heuristic', 0)}")
        for rel in RELS:
            fr, fn, n = fixed[rel]
            sr, sn, sn_n = self_[rel]
            b, w = diff[rel]
            lines.append(
                f"  {rel}  固定口径 norm={fn:.4f}（raw {fr:.4f}, n={n}）  自洽口径 raw→norm {sr:.4f}→{sn:.4f}（n={sn_n}）"
                + ("" if mode == "off" else f"  逐查询 vs 现有系统: 变好 {b} / 变差 {w}")
            )
        lines.append("")

    configure_normalization(supervised_cdn="fallback")
    freq = collect_freq(pred_docs).get("疾病和诊断", {})
    sup_rows = []
    for text, c in Counter(freq).most_common():
        canon, src = lookup_with_source(text, "疾病和诊断")
        if src == "cdn_sup" and canon != text:
            sup_rows.append({"surface": text, "to": canon, "freq": c})
    lines.append(f"兜底模式下由监督模型新链接的表面词（共 {len(sup_rows)} 个，按出现次数前 30，需人工核对）")
    for r in sup_rows[:30]:
        lines.append(f"  - {r['surface']} → {r['to']}  (freq={r['freq']})")
    report["fallback_links"] = sup_rows
    configure_normalization()

    from cdn_linker import get_cdn_linker

    get_cdn_linker().save()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "supervised_graph.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "supervised_graph.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
