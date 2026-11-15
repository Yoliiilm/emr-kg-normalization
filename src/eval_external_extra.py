"""Follow-up experiments judged by the external knowledge sources (see eval_external_kg.py).

E1  Ablation: each cumulative normalization setting of eval_ablation.py, scored by external support
    of disease->drug edges instead of our own standard answer (paired against G_raw).
E2  NER noise: the same G_raw vs G_norm comparison on graphs built from gold entities, on the
    CCKS2019 test set (379 docs) and on the CCKS2019 train+dev documents (1000 docs).
E3  Does an edge's evidence predict external support? Support rate by co-occurrence document count
    and by presence of a treatment cue, plus global precision@k when all edges are ranked together.
E4  SapBERT fallback in the graph without the site/morphology filters (threshold / always top-1).
E5  Disease normalization alone (drug/surgery alias tables off): the drug alias table was curated on
    CCKS test surfaces, so this is the setting without test-set-specific drug knowledge.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_ablation import FULL, MODES
from eval_external_kg import (
    DISEASE,
    disease_classes,
    doc_level_graph,
    edge_support,
    load_cmeie,
    load_medicalkg,
    paired_support,
    supported,
    wilson,
)
from eval_g_raw_vs_g_norm import collect_freq, compute_stats
from ner_data import load_bio_docs
from normalize import build_canonical_map, configure_normalization

REL = "使用药物"
OUT = ROOT / "outputs" / "metrics"
PROCESSED = ROOT / "data" / "processed"


def norm_graph(docs, settings: dict) -> dict:
    configure_normalization(**settings)
    canon = build_canonical_map(collect_freq(docs))
    g = doc_level_graph(docs, canon)
    configure_normalization()
    return g


def compare(raw, norm, ext) -> dict:
    s_raw, s_norm = edge_support(raw, REL, ext), edge_support(norm, REL, ext)
    keep = ("n_covered", "covered_mention_share", "n_edges", "n_supported", "support_rate", "support_rate_ci95",
            "ext_found", "ext_findable", "ext_recall")
    return {"g_raw": {k: s_raw[k] for k in keep}, "g_norm": {k: s_norm[k] for k in keep},
            "paired": paired_support({"g_raw": raw, "g_norm": norm}, REL, ext)}


def line(title: str, r: dict) -> str:
    p = r["paired"]
    return (f"  {title:30s} 印证 {r['g_norm']['n_supported']:>3d}/{r['g_norm']['n_edges']:<4d}={r['g_norm']['support_rate']:.3f}"
            f"  召回 {r['g_norm']['ext_found']}/{r['g_norm']['ext_findable']}={r['g_norm']['ext_recall']:.3f}"
            f"  配对差(vs G_raw) {p['delta']:+.3f} [{p['delta_ci95'][0]:+.3f}, {p['delta_ci95'][1]:+.3f}]")


def e1_ablation(pred_docs, raw, sources) -> dict:
    res = {}
    for key, cfg, title in MODES:
        g = norm_graph(pred_docs, {**cfg})
        res[key] = {"title": title} | {name: compare(raw, g, ext) for name, ext in sources.items()}
        print("E1", key, {n: round(res[key][n]["g_norm"]["support_rate"], 4) for n in sources})
    return res


def e2_gold_ner(sources) -> dict:
    res = {}
    sets = {
        "test_gold_ner": load_bio_docs(PROCESSED / "test.bio"),
        "train_dev_gold_ner": load_bio_docs(PROCESSED / "train.bio") + load_bio_docs(PROCESSED / "dev.bio"),
    }
    for name, docs in sets.items():
        raw = doc_level_graph(docs, None)
        norm = norm_graph(docs, FULL)
        res[name] = {"n_docs": len(docs)} | {s: compare(raw, norm, ext) for s, ext in sources.items()}
        print("E2", name, len(docs), {s: (round(res[name][s]["g_raw"]["support_rate"], 4),
                                         round(res[name][s]["g_norm"]["support_rate"], 4)) for s in sources})
    return res


def labelled_edges(g, ext) -> list[dict]:
    rows = []
    for (d, r, m), c in g["co"].items():
        if r != REL or not d.startswith(DISEASE + "::"):
            continue
        if not (disease_classes(g, d, ext) & ext["subjects"][REL]):
            continue
        n = g["n_docs"]
        rows.append({"count": c, "cue": g["cue"][(d, r, m)],
                     "lift": c * n / (g["df"][d] * g["df"][m]),
                     "label": supported(g, d, m, REL, ext)})
    return rows


def e3_evidence(norm, sources) -> dict:
    res = {}
    buckets = (("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), (">=5", 5, 10**9))
    for name, ext in sources.items():
        rows = labelled_edges(norm, ext)
        out = {"n_edges": len(rows), "by_count": {}, "by_cue": {}, "global_p_at_k": {}}
        for b, lo, hi in buckets:
            sel = [r for r in rows if lo <= r["count"] <= hi]
            k = sum(r["label"] for r in sel)
            out["by_count"][b] = {"n": len(sel), "supported": k, "rate": k / len(sel) if sel else 0.0,
                                  "ci95": wilson(k, len(sel))}
        for b, cond in (("有线索", lambda r: r["cue"] > 0), ("无线索", lambda r: r["cue"] == 0)):
            sel = [r for r in rows if cond(r)]
            k = sum(r["label"] for r in sel)
            out["by_cue"][b] = {"n": len(sel), "supported": k, "rate": k / len(sel) if sel else 0.0,
                                "ci95": wilson(k, len(sel))}
        rng = random.Random(3)
        base = sum(r["label"] for r in rows) / len(rows) if rows else 0.0
        for method in ("count", "lift", "cue"):
            acc = {k: 0.0 for k in (20, 50, 100)}
            for _ in range(200):
                order = sorted(rows, key=lambda r: (-r[method], rng.random()))
                for k in acc:
                    top = order[:k]
                    acc[k] += sum(r["label"] for r in top) / max(len(top), 1) / 200
            out["global_p_at_k"][method] = acc
        out["global_p_at_k"]["random"] = {k: base for k in (20, 50, 100)}
        res[name] = out
    return res


def e4_sapbert_fallback(gold_docs, pred_docs, raw, sources) -> dict:
    res = {}
    configure_normalization(**FULL)
    for key, extra in (("roberta (现系统)", {}),
                       ("sapbert + 过滤", {"retriever": "sapbert"}),
                       ("sapbert + 阈值兜底(无过滤)", {"retriever": "sapbert", "retriever_fallback": "threshold"}),
                       ("sapbert + top1 兜底(无过滤)", {"retriever": "sapbert", "retriever_fallback": "top1"})):
        settings = {**FULL, **extra}
        stats = compute_stats(gold_docs, pred_docs, pred_settings=settings)
        vg = stats["vs_gold_graph"]
        g = norm_graph(pred_docs, settings)
        res[key] = {
            "drug_f1": vg["使用药物"]["g_norm"]["f1"], "surg_f1": vg["实施手术"]["g_norm"]["f1"],
            "source_counts": stats["dict_coverage"]["source_counts"],
            "n_disease_nodes": stats["g_norm"]["type_counts"].get(DISEASE, 0),
        } | {name: compare(raw, g, ext) for name, ext in sources.items()}
        print("E4", key, round(res[key]["drug_f1"], 4), round(res[key]["surg_f1"], 4))
    configure_normalization()
    return res


E5_SETS = (("test_pred_ner", "测试集 预测实体"), ("test_gold_ner", "测试集 标注实体"),
           ("train_dev_gold_ner", "训练+开发集 标注实体"))


def e5_disease_only(pred_docs, sources) -> dict:
    sets = {
        "test_pred_ner": pred_docs,
        "test_gold_ner": load_bio_docs(PROCESSED / "test.bio"),
        "train_dev_gold_ner": load_bio_docs(PROCESSED / "train.bio") + load_bio_docs(PROCESSED / "dev.bio"),
    }
    no_alias = {**FULL, "use_drug_alias": False, "use_surgery_alias": False}
    res = {}
    for name, docs in sets.items():
        raw = doc_level_graph(docs, None)
        res[name] = {}
        for key, settings in (("disease_only", no_alias), ("full", FULL)):
            g = norm_graph(docs, settings)
            res[name][key] = {s: compare(raw, g, ext) for s, ext in sources.items()}
        print("E5", name, {k: {s: round(v[s]["paired"]["delta"], 4) for s in sources} for k, v in res[name].items()})
    return res


def report(e1, e2, e3, e4, e5, sources) -> str:
    names = {"cmeie": "CMeIE", "medicalkg": "寻医问药"}
    L = ["外部知识库补充实验（疾病→药物边；判定不经过本系统的标准化）", "=" * 72, ""]
    L.append("E1 消融：每一行只换 G_norm 的标准化设置，外部印证率 + 与 G_raw 的配对差（按疾病簇 bootstrap）")
    for s in sources:
        L.append(f"【{names[s]}】")
        L.append(line("G_raw（不标准化）", {"g_norm": e1[MODES[0][0]][s]["g_raw"],
                                          "paired": {"delta": 0.0, "delta_ci95": (0.0, 0.0)}}))
        for key, _, _ in MODES:
            L.append(line(key, e1[key][s]))
    L += ["", "E2 排除 NER 误差：用人工标注实体建图（G_raw → G_norm）"]
    for name, title in (("test_gold_ner", "测试集 379 篇"), ("train_dev_gold_ner", "训练+开发集 1000 篇")):
        for s in sources:
            r = e2[name][s]
            p = r["paired"]
            L.append(f"  {title:14s}{names[s]:6s} G_raw {r['g_raw']['n_supported']}/{r['g_raw']['n_edges']}={r['g_raw']['support_rate']:.3f}"
                     f"  G_norm {r['g_norm']['n_supported']}/{r['g_norm']['n_edges']}={r['g_norm']['support_rate']:.3f}"
                     f"  召回 {r['g_raw']['ext_recall']:.3f}→{r['g_norm']['ext_recall']:.3f}"
                     f"  配对差 {p['delta']:+.3f} [{p['delta_ci95'][0]:+.3f}, {p['delta_ci95'][1]:+.3f}]")
    L += ["", "E3 边的证据强弱能否预测外部印证（G_norm，预测实体，测试集）"]
    for s in sources:
        r = e3[s]
        L.append(f"【{names[s]}】可评边 {r['n_edges']}")
        L.append("  按共现文档数: " + "；".join(
            f"{b}: {v['supported']}/{v['n']}={v['rate']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
            for b, v in r["by_count"].items()))
        L.append("  按医嘱线索:   " + "；".join(
            f"{b}: {v['supported']}/{v['n']}={v['rate']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
            for b, v in r["by_cue"].items()))
        L.append("  全部边一起排序 P@20/50/100: " + "；".join(
            f"{m} {v[20]:.3f}/{v[50]:.3f}/{v[100]:.3f}" for m, v in r["global_p_at_k"].items()))
    L += ["", "E4 SapBERT 兜底不加过滤规则（查询 F1 仍以原系统标准答案为准）"]
    for key, r in e4.items():
        sc = r["source_counts"]
        ext = "  ".join(f"{names[s]} 印证 {r[s]['g_norm']['support_rate']:.3f}" for s in sources)
        L.append(f"  {key:26s} 用药 F1 {r['drug_f1']:.4f}  手术 F1 {r['surg_f1']:.4f}  疾病节点 {r['n_disease_nodes']}"
                 f"  走检索 {sc.get('bert_retrieve', 0)}  兜底 {sc.get('retrieve_fallback', 0)}  未链接 {sc.get('heuristic', 0)}  {ext}")
    L += ["", "E5 只做疾病标准化（关掉药物/手术别名表；药物别名表是照着测试集药名整理的）",
          "   配对差 = G_norm 印证率 − G_raw 印证率，按外部疾病簇 bootstrap 95% CI"]
    for name, title in E5_SETS:
        for s in sources:
            parts = []
            for key, kt in (("disease_only", "只疾病"), ("full", "完整")):
                r = e5[name][key][s]
                p = r["paired"]
                parts.append(f"{kt} {r['g_norm']['support_rate']:.3f} 差 {p['delta']:+.3f} "
                             f"[{p['delta_ci95'][0]:+.3f}, {p['delta_ci95'][1]:+.3f}]")
            g_raw = e5[name]["full"][s]["g_raw"]["support_rate"]
            L.append(f"  {title:16s}{names[s]:6s} G_raw {g_raw:.3f}  " + "  ".join(parts))
    return "\n".join(L) + "\n"


def main() -> None:
    sources = {"cmeie": load_cmeie(ROOT / "data" / "dicts" / "raw" / "CMeIE"),
               "medicalkg": load_medicalkg(ROOT / "data" / "dicts" / "raw" / "medicalkg" / "medical.json")}
    gold_docs = load_bio_docs(PROCESSED / "test.bio")
    pred_docs = load_bio_docs(PROCESSED / "test_pred.bio")
    configure_normalization()
    raw = doc_level_graph(pred_docs, None)
    full = norm_graph(pred_docs, FULL)

    e1 = e1_ablation(pred_docs, raw, sources)
    e2 = e2_gold_ner(sources)
    e3 = e3_evidence(full, sources)
    e4 = e4_sapbert_fallback(gold_docs, pred_docs, raw, sources)
    e5 = e5_disease_only(pred_docs, sources)
    text = report(e1, e2, e3, e4, e5, sources)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "external_extra.txt").write_text(text, encoding="utf-8")
    (OUT / "external_extra.json").write_text(
        json.dumps({"e1_ablation": e1, "e2_gold_ner": e2, "e3_evidence": e3, "e4_sapbert_fallback": e4,
                    "e5_disease_only": e5},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
