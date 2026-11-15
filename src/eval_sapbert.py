"""Off-the-shelf RoBERTa vs SapBERT-style fine-tuned ICD retriever (src/train_sapbert.py).

A. CHIP-CDN dev (never seen in training or threshold calibration):
   retriever alone (top-1, Recall@k), retriever + acceptance filters, and the full lookup with
   either retriever in the fallback slot. Lexicon is CHIP-CDN train only, as in eval_cdn_baselines.
B. EMR graph on CCKS2019 test: query F1 against the fixed standard answer (gold built with the
   current system), and against a gold rebuilt with the same retriever.
C. External KG support of disease->drug edges (CMeIE / 寻医问药), paired by external disease.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bert_linker import get_linker, mean_pool
from eval_cdn_baselines import DEV, acc_hit, gold_norms, load_train_alias, strict_hit
from eval_external_kg import doc_level_graph, edge_support, load_cmeie, load_medicalkg, paired_support
from eval_g_raw_vs_g_norm import build_graphs, compute_stats
from ner_data import load_bio_docs
from normalize import DictIndex, clean_surface, configure_normalization, icd_variants, strip_grade

KINDS = ("roberta", "sapbert")
FULL = {"use_chip_cdn": True, "use_drug_alias": True, "use_surgery_alias": True,
        "disease_rules_only": False, "use_split": True, "use_context": True}
OUT = ROOT / "outputs" / "metrics"


@torch.no_grad()
def topk(kind: str, texts: list[str], k: int = 10, batch_size: int = 256) -> list[list[str]]:
    linker = get_linker(kind)
    emb = linker.emb.to(linker.device)
    out = []
    for i in range(0, len(texts), batch_size):
        tok = linker.tokenizer(texts[i : i + batch_size], truncation=True, max_length=32, padding=True,
                               return_tensors="pt").to(linker.device)
        hidden = linker.model(**tok).last_hidden_state
        vec = torch.nn.functional.normalize(mean_pool(hidden, tok["attention_mask"]), p=2, dim=1)
        idx = (vec @ emb.T).topk(k, dim=1).indices.tolist()
        out.extend([[linker.names[j] for j in row] for row in idx])
    return out


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def part_a(samples) -> dict:
    train_alias = load_train_alias()
    res = {}
    for kind in KINDS:
        queries = [strip_grade(clean_surface(t)) or t for t, _ in samples]
        ranked = topk(kind, queries)
        top1 = {q: r[0] for q, r in zip(queries, ranked)}
        m = {"n": len(samples)}
        m["top1_acc"] = sum(acc_hit(r[0], g) for r, (_, g) in zip(ranked, samples)) / len(samples)
        m["top1_strict"] = sum(strict_hit(r[0], g) for r, (_, g) in zip(ranked, samples)) / len(samples)
        for k in (5, 10):
            m[f"recall@{k}"] = sum(any(norm(x) in {norm(y) for y in g} for x in r[:k])
                                   for r, (_, g) in zip(ranked, samples)) / len(samples)

        linker = get_linker(kind)
        answered = correct = correct_strict = 0
        for text, golds in samples:
            q = strip_grade(clean_surface(text)) or text
            pred = None
            for cand in icd_variants(q):
                hit = linker.retrieve(cand)
                if hit:
                    pred = hit[0]
                    break
            if pred:
                answered += 1
                correct += acc_hit(pred, golds)
                correct_strict += strict_hit(pred, golds)
        m["filtered_min_score"] = linker.min_score
        m["filtered_coverage"] = answered / len(samples)
        m["filtered_precision"] = correct / answered if answered else 0.0
        m["filtered_acc"] = correct / len(samples)

        configure_normalization(use_chip_cdn=False, use_drug_alias=False, retriever=kind)
        idx = DictIndex.load()
        idx.alias = train_alias
        idx.alias_clean = {clean_surface(k): v for k, v in train_alias.items() if clean_surface(k)}
        n_ours = n_ours_s = n_fb = n_fb_s = n_linked = n_bert = 0
        for text, golds in samples:
            pred, src = idx.lookup_disease(text)
            n_linked += src != "heuristic"
            n_bert += src == "bert_retrieve"
            n_ours += acc_hit(pred, golds)
            n_ours_s += strict_hit(pred, golds)
            fb = top1[strip_grade(clean_surface(text)) or text] if src == "heuristic" else pred
            n_fb += acc_hit(fb, golds)
            n_fb_s += strict_hit(fb, golds)
        n = len(samples)
        m.update({"ours_acc": n_ours / n, "ours_strict": n_ours_s / n, "ours_linked": n_linked / n,
                  "ours_bert_retrieve": n_bert, "ours_top1_fallback_acc": n_fb / n,
                  "ours_top1_fallback_strict": n_fb_s / n})
        res[kind] = m
        print(kind, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
    configure_normalization()
    return res


def f1s(stats: dict) -> dict:
    vg = stats["vs_gold_graph"]
    return {rel: {g: {k: vg[rel][g][k] for k in ("precision", "recall", "f1")} for g in ("g_raw", "g_norm")}
            for rel in vg} | {"source_counts": stats["dict_coverage"]["source_counts"],
                              "n_disease_merges": stats["n_disease_merges"]}


def part_b(gold_docs, pred_docs) -> dict:
    res = {}
    configure_normalization(**FULL, retriever="roberta")
    res["fixed_gold_roberta"] = f1s(compute_stats(gold_docs, pred_docs))
    res["fixed_gold_sapbert"] = f1s(compute_stats(gold_docs, pred_docs, pred_settings={**FULL, "retriever": "sapbert"}))
    configure_normalization(**FULL, retriever="sapbert")
    res["own_gold_sapbert"] = f1s(compute_stats(gold_docs, pred_docs))
    configure_normalization()
    return res


def part_c(gold_docs, pred_docs) -> dict:
    graphs = {}
    for kind in KINDS:
        configure_normalization(**FULL, retriever=kind)
        built = build_graphs(gold_docs, pred_docs)
        graphs[kind] = doc_level_graph(pred_docs, built["pred_canon"])
    configure_normalization()
    res = {}
    for name, ext in (("cmeie", load_cmeie(ROOT / "data" / "dicts" / "raw" / "CMeIE")),
                      ("medicalkg", load_medicalkg(ROOT / "data" / "dicts" / "raw" / "medicalkg" / "medical.json"))):
        rel = "使用药物"
        per = {kind: edge_support(graphs[kind], rel, ext) for kind in KINDS}
        paired = paired_support({"g_raw": graphs["roberta"], "g_norm": graphs["sapbert"]}, rel, ext)
        res[name] = {
            kind: {k: per[kind][k] for k in ("n_covered", "n_edges", "n_supported", "support_rate",
                                             "support_rate_ci95", "ext_found", "ext_findable", "ext_recall")}
            for kind in KINDS
        } | {"paired_roberta_vs_sapbert": paired}
    return res


def report(a, b, c) -> str:
    L = ["ICD 兜底检索器：现成 RoBERTa vs SapBERT 式对比学习微调（CHIP-CDN train 同义词对）", "=" * 72,
         "", "A. CHIP-CDN dev（训练和阈值标定都没用过 dev；词表只用 train）",
         f"{'':28s}{'roberta':>10s}{'sapbert':>10s}"]
    rows = [("检索单独 top1 Acc宽松", "top1_acc"), ("检索单独 top1 Acc严格", "top1_strict"),
            ("检索单独 Recall@5", "recall@5"), ("检索单独 Recall@10", "recall@10"),
            ("检索+过滤 接受阈值", "filtered_min_score"), ("检索+过滤 覆盖率", "filtered_coverage"),
            ("检索+过滤 答准率", "filtered_precision"), ("检索+过滤 Acc", "filtered_acc"),
            ("完整流程 Acc宽松", "ours_acc"), ("完整流程 Acc严格", "ours_strict"),
            ("完整流程 链接率", "ours_linked"), ("完整流程 走检索的条数", "ours_bert_retrieve"),
            ("完整流程+top1兜底 Acc宽松", "ours_top1_fallback_acc"),
            ("完整流程+top1兜底 Acc严格", "ours_top1_fallback_strict")]
    for title, key in rows:
        r, s = a["roberta"][key], a["sapbert"][key]
        fmt = (lambda v: f"{v:>10d}") if isinstance(r, int) else (lambda v: f"{v:>10.4f}")
        L.append(f"{title:28s}{fmt(r)}{fmt(s)}")
    L += ["", "B. CCKS2019 测试集图谱查询 F1（g_norm）"]
    for key, title in (("fixed_gold_roberta", "现成 RoBERTa（原系统）"),
                       ("fixed_gold_sapbert", "SapBERT，标准答案不变（仍用原系统）"),
                       ("own_gold_sapbert", "SapBERT，标准答案也用 SapBERT 重建")):
        r = b[key]
        L.append(f"  {title:34s} 用药 F1 {r['使用药物']['g_norm']['f1']:.4f}  手术 F1 {r['实施手术']['g_norm']['f1']:.4f}"
                 f"  走检索 {r['source_counts'].get('bert_retrieve', 0)}  未链接 {r['source_counts'].get('heuristic', 0)}"
                 f"  疾病合并 {r['n_disease_merges']}")
    L += ["", "C. 外部知识库印证率（疾病→药物；G_norm 分别用两种检索器）"]
    for name, title in (("cmeie", "CMeIE"), ("medicalkg", "寻医问药")):
        L.append(f"【{title}】")
        for kind in KINDS:
            r = c[name][kind]
            L.append(f"  {kind:8s} 覆盖疾病 {r['n_covered']}  被印证 {r['n_supported']}/{r['n_edges']}={r['support_rate']:.3f}"
                     f"  外部关系召回 {r['ext_found']}/{r['ext_findable']}")
        p = c[name]["paired_roberta_vs_sapbert"]
        L.append(f"  配对（{p['n_common_ext_diseases']} 个共同外部疾病）：roberta {p['g_raw']['rate']:.3f} → sapbert "
                 f"{p['g_norm']['rate']:.3f}  差 {p['delta']:+.3f} [{p['delta_ci95'][0]:.3f}, {p['delta_ci95'][1]:.3f}]")
    return "\n".join(L) + "\n"


def main() -> None:
    samples = []
    for row in json.loads(DEV.read_text(encoding="utf-8")):
        text = str(row.get("text", "")).lstrip("\ufeff").strip()
        golds = gold_norms(row)
        if text and golds:
            samples.append((text, golds))
    processed = ROOT / "data" / "processed"
    gold_docs = load_bio_docs(processed / "test.bio")
    pred_docs = load_bio_docs(processed / "test_pred.bio")

    a = part_a(samples)
    b = part_b(gold_docs, pred_docs)
    c = part_c(gold_docs, pred_docs)
    text = report(a, b, c)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sapbert_eval.txt").write_text(text, encoding="utf-8")
    (OUT / "sapbert_eval.json").write_text(json.dumps({"chip_cdn_dev": a, "graph": b, "external_kg": c},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
