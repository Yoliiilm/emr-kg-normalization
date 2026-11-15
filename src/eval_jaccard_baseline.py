"""Jaccard + BERT baseline on CHIP-CDN dev (Feng et al., biomedRxiv 2023, bmr.202305.00015).

Two stages as in that paper:
  1. candidate generation: character-set Jaccard |A∩B| / |A∪B| against all labels, top-15;
  2. disambiguation: BERT sentence-pair classifier over the 15 candidates.

The pair classifier is the RoBERTa-wwm-ext ranker already trained by train_cdn_cblue.py
(outputs/models/cdn_cblue/cls.pt, CHIP-CDN train only), so stage 2 is not retrained on Jaccard
candidates. Labels = ICD-10 Beijing + train labels, same as the supervised baseline.
Metrics follow eval_cdn_baselines.py (gold without the 「癌」 label, whitespace-insensitive).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_cdn_baselines import gold_norms
from train_cdn_cblue import (
    ENCODER,
    ICD,
    MODEL_DIR,
    OUT,
    RAW,
    commit,
    load_split,
    process_text,
    score_pairs,
    str_q2b,
)

K = 15
MAX_LEN = 64


def jaccard_topk(texts: list[str], labels: list[str], k: int, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    vec = CountVectorizer(analyzer="char", lowercase=False, binary=True)
    lab = vec.fit_transform(labels).T.tocsr()
    lab_len = np.asarray(lab.sum(axis=0)).ravel()
    idx_out = np.zeros((len(texts), k), dtype=np.int64)
    sc_out = np.zeros((len(texts), k))
    for s in range(0, len(texts), chunk):
        xc = vec.transform(texts[s : s + chunk])
        inter = (xc @ lab).toarray().astype(float)
        xlen = np.asarray(xc.sum(axis=1)).ravel()
        union = xlen[:, None] + lab_len[None, :] - inter
        jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        top = np.argpartition(-jac, k, axis=1)[:, :k]
        top_sc = np.take_along_axis(jac, top, axis=1)
        order = np.argsort(-top_sc, axis=1, kind="stable")
        idx_out[s : s + chunk] = np.take_along_axis(top, order, axis=1)
        sc_out[s : s + chunk] = np.take_along_axis(top_sc, order, axis=1)
    return idx_out, sc_out


def norm(x: str) -> str:
    return re.sub(r"\s+", "", str_q2b(x))


def evaluate(preds: list[list[str]], golds: list[list[str]]) -> dict:
    n = len(golds)
    strict = tp = n_p = n_g = 0
    for p, g in zip(preds, golds):
        ps = [norm(x) for x in p if x and x != "癌"]
        gs = {norm(x) for x in g}
        strict += bool(ps) and ps[0] in gs
        tp += len(set(ps) & gs)
        n_p += len(set(ps))
        n_g += len(gs)
    sp = tp / n_p if n_p else 0.0
    sr = tp / n_g if n_g else 0.0
    return {"acc_strict": strict / n, "set_p": sp, "set_r": sr, "set_f1": 2 * sp * sr / (sp + sr) if sp + sr else 0.0}


def score_graph(icd_labels: list[str]) -> tuple[list[str], dict]:
    """Jaccard top-1 (ICD only) as the fallback for unlinked disease mentions, through the
    llm_cdn hook: same surfaces as outputs/llm_norm/map.json, answers swapped for Jaccard's."""
    import normalize
    from eval_g_raw_vs_g_norm import build_graphs
    from llm_norm_score import GRAPH_SETS, PROCESSED, RELS, paired, rows_by_key
    from ner_data import load_bio_docs

    surfaces = list(json.loads(normalize.LLM_MAP.read_text(encoding="utf-8")))
    top, _ = jaccard_topk([process_text(s) for s in surfaces], icd_labels, 1)
    jmap = {s: [icd_labels[row[0]]] for s, row in zip(surfaces, top)}
    path = normalize.ROOT / "outputs" / "jaccard" / "map.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jmap, ensure_ascii=False, indent=0), encoding="utf-8")

    llm_map = normalize.LLM_MAP
    normalize.LLM_MAP = path
    normalize.reload_llm_map()
    normalize.LLM_MISSING.clear()
    lines = ["", "接入建图：未链接的疾病写法用 Jaccard top1（只查 ICD）兜底；固定口径，差 = 与本方法逐查询配对（整簇 bootstrap）"]
    report: dict = {}
    try:
        for label, gold_name, pred_name in GRAPH_SETS:
            gold, pred = load_bio_docs(PROCESSED / gold_name), load_bio_docs(PROCESSED / pred_name)
            normalize.configure_normalization()
            base = build_graphs(gold, pred)
            jac = build_graphs(gold, pred, pred_settings={"llm_cdn": "fallback"})
            normalize.configure_normalization(llm_cdn="fallback")
            selfg = build_graphs(gold, pred)
            normalize.configure_normalization()
            res = {"icd_hit_rate": jac["coverage"]["icd_hit_rate"]}
            parts = [f"  【{label}】疾病链接率 {base['coverage']['icd_hit_rate']:.3f}→{res['icd_hit_rate']:.3f}"]
            for rel in RELS:
                b, j = rows_by_key(base, rel), rows_by_key(jac, rel)
                v = paired(b, j)
                s_rows = list(rows_by_key(selfg, rel).values())
                res[rel] = {
                    "f1_ours": sum(x["f_norm"] for x in b.values()) / len(b),
                    "f1_jaccard": sum(x["f_norm"] for x in j.values()) / len(j),
                    "vs_ours": v,
                    "self_gold_f1": sum(x["f_norm"] for x in s_rows) / len(s_rows),
                }
                parts.append(f"{rel} {res[rel]['f1_ours']:.3f}→{res[rel]['f1_jaccard']:.3f}"
                             f"（{v['delta']:+.3f} [{v['delta_ci95'][0]:+.3f}, {v['delta_ci95'][1]:+.3f}] "
                             f"{v['better']}/{v['worse']}；自洽口径 {res[rel]['self_gold_f1']:.3f}）")
            report[label] = res
            lines.append("  ".join(parts))
            print(lines[-1], flush=True)
    finally:
        normalize.LLM_MAP = llm_map
        normalize.reload_llm_map()
        normalize.configure_normalization()
    missing = sorted(t for t in normalize.LLM_MISSING if t.strip())
    lines.append(f"  映射表缺失的疾病写法：{len(missing)}")
    report["missing"] = len(missing)
    return lines, report


def main() -> None:
    dev_rows = json.loads((RAW / "CHIP-CDN_dev.json").read_text(encoding="utf-8"))
    dev = load_split("dev")
    golds = [gold_norms(r) for r in dev_rows if str(r.get("text", "")).lstrip("\ufeff").strip()]
    keep = [i for i, g in enumerate(golds) if g]
    dev = [dev[i] for i in keep]
    golds = [golds[i] for i in keep]

    train_all = load_split("train")
    icd_names = list(json.loads(ICD.read_text(encoding="utf-8"))["name_to_codes"].keys())
    labels = list(dict.fromkeys([str_q2b(n) for n in icd_names] + [g for _, _, gs in train_all for g in gs]))
    texts = [t for _, t, _ in dev]

    cands, jsc = jaccard_topk(texts, labels, K)
    gsets = [{norm(g) for g in gs} for gs in golds]
    cand_sets = [{norm(labels[c]) for c in row} for row in cands]
    recall_any = float(np.mean([bool(g & c) for g, c in zip(gsets, cand_sets)]))
    recall_all = float(np.mean([g <= c for g, c in zip(gsets, cand_sets)]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ENCODER)
    cls = AutoModelForSequenceClassification.from_pretrained(ENCODER, num_labels=2)
    cls.load_state_dict(torch.load(MODEL_DIR / "cls.pt", map_location="cpu"))
    cls.to(device)
    probs = score_pairs(cls, tok, texts, cands, labels, device, MAX_LEN)

    num = AutoModelForSequenceClassification.from_pretrained(ENCODER, num_labels=3)
    num.load_state_dict(torch.load(MODEL_DIR / "num.pt", map_location="cpu"))
    num.to(device)
    from train_cdn_cblue import predict_num

    nums = predict_num(num, tok, texts, device, MAX_LEN)

    icd_labels = list(dict.fromkeys(str_q2b(n) for n in icd_names))
    icd_top1, _ = jaccard_topk(texts, icd_labels, 1)

    methods = {
        "jaccard_top1_icd_only": [[icd_labels[row[0]]] for row in icd_top1],
        "jaccard_top1": [[labels[row[0]]] for row in cands],
        "jaccard_top15+bert_argmax": [[labels[row[int(np.argmax(p))]]] for row, p in zip(cands, probs)],
        "jaccard_top15+bert_cblue_commit": [
            commit(t, p, int(n), row, s, labels) for t, p, n, row, s in zip(texts, probs, nums, cands, jsc)
        ],
    }
    report = {"n": len(dev), "k": K, "recall_any_at_k": recall_any, "recall_all_at_k": recall_all, "methods": {}}
    for name, preds in methods.items():
        report["methods"][name] = evaluate(preds, golds)

    lines = [
        "Jaccard + BERT 基线（Feng 等 2023 的两阶段做法）在 CHIP-CDN dev 上",
        "=" * 72,
        f"n={len(dev)}；候选 = 字符集合 Jaccard top{K}（标签 = ICD 北京临床版 + CHIP train 标签）",
        f"候选上限：至少一个 gold 在候选里 {recall_any:.3f}；全部 gold 都在候选里 {recall_all:.3f}",
        "第二步 BERT = train_cdn_cblue.py 训练好的 RoBERTa-wwm-ext 句对排序器（只用 CHIP train，未在 Jaccard 候选上重训）",
        "",
        f"{'方法':36s}{'Acc严格':>9s}{'集合P':>8s}{'集合R':>8s}{'集合F1':>8s}",
    ]
    for name, m in report["methods"].items():
        lines.append(f"{name:36s}{m['acc_strict']:>9.4f}{m['set_p']:>8.4f}{m['set_r']:>8.4f}{m['set_f1']:>8.4f}")
    lines += [
        "",
        "- jaccard_top1_icd_only：只用 Jaccard、只查 ICD 词表（与 cdn_baselines 里的无监督行同一词表）",
        "- jaccard_top1：只用 Jaccard，取最相似的一个标准名",
        "- jaccard_top15+bert_argmax：论文做法，BERT 在 15 个候选里选概率最高的一个",
        "- jaccard_top15+bert_cblue_commit：同样候选，按 CBLUE 规则（个数模型 + p≥0.5）可输出多个",
    ]
    graph_lines, report["graph"] = score_graph(icd_labels)
    lines += graph_lines
    text = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jaccard_baseline.txt").write_text(text, encoding="utf-8")
    (OUT / "jaccard_baseline.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
