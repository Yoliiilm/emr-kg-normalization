"""Replicate the graph-query experiments on CCKS2017 CNER (second provider, 300 records).

Pred entities come from the CCKS2019-trained NER (outputs/models/best.pt) run on CCKS2017 text,
so this is also a cross-domain NER setting. Gold entities and the label mapping follow
eval_cross_dataset.load_ccks2017. The standard answer is fixed to the current full system
(as in eval_ablation / eval_supervised_graph), so normalizer rows are directly comparable.

Reports
  1. NER P/R/F1 on the shared types,
  2. G_raw vs G_norm query F1 (drug / surgery) with disease-cluster bootstrap and sign-flip test,
  3. normalizers that score higher on CHIP-CDN plugged into graph building
     (supervised CBLUE fallback / primary, SapBERT top-1 fallback), paired against the full system,
next to the CCKS2019 figures.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from seqeval.metrics import classification_report
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from eval_bootstrap import cluster_bootstrap, sign_flip_test
from eval_cross_dataset import load_ccks2017
from eval_g_raw_vs_g_norm import build_graphs, query_rows
from ner_data import load_bio_docs
from ner_model import BertBiLstmCrf
from normalize import configure_normalization
from train_ner import predict_docs, write_bio

PROCESSED = ROOT / "data" / "processed"
GOLD_BIO = PROCESSED / "ccks2017_gold.bio"
PRED_BIO = PROCESSED / "ccks2017_pred.bio"
OUT = ROOT / "outputs" / "metrics"
SHARED = {"疾病和诊断", "解剖部位", "药物", "手术"}
RELS = ("使用药物", "实施手术")
MODES = [
    ("off", {}, "本方法（完整系统）"),
    ("cdn_fallback", {"supervised_cdn": "fallback"}, "+ 监督 CDN 兜底（只处理未链接的）"),
    ("cdn_primary", {"supervised_cdn": "primary"}, "监督 CDN 优先"),
    ("sapbert_top1", {"retriever": "sapbert", "retriever_fallback": "top1"}, "SapBERT 检索 + top1 兜底"),
]


def predict(gold_docs) -> list:
    if PRED_BIO.exists():
        return load_bio_docs(PRED_BIO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ROOT / "outputs" / "models" / "best.pt", map_location=device, weights_only=False)
    args = ckpt["args"]
    id2label = {i: lab for lab, i in ckpt["label2id"].items()}
    model = BertBiLstmCrf(args["model_name"], num_labels=len(id2label),
                          lstm_hidden=args["lstm_hidden"], dropout=args["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    tags = predict_docs(model, tokenizer, gold_docs, id2label, args["max_len"], device)
    pred_docs = [(chars, t) for (chars, _), t in zip(gold_docs, tags)]
    write_bio(PRED_BIO, pred_docs)
    return pred_docs


def shared_only(tags: list[str]) -> list[str]:
    return [t if t == "O" or t.split("-", 1)[1] in SHARED else "O" for t in tags]


def rows_by_key(g, rel) -> dict:
    return {(r["query"], r["canonical"]): r for r in query_rows(g, rel)}


def paired(base: dict, other: dict) -> dict:
    keys = sorted(base.keys() & other.keys())
    rows = [{"canonical": k[1], "f_raw": base[k]["f_norm"], "f_norm": other[k]["f_norm"]} for k in keys]
    bs = cluster_bootstrap(rows, "canonical")
    d = [r["f_norm"] - r["f_raw"] for r in rows]
    return {"n": len(rows), "delta": sum(d) / len(d), "delta_ci95": bs["delta_ci95"],
            "better": sum(x > 1e-9 for x in d), "worse": sum(x < -1e-9 for x in d)}


def main() -> None:
    gold_docs, load_stats = load_ccks2017()
    write_bio(GOLD_BIO, gold_docs)
    pred_docs = predict(gold_docs)
    ner_report = classification_report([shared_only(t) for _, t in gold_docs],
                                       [shared_only(t) for _, t in pred_docs], digits=3)

    configure_normalization()
    results, base_rows = {}, {}
    for mode, settings, title in MODES:
        g = build_graphs(gold_docs, pred_docs, pred_settings=settings or None)
        cov = g["coverage"]
        res = {"title": title, "icd_hit_rate": cov["icd_hit_rate"], "source_counts": cov["source_counts"], "rels": {}}
        for rel in RELS:
            rows = rows_by_key(g, rel)
            vals = list(rows.values())
            entry = {"n": len(vals),
                     "f1_raw": sum(r["f_raw"] for r in vals) / len(vals),
                     "f1_norm": sum(r["f_norm"] for r in vals) / len(vals)}
            if mode == "off":
                base_rows[rel] = rows
                bs = cluster_bootstrap(vals, "canonical")
                entry |= {"raw_vs_norm_ci95": bs["delta_ci95"],
                          "p_sign_flip": sign_flip_test([r["delta"] for r in vals]),
                          "better": sum(r["delta"] > 1e-9 for r in vals),
                          "worse": sum(r["delta"] < -1e-9 for r in vals)}
            else:
                entry["vs_off"] = paired(base_rows[rel], rows)
            res["rels"][rel] = entry
        results[mode] = res
        print(mode, {rel: round(e["f1_norm"], 4) for rel, e in res["rels"].items()}, flush=True)

    from cdn_linker import get_cdn_linker

    get_cdn_linker().save()

    ref = json.loads((OUT / "supervised_graph.json").read_text(encoding="utf-8"))["modes"]
    ref_line = (f"CCKS2019 对照（固定口径，用药 / 手术 F1）：不标准化 {ref['off']['fixed_gold']['使用药物'][0]:.3f} / "
                f"{ref['off']['fixed_gold']['实施手术'][0]:.3f}；本方法 {ref['off']['fixed_gold']['使用药物'][1]:.3f} / "
                f"{ref['off']['fixed_gold']['实施手术'][1]:.3f}；监督兜底 {ref['fallback']['fixed_gold']['使用药物'][1]:.3f} / "
                f"{ref['fallback']['fixed_gold']['实施手术'][1]:.3f}；监督优先 {ref['primary']['fixed_gold']['使用药物'][1]:.3f} / "
                f"{ref['primary']['fixed_gold']['实施手术'][1]:.3f}")

    L = ["第二个数据集：CCKS2017 CNER 训练集（极目云健康，300 份）上的建图查询 F1", "=" * 72,
         "预测实体 = CCKS2019 上训练的 NER 直接用于 CCKS2017（跨医院）；标准答案 = 人工标注实体建图，固定用本方法归一化。",
         f"「治疗」按关键词拆为 药物 {load_stats.get('治疗→药物', 0)} / 手术 {load_stats.get('治疗→手术', 0)}；症状、检查不参与。",
         "", "一、NER 在 CCKS2017 上的效果（只看两边都有的类型）", ner_report,
         "二、标准化前后（差按疾病整簇 bootstrap 2000 次；p 为符号翻转检验）"]
    off = results["off"]["rels"]
    for rel in RELS:
        e = off[rel]
        L.append(f"  {rel}  查询 {e['n']}  不标准化 {e['f1_raw']:.3f} → 本方法 {e['f1_norm']:.3f}  "
                 f"差 {e['f1_norm'] - e['f1_raw']:+.3f} [{e['raw_vs_norm_ci95'][0]:+.3f}, {e['raw_vs_norm_ci95'][1]:+.3f}]  "
                 f"变好/变差 {e['better']}/{e['worse']}  p={e['p_sign_flip']:.4f}")
    L += ["", "三、标准化评测上更强的方法接入建图（与本方法逐查询配对）"]
    for mode, _, title in MODES:
        r = results[mode]
        parts = [f"【{title}】疾病链接率 {r['icd_hit_rate']:.3f}"]
        for rel in RELS:
            e = r["rels"][rel]
            s = f"{rel} F1 {e['f1_norm']:.3f}"
            if "vs_off" in e:
                v = e["vs_off"]
                s += f"（差 {v['delta']:+.3f} [{v['delta_ci95'][0]:+.3f}, {v['delta_ci95'][1]:+.3f}]，变好/变差 {v['better']}/{v['worse']}）"
            parts.append(s)
        L.append("  " + "  ".join(parts))
    L += ["", ref_line]
    text = "\n".join(L) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ccks2017_graph.txt").write_text(text, encoding="utf-8")
    (OUT / "ccks2017_graph.json").write_text(json.dumps({"load": load_stats, "results": results},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
