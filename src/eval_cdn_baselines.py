"""External normalization baselines on CHIP-CDN dev (Accuracy@1 / fuzzy Acc)."""

from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from normalize import (
    DictIndex,
    clean_surface,
    configure_normalization,
    icd_variants,
    rule_disease,
    strip_grade,
)

DEV = ROOT / "data" / "dicts" / "raw" / "CHIP-CDN_dev.json"
TRAIN = ROOT / "data" / "dicts" / "raw" / "CHIP-CDN_train.json"
OUT = ROOT / "outputs" / "metrics"
SUPERVISED = OUT / "cdn_supervised_dev_preds.json"


def gold_norms(row: dict) -> list[str]:
    raw = str(row.get("normalized_result", "")).strip()
    return [x.strip() for x in raw.split("##") if x.strip() and x.strip() != "癌"]


def acc_hit(pred: str, golds: list[str]) -> bool:
    if not pred or not golds:
        return False
    p = re.sub(r"\s+", "", pred)
    for g in golds:
        g2 = re.sub(r"\s+", "", g)
        if p == g2 or p in g2 or g2 in p:
            return True
        if SequenceMatcher(None, p, g2).ratio() >= 0.92:
            return True
    return False


def strict_hit(pred: str, golds: list[str]) -> bool:
    if not pred:
        return False
    p = re.sub(r"\s+", "", pred)
    return any(p == re.sub(r"\s+", "", g) for g in golds)


def bert_top1_batch(texts: list[str], batch_size: int = 128) -> list[tuple[str, float]]:
    """Plain RoBERTa mean-pool retrieval against ICD names; no acceptance filters."""
    import torch

    from bert_linker import get_linker, mean_pool

    linker = get_linker()
    out: list[tuple[str, float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            tok = linker.tokenizer(
                chunk, truncation=True, max_length=32, padding=True, return_tensors="pt"
            )
            tok = {k: v.to(linker.device) for k, v in tok.items()}
            hidden = linker.model(**tok).last_hidden_state
            vec = torch.nn.functional.normalize(
                mean_pool(hidden, tok["attention_mask"]), p=2, dim=1
            ).cpu()
            scores = vec @ linker.emb.T
            topv, topi = scores.max(dim=1)
            out.extend((linker.names[j], float(v)) for v, j in zip(topv.tolist(), topi.tolist()))
    return out


def load_train_alias() -> dict[str, str]:
    alias: dict[str, str] = {}
    rows = json.loads(TRAIN.read_text(encoding="utf-8"))
    for row in rows:
        text = str(row.get("text", "")).strip()
        norms = gold_norms(row)
        if text and norms:
            alias[text] = norms[0]
    return alias


def main() -> None:
    rows = json.loads(DEV.read_text(encoding="utf-8"))
    train_alias = load_train_alias()

    configure_normalization(use_chip_cdn=False, use_drug_alias=False)
    idx = DictIndex.load()
    # attach train-only CHIP lexicon for fair lexicon baseline / ours+chip
    idx.alias = train_alias
    idx.alias_clean = {clean_surface(k): v for k, v in train_alias.items() if clean_surface(k)}

    def exact(text: str) -> str:
        return idx.resolve_name(text.strip()) or idx.resolve_name(clean_surface(text)) or ""

    def variant(text: str) -> str:
        cleaned = clean_surface(text)
        for cand in icd_variants(cleaned):
            hit = idx.resolve_name(cand)
            if hit:
                return hit
        return idx.longest_prefix_hit(cleaned) or ""

    def edit(text: str) -> str:
        q = clean_surface(text) or text.strip()
        if len(q) < 2:
            return ""
        best, best_r = "", 0.0
        for name in idx.by_head.get(q[:2], ()):
            if abs(len(name) - len(q)) > 6:
                continue
            r = SequenceMatcher(None, q, name).ratio()
            if r > best_r:
                best, best_r = name, r
        return best if best_r >= 0.80 else ""

    def chip_lex(text: str) -> str:
        raw = text.strip()
        cleaned = clean_surface(raw)
        for key in (raw, cleaned):
            hit = idx.alias.get(key) or idx.alias_clean.get(key)
            if hit:
                return idx.resolve_name(hit) or hit
        return ""

    def ours(text: str) -> str:
        # enable chip path using train-only alias already on idx
        return idx.lookup_disease(text)[0]

    def ours_linked(text: str) -> str:
        pred, src = idx.lookup_disease(text)
        return "" if src == "heuristic" else pred

    def ours_bert_fallback(text: str) -> str:
        pred, src = idx.lookup_disease(text)
        return bert_top1(text) if src == "heuristic" else pred

    def ours_split(text: str) -> list[str]:
        return [n for n, _ in idx.lookup_disease_multi(text)]

    def ours_split_bert(text: str) -> list[str]:
        hits = idx.lookup_disease_multi(text)
        if all(s == "heuristic" for _, s in hits):
            return [bert_top1(text)]
        return [n for n, s in hits if s != "heuristic"]

    def rules_only(text: str) -> str:
        return rule_disease(text)

    samples = []
    for row in rows:
        text = str(row.get("text", "")).lstrip("\ufeff").strip()
        golds = gold_norms(row)
        if text and golds:
            samples.append((text, golds))

    bert_queries = [strip_grade(clean_surface(t)) or t for t, _ in samples]
    bert_preds = dict(zip(bert_queries, bert_top1_batch(bert_queries)))

    def bert_top1(text: str) -> str:
        q = strip_grade(clean_surface(text)) or text
        return bert_preds[q][0]

    def bert_filtered(text: str) -> str:
        from bert_linker import get_linker

        q = strip_grade(clean_surface(text)) or text
        for cand in icd_variants(q):
            hit = get_linker().retrieve(cand)
            if hit:
                return hit[0]
        return ""

    methods = [
        ("rules_only", rules_only, "仅规则：去修饰/分期/分化 + 癌→恶性肿瘤等改写，不查任何词表"),
        ("exact_icd", exact, "仅词典：与 ICD 标准名完全一致"),
        ("icd_variant_prefix", variant, "规则 + 词典：变体改写后查 ICD，再试最长前缀"),
        ("edit_distance_0.80", edit, "编辑距离：同首两字 ICD 名中相似度≥0.80 取最近"),
        ("chip_train_lexicon", chip_lex, "仅 CHIP 训练集别名表"),
        ("bert_top1", bert_top1, "RoBERTa 向量检索 ICD，直接取 top1（无过滤）"),
        ("bert_filtered", bert_filtered, "RoBERTa 检索 + 部位/形态/侧别过滤（本方法的检索模块单独用）"),
        ("ours_icd_curated_fuzzy_chip", ours, "本方法：ICD优先 + curated长尾 + BERT + 过滤CHIP + fuzzy"),
        ("ours_linked_only", ours_linked, "本方法只算真正链接上的（heuristic 原样保留记为空）"),
        ("ours+bert_top1_fallback", ours_bert_fallback, "本方法未链接时改用 bert_top1 兜底"),
        ("ours_split", ours_split, "本方法 + 多诊断切分（可输出多个标准名）"),
        ("ours_split+bert_fallback", ours_split_bert, "本方法 + 切分，整条都未链接时 bert_top1 兜底"),
    ]

    if SUPERVISED.exists():
        # "癌" is dropped from gold (see gold_norms), so drop it from predictions too
        sup = {
            r["text"]: [p for p in r["pred"] if p != "癌"]
            for r in json.loads(SUPERVISED.read_text(encoding="utf-8"))
        }

        def supervised(text: str) -> list[str]:
            return sup.get(text, [])

        def ours_split_sup(text: str) -> list[str]:
            hits = idx.lookup_disease_multi(text)
            if all(s == "heuristic" for _, s in hits):
                return supervised(text)
            return [n for n, s in hits if s != "heuristic"]

        methods += [
            ("cblue_supervised", supervised, "CBLUE 官方监督基线（召回 + RoBERTa 排序 + 个数模型，train 上训练）"),
            ("ours_split+supervised_fallback", ours_split_sup, "本方法 + 切分，整条都未链接时用监督模型兜底"),
        ]

    n = len(samples)
    report = {"n": n, "methods": {}, "note": "CHIP lexicon = train only; eval on dev"}
    for name, fn, desc in methods:
        m = {"desc": desc, "correct": 0, "correct_strict": 0, "pred_empty": 0, "tp": 0, "n_pred": 0, "n_gold": 0}
        for text, golds in samples:
            out = fn(text)
            preds = [p for p in (out if isinstance(out, list) else [out]) if p]
            pred = preds[0] if preds else ""
            if not pred:
                m["pred_empty"] += 1
            if acc_hit(pred, golds):
                m["correct"] += 1
            if strict_hit(pred, golds):
                m["correct_strict"] += 1
            pset = {re.sub(r"\s+", "", p) for p in preds}
            gset = {re.sub(r"\s+", "", g) for g in golds}
            m["tp"] += len(pset & gset)
            m["n_pred"] += len(pset)
            m["n_gold"] += len(gset)
        answered = n - m["pred_empty"]
        m["accuracy"] = m["correct"] / n if n else 0.0
        m["accuracy_strict"] = m["correct_strict"] / n if n else 0.0
        m["coverage"] = answered / n if n else 0.0
        m["precision_answered"] = m["correct"] / answered if answered else 0.0
        sp = m["tp"] / m["n_pred"] if m["n_pred"] else 0.0
        sr = m["tp"] / m["n_gold"] if m["n_gold"] else 0.0
        m["set_p"], m["set_r"] = sp, sr
        m["set_f1"] = 2 * sp * sr / (sp + sr) if sp + sr else 0.0
        report["methods"][name] = m
        print(f"{name}: Acc={m['accuracy']:.4f} setF1={m['set_f1']:.4f}")

    lines = [
        "CHIP-CDN_dev 术语归一化基线对比（Accuracy@1）",
        "=" * 72,
        f"n={n}  lexicon=CHIP-CDN_train only（避免 dev 泄漏）",
        "Acc宽松: 与任一 gold 相同/互相包含/相似度≥0.92；Acc严格: 与任一 gold 完全相同",
        "覆盖率: 给出非空答案的比例；答准率: 在给出答案的条目里答对的比例（宽松）",
        "集合F1: CBLUE 口径，预测集合与 gold 集合（## 分隔）完全匹配的 micro P/R/F1",
        "",
        f"{'方法':30s}{'Acc宽松':>9s}{'Acc严格':>9s}{'覆盖率':>9s}{'答准率':>9s}{'集合P':>8s}{'集合R':>8s}{'集合F1':>8s}",
    ]
    for name, _, _ in methods:
        m = report["methods"][name]
        lines.append(
            f"{name:30s}{m['accuracy']:>9.4f}{m['accuracy_strict']:>9.4f}"
            f"{m['coverage']:>9.4f}{m['precision_answered']:>9.4f}"
            f"{m['set_p']:>8.4f}{m['set_r']:>8.4f}{m['set_f1']:>8.4f}"
        )
    lines.append("")
    lines.append("方法说明:")
    for name, _, desc in methods:
        lines.append(f"- {name}: {desc}")
    lines.append("")
    if SUPERVISED.exists():
        lines.append("注: cblue_supervised 由 src/train_cdn_cblue.py 在 CHIP-CDN train 上训练（移植 CBLUE 官方流程），")
        lines.append("    dev 集合 F1 按官方口径（含「癌」标签）见 cdn_supervised.json；本表 gold 去掉了「癌」。")
    else:
        lines.append("注: 监督基线未训练；运行 python src/train_cdn_cblue.py 后重跑本脚本。")

    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (OUT / "cdn_baselines.txt").write_text(text, encoding="utf-8")
    (OUT / "cdn_baselines.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    configure_normalization()
    print(text)


if __name__ == "__main__":
    main()
