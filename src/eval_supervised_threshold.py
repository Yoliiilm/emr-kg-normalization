"""Supervised CHIP-CDN linker with a ranker-probability threshold, plugged into graph normalization.

The threshold is chosen on CHIP-CDN dev only (never on the CCKS test graph):
  t* = smallest grid value at which the supervised model, on the dev items our own system leaves
       unlinked, is at least as precise (set P) as our system is on the items it does link.
Then reported for every grid value (sensitivity):
  - CHIP-CDN dev: coverage / set P / R / F1 of the supervised model on that unlinked subset;
  - CCKS graph query F1 in fallback mode, fixed gold and self-consistent gold;
  - the 200-row normalization sample: how many kept-as-is surfaces get linked, and how many correctly.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cdn_linker import get_cdn_linker
from eval_cdn_baselines import DEV, gold_norms, load_train_alias
from eval_g_raw_vs_g_norm import build_graphs, eval_vs_gold, query_rows
from ner_data import load_bio_docs
from normalize import DictIndex, clean_surface, configure_normalization

OUT = ROOT / "outputs" / "metrics"
ANN = OUT
GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
RELS = ("使用药物", "实施手术")
LABEL_COL = "判定(1=对/0=错)"
GOLD_COL = "正确标准名(判错时填)"


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def set_prf(pairs: list[tuple[list[str], list[str]]]) -> dict:
    tp = n_p = n_g = answered = 0
    for preds, golds in pairs:
        ps, gs = {norm(p) for p in preds}, {norm(g) for g in golds}
        tp += len(ps & gs)
        n_p += len(ps)
        n_g += len(gs)
        answered += bool(ps)
    p = tp / n_p if n_p else 0.0
    r = tp / n_g if n_g else 0.0
    return {"n": len(pairs), "coverage": answered / len(pairs) if pairs else 0.0,
            "set_p": p, "set_r": r, "set_f1": 2 * p * r / (p + r) if p + r else 0.0}


def dev_part(linker) -> tuple[dict, float, float, list[str]]:
    """Supervised precision on the CHIP-CDN dev items our system leaves unlinked, per threshold."""
    rows = json.loads(DEV.read_text(encoding="utf-8"))
    samples = [(str(r.get("text", "")).lstrip("\ufeff").strip(), gold_norms(r)) for r in rows]
    samples = [(t, g) for t, g in samples if t and g]

    configure_normalization(use_chip_cdn=False, use_drug_alias=False)
    idx = DictIndex.load()
    alias = load_train_alias()
    idx.alias = alias
    idx.alias_clean = {clean_surface(k): v for k, v in alias.items() if clean_surface(k)}
    ours, unlinked = [], []
    for text, golds in samples:
        hits = idx.lookup_disease_multi(text)
        linked = [n for n, s in hits if s != "heuristic"]
        if linked:
            ours.append((linked, golds))
        else:
            unlinked.append((text, golds))
    configure_normalization()
    ours_p = set_prf(ours)["set_p"]

    linker.link_many([t for t, _ in unlinked])
    linker.save()
    per_t = {t: set_prf([(linker.link(x, t), g) for x, g in unlinked]) for t in GRID}
    chosen = next((t for t in GRID if per_t[t]["set_p"] >= ours_p), GRID[-1])

    lines = [
        "一、阈值选择（CHIP-CDN dev，不看 CCKS 测试集）",
        f"  本方法已链接的 dev 条目 {len(ours)} 条，集合 P = {ours_p:.4f}（监督模型需达到的精确率）",
        f"  本方法未链接的 dev 条目 {len(unlinked)} 条，监督模型在其上的表现：",
        f"  {'阈值':>6s}{'覆盖率':>9s}{'集合P':>9s}{'集合R':>9s}{'集合F1':>9s}",
    ]
    for t in GRID:
        m = per_t[t]
        mark = "  ← 选定" if t == chosen else ""
        lines.append(f"  {t:>6.2f}{m['coverage']:>9.4f}{m['set_p']:>9.4f}{m['set_r']:>9.4f}{m['set_f1']:>9.4f}{mark}")
    lines.append(f"  规则：取集合 P ≥ {ours_p:.4f} 的最小阈值 → t* = {chosen}")
    lines.append("")
    return {str(t): per_t[t] for t in GRID}, ours_p, chosen, lines


def graph_scores(g) -> dict:
    return {rel: eval_vs_gold(g, rel)["g_norm"]["f1"] for rel in RELS}


def graph_part(chosen: float) -> tuple[dict, list[str]]:
    processed = ROOT / "data" / "processed"
    gold_docs = load_bio_docs(processed / "test.bio")
    pred_docs = load_bio_docs(processed / "test_pred.bio")

    configure_normalization()
    g_off = build_graphs(gold_docs, pred_docs)
    base = {rel: {(r["query"], r["canonical"]): r["f_norm"] for r in query_rows(g_off, rel)} for rel in RELS}
    off = graph_scores(g_off)

    settings = [("off", None, "现有完整系统")]
    settings += [("fallback", t, f"兜底 p≥{t}") for t in GRID]
    settings += [("primary", 0.5, "优先 p≥0.5（原设定）"), ("primary", chosen, f"优先 p≥{chosen}（选定阈值）")]

    report, lines = {}, [
        "二、CCKS 建图查询 F1（疾病归一化接入监督模型）",
        "  固定口径：标准答案由现有系统生成（行间可比，但现有系统没合并的词，新合并即使正确也会扣分）",
        "  自洽口径：标准答案与预测图用同一设定（会奖励自身的合并，偏乐观）",
        f"  {'设定':24s}{'监督链接':>8s}{'用药-固定':>10s}{'手术-固定':>10s}{'用药-自洽':>10s}{'手术-自洽':>10s}  逐查询变好/变差(用药;手术)",
    ]
    for mode, t, title in settings:
        if mode == "off":
            fixed, self_, n_sup, diff = off, off, 0, None
        else:
            cfg = {"supervised_cdn": mode, "supervised_min_p": t}
            configure_normalization()
            g_fixed = build_graphs(gold_docs, pred_docs, pred_settings=cfg)
            fixed = graph_scores(g_fixed)
            n_sup = g_fixed["coverage"]["source_counts"].get("cdn_sup", 0)
            diff = {}
            for rel in RELS:
                rows = {(r["query"], r["canonical"]): r["f_norm"] for r in query_rows(g_fixed, rel)}
                common = rows.keys() & base[rel].keys()
                diff[rel] = (sum(rows[k] > base[rel][k] + 1e-9 for k in common),
                             sum(rows[k] < base[rel][k] - 1e-9 for k in common))
            configure_normalization(**cfg)
            self_ = graph_scores(build_graphs(gold_docs, pred_docs))
        key = mode if t is None else f"{mode}@{t}"
        report[key] = {"supervised_links": n_sup, "fixed_gold": fixed, "self_gold": self_, "better_worse": diff}
        d = "" if diff is None else f"  {diff['使用药物'][0]}/{diff['使用药物'][1]}; {diff['实施手术'][0]}/{diff['实施手术'][1]}"
        lines.append(
            f"  {title:24s}{n_sup:>8d}{fixed['使用药物']:>10.4f}{fixed['实施手术']:>10.4f}"
            f"{self_['使用药物']:>10.4f}{self_['实施手术']:>10.4f}{d}"
        )
    configure_normalization()
    lines.append("")
    return report, lines


def sample_part(linker) -> tuple[dict, list[str]]:
    """Kept-as-is rows of the normalization sample: does the supervised model link them, correctly?"""
    human = ANN / "manual_norm_sample.csv"
    draft = ANN / "manual_norm_sample_AI初判.csv"
    path, tag = human, "抽检表，AI 按 CHIP-CDN 口径判定"
    with human.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not any((r.get(LABEL_COL) or "").strip() for r in rows):
        path, tag = draft, "AI 预标注，未经人工校对"
        with draft.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    kept = [r for r in rows if r["系统来源"] == "heuristic" and (r.get(LABEL_COL) or "").strip() in ("0", "1")]
    texts = [r["原文表面词"] for r in kept]
    linker.link_many(texts + [clean_surface(t) or t for t in texts])
    linker.save()

    def sup(text: str, t: float) -> list[str]:
        return linker.link(text, t) or linker.link(clean_surface(text) or text, t)

    report, lines = {}, [
        f"三、抽检样本中「原样保留」的 {len(kept)} 个表面词（标签来源：{path.name}，{tag}）",
        "  应链接 = 标注给出了 ICD 标准名；不应链接 = 无对应 / 非疾病（保留原词才对）",
        "  链对 = 监督模型输出集合与标注标准名集合完全一致",
        f"  {'阈值':>6s}{'链接数':>8s}{'链对':>6s}{'链错':>6s}{'链接精确率':>12s}{'应链接召回':>12s}{'误链不应链接':>14s}",
    ]
    should = [r for r in kept if r[LABEL_COL].strip() == "0" and r[GOLD_COL].strip()]
    for t in GRID:
        linked = right = over = 0
        for r in kept:
            out = sup(r["原文表面词"], t)
            if not out:
                continue
            linked += 1
            gold = [g for g in r[GOLD_COL].split("##") if g.strip()]
            if r[LABEL_COL].strip() == "0" and gold and {norm(x) for x in out} == {norm(g) for g in gold}:
                right += 1
            elif not (r[LABEL_COL].strip() == "0" and gold):
                over += 1
        wrong = linked - right
        prec = right / linked if linked else 0.0
        rec = right / len(should) if should else 0.0
        report[str(t)] = {"linked": linked, "right": right, "wrong": wrong, "over_link": over,
                          "precision": prec, "recall_should_link": rec}
        lines.append(f"  {t:>6.2f}{linked:>8d}{right:>6d}{wrong:>6d}{prec:>12.4f}{rec:>12.4f}{over:>14d}")
    lines.append(f"  （应链接 {len(should)} 个；不应链接 {len(kept) - len(should)} 个）")
    lines.append("")
    return {"label_file": path.name, "label_source": tag, "rows": report}, lines


def main() -> None:
    linker = get_cdn_linker()
    dev, ours_p, chosen, l1 = dev_part(linker)
    print("\n".join(l1))
    sample, l3 = sample_part(linker)
    print("\n".join(l3))
    graph, l2 = graph_part(chosen)

    lines = ["监督 CHIP-CDN 模型 + 置信度阈值 接入建图", "=" * 72, ""] + l1 + l2 + l3
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "supervised_threshold.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "supervised_threshold.json").write_text(
        json.dumps({"grid": GRID, "chosen": chosen, "ours_linked_set_p": ours_p, "dev": dev,
                    "graph": graph, "sample": sample}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    get_cdn_linker().save()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
