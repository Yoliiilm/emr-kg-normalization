"""Score LLM disease normalization (retrieve-then-select) against the existing normalizers.

Reads outputs/llm_norm/key.json and labels_gpt_XX.csv (id, answer = candidate numbers joined by
## in importance order, or 0 = keep as is), writes outputs/llm_norm/map.json for
normalize.llm_names, then reports
  1. CHIP-CDN dev: LLM vs ours (+split), SapBERT top-1, CBLUE supervised, ours + LLM fallback,
     plus the candidate-list ceiling (any gold among the LLM's candidates);
  2. graph query F1 (fixed gold = full system) with llm_cdn fallback / primary next to the
     supervised CDN model, on CCKS2019 test and CCKS2017, paired against the full system.

  python src/llm_norm_score.py           # GPT answers → map.json, llm_norm.{txt,json}
  python src/llm_norm_score.py opus      # labels_opus_*.csv → map_opus.json, llm_norm_opus.{txt,json}
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

import normalize
from eval_bootstrap import cluster_bootstrap
from eval_cdn_baselines import DEV, SUPERVISED, acc_hit, gold_norms, load_train_alias, strict_hit
from eval_g_raw_vs_g_norm import build_graphs, query_rows
from ner_data import load_bio_docs
from normalize import DictIndex, clean_surface, configure_normalization

LLM_DIR = ROOT / "outputs" / "llm_norm"
OUT = ROOT / "outputs" / "metrics"
PROCESSED = ROOT / "data" / "processed"
GRAPH_SETS = (("CCKS2019 test", "test.bio", "test_pred.bio"),
              ("CCKS2017", "ccks2017_gold.bio", "ccks2017_pred.bio"))
RELS = ("使用药物", "实施手术")
MODES = [
    ("off", {}, "本方法（完整系统）"),
    ("llm_fallback", {"llm_cdn": "fallback"}, "+ 大模型兜底（只处理未链接的）"),
    ("llm_primary", {"llm_cdn": "primary"}, "大模型优先"),
    ("cdn_fallback", {"supervised_cdn": "fallback"}, "+ 监督 CDN 兜底"),
    ("cdn_primary", {"supervised_cdn": "primary"}, "监督 CDN 优先"),
]
ANSWER_RE = re.compile(r"^\d+(##\d+)*$")
JUDGE = sys.argv[1] if len(sys.argv) > 1 else "gpt"
SUFFIX = "" if JUDGE == "gpt" else f"_{JUDGE}"
MODEL_NAME = {"gpt": "GPT", "opus": "Claude Opus"}.get(JUDGE, JUDGE)


def build_map() -> tuple[dict[str, list[str]], list[dict], dict]:
    items = json.loads((LLM_DIR / "key.json").read_text(encoding="utf-8"))
    answers: dict[str, str] = {}
    for path in sorted(LLM_DIR.glob(f"labels_{JUDGE}_*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                answers[row["id"].strip()] = re.sub(r"\s+", "", row["answer"])
    mapping: dict[str, list[str]] = {}
    stats = {"items": len(items), "answered": 0, "missing": 0, "bad": 0, "zero": 0, "multi": 0}
    for it in items:
        ans = answers.get(it["id"])
        if ans is None:
            stats["missing"] += 1
            continue
        stats["answered"] += 1
        if not ANSWER_RE.match(ans):
            stats["bad"] += 1
            continue
        idx = [int(x) for x in ans.split("##") if x != "0"]
        if any(i > len(it["cands"]) for i in idx):
            stats["bad"] += 1
            idx = [i for i in idx if i <= len(it["cands"])]
        names = list(dict.fromkeys(it["cands"][i - 1] for i in idx))
        stats["zero"] += not names
        stats["multi"] += len(names) > 1
        mapping[it["text"]] = names
    normalize.LLM_MAP = LLM_DIR / f"map{SUFFIX}.json"
    normalize.LLM_MAP.write_text(json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8")
    normalize.reload_llm_map()
    return mapping, items, stats


def score_cdn(mapping: dict[str, list[str]], items: list[dict]) -> tuple[list[str], dict]:
    rows = json.loads(DEV.read_text(encoding="utf-8"))
    samples = [(t, g) for t, g in ((str(r.get("text", "")).lstrip("\ufeff").strip(), gold_norms(r)) for r in rows)
               if t and g]
    cands = {it["text"]: it["cands"] for it in items}

    configure_normalization(use_chip_cdn=False, use_drug_alias=False)
    idx = DictIndex.load()
    train_alias = load_train_alias()
    idx.alias = train_alias
    idx.alias_clean = {clean_surface(k): v for k, v in train_alias.items() if clean_surface(k)}

    sup = {}
    if SUPERVISED.exists():
        sup = {r["text"]: [p for p in r["pred"] if p != "癌"]
               for r in json.loads(SUPERVISED.read_text(encoding="utf-8"))}

    def ours_split(text):
        return [n for n, _ in idx.lookup_disease_multi(text)]

    def ours_split_with(other):
        def fn(text):
            hits = idx.lookup_disease_multi(text)
            if all(s == "heuristic" for _, s in hits):
                return other(text)
            return [n for n, s in hits if s != "heuristic"]
        return fn

    def llm(text):
        return mapping.get(text, [])

    def sapbert_top1(text):
        return cands.get(text, [])[:1]

    def supervised(text):
        return sup.get(text, [])

    methods = [
        ("ours_split", ours_split, "本方法 + 多诊断切分"),
        ("sapbert_top1", sapbert_top1, "SapBERT 检索 top1（即大模型候选表第 1 个）"),
        ("cblue_supervised", supervised, "CBLUE 监督基线（在 CHIP-CDN train 上训练）"),
        ("llm_select", llm, "大模型从 SapBERT 候选里选（零样本，不用 CHIP 训练数据）"),
        ("ours_split+supervised_fb", ours_split_with(supervised), "本方法 + 切分，未链接时监督兜底"),
        ("ours_split+llm_fb", ours_split_with(llm), "本方法 + 切分，未链接时大模型兜底"),
    ]
    norm = lambda s: re.sub(r"\s+", "", s)
    n = len(samples)
    report = {"n": n, "methods": {}}
    for name, fn, desc in methods:
        m = {"desc": desc, "correct": 0, "strict": 0, "empty": 0, "tp": 0, "n_pred": 0, "n_gold": 0}
        for text, golds in samples:
            preds = [p for p in fn(text) if p]
            pred = preds[0] if preds else ""
            m["empty"] += not pred
            m["correct"] += acc_hit(pred, golds)
            m["strict"] += strict_hit(pred, golds)
            pset, gset = {norm(p) for p in preds}, {norm(g) for g in golds}
            m["tp"] += len(pset & gset)
            m["n_pred"] += len(pset)
            m["n_gold"] += len(gset)
        sp = m["tp"] / m["n_pred"] if m["n_pred"] else 0.0
        sr = m["tp"] / m["n_gold"] if m["n_gold"] else 0.0
        m |= {"acc": m["correct"] / n, "acc_strict": m["strict"] / n, "coverage": 1 - m["empty"] / n,
              "set_p": sp, "set_r": sr, "set_f1": 2 * sp * sr / (sp + sr) if sp + sr else 0.0}
        report["methods"][name] = m

    in_cands = sum(any(norm(g) in {norm(c) for c in cands.get(t, [])} for g in golds) for t, golds in samples)
    report["candidate_ceiling"] = in_cands / n
    configure_normalization()

    L = [f"一、CHIP-CDN dev（n={n}；CHIP 词表只用 train；Acc 宽松/严格同 cdn_baselines.txt）",
         f"  大模型候选表上限：gold 至少一个出现在候选里的比例 {in_cands / n:.4f}",
         f"  {'方法':28s}{'Acc宽松':>9s}{'Acc严格':>9s}{'覆盖率':>9s}{'集合P':>8s}{'集合R':>8s}{'集合F1':>8s}"]
    for name, _, desc in methods:
        m = report["methods"][name]
        L.append(f"  {name:28s}{m['acc']:>9.4f}{m['acc_strict']:>9.4f}{m['coverage']:>9.4f}"
                 f"{m['set_p']:>8.4f}{m['set_r']:>8.4f}{m['set_f1']:>8.4f}  {desc}")
    return L, report


def rows_by_key(g, rel) -> dict:
    return {(r["query"], r["canonical"]): r for r in query_rows(g, rel)}


def paired(base: dict, other: dict) -> dict:
    keys = sorted(base.keys() & other.keys())
    rows = [{"canonical": k[1], "f_raw": base[k]["f_norm"], "f_norm": other[k]["f_norm"]} for k in keys]
    bs = cluster_bootstrap(rows, "canonical")
    d = [r["f_norm"] - r["f_raw"] for r in rows]
    return {"n": len(rows), "delta": sum(d) / len(d), "delta_ci95": bs["delta_ci95"],
            "better": sum(x > 1e-9 for x in d), "worse": sum(x < -1e-9 for x in d)}


def score_graphs() -> tuple[list[str], dict]:
    L, report = ["", "二、接入建图后的查询 F1（固定口径：标准答案用本方法归一化；差 = 与本方法逐查询配对，按疾病整簇 bootstrap）",
                 "    自洽口径：标准答案也用同一配置归一化，列出 不标准化→标准化 的 F1（循环评测，只作对照）"], {}
    normalize.LLM_MISSING.clear()
    for label, gold_name, pred_name in GRAPH_SETS:
        gold, pred = load_bio_docs(PROCESSED / gold_name), load_bio_docs(PROCESSED / pred_name)
        configure_normalization()
        base, res = {}, {}
        L.append(f"  【{label}】")
        for mode, settings, title in MODES:
            g = build_graphs(gold, pred, pred_settings=settings or None)
            r = {"title": title, "icd_hit_rate": g["coverage"]["icd_hit_rate"], "rels": {}}
            parts = [f"{title:18s}疾病链接率 {r['icd_hit_rate']:.3f}"]
            for rel in RELS:
                rows = rows_by_key(g, rel)
                f1 = sum(x["f_norm"] for x in rows.values()) / len(rows)
                e = {"n": len(rows), "f1": f1}
                s = f"{rel} {f1:.3f}"
                if mode == "off":
                    base[rel] = rows
                else:
                    v = e["vs_off"] = paired(base[rel], rows)
                    s += f"（{v['delta']:+.3f} [{v['delta_ci95'][0]:+.3f}, {v['delta_ci95'][1]:+.3f}] {v['better']}/{v['worse']}）"
                parts.append(s)
                r["rels"][rel] = e
            if mode != "off":
                configure_normalization(**settings)
                gs = build_graphs(gold, pred)
                configure_normalization()
                r["self_gold"] = {}
                for rel in RELS:
                    rows = query_rows(gs, rel)
                    r["self_gold"][rel] = {"n": len(rows),
                                           "f1_raw": sum(x["f_raw"] for x in rows) / len(rows),
                                           "f1_norm": sum(x["f_norm"] for x in rows) / len(rows)}
                parts.append("自洽口径 " + "  ".join(
                    f"{rel} {v['f1_raw']:.3f}→{v['f1_norm']:.3f}（{v['n']} 查询）" for rel, v in r["self_gold"].items()))
            else:
                parts.append("自洽口径 " + "  ".join(
                    f"{rel} {sum(x['f_raw'] for x in base[rel].values()) / len(base[rel]):.3f}→"
                    f"{r['rels'][rel]['f1']:.3f}（{len(base[rel])} 查询）" for rel in RELS))
            res[mode] = r
            L.append("    " + "  ".join(parts))
            print(label, mode, {k: round(v["f1"], 4) for k, v in r["rels"].items()}, flush=True)
        report[label] = res
    configure_normalization()
    missing = sorted(t for t in normalize.LLM_MISSING if t.strip())
    L.append(f"  大模型映射表缺失的疾病写法：{len(missing)}" + (f"（例：{'、'.join(missing[:5])}）" if missing else ""))
    report["llm_missing"] = missing
    return L, report


def main() -> None:
    mapping, items, stats = build_map()
    print("map", stats, flush=True)
    head = ["大模型做疾病标准化（检索 + 选择）与现有方法对比", "=" * 72,
            "做法：SapBERT 为每条写法检索 ICD-10 北京临床版候选（整条 top15 + 切分后每段 top5，最多 20 个），",
            f"{MODEL_NAME} 逐条选出正确标准名（可多选，按主次排序）或答 0（保持原样/不是疾病）。零样本，未用 CHIP 训练数据。",
            f"条目 {stats['items']}，已作答 {stats['answered']}，缺答 {stats['missing']}，格式错 {stats['bad']}，"
            f"答 0 {stats['zero']}，多选 {stats['multi']}", ""]
    cdn_lines, cdn = score_cdn(mapping, items)
    graph_lines, graph = score_graphs()
    text = "\n".join(head + cdn_lines + graph_lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"llm_norm{SUFFIX}.txt").write_text(text, encoding="utf-8")
    (OUT / f"llm_norm{SUFFIX}.json").write_text(json.dumps({"map": stats, "cdn": cdn, "graph": graph},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
