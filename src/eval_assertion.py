"""Negation / assertion filtering before graph building (src/assertion.py).

For CCKS2019 test and CCKS2017 (gold and predicted entities):
  1. assertion counts of disease mentions;
  2. how many gold disease→drug / surgery edges exist only because of negated mentions;
  3. query F1 when the standard answer is corrected (negated mentions removed from the gold graph):
     predicted graph unfiltered vs filtered, paired per query, disease-cluster bootstrap;
  4. external-KB support of the predicted G_norm before vs after filtering;
  5. rule check: if outputs/assertion/labels_gpt.csv exists (LLM labels for sample.csv), the rule's
     precision per class and the share of rule-affirmed mentions that are not affirmed.
Writes outputs/assertion/{sample.csv, key.json} (once) and outputs/metrics/assertion.{txt,json}.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assertion import DISEASE, classify, filter_docs, mentions
from eval_ablation import FULL
from eval_bootstrap import cluster_bootstrap
from eval_external_extra import compare, norm_graph
from eval_external_kg import load_cmeie, load_medicalkg
from eval_g_raw_vs_g_norm import build_graphs, neighbors_by_rel, query_rows
from ner_data import load_bio_docs
from normalize import configure_normalization

PROCESSED = ROOT / "data" / "processed"
SAMPLE_DIR = ROOT / "outputs" / "assertion"
OUT = ROOT / "outputs" / "metrics"
SETS = (("CCKS2019 test", "test.bio", "test_pred.bio"), ("CCKS2017", "ccks2017_gold.bio", "ccks2017_pred.bio"))
RELS = ("使用药物", "实施手术")
FILTERS = (("neg", ("negated",), "去掉否定"), ("all", ("negated", "uncertain", "family"), "再去掉可疑和家族史"))
SAMPLE_FLAGGED, SAMPLE_AFFIRMED = 150, 150


def contexts(docs, label: str) -> list[dict]:
    out = []
    for d, (chars, tags) in enumerate(docs):
        text = "".join(chars)
        for s, e in mentions(chars, tags):
            out.append({"set": label, "doc": d, "start": s, "rule": classify(text, s, e),
                        "context": text[max(0, s - 30):s] + "【" + text[s:e] + "】" + text[e:e + 12]})
    return out


def write_sample(all_ctx: list[dict]) -> None:
    if (SAMPLE_DIR / "key.json").exists():
        return
    rng = random.Random(2026)
    flagged = [c for c in all_ctx if c["rule"] != "affirmed"]
    affirmed = [c for c in all_ctx if c["rule"] == "affirmed"]
    pick = rng.sample(flagged, min(SAMPLE_FLAGGED, len(flagged))) + rng.sample(affirmed, SAMPLE_AFFIRMED)
    rng.shuffle(pick)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    with (SAMPLE_DIR / "sample.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "context"])
        for i, c in enumerate(pick):
            w.writerow([f"A{i:03d}", c["context"]])
    key = [{"id": f"A{i:03d}"} | c for i, c in enumerate(pick)]
    (SAMPLE_DIR / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=0), encoding="utf-8")


def rule_check() -> tuple[list[str], dict]:
    lab_path = SAMPLE_DIR / "labels_gpt.csv"
    if not lab_path.exists():
        return ["", "五、规则核对：尚无 outputs/assertion/labels_gpt.csv"], {}
    key = {k["id"]: k for k in json.loads((SAMPLE_DIR / "key.json").read_text(encoding="utf-8"))}
    with lab_path.open(encoding="utf-8-sig") as f:
        lab = {r["id"].strip(): r["label"].strip() for r in csv.DictReader(f)}
    pairs = [(key[i]["rule"], lab[i]) for i in key if i in lab]
    res = {"n": len(pairs), "confusion": Counter(f"{r}->{l}" for r, l in pairs)}
    L = ["", f"五、规则核对（GPT 按上下文判定 {len(pairs)} 条；抽样 = 规则标为非肯定的 {SAMPLE_FLAGGED} 条 + 肯定的 {SAMPLE_AFFIRMED} 条）"]
    for cls in ("negated", "uncertain", "family", "affirmed"):
        got = [l for r, l in pairs if r == cls]
        if got:
            res[f"precision_{cls}"] = sum(l == cls for l in got) / len(got)
            L.append(f"  规则判为 {cls:9s} {len(got):3d} 条，GPT 同意 {res[f'precision_{cls}']:.2f}  分布 {dict(Counter(got))}")
    aff = [l for r, l in pairs if r == "affirmed"]
    res["affirmed_missed_negation"] = sum(l == "negated" for l in aff) / max(len(aff), 1)
    L.append(f"  规则判为肯定、GPT 判为否定（漏判）{res['affirmed_missed_negation']:.3f}")
    res["confusion"] = dict(res["confusion"])
    return L, res


def rel_edges(rels, rel: str) -> set[tuple[str, str]]:
    return {(d, m) for d, ms in neighbors_by_rel(rels, rel).items() if d.startswith(DISEASE) for m in ms}


def mean_f(rows, key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def paired(base: dict, other: dict) -> dict:
    keys = sorted(base.keys() & other.keys())
    rows = [{"canonical": k[1], "f_raw": base[k]["f_norm"], "f_norm": other[k]["f_norm"]} for k in keys]
    bs = cluster_bootstrap(rows, "canonical")
    d = [r["f_norm"] - r["f_raw"] for r in rows]
    return {"n": len(rows), "delta": sum(d) / len(d), "delta_ci95": bs["delta_ci95"],
            "better": sum(x > 1e-9 for x in d), "worse": sum(x < -1e-9 for x in d)}


def main() -> None:
    configure_normalization()
    sources = {"cmeie": load_cmeie(ROOT / "data" / "dicts" / "raw" / "CMeIE"),
               "medicalkg": load_medicalkg(ROOT / "data" / "dicts" / "raw" / "medicalkg" / "medical.json")}
    all_ctx, report = [], {}
    L = ["否定 / 可疑 / 家族史过滤：建图前去掉「否认高血压」这类疾病提及", "=" * 72,
         "规则：src/assertion.py（否定前置词 否认/无/未见/排除…，作用到本小句结束；后置 （-）/阴性；",
         "可疑 不排除/待排/？；家族史 父亲/母亲/家族…）。标准答案也用同一规则修正。"]
    for label, gold_name, pred_name in SETS:
        gold, pred = load_bio_docs(PROCESSED / gold_name), load_bio_docs(PROCESSED / pred_name)
        all_ctx += contexts(gold, f"{label} gold") + contexts(pred, f"{label} pred")
        r = {"counts": {}, "graphs": {}}
        L += ["", f"【{label}】"]
        for side, docs in (("gold", gold), ("pred", pred)):
            _, c = filter_docs(docs, drop=())
            r["counts"][side] = c
            n = sum(c.values())
            L.append(f"  {('人工标注' if side == 'gold' else 'NER 预测')} 疾病提及 {n}：" +
                     "  ".join(f"{k} {v}（{v / n:.1%}）" for k, v in c.items()))

        filtered = {f: (filter_docs(gold, drop)[0], filter_docs(pred, drop)[0]) for f, drop, _ in FILTERS}
        g0 = build_graphs(gold, pred)
        for f, _, title in FILTERS:
            gold_f, pred_f = filtered[f]
            g_b = build_graphs(gold_f, pred)
            g_c = build_graphs(gold_f, pred_f)
            fr = {"title": title, "rels": {}}
            L.append(f"  ◆ {title}")
            for rel in RELS:
                e0, ef = rel_edges(g0["gold_norm_rels"], rel), rel_edges(g_b["gold_norm_rels"], rel)
                rows_b = {(x["query"], x["canonical"]): x for x in query_rows(g_b, rel)}
                rows_c = {(x["query"], x["canonical"]): x for x in query_rows(g_c, rel)}
                vb, vc = list(rows_b.values()), list(rows_c.values())
                p = paired(rows_b, rows_c)
                fr["rels"][rel] = {"gold_edges": len(e0), "gold_edges_filtered": len(ef),
                                   "unfiltered": {"n": len(vb), "f_raw": mean_f(vb, "f_raw"), "f_norm": mean_f(vb, "f_norm")},
                                   "filtered": {"n": len(vc), "f_raw": mean_f(vc, "f_raw"), "f_norm": mean_f(vc, "f_norm")},
                                   "paired_norm": p}
                L.append(f"    {rel}：标准答案边 {len(e0)} → {len(ef)}（去掉 {len(e0) - len(ef)}，{(len(e0) - len(ef)) / max(len(e0), 1):.1%}）；"
                         f"预测不过滤 F1 {mean_f(vb, 'f_raw'):.3f}→{mean_f(vb, 'f_norm'):.3f}，过滤 {mean_f(vc, 'f_raw'):.3f}→{mean_f(vc, 'f_norm'):.3f}"
                         f"（标准化图 差 {p['delta']:+.3f} [{p['delta_ci95'][0]:+.3f}, {p['delta_ci95'][1]:+.3f}]，变好/变差 {p['better']}/{p['worse']}）")
            g_pred = norm_graph(pred, FULL)
            g_pred_f = norm_graph(pred_f, FULL)
            fr["kb"] = {s: compare(g_pred, g_pred_f, src) for s, src in sources.items()}
            for s, st in (("cmeie", "CMeIE"), ("medicalkg", "寻医问药")):
                k = fr["kb"][s]
                L.append(f"    外部库 {st}：印证率 {k['g_raw']['support_rate']:.3f}（{k['g_raw']['n_edges']} 边）→ "
                         f"{k['g_norm']['support_rate']:.3f}（{k['g_norm']['n_edges']} 边），配对差 {k['paired']['delta']:+.3f} "
                         f"[{k['paired']['delta_ci95'][0]:+.3f}, {k['paired']['delta_ci95'][1]:+.3f}]")
            r["graphs"][f] = fr
            print(label, f, {rel: round(fr["rels"][rel]["filtered"]["f_norm"], 4) for rel in RELS}, flush=True)
        report[label] = r

    write_sample(all_ctx)
    check_lines, check = rule_check()
    L += ["", "说明：F1 都按修正后的标准答案计算；「差」= 预测图过滤后 − 不过滤（标准化图，逐查询配对，按疾病整簇 bootstrap）。"]
    L += check_lines
    text = "\n".join(L) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "assertion.txt").write_text(text, encoding="utf-8")
    (OUT / "assertion.json").write_text(json.dumps({"sets": report, "rule_check": check}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
