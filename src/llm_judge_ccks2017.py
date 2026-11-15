"""Blind jury on CCKS2017, same design as src/llm_judge_sample.py / llm_judge_score.py (REPORT 3.5).

Does normalization still give more distinct correct drugs in a disease's top 5 on another hospital's notes?
Three inputs: entities predicted by the CCKS2019-trained NER (ccks2017_pred.bio), the annotated entities
(ccks2017_gold.bio), the latter to separate the effect of cross-hospital NER errors, and the NER after
two-fold fine-tuning on CCKS2017 (ccks2017_pred_ft.bio, src/finetune_ccks2017.py).
Pairs already in this directory keep their ids; new pairs are numbered on and written to new batch files.

Queries: g_norm disease nodes with ≥ MIN_DRUGS drug neighbours, the N_QUERIES most mentioned; g_raw answers
with the node's most mentioned raw surface. Drugs ranked by co-occurring notes, top K with ties, the
disease shown under the normalized name. Pairs already judged in rounds 1–2 (same disease name and drug)
reuse those labels; the rest go to outputs/llm_judge_ccks2017/batch_XX.csv for the same three judges.

  python src/llm_judge_ccks2017.py sample
  python src/llm_judge_ccks2017.py score    # → outputs/metrics/llm_judge_ccks2017.{txt,json}

Class terms (抗生素, 降压药物): the prompt does not say whether they are drug names, and judges split between
X and a real label on them. Items where at least one judge says X and another does not are re-judged by
all three with one added rule (CLASS_RULE), and the scores are recomputed with those labels:
  python src/llm_judge_ccks2017.py sample_classword   # → outputs/llm_judge_classword/batch_XX.csv
  python src/llm_judge_ccks2017.py score_classword    # → outputs/metrics/llm_judge_ccks2017_classword.{txt,json}
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import normalize
from attribution import REL
from eval_ablation import FULL
from eval_alias_public import set_drug_alias
from eval_attribution import build, leak_free_table
from eval_external_extra import PROCESSED
from eval_external_kg import DISEASE, neighbours
from eval_g_raw_vs_g_norm import collect_freq
from eval_robustness import expected_sup
from export_graph import node_id
from llm_judge_extra import JUDGES, LEVELS
from llm_judge_sample import top_candidates
from llm_judge_score import FACTS_K, boot_diff, distinct_facts
from ner_data import load_bio_docs

OUT_DIR = ROOT / "outputs" / "llm_judge_ccks2017"
CLASSWORD_DIR = ROOT / "outputs" / "llm_judge_classword"
OUT = ROOT / "outputs" / "metrics"
CLASS_RULE = ("- 药名如果是药物类别词或泛称（例如「抗生素」「降压药物」「头孢类药物」「止痛药物」「活血化瘀药物」），"
              "不要因为它不是具体药名就标 X：把它当作「这一类药」整体判断——这类药是该病的常规治疗用药标 2，"
              "是辅助、对症、并发症或预防用药标 1，与该病无关标 0。只有连是哪一类药都看不出来（例如单独的「药物」）才标 X。")
PRIOR = (ROOT / "outputs" / "llm_judge", ROOT / "outputs" / "llm_judge_extra")
SETS = (("pred", "NER 预测实体", "ccks2017_pred.bio"), ("gold", "人工标注实体", "ccks2017_gold.bio"),
        ("pred_ft", "NER 在 CCKS2017 上两折微调后的预测实体", "ccks2017_pred_ft.bio"))
N_QUERIES = 100
MIN_DRUGS = 3
BATCH = 100
KS = (1, 3, 5)


def drug_canon(docs) -> dict[str, str]:
    normalize.configure_normalization(**FULL)
    set_drug_alias(normalize.get_index(), leak_free_table())
    canon = normalize.build_canonical_map(collect_freq(docs))
    normalize.configure_normalization()
    return {s: c for (lb, s), c in canon.items() if lb == "药物"}


def prior_items() -> dict[tuple[str, str], str]:
    out = {}
    for d in (*PRIOR, OUT_DIR):
        for p in [d / "items.csv", *sorted(d.glob("batch_*.csv"))]:
            if p.exists():
                with p.open(encoding="utf-8-sig") as f:
                    out.update({(r["disease"], r["drug"]): r["id"] for r in csv.DictReader(f)})
    return out


def sample() -> None:
    table = leak_free_table()
    prior = prior_items()
    n_old = sum(i.startswith("C") for i in prior.values())
    new: dict[tuple[str, str], str] = {}
    key = {}
    for name, _, fname in SETS:
        docs = load_bio_docs(PROCESSED / fname)
        g_raw, _ = build(docs, None)
        g_norm, _ = build(docs, table)
        canon = drug_canon(docs)
        nb_raw, nb_norm = neighbours(g_raw, REL), neighbours(g_norm, REL)
        queries = []
        for d in sorted((d for d, ms in nb_norm.items() if len(ms) >= MIN_DRUGS),
                        key=lambda d: (-g_norm["mentions"][d], d)):
            raw = [node_id(n, DISEASE) for n in g_norm["names"][d] if node_id(n, DISEASE) in nb_raw]
            if not raw:
                continue
            r = max(raw, key=lambda n: (g_raw["mentions"][n], n))
            disease = d.split("::", 1)[1]
            entry = {"node": d, "raw_node": r, "disease": disease,
                     "n_surfaces": len(g_norm["names"][d]), "graphs": {}}
            for gn, g, node, nb in (("g_raw", g_raw, r, nb_raw), ("g_norm", g_norm, d, nb_norm)):
                ranked = top_candidates({m: g["co"][(node, REL, m)] for m in nb[node]})
                items = []
                for m, s in ranked:
                    drug = m.split("::", 1)[1]
                    pair = (disease, drug)
                    iid = prior.get(pair) or new.setdefault(pair, f"C{n_old + len(new):04d}")
                    items.append({"item": iid, "score": s, "drug": drug,
                                  "canon": canon.get(drug, drug) if gn == "g_raw" else drug})
                entry["graphs"][gn] = items
            queries.append(entry)
            if len(queries) == N_QUERIES:
                break
        key[name] = queries
        print(f"{name}: {len(queries)} queries", flush=True)

    rows = [{"id": iid, "disease": d, "drug": m} for (d, m), iid in new.items()]
    random.Random(2029).shuffle(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    first = len(list(OUT_DIR.glob("batch_*.csv")))
    for b in range(0, len(rows), BATCH):
        with (OUT_DIR / f"batch_{first + b // BATCH:02d}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "disease", "drug"])
            w.writeheader()
            w.writerows(rows[b:b + BATCH])
    (OUT_DIR / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    used = {x["item"] for qs in key.values() for q in qs for v in q["graphs"].values() for x in v}
    print(f"items used {len(used)}, new {len(rows)} in {(len(rows) + BATCH - 1) // BATCH} batches, "
          f"reused {len(used) - len(rows)}")


def read_labels(judge: str, classword: bool = False) -> dict[str, str]:
    out = {}
    for d in (*PRIOR, OUT_DIR, *((CLASSWORD_DIR,) if classword else ())):
        for p in sorted(d.glob(f"labels_{judge}_*.csv")):
            with p.open(encoding="utf-8-sig") as f:
                out.update({r["id"].strip(): r["label"].strip().upper() for r in csv.DictReader(f)})
    return out


def sample_classword() -> None:
    key = json.loads((OUT_DIR / "key.json").read_text(encoding="utf-8"))
    labels = {j: read_labels(j) for j, _ in JUDGES}
    text = {iid: pair for pair, iid in prior_items().items()}
    used = sorted({x["item"] for qs in key.values() for q in qs for v in q["graphs"].values() for x in v})
    disputed = [i for i in used if 0 < sum(labels[j].get(i) == "X" for j, _ in JUDGES) < len(JUDGES)]
    rows = [{"id": i, "disease": text[i][0], "drug": text[i][1]} for i in disputed]
    random.Random(2031).shuffle(rows)
    CLASSWORD_DIR.mkdir(parents=True, exist_ok=True)
    for p in CLASSWORD_DIR.glob("batch_*.csv"):
        p.unlink()
    for b in range(0, len(rows), BATCH):
        with (CLASSWORD_DIR / f"batch_{b // BATCH:02d}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "disease", "drug"])
            w.writeheader()
            w.writerows(rows[b:b + BATCH])
    (CLASSWORD_DIR / "rule.txt").write_text(CLASS_RULE + "\n", encoding="utf-8")
    print(f"items used {len(used)}, disputed X vs not-X {len(rows)} in {(len(rows) + BATCH - 1) // BATCH} batches")


def score(classword: bool = False) -> None:
    key = json.loads((OUT_DIR / "key.json").read_text(encoding="utf-8"))
    labels = {j: read_labels(j, classword) for j, _ in JUDGES}
    need = {x["item"] for qs in key.values() for q in qs for v in q["graphs"].values() for x in v}
    for j, lab in labels.items():
        bad = sorted(i for i in need if lab.get(i) not in ("0", "1", "2", "X"))
        if bad:
            sys.exit(f"{j}: {len(bad)} items unlabelled or invalid, e.g. {bad[:3]}")
    jury = {lvl: {i: sum(labels[j][i] in ok for j, _ in JUDGES) >= 2 for i in need} for lvl, ok in LEVELS.items()}

    L = ["CCKS2017 三模型盲评：原始图谱 vs 标准化图谱（与 3.5 节同一设计、同一评委和提示词）"
         + ("；评委在「是不是药名」上有分歧的题按补充的类别词规则重判" if classword else ""), "=" * 80,
         "差 = 标准化 − 原始，按疾病 bootstrap 95% 区间；不同正确药按标准化后的药名去重（原始图里同一种药的不同写法算一种）。", ""]
    res = {}
    for name, title, _ in SETS:
        qs = key[name]
        r = res[name] = {"n_queries": len(qs), "merged_queries": sum(q["n_surfaces"] > 1 for q in qs)}
        L.append(f"【{title}】{len(qs)} 个疾病（其中标准化合并了 ≥2 种写法的 {r['merged_queries']} 个）")
        for subset, st, sq in (("all", "全部", qs), ("merged", "有合并的疾病", [q for q in qs if q["n_surfaces"] > 1])):
            if not sq:
                continue
            for lvl, lt in (("strict", "严格"), ("lenient", "宽松")):
                parts = []
                for k in KS:
                    v = {}
                    for gn in ("g_raw", "g_norm"):
                        v[gn] = []
                        for q in sq:
                            edges = [(x["score"], jury[lvl][x["item"]]) for x in q["graphs"][gn]]
                            nk = min(k, len(edges))
                            v[gn].append(expected_sup(edges, nk) / nk)
                    d = [b - a for a, b in zip(v["g_raw"], v["g_norm"])]
                    cell = {"g_raw": sum(v["g_raw"]) / len(sq), "g_norm": sum(v["g_norm"]) / len(sq),
                            "delta": sum(d) / len(d), "ci95": boot_diff(d, k)}
                    r[f"{subset}_{lvl}_P@{k}"] = cell
                    parts.append(f"P@{k} {cell['g_raw']:.3f}→{cell['g_norm']:.3f}（{cell['delta']:+.3f} "
                                 f"[{cell['ci95'][0]:+.3f}, {cell['ci95'][1]:+.3f}]）")
                rngs = (random.Random(7), random.Random(8))
                f = {gn: [distinct_facts([(x["score"], jury[lvl][x["item"]], x["canon"]) for x in q["graphs"][gn]],
                                         FACTS_K, rng) for q in sq]
                     for gn, rng in (("g_raw", rngs[0]), ("g_norm", rngs[1]))}
                dist = {gn: [distinct_facts([(x["score"], True, x["canon"]) for x in q["graphs"][gn]], FACTS_K, rng)
                             for q in sq] for gn, rng in (("g_raw", random.Random(9)), ("g_norm", random.Random(10)))}
                d = [b - a for a, b in zip(f["g_raw"], f["g_norm"])]
                cell = {"g_raw": sum(f["g_raw"]) / len(sq), "g_norm": sum(f["g_norm"]) / len(sq),
                        "delta": sum(d) / len(d), "ci95": boot_diff(d, 99),
                        "distinct_raw": sum(dist["g_raw"]) / len(sq), "distinct_norm": sum(dist["g_norm"]) / len(sq)}
                r[f"{subset}_{lvl}_distinct@{FACTS_K}"] = cell
                parts.append(f"前{FACTS_K}名不同正确药 {cell['g_raw']:.2f}→{cell['g_norm']:.2f}（{cell['delta']:+.2f} "
                             f"[{cell['ci95'][0]:+.2f}, {cell['ci95'][1]:+.2f}]；不同药 {cell['distinct_raw']:.2f}→{cell['distinct_norm']:.2f}）")
                L.append(f"  {st}｜{lt}：" + "  ".join(parts))
        xr = {gn: sum(sum(labels[j][x["item"]] == "X" for j, _ in JUDGES) >= 2 for q in qs for x in q["graphs"][gn])
              / max(1, sum(len(q["graphs"][gn]) for q in qs)) for gn in ("g_raw", "g_norm")}
        r["x_rate"] = xr
        L.append(f"  不是药名或残缺：原始 {xr['g_raw']:.1%}，标准化 {xr['g_norm']:.1%}")
        L.append("  单个评委各自判（「抗生素」「降压药物」这类药物类别词，评委之间标 X 还是 2 不一致）：")
        for j, jt in JUDGES:
            parts = []
            for lvl, lt in (("strict", "严格"), ("lenient", "宽松")):
                ok = LEVELS[lvl]
                p5, fx = {}, {}
                for gn, seed in (("g_raw", 7), ("g_norm", 8)):
                    rng = random.Random(seed)
                    p5[gn], fx[gn] = [], []
                    for q in qs:
                        edges = [(x["score"], labels[j][x["item"]] in ok, x["canon"]) for x in q["graphs"][gn]]
                        nk = min(5, len(edges))
                        p5[gn].append(expected_sup([e[:2] for e in edges], nk) / nk)
                        fx[gn].append(distinct_facts(edges, FACTS_K, rng))
                dp = [b - a for a, b in zip(p5["g_raw"], p5["g_norm"])]
                df = [b - a for a, b in zip(fx["g_raw"], fx["g_norm"])]
                cell = {"P@5": [sum(p5["g_raw"]) / len(qs), sum(p5["g_norm"]) / len(qs), sum(dp) / len(dp), boot_diff(dp, 5)],
                        "distinct": [sum(fx["g_raw"]) / len(qs), sum(fx["g_norm"]) / len(qs), sum(df) / len(df), boot_diff(df, 99)]}
                r[f"judge_{j}_{lvl}"] = cell
                a, c = cell["P@5"], cell["distinct"]
                parts.append(f"{lt} P@5 {a[2]:+.3f} [{a[3][0]:+.3f}, {a[3][1]:+.3f}]，不同正确药 {c[0]:.2f}→{c[1]:.2f}"
                             f"（{c[2]:+.2f} [{c[3][0]:+.2f}, {c[3][1]:+.2f}]）")
            L.append(f"    {jt}：" + "；".join(parts))
        L.append("")
    text = "\n".join(L) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    name = "llm_judge_ccks2017" + ("_classword" if classword else "")
    (OUT / f"{name}.txt").write_text(text, encoding="utf-8")
    (OUT / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    {"sample": sample, "score": score, "sample_classword": sample_classword,
     "score_classword": lambda: score(True)}[sys.argv[1] if len(sys.argv) > 1 else "sample"]()
