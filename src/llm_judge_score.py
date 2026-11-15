"""Score the LLM jury on the blind items from src/llm_judge_sample.py (improvement 4).

Labels per judge in outputs/llm_judge/labels_<judge>_XX.csv: 2 treats the disease, 1 related (supportive /
complication), 0 unrelated, X not a drug. strict = label 2; lenient = label 1 or 2. The jury label is the
majority (≥ 2 of 3) of the judges' binary calls.

1. Calibration: sensitivity on KB-listed pairs, specificity on cross-chapter KB pairs (a lower bound).
2. Agreement: pairwise Cohen's kappa and Fleiss' kappa.
3. Precision of the answers: P@k per query (exact expectation over random tie-breaks), per graph, for each
   judge and the jury; differences bootstrapped over queries. Plus distinct correct drugs in the top 5, raw drug
   names mapped to their normalized drug so that spellings of one drug count once.
4. Jury vs the external KBs on graph items of KB-covered queries: how many KB-unsupported edges the jury accepts.
Writes outputs/metrics/llm_judge.{txt,json}.
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

from eval_robustness import boot_ci, expected_sup

JUDGE_DIR = ROOT / "outputs" / "llm_judge"
OUT = ROOT / "outputs" / "metrics"
JUDGES = (("opus", "Claude Opus"), ("gpt", "GPT"), ("gemini", "Gemini"))
GRAPHS = (("g_raw", "原始图谱"), ("g_norm", "标准化"), ("g_norm_em", "标准化+药物归因"))
PAIRS = (("g_norm", "g_raw"), ("g_norm_em", "g_norm"), ("g_norm_em", "g_raw"))
LEVELS = {"strict": {"2"}, "lenient": {"1", "2"}}
KS = (1, 3, 5)
FACTS_K = 5
N_TIE = 200
N_BOOT = 2000


def read_labels(judge: str) -> dict[str, str]:
    out = {}
    for p in sorted(JUDGE_DIR.glob(f"labels_{judge}_*.csv")):
        with p.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                out[r["id"].strip()] = r["label"].strip().upper()
    return out


def kappa(a: list, b: list) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss(rows: list[list]) -> float:
    cats = sorted({x for r in rows for x in r}, key=str)
    n, m = len(rows), len(rows[0])
    p_j = {c: sum(r.count(c) for r in rows) / (n * m) for c in cats}
    p_i = [(sum(r.count(c) ** 2 for c in cats) - m) / (m * (m - 1)) for r in rows]
    pbar, pe = sum(p_i) / n, sum(v * v for v in p_j.values())
    return (pbar - pe) / (1 - pe) if pe < 1 else 1.0


def boot_diff(diffs: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    return boot_ci([sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(N_BOOT)])


def canonical_drug_of() -> dict[str, str]:
    """Raw drug surface on the test notes → its normalized drug (leak-free table), to count distinct drugs."""
    import normalize
    from eval_ablation import FULL
    from eval_alias_public import set_drug_alias
    from eval_attribution import leak_free_table
    from eval_external_extra import PROCESSED
    from eval_g_raw_vs_g_norm import collect_freq
    from ner_data import load_bio_docs

    docs = load_bio_docs(PROCESSED / "test_pred.bio")
    normalize.configure_normalization(**FULL)
    set_drug_alias(normalize.get_index(), leak_free_table())
    canon = normalize.build_canonical_map(collect_freq(docs))
    normalize.configure_normalization()
    return {s: c for (lb, s), c in canon.items() if lb == "药物"}


def distinct_facts(edges: list[tuple[float, bool, str]], k: int, rng: random.Random) -> float:
    """Expected number of distinct correct drugs among the top k, ties broken at random."""
    total = 0.0
    for _ in range(N_TIE):
        order = sorted(edges, key=lambda e: (-e[0], rng.random()))[:k]
        total += len({d for _, ok, d in order if ok})
    return total / N_TIE


def main() -> None:
    key = json.loads((JUDGE_DIR / "key.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((JUDGE_DIR / "items.csv").open(encoding="utf-8-sig")))
    items = [r["id"] for r in rows]
    drug_name = {r["id"]: r["drug"] for r in rows}
    labels = {j: read_labels(j) for j, _ in JUDGES}
    for j, lab in labels.items():
        missing = [i for i in items if i not in lab]
        bad = [i for i in items if lab.get(i) not in (None, "0", "1", "2", "X")]
        if missing or bad:
            sys.exit(f"{j}: {len(missing)} items unlabelled (e.g. {missing[:3]}), {len(bad)} invalid (e.g. {bad[:3]})")

    calls = {lvl: {j: {i: labels[j][i] in ok for i in items} for j, _ in JUDGES} for lvl, ok in LEVELS.items()}
    for lvl in LEVELS:
        calls[lvl]["jury"] = {i: sum(calls[lvl][j][i] for j, _ in JUDGES) >= 2 for i in items}
    raters = [j for j, _ in JUDGES] + ["jury"]
    res = {"n_items": len(items), "n_queries": len(key["queries"]),
           "label_counts": {j: dict(Counter(labels[j].values())) for j, _ in JUDGES}}

    calib = key["calibration"]
    res["calibration"] = {}
    for lvl in LEVELS:
        for r in raters:
            pos = [calls[lvl][r][c["item"]] for c in calib if c["kind"] == "pos"]
            neg = [not calls[lvl][r][c["item"]] for c in calib if c["kind"] == "neg"]
            res["calibration"][f"{lvl}_{r}"] = {"sensitivity": sum(pos) / len(pos), "specificity": sum(neg) / len(neg),
                                                "n_pos": len(pos), "n_neg": len(neg)}

    res["agreement"] = {"pairwise": {}}
    for a in range(len(JUDGES)):
        for b in range(a + 1, len(JUDGES)):
            ja, jb = JUDGES[a][0], JUDGES[b][0]
            res["agreement"]["pairwise"][f"{ja}-{jb}"] = {
                "kappa4": kappa([labels[ja][i] for i in items], [labels[jb][i] for i in items]),
                "kappa_strict": kappa([calls["strict"][ja][i] for i in items], [calls["strict"][jb][i] for i in items])}
    res["agreement"]["fleiss4"] = fleiss([[labels[j][i] for j, _ in JUDGES] for i in items])
    res["agreement"]["fleiss_strict"] = fleiss([[calls["strict"][j][i] for j, _ in JUDGES] for i in items])
    res["agreement"]["fleiss_lenient"] = fleiss([[calls["lenient"][j][i] for j, _ in JUDGES] for i in items])

    res["precision"] = {}
    for subset, qs in (("all", key["queries"]), ("kb_covered", [q for q in key["queries"] if q["kb_covered"]]),
                       ("kb_uncovered", [q for q in key["queries"] if not q["kb_covered"]])):
        for lvl in LEVELS:
            for r in raters:
                per_q = []
                for q in qs:
                    row = {}
                    for gn, _ in GRAPHS:
                        edges = [(x["score"], calls[lvl][r][x["item"]]) for x in q["graphs"][gn]]
                        for k in KS:
                            n = min(k, len(edges))
                            row[(gn, k)] = expected_sup(edges, n) / n
                    per_q.append(row)
                for k in KS:
                    cell = {gn: sum(x[(gn, k)] for x in per_q) / len(per_q) for gn, _ in GRAPHS}
                    for a, b in PAIRS:
                        diffs = [x[(a, k)] - x[(b, k)] for x in per_q]
                        cell[f"{a}-{b}"] = {"delta": sum(diffs) / len(diffs), "ci95": boot_diff(diffs, k)}
                    res["precision"][f"{subset}_{lvl}_{r}_P@{k}"] = cell
        res["precision"][f"{subset}_n"] = len(qs)

    to_canon = canonical_drug_of()
    rng = random.Random(7)
    res["facts"] = {}
    for lvl in LEVELS:
        per_q = []
        for q in key["queries"]:
            row = {}
            for gn, _ in GRAPHS:
                edges = [(x["score"], calls[lvl]["jury"][x["item"]],
                          to_canon.get(drug_name[x["item"]], drug_name[x["item"]]) if gn == "g_raw" else drug_name[x["item"]])
                         for x in q["graphs"][gn]]
                row[gn] = distinct_facts(edges, FACTS_K, rng)
                row[gn + "_distinct"] = distinct_facts([(s, True, d) for s, _, d in edges], FACTS_K, rng)
            per_q.append(row)
        cell = {gn: sum(x[gn] for x in per_q) / len(per_q) for gn, _ in GRAPHS}
        cell.update({gn + "_distinct": sum(x[gn + "_distinct"] for x in per_q) / len(per_q) for gn, _ in GRAPHS})
        for a, b in PAIRS:
            diffs = [x[a] - x[b] for x in per_q]
            cell[f"{a}-{b}"] = {"delta": sum(diffs) / len(diffs), "ci95": boot_diff(diffs, 99)}
        res["facts"][lvl] = cell

    x_rate = {}
    for gn, _ in GRAPHS:
        ids = [x["item"] for q in key["queries"] for x in q["graphs"][gn]]
        x_rate[gn] = sum(sum(labels[j][i] == "X" for j, _ in JUDGES) >= 2 for i in ids) / len(ids)
    res["not_a_drug_rate"] = x_rate

    seen, kb_rows = set(), []
    for q in key["queries"]:
        if q["kb_covered"]:
            for gn, _ in GRAPHS:
                for x in q["graphs"][gn]:
                    if x["item"] not in seen:
                        seen.add(x["item"])
                        kb_rows.append((x["kb"], calls["strict"]["jury"][x["item"]], calls["lenient"]["jury"][x["item"]]))
    sup = [r for r in kb_rows if r[0]]
    uns = [r for r in kb_rows if not r[0]]
    res["vs_kb"] = {"n_supported": len(sup), "n_unsupported": len(uns),
                    "jury_strict_on_supported": sum(r[1] for r in sup) / max(len(sup), 1),
                    "jury_strict_on_unsupported": sum(r[1] for r in uns) / max(len(uns), 1),
                    "jury_lenient_on_unsupported": sum(r[2] for r in uns) / max(len(uns), 1),
                    "kb_share_of_jury_strict": sum(r[0] and r[1] for r in kb_rows) / max(sum(r[1] for r in kb_rows), 1)}

    text = report(res)
    (OUT / "llm_judge.txt").write_text(text, encoding="utf-8")
    (OUT / "llm_judge.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


def ci(c) -> str:
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def report(res) -> str:
    names = dict(JUDGES) | {"jury": "三模型多数票"}
    gnames = dict(GRAPHS)
    L = [f"改进四：大模型多评委盲评（测试集预测实体，{res['n_queries']} 个高频疾病，{res['n_items']} 条去重后的疾病-药物对）",
         "严格 = 只算「治疗用药」；宽松 = 「治疗用药」+「相关辅助用药」；评委看不到来自哪张图", "=" * 80,
         "一、标签分布：" + "；".join(f"{names[j]} " + " ".join(f"{k}={v}" for k, v in sorted(c.items()))
                               for j, c in res["label_counts"].items()),
         "", "二、评委校准（外部知识库列出的药 = 应判是；别的章节疾病的药 = 应判否，后者是下界）"]
    for lvl, lt in (("strict", "严格"), ("lenient", "宽松")):
        for r in names:
            c = res["calibration"][f"{lvl}_{r}"]
            L.append(f"  {lt} {names[r]}：灵敏度 {c['sensitivity']:.2f}（{c['n_pos']} 条）  特异度 {c['specificity']:.2f}（{c['n_neg']} 条）")
    ag = res["agreement"]
    L += ["", "三、评委之间的一致性（kappa：0.4–0.6 中等，0.6–0.8 较好，>0.8 很好）",
          "  " + "  ".join(f"{k}：四类 {v['kappa4']:.2f}，是否治疗 {v['kappa_strict']:.2f}" for k, v in ag["pairwise"].items()),
          f"  三人 Fleiss kappa：四类 {ag['fleiss4']:.2f}，严格 {ag['fleiss_strict']:.2f}，宽松 {ag['fleiss_lenient']:.2f}",
          "", "四、各图谱回答的准确率（P@k = 每个疾病排前 k 的药里有多少是对的；差值按疾病 bootstrap 95% 区间）"]
    for subset, st in (("all", "全部疾病"), ("kb_covered", "外部库覆盖的疾病"), ("kb_uncovered", "外部库没覆盖的疾病")):
        L.append(f"  [{st}，{res['precision'][f'{subset}_n']} 个]")
        for lvl, lt in (("strict", "严格"), ("lenient", "宽松")):
            for k in KS:
                c = res["precision"][f"{subset}_{lvl}_jury_P@{k}"]
                L.append(f"    {lt} P@{k}  " + "  ".join(f"{gnames[g]} {c[g]:.3f}" for g, _ in GRAPHS) + "  |  " +
                         "  ".join(f"{gnames[a]}−{gnames[b]} {c[f'{a}-{b}']['delta']:+.3f}{ci(c[f'{a}-{b}']['ci95'])}" for a, b in PAIRS))
    L += ["", "  单个评委（全部疾病，严格 P@3）："]
    for r in names:
        c = res["precision"][f"all_strict_{r}_P@3"]
        L.append(f"    {names[r]}：" + "  ".join(f"{gnames[g]} {c[g]:.3f}" for g, _ in GRAPHS) + "  " +
                 "  ".join(f"{gnames[a]}−{gnames[b]} {c[f'{a}-{b}']['delta']:+.3f}{ci(c[f'{a}-{b}']['ci95'])}" for a, b in PAIRS))
    L += ["", f"  排前 {FACTS_K} 的药里，不同的药有几种、其中正确的有几种（同一种药的不同写法只算一次）："]
    for lvl, lt in (("strict", "严格"), ("lenient", "宽松")):
        c = res["facts"][lvl]
        L.append(f"    {lt}：" + "  ".join(f"{gnames[g]} 不同药 {c[g + '_distinct']:.2f} 种 / 正确 {c[g]:.2f} 种" for g, _ in GRAPHS)
                 + "  |  " + "  ".join(f"{gnames[a]}−{gnames[b]} {c[f'{a}-{b}']['delta']:+.2f}{ci(c[f'{a}-{b}']['ci95'])}" for a, b in PAIRS))
    L += ["", "五、回答里“不是药名/名字残缺”的比例（多数票判 X）：" +
          "  ".join(f"{gnames[g]} {v:.1%}" for g, v in res["not_a_drug_rate"].items())]
    v = res["vs_kb"]
    L += ["", f"六、评委 vs 外部知识库（外部库覆盖的疾病，{v['n_supported'] + v['n_unsupported']} 条边）",
          f"  外部库证实的 {v['n_supported']} 条里，评委多数票判为治疗用药 {v['jury_strict_on_supported']:.0%}",
          f"  外部库没证实的 {v['n_unsupported']} 条里，评委判为治疗用药 {v['jury_strict_on_unsupported']:.0%}，判为治疗或相关 {v['jury_lenient_on_unsupported']:.0%}",
          f"  评委认可的治疗用药里，只有 {v['kb_share_of_jury_strict']:.0%} 被外部库收录 → 外部库印证率是真实准确率的明显低估"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
