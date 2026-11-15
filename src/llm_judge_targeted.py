"""Third jury round, targeted at the queries where LLM disease normalization changes the answer.

In rounds 1–2 (src/llm_judge_sample.py, src/llm_judge_extra.py) the 100 queries are the most frequent
G_norm diseases, mostly linked by exact ICD match, so the LLM graphs return the same top 5 for ~80 of them
and the paired difference is 0 there by construction. Here the queries are drawn where it is not.

Query = a predicted disease surface (≥ MIN_MENTIONS mentions) on the CCKS2019 test notes or CCKS2017,
i.e. what a user would type; every such query whose answer changes is used, not a sample.
In each graph the query resolves to the disease node whose names contain the surface (most mentioned one,
same rule as llm_judge_extra). Graphs (predicted entities, leak-free drug table, drugs ranked by
co-occurring notes, top K with ties):
  g_norm         full system
  g_llm_fb       full system + LLM fallback for unlinked disease mentions
  g_llm_primary  LLM first
A query is kept if g_llm_fb or g_llm_primary gives a different top-K drug set than g_norm and every graph
has at least one drug for it; queries resolving to the same node triple are deduplicated (most frequent
surface kept). The disease is shown to the judges under the query surface, the same for all graphs.

Repeat items (new ids, same text as round 1): REPEAT_CALIB round-1 calibration items per kind and
REPEAT_PLAIN round-1 query items, for calibration and test–retest agreement with round 1.

  python src/llm_judge_targeted.py sample   # → outputs/llm_judge_targeted/{batch_XX.csv, key.json}
  python src/llm_judge_targeted.py score    # → outputs/metrics/llm_judge_targeted.{txt,json}
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

from attribution import REL
from eval_external_extra import PROCESSED
from eval_g_raw_vs_g_norm import collect_freq
from eval_robustness import expected_sup
from llm_judge_extra import DISEASE, JUDGES, LEVELS, build_graph, ranked
from llm_judge_score import FACTS_K, boot_diff, distinct_facts, kappa
from ner_data import load_bio_docs
from normalize import configure_normalization, lookup_with_source
from eval_ablation import FULL

ROUND1 = ROOT / "outputs" / "llm_judge"
OUT_DIR = ROOT / "outputs" / "llm_judge_targeted"
OUT = ROOT / "outputs" / "metrics"
GRAPHS = (("g_norm", "off"), ("g_llm_fb", "fallback"), ("g_llm_primary", "primary"))
SETS = (("CCKS2019 test", "test_pred.bio"), ("CCKS2017", "ccks2017_pred.bio"))
MIN_MENTIONS = 1
REPEAT_CALIB = 30
REPEAT_PLAIN = 40
BATCH = 100
KS = (1, 3, 5)


def resolve(g, surface: str) -> str | None:
    nodes = [nid for nid in g["names"] if nid.startswith(DISEASE + "::") and surface in g["names"][nid]]
    return max(nodes, key=lambda x: (g["mentions"][x], x)) if nodes else None


def dataset_queries(label: str, fname: str) -> list[dict]:
    docs = load_bio_docs(PROCESSED / fname)
    graphs = {name: build_graph(docs, llm) for name, llm in GRAPHS}
    freq = collect_freq(docs)[DISEASE]
    surfaces = sorted((s for s, c in freq.items() if c >= MIN_MENTIONS), key=lambda s: (-freq[s], s))
    configure_normalization(**FULL)
    source = {s: lookup_with_source(s, DISEASE)[1] for s in surfaces}
    configure_normalization()

    seen, queries = set(), []
    for s in surfaces:
        nodes = {name: resolve(g, s) for name, g in graphs.items()}
        if None in nodes.values():
            continue
        lists = {name: ranked(graphs[name], nodes[name], lambda d, m, g=graphs[name]: g["co"][(d, REL, m)])
                 for name in graphs}
        if not all(lists.values()):
            continue
        sets = {name: {m for m, _ in v} for name, v in lists.items()}
        changed = {name: sets[name] != sets["g_norm"] for name in ("g_llm_fb", "g_llm_primary")}
        triple = tuple(nodes[n] for n, _ in GRAPHS)
        if not any(changed.values()) or triple in seen:
            continue
        seen.add(triple)
        queries.append({"dataset": label, "surface": s, "mentions": freq[s], "nodes": nodes, "changed": changed,
                        "unlinked_in_ours": source[s] == "heuristic",
                        "lists": {n: [(m.split("::", 1)[1], sc) for m, sc in v] for n, v in lists.items()}})
    return queries


def sample() -> None:
    queries = [q for label, fname in SETS for q in dataset_queries(label, fname)]
    items: dict[tuple[str, str], str] = {}

    def item(d: str, m: str) -> str:
        return items.setdefault((d, m), f"T{len(items):04d}")

    for q in queries:
        q["lists"] = {n: [{"item": item(q["surface"], m), "score": sc} for m, sc in v] for n, v in q["lists"].items()}

    r1 = list(csv.DictReader((ROUND1 / "items.csv").open(encoding="utf-8-sig")))
    r1_text = {r["id"]: (r["disease"], r["drug"]) for r in r1}
    key1 = json.loads((ROUND1 / "key.json").read_text(encoding="utf-8"))
    rng = random.Random(2028)
    calib_ids = {c["item"] for c in key1["calibration"]}
    repeats = []
    for kind in ("pos", "neg"):
        pool = sorted({c["item"] for c in key1["calibration"] if c["kind"] == kind})
        repeats += [(i, kind) for i in rng.sample(pool, REPEAT_CALIB)]
    plain = sorted({x["item"] for q in key1["queries"] for x in q["graphs"]["g_norm"]} - calib_ids)
    repeats += [(i, "plain") for i in rng.sample(plain, REPEAT_PLAIN)]
    repeat_key = []
    for old, kind in repeats:
        d, m = r1_text[old]
        if (d, m) in items:
            continue
        repeat_key.append({"item": item(d, m), "round1": old, "kind": kind})

    rows = [{"id": iid, "disease": d, "drug": m} for (d, m), iid in items.items()]
    rng.shuffle(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.glob("batch_*.csv"):
        p.unlink()
    for b in range(0, len(rows), BATCH):
        with (OUT_DIR / f"batch_{b // BATCH:02d}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "disease", "drug"])
            w.writeheader()
            w.writerows(rows[b:b + BATCH])
    (OUT_DIR / "key.json").write_text(json.dumps({"queries": queries, "repeats": repeat_key},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"queries {len(queries)} (changed fb {sum(q['changed']['g_llm_fb'] for q in queries)}, "
          f"primary {sum(q['changed']['g_llm_primary'] for q in queries)}, "
          f"unlinked in ours {sum(q['unlinked_in_ours'] for q in queries)}); items {len(rows)} in "
          f"{(len(rows) + BATCH - 1) // BATCH} batches, of which repeats {len(repeat_key)}")


def read_labels(judge: str, d: Path) -> dict[str, str]:
    out = {}
    for p in sorted(d.glob(f"labels_{judge}_*.csv")):
        with p.open(encoding="utf-8-sig") as f:
            out.update({r["id"].strip(): r["label"].strip().upper() for r in csv.DictReader(f)})
    return out


def compare(qs: list[dict], other: str, jury: dict, drug: dict) -> dict:
    res = {"n": len(qs)}
    for lvl in LEVELS:
        for k in KS:
            diffs, base, oth = [], [], []
            for q in qs:
                v = []
                for gn in ("g_norm", other):
                    edges = [(x["score"], jury[lvl][x["item"]]) for x in q["lists"][gn]]
                    nk = min(k, len(edges))
                    v.append(expected_sup(edges, nk) / nk)
                base.append(v[0])
                oth.append(v[1])
                diffs.append(v[1] - v[0])
            res[f"{lvl}_P@{k}"] = {"g_norm": sum(base) / len(qs), other: sum(oth) / len(qs),
                                   "delta": sum(diffs) / len(qs), "ci95": boot_diff(diffs, k)}
        rngs = (random.Random(7), random.Random(8))
        vals = {gn: [distinct_facts([(x["score"], jury[lvl][x["item"]], drug[x["item"]]) for x in q["lists"][gn]],
                                    FACTS_K, rng) for q in qs]
                for gn, rng in (("g_norm", rngs[0]), (other, rngs[1]))}
        diffs = [b - a for a, b in zip(vals["g_norm"], vals[other])]
        res[f"{lvl}_distinct@{FACTS_K}"] = {"g_norm": sum(vals["g_norm"]) / len(qs), other: sum(vals[other]) / len(qs),
                                            "delta": sum(diffs) / len(qs), "ci95": boot_diff(diffs, 99)}
    res["list_len"] = {gn: sum(len(q["lists"][gn]) for q in qs) / len(qs) for gn in ("g_norm", other)}
    return res


def score() -> None:
    key = json.loads((OUT_DIR / "key.json").read_text(encoding="utf-8"))
    labels = {j: read_labels(j, OUT_DIR) for j, _ in JUDGES}
    r1_labels = {j: read_labels(j, ROUND1) for j, _ in JUDGES}
    drug = {}
    for p in sorted(OUT_DIR.glob("batch_*.csv")):
        with p.open(encoding="utf-8-sig") as f:
            drug.update({r["id"].strip(): r["drug"] for r in csv.DictReader(f)})
    for j, lab in labels.items():
        bad = sorted(i for i in drug if lab.get(i) not in ("0", "1", "2", "X"))
        if bad:
            sys.exit(f"{j}: {len(bad)} items unlabelled or invalid, e.g. {bad[:3]}")
    jury = {lvl: {i: sum(labels[j][i] in ok for j, _ in JUDGES) >= 2 for i in drug} for lvl, ok in LEVELS.items()}

    res = {"n_items": len(drug), "label_counts": {j: dict(Counter(labels[j].values())) for j, _ in JUDGES}}
    L = ["第三轮盲评：只看大模型标准化改变了答案的查询", "=" * 80,
         "查询 = CCKS2019 测试集和 CCKS2017 上预测出的疾病写法，三张图各取包含该写法的节点；"
         "大模型兜底或优先的前 5 名药与本方法不同的查询全部使用。",
         "疾病按写法原样出示，三张图相同；评委、提示词与 3.5 节相同。差值按查询 bootstrap 95% 区间。",
         f"题目 {len(drug)} 条，其中重复题 {len(key['repeats'])} 条（第一轮原题换编号，用于校准和前后一致性）。", ""]

    rep = key["repeats"]
    L.append("一、校准与前后一致性（重复题）")
    for lvl, lt in (("strict", "严格"), ("lenient", "宽松")):
        pos = [jury[lvl][r["item"]] for r in rep if r["kind"] == "pos"]
        neg = [not jury[lvl][r["item"]] for r in rep if r["kind"] == "neg"]
        res[f"calib_{lvl}"] = {"sensitivity": sum(pos) / len(pos), "specificity": sum(neg) / len(neg)}
        L.append(f"  {lt}：外部库列出的药 判对 {sum(pos)}/{len(pos)}（灵敏度 {sum(pos) / len(pos):.2f}），"
                 f"别的章节的药 判否 {sum(neg)}/{len(neg)}（特异度 {sum(neg) / len(neg):.2f}）")
    for j, jn in JUDGES:
        a = [r1_labels[j][r["round1"]] for r in rep]
        b = [labels[j][r["item"]] for r in rep]
        res[f"retest_{j}"] = {"agree": sum(x == y for x, y in zip(a, b)) / len(a), "kappa4": kappa(a, b)}
        L.append(f"  {jn} 与自己第一轮的答案：一致 {res[f'retest_{j}']['agree']:.2f}，四类 kappa {res[f'retest_{j}']['kappa4']:.2f}")
    L.append("")

    qs_all = key["queries"]
    L.append(f"二、与本方法逐查询配对（共 {len(qs_all)} 个查询）")
    for other, title in (("g_llm_fb", "大模型兜底"), ("g_llm_primary", "大模型优先")):
        changed = [q for q in qs_all if q["changed"][other]]
        subsets = [("changed", "答案不同的全部查询", changed)]
        subsets += [(f"changed_{label}", label, [q for q in changed if q["dataset"] == label]) for label, _ in SETS]
        subsets += [("changed_unlinked", "本方法原样保留的写法", [q for q in changed if q["unlinked_in_ours"]]),
                    ("changed_linked", "本方法已链接的写法", [q for q in changed if not q["unlinked_in_ours"]])]
        for subset, st, qs in subsets:
            if not qs:
                continue
            c = compare(qs, other, jury, drug)
            res[f"{other}_{subset}"] = c
            L.append(f"  【{title}｜{st}，{c['n']} 个；平均候选药 本方法 {c['list_len']['g_norm']:.1f} / 大模型 {c['list_len'][other]:.1f}】")
            for lvl, lt in (("strict", "严格"), ("lenient", "宽松")):
                cells = [c[f"{lvl}_P@{k}"] for k in KS] + [c[f"{lvl}_distinct@{FACTS_K}"]]
                names = [f"P@{k}" for k in KS] + [f"前{FACTS_K}名不同正确药"]
                L.append(f"    {lt}：" + "  ".join(
                    f"{n} {x['g_norm']:.3f}→{x[other]:.3f}（{x['delta']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]）"
                    for n, x in zip(names, cells)))
    x_new = sum(sum(labels[j][i] == "X" for j, _ in JUDGES) >= 2 for i in drug)
    L.append("")
    L.append(f"多数票判为「不是药名」：{x_new}/{len(drug)}")
    text = "\n".join(L) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "llm_judge_targeted.txt").write_text(text, encoding="utf-8")
    (OUT / "llm_judge_targeted.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "sample"]()
