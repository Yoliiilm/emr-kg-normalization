"""Second jury round on the 100 queries of src/llm_judge_sample.py (same judges, same prompt).

New answer lists, each compared with g_norm (count ranking, full system) per query:
  edge scoring on g_norm (refs: Rotmensch 2017, Finlayson 2014, statistically validated networks)
    lift          co·N / (df_d · df_m), i.e. relative risk-like association
    pmi_disc      PMI with the Pantel–Lin discount (eval_external_kg.scorers)
    hypergeom     −log10 one-sided hypergeometric p of the co-occurrence count
  LLM disease normalization (src/llm_norm_score.py, outputs/llm_norm/map.json)
    g_llm_fb      full system + LLM fallback for unlinked disease mentions
    g_llm_primary LLM first
The query of an LLM graph is the node that contains the query's raw surface (what a user would type).

  python src/llm_judge_extra.py sample   # items not judged in round 1 → outputs/llm_judge_extra/batch_XX.csv
  python src/llm_judge_extra.py score    # labels of both rounds → outputs/metrics/llm_judge_extra.{txt,json}
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import normalize
from attribution import REL, doc_level_graph_with_pairs
from eval_ablation import FULL
from eval_alias_public import set_drug_alias
from eval_attribution import leak_free_table
from eval_external_extra import PROCESSED
from eval_external_kg import scorers
from eval_g_raw_vs_g_norm import collect_freq
from eval_robustness import boot_ci, expected_sup
from export_graph import node_id
from llm_judge_sample import top_candidates
from llm_judge_score import FACTS_K, boot_diff, distinct_facts
from ner_data import load_bio_docs
from normalize import build_canonical_map, configure_normalization

ROUND1 = ROOT / "outputs" / "llm_judge"
OUT_DIR = ROOT / "outputs" / "llm_judge_extra"
OUT = ROOT / "outputs" / "metrics"
JUDGES = (("opus", "Claude Opus"), ("gpt", "GPT"), ("gemini", "Gemini"))
DISEASE = "疾病和诊断"
BATCH = 100
LISTS = (("lift", "lift（相对风险式关联）"), ("pmi_disc", "折扣 PMI"), ("hypergeom", "超几何检验显著性"),
         ("g_llm_fb", "大模型兜底标准化"), ("g_llm_primary", "大模型优先标准化"))
LEVELS = {"strict": {"2"}, "lenient": {"1", "2"}}
KS = (1, 3, 5)
N_BOOT = 2000


def log_hypergeom_sf(k: int, n: int, K: int, N: int) -> float:
    """log P(X ≥ k), X ~ Hypergeom(N population, K successes, n draws)."""
    def logc(a, b):
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)
    terms = [logc(K, i) + logc(N - K, n - i) - logc(N, n) for i in range(k, min(n, K) + 1)]
    top = max(terms)
    return top + math.log(sum(math.exp(t - top) for t in terms))


def build_graph(docs, llm: str = "off"):
    configure_normalization(**{**FULL, "llm_cdn": llm})
    set_drug_alias(normalize.get_index(), leak_free_table())
    g, _ = doc_level_graph_with_pairs(docs, build_canonical_map(collect_freq(docs)))
    configure_normalization()
    return g


def ranked(g, node: str, score) -> list[tuple[str, float]]:
    sc = {m: score(node, m) for (d, r, m) in g["co"] if d == node and r == REL}
    return top_candidates(sc)


def answer_lists(key: dict) -> dict:
    docs = load_bio_docs(PROCESSED / "test_pred.bio")
    g_norm = build_graph(docs)
    sc = scorers(g_norm, REL)
    n = g_norm["n_docs"]

    def hypergeom(d, m):
        return -log_hypergeom_sf(g_norm["co"][(d, REL, m)], g_norm["df"][d], g_norm["df"][m], n) / math.log(10)

    score_fns = {"lift": sc["lift"], "pmi_disc": sc["pmi_discounted"], "hypergeom": hypergeom}
    llm_graphs = {"g_llm_fb": build_graph(docs, "fallback"), "g_llm_primary": build_graph(docs, "primary")}

    out = {}
    for q in key["queries"]:
        lists = {name: ranked(g_norm, q["node"], fn) for name, fn in score_fns.items()}
        surface = q["raw_node"].split("::", 1)[1]
        for name, g in llm_graphs.items():
            nodes = [nid for nid in g["names"] if nid.startswith(DISEASE + "::") and surface in g["names"][nid]]
            node = max(nodes, key=lambda x: (g["mentions"][x], x)) if nodes else None
            lists[name] = ranked(g, node, lambda d, m, g=g: g["co"][(d, REL, m)]) if node else []
            lists[name + "_node"] = node
        out[q["node"]] = lists
    return out


def round1_items() -> dict[tuple[str, str], str]:
    rows = csv.DictReader((ROUND1 / "items.csv").open(encoding="utf-8-sig"))
    return {(r["disease"], r["drug"]): r["id"] for r in rows}


def sample() -> None:
    key = json.loads((ROUND1 / "key.json").read_text(encoding="utf-8"))
    lists = answer_lists(key)
    old = round1_items()
    new: dict[tuple[str, str], str] = {}
    out_queries = []
    for q in key["queries"]:
        entry = {"node": q["node"], "disease": q["disease"], "lists": {}}
        for name, _ in LISTS:
            items = []
            for m, s in lists[q["node"]][name]:
                pair = (q["disease"], m.split("::", 1)[1])
                iid = old.get(pair) or new.setdefault(pair, f"R{len(new):04d}")
                items.append({"item": iid, "score": s})
            entry["lists"][name] = items
            if name.startswith("g_llm"):
                entry[name + "_node"] = lists[q["node"]][name + "_node"]
        out_queries.append(entry)
    rows = [{"id": iid, "disease": d, "drug": m} for (d, m), iid in new.items()]
    random.Random(2027).shuffle(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.glob("batch_*.csv"):
        p.unlink()
    for b in range(0, len(rows), BATCH):
        with (OUT_DIR / f"batch_{b // BATCH:02d}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "disease", "drug"])
            w.writeheader()
            w.writerows(rows[b:b + BATCH])
    (OUT_DIR / "key.json").write_text(json.dumps({"queries": out_queries}, ensure_ascii=False, indent=1), encoding="utf-8")
    per = {name: sum(len(q["lists"][name]) for q in out_queries) for name, _ in LISTS}
    reused = {name: sum(not x["item"].startswith("R") for q in out_queries for x in q["lists"][name]) for name, _ in LISTS}
    no_node = {name: sum(not q.get(name + "_node") for q in out_queries) for name in ("g_llm_fb", "g_llm_primary")}
    print(f"new items {len(rows)} in {(len(rows) + BATCH - 1) // BATCH} batches; per list {per}; "
          f"already judged {reused}; queries without node {no_node}")


def read_labels(judge: str) -> dict[str, str]:
    out = {}
    for d in (ROUND1, OUT_DIR):
        for p in sorted(d.glob(f"labels_{judge}_*.csv")):
            with p.open(encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    out[r["id"].strip()] = r["label"].strip().upper()
    return out


def score() -> None:
    key1 = json.loads((ROUND1 / "key.json").read_text(encoding="utf-8"))
    key2 = {q["node"]: q for q in json.loads((OUT_DIR / "key.json").read_text(encoding="utf-8"))["queries"]}
    labels = {j: read_labels(j) for j, _ in JUDGES}
    need = {x["item"] for q in key2.values() for name, _ in LISTS for x in q["lists"][name]}
    need |= {x["item"] for q in key1["queries"] for x in q["graphs"]["g_norm"]}
    for j, lab in labels.items():
        miss = sorted(i for i in need if lab.get(i) not in ("0", "1", "2", "X"))
        if miss:
            sys.exit(f"{j}: {len(miss)} items unlabelled or invalid, e.g. {miss[:3]}")
    jury = {lvl: {i: sum(labels[j][i] in ok for j, _ in JUDGES) >= 2 for i in need} for lvl, ok in LEVELS.items()}
    drug = {}
    for p in [ROUND1 / "items.csv", *sorted(OUT_DIR.glob("batch_*.csv"))]:
        with p.open(encoding="utf-8-sig") as f:
            drug.update({r["id"].strip(): r["drug"] for r in csv.DictReader(f)})

    new_ids = sorted(i for i in need if i.startswith("R"))
    res = {"n_new_items": len(new_ids), "lists": {}}
    L = ["第二轮盲评：边打分方式 / 大模型标准化，与标准化图谱（按共现病历数排序）逐病配对", "=" * 80,
         f"同 100 个疾病、同三位评委、同提示词；新判 {len(new_ids)} 对，其余复用第一轮标签。"
         "严格 = 治疗用药；宽松 = 治疗或相关；差值按疾病 bootstrap 95% 区间。", ""]
    for name, title in LISTS:
        res["lists"][name] = {}
        L.append(f"【{title}】")
        for lvl, lt in (("strict", "严格"), ("lenient", "宽松")):
            for k in KS:
                base, other = [], []
                for q in key1["queries"]:
                    e2 = key2[q["node"]]["lists"][name]
                    if not e2:
                        continue
                    for lst, acc in ((q["graphs"]["g_norm"], base), (e2, other)):
                        edges = [(x["score"], jury[lvl][x["item"]]) for x in lst]
                        nk = min(k, len(edges))
                        acc.append(expected_sup(edges, nk) / nk)
                diffs = [b - a for a, b in zip(base, other)]
                rng = random.Random(k)
                ci = boot_ci([sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(N_BOOT)])
                cell = {"n": len(diffs), "g_norm": sum(base) / len(base), name: sum(other) / len(other),
                        "delta": sum(diffs) / len(diffs), "ci95": ci}
                res["lists"][name][f"{lvl}_P@{k}"] = cell
                if k in (1, 3):
                    L.append(f"  {lt} P@{k}  标准化图谱 {cell['g_norm']:.3f} → {cell[name]:.3f}  "
                             f"差 {cell['delta']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]（{cell['n']} 个病）")
        for lvl, lt in (("strict", "严格"), ("lenient", "宽松")):
            rngs = (random.Random(7), random.Random(8))
            base, other = [], []
            for q in key1["queries"]:
                e2 = key2[q["node"]]["lists"][name]
                if not e2:
                    continue
                for lst, acc, rng in ((q["graphs"]["g_norm"], base, rngs[0]), (e2, other, rngs[1])):
                    acc.append(distinct_facts([(x["score"], jury[lvl][x["item"]], drug[x["item"]]) for x in lst],
                                              FACTS_K, rng))
            diffs = [b - a for a, b in zip(base, other)]
            cell = {"g_norm": sum(base) / len(base), name: sum(other) / len(other),
                    "delta": sum(diffs) / len(diffs), "ci95": boot_diff(diffs, 99)}
            res["lists"][name][f"{lvl}_distinct_correct@{FACTS_K}"] = cell
            L.append(f"  {lt} 前 {FACTS_K} 名里不同的正确药  标准化图谱 {cell['g_norm']:.2f} → {cell[name]:.2f}  "
                     f"差 {cell['delta']:+.2f} [{cell['ci95'][0]:+.2f}, {cell['ci95'][1]:+.2f}]")
        if name.startswith("g_llm"):
            moved = sum(key2[q["node"]].get(name + "_node") not in (None, q["node"]) for q in key1["queries"])
            res["lists"][name]["node_changed"] = moved
            L.append(f"  查询落到与本方法不同的节点：{moved} 个病")
        L.append("")
    x_new = sum(sum(labels[j][i] == "X" for j, _ in JUDGES) >= 2 for i in new_ids)
    L.append(f"新判药对中多数票判为「不是药名」：{x_new}/{len(new_ids)}")
    text = "\n".join(L) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "llm_judge_extra.txt").write_text(text, encoding="utf-8")
    (OUT / "llm_judge_extra.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "sample"]()
