"""Keep the LLM's completeness without its top-1 cost: rank our own node's drugs first, backfill from the LLM node.

REPORT 3.9: on queries the LLM changes, LLM fallback puts more distinct correct drugs in the top 5 but the
first drug gets slightly worse, because the bigger node brings in drugs of other patients filed under the
same standard name. Here, for the same 171 queries, the LLM graph's list is re-ranked lexicographically:
first by co-occurring notes in our method's node for the query surface, then by co-occurring notes in the
LLM node. Our top drugs stay where they were; the slots our node cannot fill are taken by the LLM node's
drugs. Lists: g_bf_fb (backfill from LLM fallback), g_bf_primary (from LLM first).

Pairs already judged in round 3 reuse those labels; the rest go to outputs/llm_judge_backfill/batch_XX.csv
for the same three judges and prompt.

  python src/llm_judge_backfill.py sample
  python src/llm_judge_backfill.py score    # → outputs/metrics/llm_judge_backfill.{txt,json}
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

from attribution import REL
from eval_external_extra import PROCESSED
from eval_robustness import expected_sup
from llm_judge_extra import JUDGES, LEVELS, build_graph
from llm_judge_sample import top_candidates
from llm_judge_score import FACTS_K, boot_diff, distinct_facts
from llm_judge_targeted import GRAPHS, KS, OUT_DIR as TARGETED, SETS, read_labels
from ner_data import load_bio_docs

OUT_DIR = ROOT / "outputs" / "llm_judge_backfill"
OUT = ROOT / "outputs" / "metrics"
BATCH = 100
BACKFILL = (("g_bf_fb", "g_llm_fb"), ("g_bf_primary", "g_llm_primary"))


def drugs_of(g, node: str) -> dict[str, int]:
    return {m: c for (d, r, m), c in g["co"].items() if d == node and r == REL}


def targeted_items() -> dict[tuple[str, str], str]:
    out = {}
    for p in sorted(TARGETED.glob("batch_*.csv")):
        with p.open(encoding="utf-8-sig") as f:
            out.update({(r["disease"], r["drug"]): r["id"] for r in csv.DictReader(f)})
    return out


def sample() -> None:
    key = json.loads((TARGETED / "key.json").read_text(encoding="utf-8"))
    prior = targeted_items()
    new: dict[tuple[str, str], str] = {}
    for label, fname in SETS:
        docs = load_bio_docs(PROCESSED / fname)
        graphs = {name: build_graph(docs, llm) for name, llm in GRAPHS}
        for q in key["queries"]:
            if q["dataset"] != label:
                continue
            own = drugs_of(graphs["g_norm"], q["nodes"]["g_norm"])
            for bf, src in BACKFILL:
                big = drugs_of(graphs[src], q["nodes"][src])
                cands = set(own) | set(big)
                scale = max(big.values(), default=0) + 1
                ranked = top_candidates({m: own.get(m, 0) * scale + big.get(m, 0) for m in cands})
                items = []
                for m, sc in ranked:
                    drug = m.split("::", 1)[1]
                    pair = (q["surface"], drug)
                    iid = prior.get(pair) or new.setdefault(pair, f"B{len(new):04d}")
                    items.append({"item": iid, "score": sc})
                q["lists"][bf] = items
        print(label, "done", flush=True)

    rows = [{"id": iid, "disease": d, "drug": m} for (d, m), iid in new.items()]
    random.Random(2030).shuffle(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.glob("batch_*.csv"):
        p.unlink()
    for b in range(0, len(rows), BATCH):
        with (OUT_DIR / f"batch_{b // BATCH:02d}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "disease", "drug"])
            w.writeheader()
            w.writerows(rows[b:b + BATCH])
    (OUT_DIR / "key.json").write_text(json.dumps({"queries": key["queries"]}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"queries {len(key['queries'])}, new items {len(rows)} in {(len(rows) + BATCH - 1) // BATCH} batches")


def compare(qs, a: str, b: str, jury, drug) -> dict:
    res = {"n": len(qs)}
    for lvl in LEVELS:
        for k in KS:
            va, vb = [], []
            for q in qs:
                for gn, out in ((a, va), (b, vb)):
                    edges = [(x["score"], jury[lvl][x["item"]]) for x in q["lists"][gn]]
                    nk = min(k, len(edges))
                    out.append(expected_sup(edges, nk) / nk)
            d = [y - x for x, y in zip(va, vb)]
            res[f"{lvl}_P@{k}"] = {a: sum(va) / len(qs), b: sum(vb) / len(qs), "delta": sum(d) / len(d), "ci95": boot_diff(d, k)}
        f = {gn: [distinct_facts([(x["score"], jury[lvl][x["item"]], drug[x["item"]]) for x in q["lists"][gn]], FACTS_K, rng)
                  for q in qs] for gn, rng in ((a, random.Random(7)), (b, random.Random(8)))}
        d = [y - x for x, y in zip(f[a], f[b])]
        res[f"{lvl}_distinct@{FACTS_K}"] = {a: sum(f[a]) / len(qs), b: sum(f[b]) / len(qs), "delta": sum(d) / len(d), "ci95": boot_diff(d, 99)}
    return res


def score() -> None:
    key = json.loads((OUT_DIR / "key.json").read_text(encoding="utf-8"))
    labels = {j: {**read_labels(j, TARGETED), **read_labels(j, OUT_DIR)} for j, _ in JUDGES}
    drug = {}
    for d in (TARGETED, OUT_DIR):
        for p in sorted(d.glob("batch_*.csv")):
            with p.open(encoding="utf-8-sig") as f:
                drug.update({r["id"].strip(): r["drug"] for r in csv.DictReader(f)})
    need = {x["item"] for q in key["queries"] for v in q["lists"].values() for x in v}
    for j, lab in labels.items():
        bad = sorted(i for i in need if lab.get(i) not in ("0", "1", "2", "X"))
        if bad:
            sys.exit(f"{j}: {len(bad)} items unlabelled or invalid, e.g. {bad[:3]}")
    jury = {lvl: {i: sum(labels[j][i] in ok for j, _ in JUDGES) >= 2 for i in need} for lvl, ok in LEVELS.items()}

    names = {"g_norm": "本方法", "g_llm_fb": "大模型兜底", "g_llm_primary": "大模型优先",
             "g_bf_fb": "兜底+本方法的药排前", "g_bf_primary": "优先+本方法的药排前"}
    L = ["大模型标准化 + 「本方法节点里的药排在前面」：能不能既更全、又不丢排第一的准确率", "=" * 80,
         "查询与第三轮（3.9 节）相同；新排序 = 先按本方法节点里的共现病历数，再按大模型节点里的共现病历数。",
         "差 = 后者 − 前者，按查询 bootstrap 95% 区间。", ""]
    res = {}
    for bf, src in BACKFILL:
        changed = [q for q in key["queries"] if q["changed"][src]]
        for subset, st, qs in (("all", "答案不同的全部查询", changed),
                               ("unlinked", "本方法原样保留的写法", [q for q in changed if q["unlinked_in_ours"]])):
            for a, b in (("g_norm", bf), (src, bf)):
                c = compare(qs, a, b, jury, drug)
                res[f"{bf}_{subset}_vs_{a}"] = c
                L.append(f"【{st}，{len(qs)} 个】{names[a]} → {names[b]}")
                for lvl, lt in (("strict", "严格"), ("lenient", "宽松")):
                    cells = [(f"P@{k}", c[f"{lvl}_P@{k}"]) for k in KS] + [(f"前{FACTS_K}名不同正确药", c[f"{lvl}_distinct@{FACTS_K}"])]
                    L.append(f"  {lt}：" + "  ".join(f"{n} {x[a]:.3f}→{x[b]:.3f}（{x['delta']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]）"
                                                     for n, x in cells))
            L.append("")
    text = "\n".join(L) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "llm_judge_backfill.txt").write_text(text, encoding="utf-8")
    (OUT / "llm_judge_backfill.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "sample"]()
