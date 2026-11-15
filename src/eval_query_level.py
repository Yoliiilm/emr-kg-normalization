"""Query-level and confidence-level external checks of G_raw vs G_norm (疾病→药物).

Judges: CMeIE, 寻医问药, and both combined (disease names merged through either source's classes).

A  per-query: every external disease matched by both graphs is a query "which drugs treat X"; each graph
   returns its drug neighbours ranked by document count. single = G_raw answers from its most-mentioned
   matching node (what a user typing the usual name gets); all = counts summed over every matching node
   (G_raw with perfect synonym expansion). P@k and facts@k = distinct external (disease, drug) facts in top k.
B  confidence: on the paired clusters, keep edges with count ≥ t; support rate and distinct supported facts;
   AUC of edge count as a predictor of external support.
E  configuration chosen on train+dev gold (combined judge, all-edge support delta) among drug tables
   {hand_train_only, public, public_plus_train} × ICD 3-char rollup {off, on}; reported on the test set
   next to the default FULL setting (hand_full table).

Writes outputs/metrics/query_level.{txt,json} by default; --out selects a separate JSON path.
"""

from __future__ import annotations

import bisect
import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_alias_heldout import key_forms
from eval_alias_public import norm_graph_with
from eval_external_extra import PROCESSED
from eval_external_kg import DISEASE, REL_PREDICATE, UnionFind, disease_classes, doc_level_graph, neighbours, target_keys
from eval_g_raw_vs_g_norm import collect_freq
from eval_robustness import REL, boot_ci, icd_category_names, load_sources, rollup
from ner_data import load_bio_docs

DICTS = ROOT / "data" / "dicts"
OUT = ROOT / "outputs" / "metrics"
KS = (1, 3, 5)
THRESHOLDS = (1, 2, 3, 5)
N_BOOT = 2000
N_BOOT_AUC = 500
N_SHUFFLE = 50
JUDGES = (("combined", "合并"), ("cmeie", "CMeIE"), ("medicalkg", "寻医问药"))


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def combine_sources(sources: dict) -> dict:
    uf = UnionFind()
    for ext in sources.values():
        for k in list(ext["uf"].parent):
            uf.union(k, ext["uf"].find(k))
    pairs = {rel: set() for rel in REL_PREDICATE}
    subjects = {rel: set() for rel in REL_PREDICATE}
    for ext in sources.values():
        for rel in REL_PREDICATE:
            pairs[rel] |= {(uf.find(c), o) for c, o in ext["pairs"][rel]}
            subjects[rel] |= {uf.find(c) for c in ext["subjects"][rel]}
    return {"name": "CMeIE ∪ 寻医问药", "uf": uf, "pairs": pairs, "subjects": subjects}


def ext_index(ext: dict) -> dict[str, set[str]]:
    by_cls = defaultdict(set)
    for c, o in ext["pairs"][REL]:
        by_cls[c].add(o)
    return by_cls


def node_classes(g: dict, ext: dict) -> dict[str, set[str]]:
    subj = ext["subjects"][REL]
    out = {}
    for nid in g["df"]:
        if nid.startswith(DISEASE + "::"):
            cls = disease_classes(g, nid, ext) & subj
            if cls:
                out[nid] = cls
    return out


def boot_mean_diff(diffs: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    return boot_ci([mean(diffs[rng.randrange(len(diffs))] for _ in diffs) for _ in range(N_BOOT)])


# ---------------------------------------------------------------- A: per-query


def topk_metrics(items: list[tuple[int, frozenset]], rng: random.Random) -> dict[str, float]:
    acc = defaultdict(float)
    for _ in range(N_SHUFFLE):
        order = sorted(items, key=lambda x: (-x[0], rng.random()))
        for k in KS:
            top = order[:k]
            acc[f"P@{k}"] += sum(bool(h) for _, h in top) / len(top)
            acc[f"facts@{k}"] += len(set().union(*(h for _, h in top)))
    return {k: v / N_SHUFFLE for k, v in acc.items()}


def query_eval(graphs: dict, ext: dict) -> dict:
    by_cls = ext_index(ext)
    cls_nodes = {gn: defaultdict(list) for gn in graphs}
    for gn, g in graphs.items():
        for nid, cls in node_classes(g, ext).items():
            for c in cls:
                cls_nodes[gn][c].append(nid)
    nb = {gn: neighbours(g, REL) for gn, g in graphs.items()}
    common = sorted(set(cls_nodes["g_raw"]) & set(cls_nodes["g_norm"]))
    rng = random.Random(3)
    res = {}
    for mode in ("single", "all"):
        rows = []
        for c in common:
            row = {}
            for gn, g in graphs.items():
                nodes = cls_nodes[gn][c]
                if mode == "single":
                    nodes = [max(nodes, key=lambda n: (g["mentions"][n], n))]
                agg = defaultdict(int)
                for d in nodes:
                    for m in nb[gn].get(d, []):
                        agg[m] += g["co"][(d, REL, m)]
                if not agg:
                    break
                row[gn] = topk_metrics([(cnt, frozenset(target_keys(g, m, REL) & by_cls[c])) for m, cnt in agg.items()], rng)
            else:
                rows.append(row)
        out = {"n_queries": len(rows)}
        for i, k in enumerate(rows[0]["g_raw"] if rows else []):
            a, b = mean(r["g_raw"][k] for r in rows), mean(r["g_norm"][k] for r in rows)
            out[k] = {"g_raw": a, "g_norm": b, "delta": b - a,
                      "ci95": boot_mean_diff([r["g_norm"][k] - r["g_raw"][k] for r in rows], 9 + i)}
        res[mode] = out
    return res


# ---------------------------------------------------------------- B: thresholds and AUC


def cluster_edges(graphs: dict, ext: dict) -> list[dict]:
    by_cls = ext_index(ext)
    ncls = {gn: node_classes(g, ext) for gn, g in graphs.items()}
    common = set().union(*ncls["g_raw"].values()) & set().union(*ncls["g_norm"].values())
    uf = UnionFind()
    kept = []
    for gn, m in ncls.items():
        for nid, cls in m.items():
            shared = cls & common
            if shared:
                kept.append((gn, nid, shared))
                for c in shared:
                    uf.union(f"{gn}|{nid}", f"cls|{c}")
    nb = {gn: neighbours(g, REL) for gn, g in graphs.items()}
    rows = defaultdict(lambda: {"g_raw": [], "g_norm": []})
    for gn, nid, shared in kept:
        g = graphs[gn]
        for m in nb[gn].get(nid, []):
            keys = target_keys(g, m, REL)
            facts = frozenset((c, k) for c in shared for k in keys if k in by_cls[c])
            rows[uf.find(f"{gn}|{nid}")][gn].append((g["co"][(nid, REL, m)], facts))
    return list(rows.values())


def auc(edges: list[tuple[int, frozenset]]) -> float:
    pos = [c for c, f in edges if f]
    neg = sorted(c for c, f in edges if not f)
    if not pos or not neg:
        return float("nan")
    s = sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos)
    return s / (len(pos) * len(neg))


def threshold_eval(rows: list[dict]) -> dict:
    res = {}
    for t in THRESHOLDS:
        cells = []
        for r in rows:
            cell = []
            for gn in ("g_raw", "g_norm"):
                kept = [f for c, f in r[gn] if c >= t]
                cell += [len(kept), sum(bool(f) for f in kept), len(set().union(*kept)) if kept else 0]
            cells.append(cell)

        def stats(sample):
            s = [sum(c[i] for c in sample) for i in range(6)]
            return (s[1] / s[0] if s[0] else 0.0), (s[4] / s[3] if s[3] else 0.0), s

        ra, rb, s = stats(cells)
        rng = random.Random(21 + t)
        d_rate, d_facts = [], []
        for _ in range(N_BOOT):
            a, b, ss = stats([cells[rng.randrange(len(cells))] for _ in cells])
            if ss[0] and ss[3]:
                d_rate.append(b - a)
            d_facts.append(ss[5] - ss[2])
        nan = float("nan")
        both = s[0] and s[3]
        res[t] = {"edges": [s[0], s[3]], "supported_edges": [s[1], s[4]],
                  "rate": [ra if s[0] else nan, rb if s[3] else nan], "rate_delta": rb - ra if both else nan,
                  "rate_ci95": boot_ci(d_rate) if len(d_rate) > N_BOOT // 2 else [nan, nan],
                  "facts": [s[2], s[5]], "facts_delta": s[5] - s[2],
                  "facts_ci95": boot_ci(d_facts)}
    flat = {gn: [e for r in rows for e in r[gn]] for gn in ("g_raw", "g_norm")}
    a, b = auc(flat["g_raw"]), auc(flat["g_norm"])
    rng = random.Random(77)
    diffs = []
    for _ in range(N_BOOT_AUC):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        x = auc([e for r in sample for e in r["g_raw"]])
        y = auc([e for r in sample for e in r["g_norm"]])
        if x == x and y == y:
            diffs.append(y - x)
    res["auc"] = {"g_raw": a, "g_norm": b, "delta": b - a, "ci95": boot_ci(diffs)}
    return res


def evaluate(raw: dict, norm: dict, judges: dict) -> dict:
    graphs = {"g_raw": raw, "g_norm": norm}
    return {j: {"query": query_eval(graphs, ext), "threshold": threshold_eval(cluster_edges(graphs, ext))}
            for j, ext in judges.items()}


# ---------------------------------------------------------------- main / report


def ci(c) -> str:
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def threshold_cell(k: int, v: dict) -> str:
    rate = lambda x, n: f"{x:.3f}({n}边)" if n else "无边"
    diff = f" {v['rate_delta']:+.3f}{ci(v['rate_ci95'])}" if v["rate_ci95"][0] == v["rate_ci95"][0] else ""
    return (f"共现≥{k}: 印证率 {rate(v['rate'][0], v['edges'][0])}→{rate(v['rate'][1], v['edges'][1])}{diff}"
            f" 事实 {v['facts'][0]}→{v['facts'][1]} {v['facts_delta']:+d}{ci(v['facts_ci95']).replace('.000', '')}")


def report(res: dict) -> str:
    L = ["标准化能否让「疾病→药物」查询结果更可信：按查询 / 按置信度的外部检验（差 = G_norm − G_raw，95% bootstrap 区间）",
         "=" * 90, f"合并裁判：疾病 {res['judge_sizes']['subjects']} 个，疾病-药物对 {res['judge_sizes']['pairs']} 条",
         f"协议：选择集={res['protocol']['selection_split']}；最终测试={res['protocol']['test_split']}；",
         f"测试病历={res['protocol']['n_test_docs']}；bootstrap={res['protocol']['bootstrap_replicates']} 次；",
         f"seed={res['protocol']['seed']}；输入文件 SHA-256 见 JSON。", ""]
    L.append("E 配置选择（训练+开发 标注实体，合并裁判，全部边印证率的配对差；只在这里选，测试集不参与）")
    for name, v in res["selection"].items():
        L.append(f"  {name:28s} {v['rate'][0]:.3f} → {v['rate'][1]:.3f}  差 {v['rate_delta']:+.3f} {ci(v['rate_ci95'])}")
    L.append(f"  选中：{res['best']}")
    for set_name, set_title in (("test_pred_ner", "测试集 预测实体"), ("test_gold_ner", "测试集 标注实体"),
                                ("train_dev_gold_ner", "训练+开发 标注实体（选择集）")):
        for cfg, r in res[set_name].items():
            L += ["", f"【{set_title} · {cfg}】"]
            for j, jt in JUDGES:
                q, t = r[j]["query"], r[j]["threshold"]
                for mode, mt in (("single", "原图只查最常见写法"), ("all", "原图查全部写法")):
                    m = q[mode]
                    if not m["n_queries"]:
                        continue
                    cells = "  ".join(f"{k} {m[k]['g_raw']:.3f}→{m[k]['g_norm']:.3f} {m[k]['delta']:+.3f}{ci(m[k]['ci95'])}"
                                      for k in ("P@1", "P@3", "P@5", "facts@5"))
                    L.append(f"  {jt:5s} A 查询({mt}, {m['n_queries']} 个疾病)  {cells}")
                L.append(f"  {jt:5s} B " + "  ".join(threshold_cell(k, t[k]) for k in THRESHOLDS))
                a = t["auc"]
                L.append(f"  {jt:5s} B 共现次数预测印证的 AUC {a['g_raw']:.3f} → {a['g_norm']:.3f}  差 {a['delta']:+.3f} {ci(a['ci95'])}")
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="JSON output path (TXT uses the same stem)")
    args = parser.parse_args()
    sources = load_sources()
    judges = {"combined": combine_sources(sources), **sources}
    hand = json.loads((DICTS / "drug_alias.json").read_text(encoding="utf-8"))["alias"]
    public = json.loads((DICTS / "drug_alias_public.json").read_text(encoding="utf-8"))["alias"]
    train_docs = load_bio_docs(PROCESSED / "train.bio") + load_bio_docs(PROCESSED / "dev.bio")
    allowed = set().union(*(key_forms(s) for s in collect_freq(train_docs).get("药物", {})))
    hand_train = {k: v for k, v in hand.items() if key_forms(k) & allowed}
    tables = {"hand_train_only": hand_train, "public": public, "public_plus_train": {**public, **hand_train}}
    cats = icd_category_names()
    combined = judges["combined"]
    test_pred_docs = load_bio_docs(PROCESSED / "test_pred.bio")
    test_gold_docs = load_bio_docs(PROCESSED / "test.bio")
    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    res = {
        "judge_sizes": {"subjects": len(combined["subjects"][REL]), "pairs": len(combined["pairs"][REL])},
        "protocol": {
            "selection_split": "train.bio + dev.bio only",
            "test_split": "test_pred.bio and test.bio evaluated separately; no threshold/config selection on test",
            "n_train_docs": len(load_bio_docs(PROCESSED / "train.bio")),
            "n_dev_docs": len(load_bio_docs(PROCESSED / "dev.bio")),
            "n_test_docs": len(test_pred_docs),
            "n_test_gold_docs": len(test_gold_docs),
            "bootstrap_replicates": N_BOOT,
            "shuffle_replicates_for_ties": N_SHUFFLE,
            "seed": 3,
            "paired_unit": "external disease cluster; raw and normalized scores computed on the same clusters",
            "threshold_selection_metric": "combined external judge, train+dev all-edge support-rate delta",
            "baselines": ["raw graph", "train+dev-observed hand aliases", "public alias dictionary", "public + train+dev-observed hand aliases", "ICD-3 rollup variants"],
            "diagnostic_only": {"hand_full": "full hand alias dictionary may include test-surface curation; not a strict train-only baseline"},
            "reference_limit": "external KG support is not disease-normalization gold correctness",
            "input_sha256": {p.name: digest(p) for p in (PROCESSED / "train.bio", PROCESSED / "dev.bio", PROCESSED / "test.bio", PROCESSED / "test_pred.bio")},
        },
    }

    raw_td = doc_level_graph(train_docs, None)
    built = {}
    for name, table in tables.items():
        g = norm_graph_with(train_docs, table)[0]
        built[name] = g
        built[name + "+icd3"] = rollup(g, cats)[0]
    res["selection"] = {name: threshold_eval(cluster_edges({"g_raw": raw_td, "g_norm": g}, combined))[1]
                        for name, g in built.items()}
    best = max(res["selection"], key=lambda n: res["selection"][n]["rate_delta"])
    res["best"] = best
    print("selection", {n: round(v["rate_delta"], 4) for n, v in res["selection"].items()}, "→", best, flush=True)

    def config_graph(docs, cfg):
        if cfg == "hand_full":
            return norm_graph_with(docs, hand)[0]
        name, _, icd = cfg.partition("+")
        g = norm_graph_with(docs, tables[name])[0]
        return rollup(g, cats)[0] if icd else g

    doc_sets = {"test_pred_ner": load_bio_docs(PROCESSED / "test_pred.bio"),
                "test_gold_ner": load_bio_docs(PROCESSED / "test.bio"),
                "train_dev_gold_ner": train_docs}
    for set_name, docs in doc_sets.items():
        raw = raw_td if set_name == "train_dev_gold_ner" else doc_level_graph(docs, None)
        res[set_name] = {}
        for cfg in ("hand_full", best):
            g = built[best] if (set_name == "train_dev_gold_ner" and cfg == best) else config_graph(docs, cfg)
            res[set_name][cfg] = evaluate(raw, g, judges)
            q = res[set_name][cfg]["combined"]["query"]["single"]
            print(set_name, cfg, "P@5", round(q["P@5"]["delta"], 4) if q["n_queries"] else None, flush=True)

    text = report(res)
    out_path = args.out or (OUT / "query_level.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_suffix(".txt").write_text(text, encoding="utf-8")
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
