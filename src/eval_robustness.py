"""Robustness checks for the G_raw vs G_norm external-support gain (疾病→药物 edges).

R1 random-merge control: shuffle which surface forms are merged together, keeping the real group-size
   structure (both entity types, or the drug side only); the merged node is named after its most
   frequent member. If merging per se inflated support, these graphs would gain as much as G_norm.
R2 equal-size comparison: per external-disease cluster keep only the top-n edges by document count,
   n = min(edges in G_raw, edges in G_norm); plus global top-k over the paired clusters.
R3 matching sensitivity: drug names matched exactly without dose-form/salt cleanup (stricter), with it
   (default), or by containment either way (looser).
R4 ICD category rollup: merge G_norm disease nodes that share a 3-character ICD category.
R5 data-size curve: gold-NER graphs on random subsets of all 1379 CCKS2019 notes.

Ties in top-n are resolved by expected value (partial tie groups contribute their support share).
Writes outputs/metrics/robustness.{txt,json}.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_external_kg as ekg
import normalize
from eval_ablation import FULL
from eval_external_extra import PROCESSED
from eval_external_kg import (
    DISEASE,
    N_BOOT,
    UnionFind,
    disease_classes,
    doc_level_graph,
    load_cmeie,
    load_medicalkg,
    neighbours,
    paired_support,
    supported,
    target_keys,
)
from eval_g_raw_vs_g_norm import collect_freq
from ner_data import load_bio_docs
from normalize import build_canonical_map, configure_normalization

REL = "使用药物"
DRUG = "药物"
OUT = ROOT / "outputs" / "metrics"
DICTS = ROOT / "data" / "dicts"
N_SEEDS = 20
TOP_KS = (25, 50, 100, 150, 200)
SIZES = (100, 200, 400, 700, 1000)
SIZE_REPS = 5
SOURCES = (("cmeie", "CMeIE"), ("medicalkg", "寻医问药"))


def load_sources() -> dict:
    return {"cmeie": load_cmeie(DICTS / "raw" / "CMeIE"),
            "medicalkg": load_medicalkg(DICTS / "raw" / "medicalkg" / "medical.json")}


def brief(p: dict) -> dict:
    return {"g_raw": p["g_raw"]["rate"], "g_norm": p["g_norm"]["rate"], "edges_raw": p["g_raw"]["edges"],
            "edges_norm": p["g_norm"]["edges"], "delta": p["delta"], "ci95": list(p["delta_ci95"])}


# ---------------------------------------------------------------- paired clusters with edge lists


def make_sup(kind: str, ext: dict):
    if kind == "strict":
        return lambda g, d, m: supported(g, d, m, REL, ext)
    by_cls = defaultdict(set)
    for c, o in ext["pairs"][REL]:
        by_cls[c].add(o)

    def contain(g, d, m) -> bool:
        objs = set().union(*(by_cls.get(c, set()) for c in disease_classes(g, d, ext)))
        if not objs:
            return False
        for k in target_keys(g, m, REL):
            if k in objs or (len(k) >= 2 and any(len(o) >= 2 and (k in o or o in k) for o in objs)):
                return True
        return False

    return contain


def cluster_rows(graphs: dict, ext: dict, sup) -> list[dict]:
    """Same clustering as eval_external_kg.paired_support, but keeps (count, supported) per edge."""
    subjects = ext["subjects"][REL]
    node_classes = {}
    for gname, g in graphs.items():
        for nid in g["df"]:
            if nid.startswith(DISEASE + "::"):
                cls = disease_classes(g, nid, ext) & subjects
                if cls:
                    node_classes[(gname, nid)] = cls
    by_graph = {gn: set().union(*[c for (x, _), c in node_classes.items() if x == gn]) for gn in graphs}
    common = by_graph["g_raw"] & by_graph["g_norm"]
    uf = UnionFind()
    kept = []
    for (gname, nid), cls in node_classes.items():
        shared = cls & common
        if shared:
            kept.append((gname, nid))
            for c in shared:
                uf.union(f"{gname}|{nid}", f"cls|{c}")
    nb = {gn: neighbours(g, REL) for gn, g in graphs.items()}
    rows = defaultdict(lambda: {"g_raw": [], "g_norm": []})
    for gname, nid in kept:
        g = graphs[gname]
        for m in nb[gname].get(nid, []):
            rows[uf.find(f"{gname}|{nid}")][gname].append((g["co"][(nid, REL, m)], sup(g, nid, m)))
    return list(rows.values())


def expected_sup(edges: list[tuple[int, bool]], n: int) -> float:
    if n <= 0:
        return 0.0
    if n >= len(edges):
        return float(sum(s for _, s in edges))
    by = sorted(edges, key=lambda e: -e[0])
    cut = by[n - 1][0]
    above = [s for c, s in by if c > cut]
    tie = [s for c, s in by if c == cut]
    return sum(above) + (n - len(above)) * sum(tie) / len(tie)


def boot_ci(values: list[float]) -> list[float]:
    values = sorted(values)
    return [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]]


def paired_rows(rows: list[dict], equal_n: bool) -> dict:
    cells = []
    for r in rows:
        n = min(len(r["g_raw"]), len(r["g_norm"]))
        cells.append(tuple(x for gn in ("g_raw", "g_norm")
                           for x in ((n, expected_sup(r[gn], n)) if equal_n else (len(r[gn]), expected_sup(r[gn], len(r[gn]))))))

    def rates(sample):
        er, sr, en, sn = (sum(c[i] for c in sample) for i in range(4))
        return (sr / er if er else 0.0), (sn / en if en else 0.0), er, en

    rr, rn, er, en = rates(cells)
    rng = random.Random(11)
    diffs = []
    for _ in range(N_BOOT):
        a, b, _, _ = rates([cells[rng.randrange(len(cells))] for _ in cells])
        diffs.append(b - a)
    return {"g_raw": rr, "g_norm": rn, "edges_raw": er, "edges_norm": en, "delta": rn - rr, "ci95": boot_ci(diffs)}


def global_topk(rows: list[dict]) -> dict:
    def at_k(sample, gn, k):
        flat = [e for r in sample for e in r[gn]]
        return expected_sup(flat, k) / k if len(flat) >= k else float("nan")

    out = {}
    n_min = min(sum(len(r["g_raw"]) for r in rows), sum(len(r["g_norm"]) for r in rows))
    rng = random.Random(5)
    for k in [k for k in TOP_KS if k < n_min] + [n_min]:
        diffs = []
        for _ in range(N_BOOT // 4):
            sample = [rows[rng.randrange(len(rows))] for _ in rows]
            a, b = at_k(sample, "g_raw", k), at_k(sample, "g_norm", k)
            if a == a and b == b:
                diffs.append(b - a)
        a, b = at_k(rows, "g_raw", k), at_k(rows, "g_norm", k)
        out[k] = {"g_raw": a, "g_norm": b, "delta": b - a, "ci95": boot_ci(diffs)}
    return out


# ---------------------------------------------------------------- R1


def random_canon(real: dict, freq: dict, labels: tuple[str, ...], rng: random.Random) -> dict:
    out = dict(real)
    for label in labels:
        groups = defaultdict(list)
        for (lab, s), c in real.items():
            if lab == label and isinstance(c, str):
                groups[c].append(s)
        surfaces = [s for v in groups.values() for s in v]
        rng.shuffle(surfaces)
        i = 0
        for size in (len(v) for v in groups.values()):
            chunk = surfaces[i:i + size]
            i += size
            name = max(chunk, key=lambda s: freq[label][s])
            for s in chunk:
                out[(label, s)] = name
    return out


def r1_random_merge(pred_docs, raw, real_canon, freq, sources) -> dict:
    real = doc_level_graph(pred_docs, real_canon)
    res = {"real": {s: brief(paired_support({"g_raw": raw, "g_norm": real}, REL, ext)) for s, ext in sources.items()},
           "n_nodes": {"g_raw": len(raw["df"]), "real": len(real["df"])}}
    for key, labels in (("random_both", (DISEASE, DRUG)), ("random_drug", (DRUG,))):
        runs = []
        for seed in range(N_SEEDS):
            g = doc_level_graph(pred_docs, random_canon(real_canon, freq, labels, random.Random(seed)))
            runs.append({"n_nodes": len(g["df"])} |
                        {s: brief(paired_support({"g_raw": raw, "g_norm": g}, REL, ext)) for s, ext in sources.items()})
        res[key] = runs
        print("R1", key, {s: round(sum(r[s]["delta"] for r in runs) / len(runs), 4) for s in sources}, flush=True)
    return res


# ---------------------------------------------------------------- R2 / R3


def r2_equal_size(raw, norm, sources) -> dict:
    res = {}
    for s, ext in sources.items():
        rows = cluster_rows({"g_raw": raw, "g_norm": norm}, ext, make_sup("strict", ext))
        res[s] = {"all": paired_rows(rows, False), "equal_n": paired_rows(rows, True), "global_topk": global_topk(rows)}
        print("R2", s, round(res[s]["equal_n"]["delta"], 4), flush=True)
    return res


def r3_matching(pred_docs, raw, norm, sources) -> dict:
    res = {}
    for s, ext in sources.items():
        res[s] = {kind: paired_rows(cluster_rows({"g_raw": raw, "g_norm": norm}, ext, make_sup(kind, ext)), False)
                  for kind in ("strict", "contain")}
    orig = ekg.drug_key
    ekg.drug_key = ekg.ext_key
    try:
        plain = load_sources()
        for s, ext in plain.items():
            res[s]["no_cleanup"] = paired_rows(cluster_rows({"g_raw": raw, "g_norm": norm}, ext, make_sup("strict", ext)), False)
            res[s]["no_cleanup_contain"] = paired_rows(
                cluster_rows({"g_raw": raw, "g_norm": norm}, ext, make_sup("contain", ext)), False)
    finally:
        ekg.drug_key = orig
    print("R3", {s: {k: round(v["delta"], 4) for k, v in r.items()} for s, r in res.items()}, flush=True)
    return res


# ---------------------------------------------------------------- R4


def icd_category_names() -> dict[str, str]:
    out = {}
    for name, codes in normalize.get_index().names.items():
        for c in codes:
            if len(c) == 3 and c not in out:
                out[c] = name
    return out


def rollup(g: dict, cat_names: dict[str, str]) -> tuple[dict, int]:
    idx = normalize.get_index()
    new_id = {}
    for nid in g["df"]:
        if nid.startswith(DISEASE + "::"):
            code = idx.preferred_code(nid.split("::", 1)[1])
            if code and code[:3] in cat_names:
                new_id[nid] = f"{DISEASE}::{cat_names[code[:3]]}"
    out = {"n_docs": g["n_docs"], "df": defaultdict(int), "mentions": defaultdict(int), "names": defaultdict(set),
           "co": defaultdict(int), "cue": defaultdict(int)}
    for nid in g["df"]:
        t = new_id.get(nid, nid)
        out["df"][t] += g["df"][nid]
        out["mentions"][t] += g["mentions"][nid]
        out["names"][t] |= g["names"][nid] | {t.split("::", 1)[1]}
    for key in ("co", "cue"):
        for (s, r, t), v in g[key].items():
            out[key][(new_id.get(s, s), r, new_id.get(t, t))] += v
    return out, len(set(new_id.values()))


def r4_icd_rollup(raw, norm, sources) -> dict:
    rolled, n_cats = rollup(norm, icd_category_names())
    n_dis = lambda g: sum(1 for n in g["df"] if n.startswith(DISEASE + "::"))
    res = {"disease_nodes": {"g_raw": n_dis(raw), "g_norm": n_dis(norm), "rollup": n_dis(rolled)}, "n_categories": n_cats}
    for s, ext in sources.items():
        res[s] = {"g_norm": brief(paired_support({"g_raw": raw, "g_norm": norm}, REL, ext)),
                  "rollup": brief(paired_support({"g_raw": raw, "g_norm": rolled}, REL, ext)),
                  "rollup_vs_norm": brief(paired_support({"g_raw": norm, "g_norm": rolled}, REL, ext))}
    print("R4", {s: round(res[s]["rollup"]["delta"], 4) for s in sources}, flush=True)
    return res


# ---------------------------------------------------------------- R5


def r5_data_size(sources) -> dict:
    docs = (load_bio_docs(PROCESSED / "train.bio") + load_bio_docs(PROCESSED / "dev.bio")
            + load_bio_docs(PROCESSED / "test.bio"))
    canon = build_canonical_map(collect_freq(docs))
    res = {"n_docs_total": len(docs)}
    for size in SIZES + (len(docs),):
        reps = 1 if size == len(docs) else SIZE_REPS
        runs = []
        for rep in range(reps):
            sub = random.Random(100 * size + rep).sample(docs, size)
            raw, norm = doc_level_graph(sub, None), doc_level_graph(sub, canon)
            runs.append({s: brief(paired_support({"g_raw": raw, "g_norm": norm}, REL, ext)) for s, ext in sources.items()})
        res[size] = runs
        print("R5", size, {s: round(sum(r[s]["delta"] for r in runs) / reps, 4) for s in sources}, flush=True)
    return res


# ---------------------------------------------------------------- report


def ci(c) -> str:
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def report(res: dict) -> str:
    L = ["外部印证率提升的稳健性检验（疾病→药物；差 = G_norm − G_raw，按簇 bootstrap 95% 区间）", "=" * 78]
    if "r1" in res:
        r1 = res["r1"]
        L += ["", f"R1 随机合并对照（保持真实的合并组大小，随机决定谁和谁合并，{N_SEEDS} 个随机种子；测试集预测实体）",
              f"  节点数：G_raw {r1['n_nodes']['g_raw']}  真实标准化 {r1['n_nodes']['real']}  "
              f"随机(两类) {mean(r['n_nodes'] for r in r1['random_both']):.0f}  随机(仅药物) {mean(r['n_nodes'] for r in r1['random_drug']):.0f}"]
        for s, st in SOURCES:
            real = r1["real"][s]
            L.append(f"  {st}: 真实标准化 差 {real['delta']:+.3f} {ci(real['ci95'])}")
            for key, t in (("random_both", "随机合并(疾病+药物)"), ("random_drug", "随机合并(仅药物)")):
                ds = sorted(r[s]["delta"] for r in r1[key])
                ge = sum(d >= real["delta"] for d in ds)
                sig = sum(r[s]["ci95"][0] > 0 for r in r1[key])
                L.append(f"    {t:14s} 差 均值 {mean(ds):+.3f}  范围 [{ds[0]:+.3f}, {ds[-1]:+.3f}]  "
                         f"≥真实值的种子 {ge}/{len(ds)}  自身显著>0 的种子 {sig}/{len(ds)}")
    if "r2" in res:
        L += ["", "R2 等规模比较（测试集预测实体）"]
        for s, st in SOURCES:
            r = res["r2"][s]
            a, e = r["all"], r["equal_n"]
            L.append(f"  {st}: 全部边 {a['edges_raw']}/{a['edges_norm']} 条  差 {a['delta']:+.3f} {ci(a['ci95'])}")
            L.append(f"        每簇取同样多的高频边 {e['edges_raw']}/{e['edges_norm']} 条  "
                     f"G_raw {e['g_raw']:.3f}  G_norm {e['g_norm']:.3f}  差 {e['delta']:+.3f} {ci(e['ci95'])}")
            L.append("        全局按共现数取前 k 条：" + "  ".join(
                f"k={k}: {v['g_raw']:.3f}→{v['g_norm']:.3f} 差 {v['delta']:+.3f} {ci(v['ci95'])}" for k, v in r["global_topk"].items()))
    if "r3" in res:
        L += ["", "R3 匹配规则敏感性（测试集预测实体）"]
        names = (("no_cleanup", "严格·不去剂型/盐"), ("strict", "默认·去剂型/盐"),
                 ("contain", "宽松·去剂型/盐+包含"), ("no_cleanup_contain", "宽松·不去剂型/盐+包含"))
        for s, st in SOURCES:
            for k, t in names:
                v = res["r3"][s][k]
                L.append(f"  {st:6s} {t:16s} G_raw {v['g_raw']:.3f}  G_norm {v['g_norm']:.3f}  差 {v['delta']:+.3f} {ci(v['ci95'])}")
    if "r4" in res:
        r4 = res["r4"]
        dn = r4["disease_nodes"]
        L += ["", f"R4 ICD 三位码类目汇总（测试集预测实体）：疾病节点 G_raw {dn['g_raw']} → G_norm {dn['g_norm']} → 汇总后 {dn['rollup']}"
              f"（涉及 {r4['n_categories']} 个类目）"]
        for s, st in SOURCES:
            for k, t in (("g_norm", "G_norm vs G_raw"), ("rollup", "汇总 vs G_raw"), ("rollup_vs_norm", "汇总 vs G_norm")):
                v = r4[s][k]
                L.append(f"  {st:6s} {t:16s} {v['g_raw']:.3f} → {v['g_norm']:.3f}  边 {v['edges_raw']}→{v['edges_norm']}  "
                         f"差 {v['delta']:+.3f} {ci(v['ci95'])}")
    if "r5" in res:
        r5 = res["r5"]
        L += ["", f"R5 数据量曲线（CCKS2019 全部 {r5['n_docs_total']} 份标注实体病历中随机抽样，每档 {SIZE_REPS} 次）"]
        for size in [k for k in r5 if k != "n_docs_total"]:
            runs = r5[size]
            cells = []
            for s, st in SOURCES:
                cells.append(f"{st} {mean(r[s]['g_raw'] for r in runs):.3f}→{mean(r[s]['g_norm'] for r in runs):.3f} "
                             f"差 {mean(r[s]['delta'] for r in runs):+.3f}（显著 {sum(r[s]['ci95'][0] > 0 for r in runs)}/{len(runs)}）"
                             f" 边 {mean(r[s]['edges_norm'] for r in runs):.0f}")
            L.append(f"  {size:>5} 份  " + "   ".join(cells))
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=["r1", "r2", "r3", "r4", "r5"])
    args = p.parse_args()
    sources = load_sources()
    pred_docs = load_bio_docs(PROCESSED / "test_pred.bio")
    configure_normalization(**FULL)
    freq = collect_freq(pred_docs)
    canon = build_canonical_map(freq)
    raw, norm = doc_level_graph(pred_docs, None), doc_level_graph(pred_docs, canon)

    res = {}
    if "r1" in args.only:
        res["r1"] = r1_random_merge(pred_docs, raw, canon, freq, sources)
    if "r2" in args.only:
        res["r2"] = r2_equal_size(raw, norm, sources)
    if "r3" in args.only:
        res["r3"] = r3_matching(pred_docs, raw, norm, sources)
    if "r4" in args.only:
        res["r4"] = r4_icd_rollup(raw, norm, sources)
    if "r5" in args.only:
        res["r5"] = r5_data_size(sources)
    configure_normalization()

    text = report(res)
    suffix = "" if len(args.only) == 5 else "_" + "_".join(args.only)
    (OUT / f"robustness{suffix}.txt").write_text(text, encoding="utf-8")
    (OUT / f"robustness{suffix}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
