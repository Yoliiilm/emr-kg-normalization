"""Improvement 1: does drug→disease attribution make 疾病→药物 edges more trustworthy than co-occurrence?

Edge scores on the same edge set: count (co-occurring notes), lift, oneshot (one allocation step
weighted by lift), em (src/attribution.py, background cause on, no smoothing), em_nobg,
em_tuned (smoothing α and iterations chosen with --tune on train+dev gold notes only).
G_norm uses the leak-free drug table (public brand table + hand entries reachable from train notes).
Held-out evaluation is on the test notes (predicted and gold entities); train+dev is only used for tuning.

Metrics against the external judges (combined CMeIE ∪ 寻医问药, and each alone):
  per-disease  P@1/P@3/P@5/MAP over each covered disease's drugs (ties averaged), paired bootstrap vs count
  global       precision of the top-k edges over all covered diseases
  strata       P@3 gain by comorbidity load (mean number of other diseases in the disease's notes)
  2x2          {G_raw, G_norm} × {count, em}: does attribution gain more on the normalized graph?
Writes outputs/metrics/attribution.{txt,json}.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import normalize
from attribution import REL, attribute, comorbidity_load, doc_level_graph_with_pairs
from eval_ablation import FULL
from eval_alias_heldout import key_forms
from eval_alias_public import set_drug_alias
from eval_external_extra import PROCESSED
from eval_external_kg import DISEASE, disease_classes, neighbours, rank_metrics, supported
from eval_g_raw_vs_g_norm import collect_freq
from eval_query_level import boot_mean_diff, combine_sources
from eval_robustness import boot_ci, load_sources
from ner_data import load_bio_docs
from normalize import build_canonical_map, configure_normalization

DICTS = ROOT / "data" / "dicts"
OUT = ROOT / "outputs" / "metrics"
METHODS = (("count", "共现计数"), ("lift", "lift"), ("oneshot", "一次分配(lift 加权)"),
           ("em_nobg", "EM(无背景无平滑)"), ("em", "EM(无平滑)"), ("em_flat", "EM+平滑(≈Li 2019 MCEM)"),
           ("em_tuned", "EM+平滑+ICD层级"))
TUNE_FILE = OUT / "attribution_tune.json"
ALPHAS = (0, 2, 5, 10, 20)
ITERS = (3, 10, 30)
GROUP_LEVELS = ("none", "cat3", "cat2", "cat2_c")
GLOBAL_KS = (20, 50, 100, 200)
N_SHUFFLE = 50
N_BOOT = 2000
JUDGES = (("combined", "合并"), ("cmeie", "CMeIE"), ("medicalkg", "寻医问药"))
SHOWCASE = ("高血压", "原发性高血压", "2型糖尿病", "糖尿病", "冠状动脉粥样硬化性心脏病", "肝继发恶性肿瘤", "肺继发恶性肿瘤")


def leak_free_table() -> dict[str, str]:
    hand = json.loads((DICTS / "drug_alias.json").read_text(encoding="utf-8"))["alias"]
    public = json.loads((DICTS / "drug_alias_public.json").read_text(encoding="utf-8"))["alias"]
    train = load_bio_docs(PROCESSED / "train.bio") + load_bio_docs(PROCESSED / "dev.bio")
    allowed = set().union(*(key_forms(s) for s in collect_freq(train).get("药物", {})))
    return {**public, **{k: v for k, v in hand.items() if key_forms(k) & allowed}}


def build(docs, table: dict | None):
    if table is None:
        return doc_level_graph_with_pairs(docs, None)
    configure_normalization(**FULL)
    set_drug_alias(normalize.get_index(), table)
    canon = build_canonical_map(collect_freq(docs))
    out = doc_level_graph_with_pairs(docs, canon)
    configure_normalization()
    return out


def icd_groups(g: dict, level: str) -> dict[str, str] | None:
    """Disease node → ICD group: cat3 = 3-character category (E11), cat2 = letter + first digit
    (E1 ≈ block E10–E14 diabetes, I1 ≈ hypertensive diseases), cat2_c = cat2 with all of chapter C as one."""
    if level == "none":
        return None
    idx = normalize.get_index()
    out = {}
    for nid in g["df"]:
        if nid.startswith(DISEASE + "::"):
            code = idx.preferred_code(nid.split("::", 1)[1])
            if code:
                out[nid] = "C" if level == "cat2_c" and code[0] == "C" else code[:3 if level == "cat3" else 2]
    return out


def all_scores(g: dict, pairs, tuned: dict) -> dict[str, dict]:
    n = g["n_docs"]
    co = {k: v for k, v in g["co"].items() if k[1] == REL}
    a, i = tuned["alpha"], tuned["iters"]
    return {"count": co,
            "lift": {(d, r, m): c * n / (g["df"][d] * g["df"][m]) for (d, r, m), c in co.items()},
            "oneshot": attribute(g, pairs, iters=1, background=False, init="lift"),
            "em_nobg": attribute(g, pairs, iters=30, background=False),
            "em": attribute(g, pairs, iters=30, background=True),
            "em_flat": attribute(g, pairs, iters=i, background=True, alpha=a),
            "em_tuned": attribute(g, pairs, iters=i, background=True, alpha=a, group=icd_groups(g, tuned["level"]))}


def tune(judges) -> dict:
    """Grid over (group level, alpha, iters) on train+dev gold G_norm; criterion = per-disease MAP, combined judge."""
    docs = load_bio_docs(PROCESSED / "train.bio") + load_bio_docs(PROCESSED / "dev.bio")
    g, pairs = build(docs, leak_free_table())
    ext = judges["combined"]
    grid = {}
    for lvl in GROUP_LEVELS:
        grp = icd_groups(g, lvl)
        for a in ALPHAS:
            for i in ITERS:
                grid[f"{lvl}_a{a}_i{i}"] = attribute(g, pairs, iters=i, background=True, alpha=a, group=grp)
    grid["count"] = {k: v for k, v in g["co"].items() if k[1] == REL}
    rows = per_disease(g, grid, ext)
    labels = labelled_edges(g, ext)
    res = {}
    for name, sc in grid.items():
        res[name] = {"MAP": sum(r[name]["AP"] for r in rows.values()) / len(rows),
                     "P@3": sum(r[name]["P@3"] for r in rows.values()) / len(rows),
                     "global@100": global_topk_fast(labels, sc, list(labels)).get(100, float("nan"))}
    best = max((k for k in res if k != "count"), key=lambda k: res[k]["MAP"])
    lvl, rest = best.rsplit("_a", 1)
    a, i = rest.split("_i")
    out = {"n_diseases": len(rows), "grid": res, "best": best, "level": lvl, "alpha": float(a), "iters": int(i)}
    TUNE_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: {m: round(v, 4) for m, v in r.items()} for k, r in res.items()}, ensure_ascii=False))
    print("best", best, res[best], "count", res["count"])
    return out


def ordered(cands, score, rng):
    return sorted(cands, key=lambda m: (-score[m], rng.random()))


def per_disease(g, scores, ext) -> dict[str, dict]:
    nb = neighbours(g, REL)
    rng = random.Random(42)
    rows = {}
    for d, cands in nb.items():
        if len(cands) < 2 or not (disease_classes(g, d, ext) & ext["subjects"][REL]):
            continue
        labels = {m: supported(g, d, m, REL, ext) for m in cands}
        if not any(labels.values()):
            continue
        rows[d] = {}
        for name, sc in scores.items():
            s = {m: sc[(d, REL, m)] for m in cands}
            acc = defaultdict(float)
            for _ in range(N_SHUFFLE):
                for k, v in rank_metrics([labels[m] for m in ordered(cands, s, rng)]).items():
                    acc[k] += v / N_SHUFFLE
            rows[d][name] = dict(acc)
    return rows


def labelled_edges(g, ext) -> dict[tuple, bool]:
    nb = neighbours(g, REL)
    out = {}
    for d, cands in nb.items():
        if disease_classes(g, d, ext) & ext["subjects"][REL]:
            for m in cands:
                out[(d, REL, m)] = supported(g, d, m, REL, ext)
    return out


def global_topk(labels: dict, score: dict, keys=None) -> dict[int, float]:
    keys = list(labels) if keys is None else keys
    rng = random.Random(5)
    out = {}
    for k in GLOBAL_KS:
        if k > len(keys):
            continue
        acc = 0.0
        for _ in range(N_SHUFFLE):
            top = sorted(keys, key=lambda e: (-score[e], rng.random()))[:k]
            acc += sum(labels[e] for e in top) / k / N_SHUFFLE
        out[k] = acc
    return out


def global_boot(labels, scores, method) -> dict[int, list[float]]:
    by_d = defaultdict(list)
    for e in labels:
        by_d[e[0]].append(e)
    diseases = list(by_d)
    rng = random.Random(13)
    diffs = defaultdict(list)
    for _ in range(200):
        sample = [e for _ in diseases for e in by_d[diseases[rng.randrange(len(diseases))]]]
        a = global_topk_fast(labels, scores["count"], sample)
        b = global_topk_fast(labels, scores[method], sample)
        for k in a:
            if k in b:
                diffs[k].append(b[k] - a[k])
    return {k: boot_ci(v) for k, v in diffs.items()}


def global_topk_fast(labels, score, keys) -> dict[int, float]:
    """Expected precision@k with ties resolved by expectation (no shuffling)."""
    by = sorted(keys, key=lambda e: -score[e])
    out = {}
    for k in GLOBAL_KS:
        if k > len(by):
            continue
        cut = score[by[k - 1]]
        above = [labels[e] for e in by if score[e] > cut]
        tie = [labels[e] for e in by if score[e] == cut]
        out[k] = (sum(above) + (k - len(above)) * sum(tie) / len(tie)) / k
    return out


def evaluate(g, pairs, judges, tuned) -> dict:
    scores = all_scores(g, pairs, tuned)
    load = comorbidity_load(pairs)
    res = {}
    for j, ext in judges.items():
        rows = per_disease(g, scores, ext)
        metrics = ("P@1", "P@3", "P@5", "AP")
        summ = {name: {k: sum(r[name][k] for r in rows.values()) / len(rows) for k in metrics} for name in scores}
        vs = {name: {k: {"delta": summ[name][k] - summ["count"][k],
                         "ci95": boot_mean_diff([r[name][k] - r["count"][k] for r in rows.values()], 31 + i)}
                     for i, k in enumerate(("P@1", "P@3", "AP"))} for name in scores if name != "count"}
        vs["hier_vs_flat"] = {k: {"delta": summ["em_tuned"][k] - summ["em_flat"][k],
                                  "ci95": boot_mean_diff([r["em_tuned"][k] - r["em_flat"][k] for r in rows.values()], 41 + i)}
                              for i, k in enumerate(("P@1", "P@3", "AP"))}
        labels = labelled_edges(g, ext)
        glob = {name: global_topk_fast(labels, sc, list(labels)) for name, sc in scores.items()}
        glob_ci = {name: global_boot(labels, scores, name) for name in ("oneshot", "em_flat", "em_tuned")}
        ds = sorted(rows, key=lambda d: load.get(d, 0))
        third = max(1, len(ds) // 3)
        strata = {}
        for sname, part in (("低", ds[:third]), ("中", ds[third:2 * third]), ("高", ds[2 * third:])):
            if part:
                strata[sname] = {"n": len(part), "load": sum(load.get(d, 0) for d in part) / len(part),
                                 **{name: sum(rows[d][name]["P@3"] for d in part) / len(part) for name in scores}}
        res[j] = {"n_diseases": len(rows), "n_edges": len(labels), "n_supported": sum(labels.values()),
                  "per_disease": summ, "vs_count": vs, "global": glob, "global_ci_vs_count": glob_ci, "strata": strata}
    return res, scores


def showcase(g, scores) -> list[str]:
    nb = neighbours(g, REL)
    lines = []
    for name in SHOWCASE:
        d = f"{DISEASE}::{name}"
        if d not in nb:
            continue
        cells = []
        for method in ("count", "em", "em_tuned"):
            top = sorted(nb[d], key=lambda m: -scores[method][(d, REL, m)])[:4]
            cells.append(f"{method}: " + "、".join(f"{m.split('::', 1)[1]}({scores[method][(d, REL, m)]:.1f})" for m in top))
        lines.append(f"  {name}（{g['df'][d]} 份）  " + "   ".join(cells))
    return lines


def ci(c) -> str:
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def report(res, tuned) -> str:
    L = ["改进一：药物→疾病归因建图 vs 共现计数（同一批边，只改边的权重；外部库判定）", "=" * 80,
         f"EM 参数在训练+开发集标注实体上选（按每病 MAP）：层级 {tuned['level']}，α={tuned['alpha']:g}，迭代 {tuned['iters']} 次；"
         f"调参集上 MAP {tuned['grid'][tuned['best']]['MAP']:.3f}，共现计数 {tuned['grid']['count']['MAP']:.3f}"]
    for set_name, title in (("test_pred_ner", "测试集 预测实体（379 份）"), ("test_gold_ner", "测试集 标注实体（379 份）")):
        r = res[set_name]
        for gname in ("g_norm", "g_raw"):
            L += ["", f"【{title} · {'标准化图谱' if gname == 'g_norm' else '原始图谱'}】"]
            for j, jt in JUDGES:
                e = r[gname][j]
                L.append(f"  [{jt}] 评测疾病 {e['n_diseases']} 个，边 {e['n_edges']} 条（被印证 {e['n_supported']}）")
                L.append(f"    {'方法':18s}{'P@1':>7s}{'P@3':>7s}{'P@5':>7s}{'MAP':>7s}   全局前 k 条精确率 " +
                         " ".join(f"k={k}" for k in GLOBAL_KS))
                for m, mt in METHODS:
                    p = e["per_disease"][m]
                    gl = " ".join(f"{e['global'][m].get(k, float('nan')):.3f}" for k in GLOBAL_KS)
                    L.append(f"    {mt:16s}{p['P@1']:7.3f}{p['P@3']:7.3f}{p['P@5']:7.3f}{p['AP']:7.3f}   {gl}")
                for m in ("oneshot", "em_flat", "em_tuned"):
                    v = e["vs_count"][m]
                    g_ci = e["global_ci_vs_count"][m]
                    L.append(f"    {dict(METHODS)[m]} − 共现计数：P@1 {v['P@1']['delta']:+.3f}{ci(v['P@1']['ci95'])}  "
                             f"P@3 {v['P@3']['delta']:+.3f}{ci(v['P@3']['ci95'])}  MAP {v['AP']['delta']:+.3f}{ci(v['AP']['ci95'])}  "
                             f"全局 k=50 {ci(g_ci.get(50, [float('nan')] * 2))} k=100 {ci(g_ci.get(100, [float('nan')] * 2))}")
                v = e["vs_count"]["hier_vs_flat"]
                L.append(f"    层级分组的单独贡献（层级 − 无层级）：P@1 {v['P@1']['delta']:+.3f}{ci(v['P@1']['ci95'])}  "
                         f"P@3 {v['P@3']['delta']:+.3f}{ci(v['P@3']['ci95'])}  MAP {v['AP']['delta']:+.3f}{ci(v['AP']['ci95'])}")
                if gname == "g_norm" and j == "combined":
                    L.append("    按合并症负担分层（P@3）：" + "  ".join(
                        f"{s}(n={v['n']}, 平均合并病 {v['load']:.1f}) 计数 {v['count']:.3f} → 加层级 {v['em_tuned']:.3f}"
                        for s, v in e["strata"].items()))
        L += ["", "  2×2（合并裁判，全局前 50 / 100 条精确率）"]
        for gname, gt in (("g_raw", "原始图谱"), ("g_norm", "标准化图谱")):
            gl = r[gname]["combined"]["global"]
            L.append(f"    {gt}: 计数 {gl['count'].get(50, float('nan')):.3f} / {gl['count'].get(100, float('nan')):.3f}   "
                     f"Li 2019 式 {gl['em_flat'].get(50, float('nan')):.3f} / {gl['em_flat'].get(100, float('nan')):.3f}   "
                     f"加层级 {gl['em_tuned'].get(50, float('nan')):.3f} / {gl['em_tuned'].get(100, float('nan')):.3f}")
    if res.get("showcase"):
        L += ["", "全部 1379 份标注实体上的示例：合并症疾病的前 4 个药（计数 / EM 无平滑 / EM+平滑+层级；括号内为期望计数）",
              *res["showcase"]]
    return "\n".join(L) + "\n"


def main() -> None:
    sources = load_sources()
    judges = {"combined": combine_sources(sources), **sources}
    if "--tune" in sys.argv or not TUNE_FILE.exists():
        tune(judges)
        if "--tune" in sys.argv:
            return
    tuned = json.loads(TUNE_FILE.read_text(encoding="utf-8"))
    table = leak_free_table()
    doc_sets = {"test_pred_ner": load_bio_docs(PROCESSED / "test_pred.bio"),
                "test_gold_ner": load_bio_docs(PROCESSED / "test.bio")}
    res = {}
    for set_name, docs in doc_sets.items():
        res[set_name] = {}
        for gname, tbl in (("g_norm", table), ("g_raw", None)):
            g, pairs = build(docs, tbl)
            res[set_name][gname], scores = evaluate(g, pairs, judges, tuned)
            c = res[set_name][gname]["combined"]
            print(set_name, gname, {m: round(c["per_disease"][m]["P@3"], 4) for m, _ in METHODS},
                  {m: round(c["global"][m].get(100, -1), 4) for m, _ in METHODS}, flush=True)
    all_docs = sum((load_bio_docs(PROCESSED / f"{s}.bio") for s in ("train", "dev", "test")), [])
    g, pairs = build(all_docs, table)
    res["showcase"] = showcase(g, all_scores(g, pairs, tuned))
    text = report(res, tuned)
    (OUT / "attribution.txt").write_text(text, encoding="utf-8")
    (OUT / "attribution.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

