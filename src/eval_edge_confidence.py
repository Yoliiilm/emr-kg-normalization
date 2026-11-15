"""Improvement 2: statistical confidence for 疾病→药物 edges.

Three confidence measures per co-occurrence edge (d, m) with count c, expected E = df_d · df_m / n:
  fisher   one-sided hypergeometric p of ≥ c shared notes under independence, Benjamini–Hochberg q over all
           使用药物 edges; confident = q < 0.05 and c ≥ 2 (with c = 1 a pair of rare names is "significant"
           just because both are rare)
  eb05     Gamma–Poisson shrinker (DuMouchel 1999, single-gamma prior fitted by zero-truncated ML): 5th
           percentile of the posterior of c / E; confident = EB05 ≥ 2, the usual pharmacovigilance cutoff
  eb05_attr  the same on the attributed count from src/attribution.py (Li et al. 2019 use MCEM + EB05)

Questions, on G_raw vs G_norm and a random-merge negative control (same merge-group sizes as G_norm):
  1. how many edges are confident, and how many of those the external judges support
  2. are confident edges more often supported than the others (gap, bootstrap over diseases)
  3. does the continuous score rank supported edges higher than the raw count (AUC), and is it calibrated
Writes outputs/metrics/edge_confidence.{txt,json}.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import false_discovery_control, gamma, hypergeom

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import normalize
from attribution import REL, attribute, doc_level_graph_with_pairs
from eval_ablation import FULL
from eval_alias_public import set_drug_alias
from eval_attribution import JUDGES, OUT, TUNE_FILE, labelled_edges, leak_free_table
from eval_external_extra import PROCESSED
from eval_external_kg import DISEASE
from eval_g_raw_vs_g_norm import collect_freq
from eval_query_level import combine_sources
from eval_robustness import DRUG, boot_ci, load_sources, random_canon
from ner_data import load_bio_docs
from normalize import build_canonical_map, configure_normalization

MEASURES = (("fisher", "超几何+FDR"), ("eb05", "EB05(计数)"), ("eb05_attr", "EB05(归因计数)"))
EB_BINS = ((0, 1), (1, 2), (2, 4), (4, math.inf))
N_BOOT = 1000
N_RANDOM = 10


def fit_gps(c: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    """ML fit of a Gamma(a, b) prior on λ for c ~ Poisson(λ·E), zero-truncated (only observed pairs)."""
    def nll(x):
        a, b = np.exp(x)
        logp = gammaln(a + c) - gammaln(a) - gammaln(c + 1) + a * np.log(b / (b + e)) + c * np.log(e / (b + e))
        p0 = (b / (b + e)) ** a
        return -(logp - np.log1p(-p0)).sum()
    res = minimize(nll, x0=np.log([1.0, 0.1]), method="Nelder-Mead", options={"maxiter": 2000})
    a, b = np.exp(res.x)
    return float(a), float(b)


def confidences(g: dict, attr: dict) -> dict[str, dict]:
    n = g["n_docs"]
    keys = [k for k in g["co"] if k[1] == REL]
    c = np.array([g["co"][k] for k in keys], dtype=float)
    dd = np.array([g["df"][k[0]] for k in keys], dtype=float)
    dm = np.array([g["df"][k[2]] for k in keys], dtype=float)
    e = dd * dm / n
    p = hypergeom.sf(c - 1, n, dm, dd)
    q = false_discovery_control(p, method="bh")
    out = {"p": dict(zip(keys, p.tolist())), "q": dict(zip(keys, q.tolist()))}
    a, b = fit_gps(c, e)
    out["eb05"] = dict(zip(keys, gamma.ppf(0.05, a + c, scale=1 / (b + e)).tolist()))
    ca = np.array([attr[k] for k in keys])
    a2, b2 = fit_gps(np.maximum(np.round(ca), 1), e)
    out["eb05_attr"] = dict(zip(keys, gamma.ppf(0.05, a2 + ca, scale=1 / (b2 + e)).tolist()))
    out["prior"] = {"count": [a, b], "attr": [a2, b2]}
    out["confident"] = {"fisher": {k: out["q"][k] < 0.05 and g["co"][k] >= 2 for k in keys},
                        "eb05": {k: out["eb05"][k] >= 2 for k in keys},
                        "eb05_attr": {k: out["eb05_attr"][k] >= 2 for k in keys}}
    out["score"] = {"fisher": {k: -math.log10(max(out["p"][k], 1e-300)) if g["co"][k] >= 2 else 0.0 for k in keys},
                    "eb05": out["eb05"], "eb05_attr": out["eb05_attr"]}
    return out


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    neg = np.sort(np.array(neg))
    pos = np.array(pos)
    lo, hi = np.searchsorted(neg, pos, "left"), np.searchsorted(neg, pos, "right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))


def measure_cell(labels: dict, conf: dict, score: dict, count: dict) -> dict:
    by_d = defaultdict(list)
    for e in labels:
        by_d[e[0]].append(e)
    diseases = list(by_d)

    def stats(edges):
        sig = [labels[e] for e in edges if conf[e]]
        rest = [labels[e] for e in edges if not conf[e]]
        ys = [labels[e] for e in edges]
        a_s = auc([score[e] for e, y in zip(edges, ys) if y], [score[e] for e, y in zip(edges, ys) if not y])
        a_c = auc([count[e] for e, y in zip(edges, ys) if y], [count[e] for e, y in zip(edges, ys) if not y])
        return (len(sig), sum(sig), sum(sig) / len(sig) if sig else float("nan"),
                sum(rest) / len(rest) if rest else float("nan"), a_s, a_c)

    n_sig, n_sup, p_sig, p_rest, a_s, a_c = stats(list(labels))
    rng = random.Random(7)
    gaps, d_auc = [], []
    for _ in range(N_BOOT):
        st = stats([e for _ in diseases for e in by_d[diseases[rng.randrange(len(diseases))]]])
        if not (math.isnan(st[2]) or math.isnan(st[3])):
            gaps.append(st[2] - st[3])
        if not (math.isnan(st[4]) or math.isnan(st[5])):
            d_auc.append(st[4] - st[5])
    return {"n_conf": n_sig, "n_conf_supported": n_sup, "prec_conf": p_sig, "prec_rest": p_rest,
            "gap_ci95": boot_ci(gaps) if len(gaps) > N_BOOT // 2 else None,
            "auc": a_s, "auc_count": a_c, "auc_diff_ci95": boot_ci(d_auc) if len(d_auc) > N_BOOT // 2 else None}


def graph_cell(g, pairs, judges, tuned) -> dict:
    count = {k: v for k, v in g["co"].items() if k[1] == REL}
    attr = attribute(g, pairs, iters=tuned["iters"], background=True, alpha=tuned["alpha"])
    cf = confidences(g, attr)
    out = {"n_edges": len(count), "prior": cf["prior"],
           "n_conf": {m: sum(cf["confident"][m].values()) for m, _ in MEASURES}, "judges": {}}
    for j, ext in judges.items():
        labels = labelled_edges(g, ext)
        cell = {"n_labelled": len(labels), "n_supported": sum(labels.values())}
        for m, _ in MEASURES:
            cell[m] = measure_cell(labels, cf["confident"][m], cf["score"][m], count)
        if j == "combined":
            cell["eb_bins"] = []
            for lo, hi in EB_BINS:
                ys = [labels[e] for e in labels if lo <= cf["eb05"][e] < hi]
                cell["eb_bins"].append({"bin": [lo, hi], "n": len(ys), "rate": sum(ys) / len(ys) if ys else float("nan")})
        out["judges"][j] = cell
    return out


def random_cells(docs, canon, freq, judges, tuned) -> list[dict]:
    runs = []
    for seed in range(N_RANDOM):
        g, pairs = doc_level_graph_with_pairs(docs, random_canon(canon, freq, (DISEASE, DRUG), random.Random(seed)))
        cell = graph_cell(g, pairs, {"combined": judges["combined"]}, tuned)
        runs.append({"n_edges": cell["n_edges"], "n_conf": cell["n_conf"],
                     "combined": {m: {k: cell["judges"]["combined"][m][k] for k in ("n_conf", "n_conf_supported", "prec_conf", "auc")}
                                  for m, _ in MEASURES}})
    return runs


def evaluate(docs, judges, tuned, with_random: bool) -> dict:
    res = {"g_raw": graph_cell(*doc_level_graph_with_pairs(docs, None), judges, tuned)}
    configure_normalization(**FULL)
    set_drug_alias(normalize.get_index(), leak_free_table())
    freq = collect_freq(docs)
    canon = build_canonical_map(freq)
    res["g_norm"] = graph_cell(*doc_level_graph_with_pairs(docs, canon), judges, tuned)
    if with_random:
        res["random"] = random_cells(docs, canon, freq, judges, tuned)
    configure_normalization()
    return res


def fmt(x, nd=3) -> str:
    return "  nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def ci(c) -> str:
    return "[  n/a  ]" if not c else f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def rng_str(xs, nd=3) -> str:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return "nan" if not xs else (f"{min(xs)}–{max(xs)}" if nd == 0 else f"{min(xs):.{nd}f}–{max(xs):.{nd}f}")


def report(res) -> str:
    L = ["改进二：边的统计置信度（外部库判定是否被印证；精确率只在外部库覆盖的疾病上算）", "=" * 80,
         "可信边的判定：超几何+FDR = q<0.05 且共现 ≥2；EB05 = Gamma–Poisson 收缩后观测/期望比的 5% 下界 ≥2。",
         "AUC：该置信分数把被印证的边排在前面的能力，与直接用共现计数比较（差的 95% CI 按疾病自助）。"]
    for set_name, title in (("test_pred_ner", "测试集 预测实体（379 份）"), ("test_gold_ner", "测试集 标注实体（379 份）"),
                            ("all_gold_ner", "全部 1379 份 标注实体")):
        r = res[set_name]
        L += ["", f"【{title}】  边数 原始 {r['g_raw']['n_edges']} / 标准化 {r['g_norm']['n_edges']}"]
        for m, mt in MEASURES:
            L.append(f"  [{mt}] 可信边 原始 {r['g_raw']['n_conf'][m]} / 标准化 {r['g_norm']['n_conf'][m]}")
            for j, jt in JUDGES:
                for gname, gt in (("g_raw", "原始"), ("g_norm", "标准化")):
                    c = r[gname]["judges"][j]
                    x = c[m]
                    L.append(f"    {jt:5s} {gt:3s} 可判定 {c['n_labelled']:4d}（印证 {c['n_supported']:3d}）  可信 {x['n_conf']:4d}"
                             f"、其中印证 {x['n_conf_supported']:3d}，精确率 {fmt(x['prec_conf'])} vs 其余 {fmt(x['prec_rest'])}"
                             f"，差 {ci(x['gap_ci95'])}   AUC {fmt(x['auc'])} vs 计数 {fmt(x['auc_count'])}，差 {ci(x['auc_diff_ci95'])}")
        c = r["g_norm"]["judges"]["combined"]
        L.append("  EB05 校准（标准化图谱，合并裁判）：" + "  ".join(
            f"[{b['bin'][0]:g},{b['bin'][1]:g}) n={b['n']} 印证率 {fmt(b['rate'])}" for b in c["eb_bins"]))
        if "random" in r:
            runs = r["random"]
            L.append(f"  随机合并对照（{len(runs)} 次，组大小同标准化；合并裁判）：")
            for m, mt in MEASURES:
                real = r["g_norm"]["judges"]["combined"][m]
                raw = r["g_raw"]["judges"]["combined"][m]
                L.append(f"    [{mt}] 可信且被印证：原始 {raw['n_conf_supported']}，真实标准化 {real['n_conf_supported']}，随机 "
                         f"{rng_str([x['combined'][m]['n_conf_supported'] for x in runs], 0)}；可信边精确率：原始 "
                         f"{fmt(raw['prec_conf'])}，真实 {fmt(real['prec_conf'])}，随机 "
                         f"{rng_str([x['combined'][m]['prec_conf'] for x in runs])}；AUC 真实 {fmt(real['auc'])}，随机 "
                         f"{rng_str([x['combined'][m]['auc'] for x in runs])}")
    return "\n".join(L) + "\n"


def main() -> None:
    sources = load_sources()
    judges = {"combined": combine_sources(sources), **sources}
    tuned = json.loads(TUNE_FILE.read_text(encoding="utf-8"))
    doc_sets = {"test_pred_ner": load_bio_docs(PROCESSED / "test_pred.bio"),
                "test_gold_ner": load_bio_docs(PROCESSED / "test.bio"),
                "all_gold_ner": sum((load_bio_docs(PROCESSED / f"{s}.bio") for s in ("train", "dev", "test")), [])}
    res = {}
    for set_name, docs in doc_sets.items():
        res[set_name] = evaluate(docs, judges, tuned, with_random=set_name != "test_gold_ner")
        print(set_name, {g: res[set_name][g]["n_conf"] for g in ("g_raw", "g_norm")}, flush=True)
    text = report(res)
    (OUT / "edge_confidence.txt").write_text(text, encoding="utf-8")
    (OUT / "edge_confidence.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
