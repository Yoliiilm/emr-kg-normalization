"""Improvement 3: does checking merges against the graph (src/graph_feedback.py) make G_norm more trustworthy?

0. Context type (graph_feedback.CTX_MODES) chosen on train+dev by the injected-error AUC.
1. Injected errors: K surfaces per type are deliberately merged into a random wrong node; how well does the
   rank percentile separate them from the pipeline's own merges (AUC, share of errors among the most suspect),
   and what share is vetoed?
2. Selection on train+dev gold notes only: (tau, min_docs) by the paired external support rate of
   G_fb vs G_norm (combined judge).
3. Test notes (predicted and gold entities): G_norm vs G_fb on edge support (paired over disease clusters),
   query-level P@k / facts@k and support at count thresholds; plus the list of vetoed merges for inspection.
Writes outputs/metrics/graph_feedback.{txt,json}.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import normalize
from eval_ablation import FULL
from eval_alias_public import set_drug_alias
from eval_attribution import JUDGES, OUT, leak_free_table
from eval_external_extra import PROCESSED
from eval_external_kg import doc_level_graph, paired_support
from eval_g_raw_vs_g_norm import collect_freq
from eval_query_level import combine_sources, query_eval, threshold_eval, cluster_edges
from eval_robustness import load_sources
from export_graph import node_id
from graph_feedback import CTX_MODES, LABELS, apply_feedback, merge_scores, surface_profiles
from ner_data import load_bio_docs
from normalize import build_canonical_map, configure_normalization, lookup_with_source

REL = "使用药物"
TAUS = (0.1, 0.25, 0.5)
MIN_DOCS = (2, 3)
K_INJECT = 30
N_SEEDS = 3


def setup(docs):
    configure_normalization(**FULL)
    set_drug_alias(normalize.get_index(), leak_free_table())
    return build_canonical_map(collect_freq(docs))


def auc(pos, neg) -> float:
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def injected(docs, canon, base, seed: int, ctx_mode: str = "all", scorer=merge_scores, labels=LABELS,
             cand_ok=lambda k: True, node_ok=lambda n: True, taus=TAUS, min_docs=MIN_DOCS) -> dict:
    """Merge K surfaces per type into a random wrong node; score injected vs the pipeline's own merges."""
    rng = random.Random(seed)
    inj = dict(canon)
    keys = set()
    for label in labels:
        cands = sorted(k for k in base["prof"] if k[0] == label and base["ndoc"][k] >= 2 and cand_ok(k))
        nodes = sorted({base["member"][k] for k in base["member"] if k[0] == label and node_ok(base["member"][k])})
        for k in rng.sample(cands, min(K_INJECT, len(cands))):
            target = rng.choice([n for n in nodes if n != base["member"][k]])
            inj[k] = target.split("::", 1)[1]
            keys.add(k)
    scores = scorer(surface_profiles(docs, inj, ctx_mode=ctx_mode))
    out = {}
    for label in labels:
        pos = [-s["pct"] for k, s in scores.items() if k in keys and k[0] == label]
        neg = [-s["pct"] for k, s in scores.items() if k not in keys and k[0] == label]
        ranked = sorted((k for k in scores if k[0] == label), key=lambda k: (scores[k]["pct"], -scores[k]["ndoc"]))
        top = ranked[:len(pos)]
        out[label] = {"n_injected_scored": len(pos), "n_own": len(neg), "auc": auc(pos, neg),
                      "top_hit": sum(k in keys for k in top) / max(len(top), 1),
                      "base_rate": len(pos) / max(len(pos) + len(neg), 1),
                      "veto": {f"{t}_{m}": {"injected": sum(1 for k, s in scores.items() if k in keys and k[0] == label
                                                          and s["ndoc"] >= m and s["pct"] < t) / max(len(pos), 1),
                                            "own": sum(1 for k, s in scores.items() if k not in keys and k[0] == label
                                                       and s["ndoc"] >= m and s["pct"] < t) / max(len(neg), 1)}
                               for t in taus for m in min_docs}}
    return out


def compare(docs, canon, canon_fb, judges, query_on=("combined",)) -> dict:
    graphs = {"g_raw": doc_level_graph(docs, canon), "g_norm": doc_level_graph(docs, canon_fb)}
    res = {"n_nodes": {k: len(g["df"]) for k, g in graphs.items()}}
    for j, ext in judges.items():
        p = paired_support(graphs, REL, ext)
        res[j] = {"support": {"before": p["g_raw"]["rate"], "after": p["g_norm"]["rate"], "delta": p["delta"],
                              "ci95": list(p["delta_ci95"]), "edges": [p["g_raw"]["edges"], p["g_norm"]["edges"]]}}
        if j in query_on:
            res[j]["query"] = query_eval(graphs, ext)
            res[j]["threshold"] = threshold_eval(cluster_edges(graphs, ext))
    return res


def vetoed_rows(vetoed, prof) -> list[dict]:
    rows = []
    for (label, s), sc in sorted(vetoed, key=lambda v: (-v[1]["ndoc"], v[1]["pct"])):
        top = [x.split("::", 1)[1] for x, _ in prof[(label, s)].most_common(6)]
        rows.append({"label": label, "surface": s, "merged_into": sc["node"].split("::", 1)[1],
                     "source": lookup_with_source(s, label)[1], "ndoc": sc["ndoc"], "pct": round(sc["pct"], 3),
                     "context": top})
    return rows


def main() -> None:
    sources = load_sources()
    judges = {"combined": combine_sources(sources), **sources}
    sets = {"train_dev": load_bio_docs(PROCESSED / "train.bio") + load_bio_docs(PROCESSED / "dev.bio"),
            "test_pred_ner": load_bio_docs(PROCESSED / "test_pred.bio"),
            "test_gold_ner": load_bio_docs(PROCESSED / "test.bio")}
    res = {}

    docs = sets["train_dev"]
    canon = setup(docs)
    res["modes"] = {}
    for m in CTX_MODES:
        b = surface_profiles(docs, canon, ctx_mode=m)
        runs = [injected(docs, canon, b, s, ctx_mode=m) for s in range(N_SEEDS)]
        res["modes"][m] = {"n_scored": Counter(k[0] for k in merge_scores(b)),
                           "auc": {lb: sum(r[lb]["auc"] for r in runs) / len(runs) for lb in LABELS}}
    mode = max(res["modes"], key=lambda m: sum(res["modes"][m]["auc"].values()))
    res["ctx_mode"] = mode
    base = surface_profiles(docs, canon, ctx_mode=mode)
    scores = merge_scores(base)
    res["train_dev"] = {"n_scored": Counter(k[0] for k in scores),
                        "injected": [injected(docs, canon, base, s, ctx_mode=mode) for s in range(N_SEEDS)], "grid": {}}
    for t in TAUS:
        for m in MIN_DOCS:
            canon_fb, vetoed = apply_feedback(canon, scores, t, m)
            p = paired_support({"g_raw": doc_level_graph(docs, canon), "g_norm": doc_level_graph(docs, canon_fb)},
                               REL, judges["combined"])
            res["train_dev"]["grid"][f"{t}_{m}"] = {"n_vetoed": Counter(k[0] for k, _ in vetoed), "delta": p["delta"],
                                                    "ci95": list(p["delta_ci95"]), "before": p["g_raw"]["rate"],
                                                    "after": p["g_norm"]["rate"]}
            print("grid", t, m, len(vetoed), round(p["delta"], 4), flush=True)
    grid = res["train_dev"]["grid"]
    best = max(grid, key=lambda k: (grid[k]["delta"], -sum(grid[k]["n_vetoed"].values())))
    tau, min_docs = float(best.split("_")[0]), int(best.split("_")[1])
    res["best"] = {"tau": tau, "min_docs": min_docs}
    configure_normalization()

    for set_name in ("test_pred_ner", "test_gold_ner"):
        docs = sets[set_name]
        canon = setup(docs)
        base = surface_profiles(docs, canon, ctx_mode=mode)
        scores = merge_scores(base)
        canon_fb, vetoed = apply_feedback(canon, scores, tau, min_docs)
        res[set_name] = {"n_scored": Counter(k[0] for k in scores), "n_vetoed": Counter(k[0] for k, _ in vetoed),
                         "compare": compare(docs, canon, canon_fb, judges), "vetoed": vetoed_rows(vetoed, base["prof"])}
        if set_name == "test_pred_ner":
            res[set_name]["injected"] = [injected(docs, canon, base, 100 + s, ctx_mode=mode) for s in range(N_SEEDS)]
        configure_normalization()
        print(set_name, res[set_name]["n_vetoed"], flush=True)

    all_docs = sets["train_dev"] + sets["test_gold_ner"]
    canon = setup(all_docs)
    base = surface_profiles(all_docs, canon, ctx_mode=mode)
    _, vetoed = apply_feedback(canon, merge_scores(base), tau, min_docs)
    res["all_gold_vetoed"] = vetoed_rows(vetoed, base["prof"])
    configure_normalization()

    text = report(res)
    (OUT / "graph_feedback.txt").write_text(text, encoding="utf-8")
    (OUT / "graph_feedback.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(text)


def ci(c) -> str:
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def inj_line(runs, label, key) -> str:
    a = [r[label]["auc"] for r in runs]
    vi = [r[label]["veto"][key]["injected"] for r in runs]
    vo = [r[label]["veto"][key]["own"] for r in runs]
    n = runs[0][label]["n_injected_scored"]
    th = sum(r[label]["top_hit"] for r in runs) / len(runs)
    br = sum(r[label]["base_rate"] for r in runs) / len(runs)
    return (f"{label}：AUC {sum(a) / len(a):.3f}（{min(a):.3f}–{max(a):.3f}），可打分的注入错误约 {n} 个；"
            f"最可疑的前 {n} 个里错误占 {th:.0%}（随便抽是 {br:.0%}）；"
            f"按选中阈值，注入错误被拦下 {sum(vi) / len(vi):.0%}，流程自己的合并被拦下 {sum(vo) / len(vo):.0%}")


def report(res) -> str:
    b = res["best"]
    key = f"{b['tau']}_{b['min_docs']}"
    mode_names = {"all": "同病历里的所有实体", "treat": "只看用药（疾病看药，药看疾病）", "treat_surg": "用药+手术"}
    L = ["改进三：用图谱反过来检查标准化（名字的病历上下文与被合并的节点不像，就撤销这次合并）", "=" * 80,
         "〇、上下文用什么（只用训练+开发，按注入错误 AUC 选）"]
    for m, v in res["modes"].items():
        L.append(f"  {mode_names[m]}：可检查的合并 {dict(v['n_scored'])}，AUC " +
                 "  ".join(f"{lb} {a:.3f}" for lb, a in v["auc"].items()))
    L += [f"  选中：{mode_names[res['ctx_mode']]}", "",
          "一、注入错误测试（故意把名字合并到随机的错误节点，看能否识别；AUC=把错误合并排在流程自身合并之前的概率）"]
    for set_name, title in (("train_dev", "训练+开发"), ("test_pred_ner", "测试集 预测实体")):
        runs = res[set_name]["injected"]
        L.append(f"  [{title}，{len(runs)} 次]")
        for label in LABELS:
            L.append("    " + inj_line(runs, label, key))
    L += ["", "二、参数选择（只用训练+开发 标注实体；合并裁判，撤销后 − 撤销前 的边印证率）"]
    for k, v in res["train_dev"]["grid"].items():
        t, m = k.split("_")
        L.append(f"  排名分位 < {t}，至少 {m} 份病历：撤销 {dict(v['n_vetoed'])}  印证率 {v['before']:.3f} → {v['after']:.3f}"
                 f"  差 {v['delta']:+.3f} {ci(v['ci95'])}")
    L.append(f"  选中：排名分位 < {b['tau']}，至少 {b['min_docs']} 份病历")
    for set_name, title in (("test_pred_ner", "测试集 预测实体"), ("test_gold_ner", "测试集 标注实体")):
        r = res[set_name]
        c = r["compare"]
        L += ["", f"三、{title}：可检查的合并 {dict(r['n_scored'])}，撤销 {dict(r['n_vetoed'])}；节点数 {c['n_nodes']['g_raw']} → {c['n_nodes']['g_norm']}"]
        for j, jt in JUDGES:
            s = c[j]["support"]
            L.append(f"  {jt:5s} 边印证率 {s['before']:.3f} → {s['after']:.3f}  差 {s['delta']:+.3f} {ci(s['ci95'])}  边数 {s['edges'][0]} → {s['edges'][1]}")
        q = c["combined"]["query"]["single"]
        if q["n_queries"]:
            L.append(f"  合并裁判 按查询（{q['n_queries']} 个疾病）：" + "  ".join(
                f"{k} {q[k]['g_raw']:.3f}→{q[k]['g_norm']:.3f} {q[k]['delta']:+.3f}{ci(q[k]['ci95'])}" for k in ("P@1", "P@3", "P@5", "facts@5")))
        t = c["combined"]["threshold"]
        L.append("  合并裁判 按共现阈值：" + "  ".join(
            f"≥{k}: 印证率 {t[k]['rate'][0]:.3f}→{t[k]['rate'][1]:.3f} 事实 {t[k]['facts'][0]}→{t[k]['facts'][1]}" for k in (1, 2, 3) if k in t))
        for row in r["vetoed"][:15]:
            L.append(f"    撤销 [{row['label']}] {row['surface']} ↛ {row['merged_into']}（来源 {row['source']}，{row['ndoc']} 份，分位 {row['pct']}）"
                     f" 上下文：{'、'.join(row['context'])}")
    L += ["", f"四、全部 1379 份标注病历上会被撤销的合并（共 {len(res['all_gold_vetoed'])} 个，按病历数排序，供人工核对）"]
    for row in res["all_gold_vetoed"][:40]:
        L.append(f"  [{row['label']}] {row['surface']} ↛ {row['merged_into']}（来源 {row['source']}，{row['ndoc']} 份，分位 {row['pct']}）"
                 f" 上下文：{'、'.join(row['context'])}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
