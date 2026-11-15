"""Improvement 3b: check disease merges against one external disease→drug list, judge with the other.

A merge s → G is checked by whether the drugs given with s in the notes are ones the checking KB lists for G
(graph_feedback.kb_merge_scores). To avoid grading the check with the list it used, the two KBs swap roles:
check with 寻医问药 and judge with CMeIE, and the reverse.

For each direction:
1. Injected errors (surfaces with note drugs merged into a random wrong KB-covered node): AUC and share vetoed.
2. Selection on train+dev gold notes only: (tau, min_docs) by the paired support rate under the judging KB.
3. Test notes (predicted and gold entities): G_norm vs G_fb under the judging KB; list of vetoed merges.
Writes outputs/metrics/graph_feedback_kb.{txt,json}.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_attribution import OUT
from eval_external_extra import PROCESSED
from eval_external_kg import doc_level_graph, ext_key, paired_support
from eval_graph_feedback import N_SEEDS, REL, ci, compare, injected, setup, vetoed_rows
from eval_robustness import load_sources
from graph_feedback import apply_feedback, kb_merge_scores, surface_profiles
from ner_data import load_bio_docs
from normalize import configure_normalization

DIS = "疾病和诊断"
CTX = "treat"
TAUS = (0.25, 0.4, 0.45, 0.5)
MIN_DOCS = (1, 2)
DIRECTIONS = (("medicalkg", "cmeie"), ("cmeie", "medicalkg"))
KB_NAMES = {"medicalkg": "寻医问药", "cmeie": "CMeIE"}


def covered_nodes(base, ext) -> set[str]:
    subj = ext["subjects"][REL]
    names = {}
    for (lb, s), nid in base["member"].items():
        if lb == DIS:
            names.setdefault(nid, {nid.split("::", 1)[1]}).add(s)
    return {nid for nid, ns in names.items() if any(ext["uf"].find(ext_key(n)) in subj for n in ns)}


def inject_runs(docs, canon, base, ext, seed0: int) -> list[dict]:
    covered = covered_nodes(base, ext)
    has_drug = lambda k: any(x.startswith("药物::") for x in base["prof"][k])
    return [injected(docs, canon, base, seed0 + s, ctx_mode=CTX, scorer=lambda p: kb_merge_scores(p, ext),
                     labels=(DIS,), cand_ok=has_drug, node_ok=lambda n: n in covered, taus=TAUS, min_docs=MIN_DOCS)[DIS]
            for s in range(N_SEEDS)]


def main() -> None:
    sources = load_sources()
    sets = {"train_dev": load_bio_docs(PROCESSED / "train.bio") + load_bio_docs(PROCESSED / "dev.bio"),
            "test_pred_ner": load_bio_docs(PROCESSED / "test_pred.bio"),
            "test_gold_ner": load_bio_docs(PROCESSED / "test.bio")}
    res = {}
    for check, judge in DIRECTIONS:
        ext, jext = sources[check], sources[judge]
        r = res[f"{check}->{judge}"] = {}
        docs = sets["train_dev"]
        canon = setup(docs)
        base = surface_profiles(docs, canon, ctx_mode=CTX)
        scores = kb_merge_scores(base, ext)
        r["train_dev"] = {"n_scored": len(scores), "injected": inject_runs(docs, canon, base, ext, 0), "grid": {}}
        g_norm = doc_level_graph(docs, canon)
        for t in TAUS:
            for m in MIN_DOCS:
                canon_fb, vetoed = apply_feedback(canon, scores, t, m)
                p = paired_support({"g_raw": g_norm, "g_norm": doc_level_graph(docs, canon_fb)}, REL, jext)
                r["train_dev"]["grid"][f"{t}_{m}"] = {"n_vetoed": len(vetoed), "delta": p["delta"],
                                                      "ci95": list(p["delta_ci95"]), "before": p["g_raw"]["rate"],
                                                      "after": p["g_norm"]["rate"]}
                print(check, "grid", t, m, len(vetoed), round(p["delta"], 4), flush=True)
        grid = r["train_dev"]["grid"]
        best = max(grid, key=lambda k: (grid[k]["delta"], -grid[k]["n_vetoed"]))
        tau, min_docs = float(best.split("_")[0]), int(best.split("_")[1])
        r["best"] = {"tau": tau, "min_docs": min_docs}
        configure_normalization()

        for set_name in ("test_pred_ner", "test_gold_ner"):
            docs = sets[set_name]
            canon = setup(docs)
            base = surface_profiles(docs, canon, ctx_mode=CTX)
            scores = kb_merge_scores(base, ext)
            canon_fb, vetoed = apply_feedback(canon, scores, tau, min_docs)
            r[set_name] = {"n_scored": len(scores), "n_vetoed": len(vetoed),
                           "compare": compare(docs, canon, canon_fb, {judge: jext}, query_on=(judge,))[judge],
                           "vetoed": vetoed_rows(vetoed, base["prof"])}
            if set_name == "test_pred_ner":
                r[set_name]["injected"] = inject_runs(docs, canon, base, ext, 100)
            configure_normalization()
            print(check, set_name, len(vetoed), flush=True)

        docs = sets["train_dev"] + sets["test_gold_ner"]
        canon = setup(docs)
        base = surface_profiles(docs, canon, ctx_mode=CTX)
        _, vetoed = apply_feedback(canon, kb_merge_scores(base, ext), tau, min_docs)
        r["all_gold_vetoed"] = vetoed_rows(vetoed, base["prof"])
        configure_normalization()

    text = report(res)
    (OUT / "graph_feedback_kb.txt").write_text(text, encoding="utf-8")
    (OUT / "graph_feedback_kb.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                                encoding="utf-8")
    print(text)


def inj_line(runs, key) -> str:
    a = [x["auc"] for x in runs]
    vi = sum(x["veto"][key]["injected"] for x in runs) / len(runs)
    vo = sum(x["veto"][key]["own"] for x in runs) / len(runs)
    th = sum(x["top_hit"] for x in runs) / len(runs)
    br = sum(x["base_rate"] for x in runs) / len(runs)
    n = runs[0]["n_injected_scored"]
    return (f"AUC {sum(a) / len(a):.3f}（{min(a):.3f}–{max(a):.3f}），可检查的注入错误约 {n} 个；"
            f"最可疑的前 {n} 个里错误占 {th:.0%}（随便抽是 {br:.0%}）；"
            f"按选中阈值，注入错误被拦下 {vi:.0%}，流程自己的合并被拦下 {vo:.0%}")


def report(res) -> str:
    L = ["改进三(B)：用外部知识库的药物清单检查疾病合并（病历里给这个名字用的药，知识库是否列在被合并的节点下）",
         "用一个知识库检查、另一个知识库打分，避免“自己考自己”", "=" * 80]
    for d, r in res.items():
        check, judge = d.split("->")
        b = r["best"]
        key = f"{b['tau']}_{b['min_docs']}"
        L += ["", f"【用 {KB_NAMES[check]} 检查，用 {KB_NAMES[judge]} 打分】",
              f"一、注入错误测试（只注入能检查的：名字在病历里有用药，错误节点在知识库里）",
              f"  训练+开发：{inj_line(r['train_dev']['injected'], key)}",
              f"  测试集 预测实体：{inj_line(r['test_pred_ner']['injected'], key)}",
              f"二、参数选择（训练+开发；可检查的合并 {r['train_dev']['n_scored']} 个；{KB_NAMES[judge]} 边印证率 撤销后 − 撤销前）"]
        for k, v in r["train_dev"]["grid"].items():
            t, m = k.split("_")
            L.append(f"  排名分位 < {t}，至少 {m} 份病历：撤销 {v['n_vetoed']}  印证率 {v['before']:.3f} → {v['after']:.3f}"
                     f"  差 {v['delta']:+.3f} {ci(v['ci95'])}")
        L.append(f"  选中：排名分位 < {b['tau']}，至少 {b['min_docs']} 份病历")
        for set_name, title in (("test_pred_ner", "测试集 预测实体"), ("test_gold_ner", "测试集 标注实体")):
            x = r[set_name]
            c = x["compare"]
            s = c["support"]
            L.append(f"三、{title}：可检查 {x['n_scored']}，撤销 {x['n_vetoed']}；{KB_NAMES[judge]} 边印证率 "
                     f"{s['before']:.3f} → {s['after']:.3f}  差 {s['delta']:+.3f} {ci(s['ci95'])}  边数 {s['edges'][0]} → {s['edges'][1]}")
            q = c["query"]["single"]
            if q["n_queries"]:
                L.append(f"  按查询（{q['n_queries']} 个疾病）：" + "  ".join(
                    f"{k} {q[k]['g_raw']:.3f}→{q[k]['g_norm']:.3f} {q[k]['delta']:+.3f}{ci(q[k]['ci95'])}"
                    for k in ("P@1", "P@3", "P@5", "facts@5")))
            for row in x["vetoed"][:10]:
                L.append(f"    撤销 {row['surface']} ↛ {row['merged_into']}（来源 {row['source']}，{row['ndoc']} 份，分位 {row['pct']}）"
                         f" 同病历用药：{'、'.join(row['context'])}")
        L.append(f"四、全部 1379 份标注病历上会被撤销的合并（共 {len(r['all_gold_vetoed'])} 个，供人工核对）")
        for row in r["all_gold_vetoed"][:30]:
            L.append(f"  {row['surface']} ↛ {row['merged_into']}（来源 {row['source']}，{row['ndoc']} 份，分位 {row['pct']}）"
                     f" 同病历用药：{'、'.join(row['context'])}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
