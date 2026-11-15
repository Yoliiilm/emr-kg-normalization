"""Multiple-comparison correction over the comparisons the report's conclusions rest on.

The family is fixed by hand: every paired comparison quoted as a finding in REPORT 2.x / 3.0–3.9,
significant or not (sub-analyses, per-judge and per-KB robustness checks are left out). Where a
script stored an exact p (sign-flip test, bootstrap one-sided share) it is used, two-sided;
otherwise p is recovered from the 95% bootstrap interval as in Altman & Bland (BMJ 2011):
SE = (upper − lower) / (2 × 1.96), z = estimate / SE. Holm controls the family-wise error rate,
Benjamini–Hochberg the false discovery rate, both at 0.05.

  python src/eval_multiplicity.py          # → outputs/metrics/multiplicity.{txt,json}
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "outputs" / "metrics"
ALPHA = 0.05

FAMILY = [
    ("2.4", "CCKS2019 用药查询 F1：标准化 − 原始", "bootstrap_query_f1", "relations/使用药物/bootstrap_by_disease", "boot"),
    ("2.4", "CCKS2019 手术查询 F1：标准化 − 原始", "bootstrap_query_f1", "relations/实施手术/bootstrap_by_disease", "boot"),
    ("2.10", "CCKS2017 用药查询 F1：标准化 − 原始", "ccks2017_graph", "results/off/rels/使用药物", "flip"),
    ("2.10", "CCKS2017 手术查询 F1：标准化 − 原始", "ccks2017_graph", "results/off/rels/实施手术", "flip"),
    ("2.10", "CCKS2017 用药：监督 CDN 优先 − 本方法", "ccks2017_graph", "results/cdn_primary/rels/使用药物/vs_off", "ci:delta_ci95"),
    ("2.5", "CCKS2019 用药：Jaccard 兜底 − 本方法", "jaccard_baseline", "graph/CCKS2019 test/使用药物/vs_ours", "ci:delta_ci95"),
    ("2.5", "CCKS2019 手术：Jaccard 兜底 − 本方法", "jaccard_baseline", "graph/CCKS2019 test/实施手术/vs_ours", "ci:delta_ci95"),
    ("2.11", "CCKS2019 用药：大模型兜底 − 本方法（固定口径）", "llm_norm", "graph/CCKS2019 test/llm_fallback/rels/使用药物/vs_off", "ci:delta_ci95"),
    ("2.11", "CCKS2019 手术：大模型兜底 − 本方法（固定口径）", "llm_norm", "graph/CCKS2019 test/llm_fallback/rels/实施手术/vs_off", "ci:delta_ci95"),
    ("2.13", "CCKS2017 手术查询 F1：否定过滤 − 不过滤", "assertion", "sets/CCKS2017/graphs/neg/rels/实施手术/paired_norm", "ci:delta_ci95"),
    ("2.13", "CCKS2017 外部库印证率：否定过滤 − 不过滤", "assertion", "sets/CCKS2017/graphs/neg/kb/medicalkg/paired", "ci:delta_ci95"),
    ("3.1", "外部库 P@1：标准化 − 原始", "query_level", "test_pred_ner/public_plus_train/combined/query/single/P@1", "ci:ci95"),
    ("3.2", "外部库 P@3：药物归因 − 共现计数", "attribution", "test_pred_ner/g_norm/combined/vs_count/em_tuned/P@3", "ci:ci95"),
    ("3.3", "可信边精确率 − 其余边（标准化图，1379 份）", "edge_confidence", "all_gold_ner/g_norm/judges/combined/fisher", "gap"),
    ("3.5", "盲评 严格 P@3：标准化 − 原始", "llm_judge", "precision/all_strict_jury_P@3/g_norm-g_raw", "ci:ci95"),
    ("3.5", "盲评 前5名不同正确药（严格）：标准化 − 原始", "llm_judge", "facts/strict/g_norm-g_raw", "ci:ci95"),
    ("3.5", "盲评 前5名不同正确药（宽松）：标准化 − 原始", "llm_judge", "facts/lenient/g_norm-g_raw", "ci:ci95"),
    ("3.5", "盲评 前5名不同正确药（严格）：标准化+归因 − 原始", "llm_judge", "facts/strict/g_norm_em-g_raw", "ci:ci95"),
    ("3.6", "盲评 严格 P@1，外部库覆盖的病：标准化 − 原始", "llm_judge", "precision/kb_covered_strict_jury_P@1/g_norm-g_raw", "ci:ci95"),
    ("3.7", "盲评 严格 P@1：lift − 共现计数", "llm_judge_extra", "lists/lift/strict_P@1", "ci:ci95"),
    ("3.7", "盲评 严格 P@1：超几何 − 共现计数", "llm_judge_extra", "lists/hypergeom/strict_P@1", "ci:ci95"),
    ("3.7", "盲评 前5名不同正确药（严格）：超几何 − 共现计数", "llm_judge_extra", "lists/hypergeom/strict_distinct_correct@5", "ci:ci95"),
    ("3.7", "盲评 严格 P@1：大模型兜底 − 本方法（高频病）", "llm_judge_extra", "lists/g_llm_fb/strict_P@1", "ci:ci95"),
    ("3.8", "CCKS2017 盲评 不同正确药（宽松），预测实体", "llm_judge_ccks2017", "pred/all_lenient_distinct@5", "ci:ci95"),
    ("3.8", "CCKS2017 盲评 不同正确药（宽松），人工标注实体", "llm_judge_ccks2017", "gold/all_lenient_distinct@5", "ci:ci95"),
    ("3.8", "CCKS2017 盲评 不同正确药（严格），人工标注实体", "llm_judge_ccks2017", "gold/all_strict_distinct@5", "ci:ci95"),
    ("3.9", "第三轮 不同正确药（严格）：大模型兜底 − 本方法", "llm_judge_targeted", "g_llm_fb_changed/strict_distinct@5", "ci:ci95"),
    ("3.9", "第三轮 不同正确药（宽松）：大模型兜底 − 本方法", "llm_judge_targeted", "g_llm_fb_changed/lenient_distinct@5", "ci:ci95"),
    ("3.9", "第三轮 严格 P@1：大模型兜底 − 本方法", "llm_judge_targeted", "g_llm_fb_changed/strict_P@1", "ci:ci95"),
    ("3.9", "第三轮 严格 P@1：大模型优先 − 本方法", "llm_judge_targeted", "g_llm_primary_changed/strict_P@1", "ci:ci95"),
    ("3.9", "第三轮 不同正确药（严格）：大模型优先 − 本方法", "llm_judge_targeted", "g_llm_primary_changed/strict_distinct@5", "ci:ci95"),
    ("2.11", "CCKS2019 用药：Claude 兜底 − 本方法（固定口径）", "llm_norm_opus", "graph/CCKS2019 test/llm_fallback/rels/使用药物/vs_off", "ci:delta_ci95"),
    ("2.15", "CCKS2017 微调 NER 用药查询 F1：标准化 − 原始", "ccks2017_finetune", "graph/finetuned/使用药物", "ft"),
    ("2.15", "CCKS2017 微调 NER 手术查询 F1：标准化 − 原始", "ccks2017_finetune", "graph/finetuned/实施手术", "ft"),
    ("3.8", "CCKS2017 盲评 不同正确药（宽松），微调后预测实体", "llm_judge_ccks2017", "pred_ft/all_lenient_distinct@5", "ci:ci95"),
    ("3.11", "严格 P@1：兜底+本方法的药排前 − 普通兜底", "llm_judge_backfill", "g_bf_fb_all_vs_g_llm_fb/strict_P@1", "ci:ci95"),
    ("3.11", "不同正确药（严格）：兜底+本方法的药排前 − 本方法", "llm_judge_backfill", "g_bf_fb_all_vs_g_norm/strict_distinct@5", "ci:ci95"),
    ("3.11", "不同正确药（宽松）：兜底+本方法的药排前 − 本方法", "llm_judge_backfill", "g_bf_fb_all_vs_g_norm/lenient_distinct@5", "ci:ci95"),
]


def get(obj, path: str):
    for part in path.split("/"):
        obj = obj[part]
    return obj


def p_from_ci(est: float, lo: float, hi: float) -> float:
    se = (hi - lo) / (2 * 1.96)
    return 1.0 if se <= 0 else math.erfc(abs(est / se) / math.sqrt(2))


def evaluate(kind: str, node: dict) -> tuple[float, list[float], float, str]:
    if kind == "boot":
        est = None
        ci = node["delta_ci95"]
        est = (ci[0] + ci[1]) / 2
        return est, ci, min(1.0, 2 * node["p_one_sided"]), "bootstrap 单侧 ×2"
    if kind == "flip":
        return node["f1_norm"] - node["f1_raw"], node["raw_vs_norm_ci95"], node["p_sign_flip"], "符号翻转检验"
    if kind == "ft":
        return node["f1_norm"] - node["f1_raw"], node["ci95"], node["p"], "符号翻转检验"
    if kind == "gap":
        est = node["prec_conf"] - node["prec_rest"]
        ci = node["gap_ci95"]
        return est, ci, p_from_ci(est, *ci), "由区间反推"
    key = kind.split(":", 1)[1]
    ci = node[key]
    est = node["delta"]
    return est, ci, p_from_ci(est, *ci), "由区间反推"


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=ps.__getitem__)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def bh(ps: list[float]) -> list[float]:
    n = len(ps)
    order = sorted(range(n), key=ps.__getitem__, reverse=True)
    adj, run = [0.0] * n, 1.0
    for k, i in enumerate(order):
        run = min(run, ps[i] * n / (n - k))
        adj[i] = run
    return adj


def main() -> None:
    cache: dict[str, dict] = {}
    rows = []
    for sec, title, f, path, kind in FAMILY:
        if f not in cache:
            cache[f] = json.loads((M / f"{f}.json").read_text(encoding="utf-8"))
        est, ci, p, how = evaluate(kind, get(cache[f], path))
        rows.append({"section": sec, "title": title, "estimate": est, "ci95": ci, "p": p, "p_source": how})
    for r, ph, pb in zip(rows, holm([r["p"] for r in rows]), bh([r["p"] for r in rows])):
        r["p_holm"], r["p_bh"] = ph, pb

    sig = lambda p: "是" if p < ALPHA else "否"
    n_raw = sum(r["p"] < ALPHA for r in rows)
    n_holm = sum(r["p_holm"] < ALPHA for r in rows)
    n_bh = sum(r["p_bh"] < ALPHA for r in rows)
    L = ["多重比较校正：报告结论所依据的比较", "=" * 72,
         f"共 {len(rows)} 个比较（报告里作为结论引用的全部配对比较，显著和不显著的都算；单个评委、单个知识库等稳健性检验不计入）。",
         "p 值：有精确值的用精确值（双侧），其余由 95% bootstrap 区间反推（Altman & Bland 2011）。Holm 控制族错误率，BH 控制错误发现率，均取 0.05。",
         f"未校正显著 {n_raw} 个；Holm 校正后 {n_holm} 个；BH 校正后 {n_bh} 个。", "",
         f"{'节':<5}{'比较':<44}{'估计':>9}  {'95% 区间':<20}{'p':>9}{'Holm':>9}{'BH':>9}  显著(原/Holm/BH)"]
    for r in rows:
        L.append(f"{r['section']:<5}{r['title']:<44}{r['estimate']:>+9.3f}  [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
                 f"{r['p']:>9.4f}{r['p_holm']:>9.4f}{r['p_bh']:>9.4f}  {sig(r['p'])}/{sig(r['p_holm'])}/{sig(r['p_bh'])}")
    lost_holm = [r for r in rows if r["p"] < ALPHA <= r["p_holm"]]
    lost_bh = [r for r in rows if r["p"] < ALPHA <= r["p_bh"]]
    L += ["", "校正后不再显著的（BH）：" + ("无" if not lost_bh else "；".join(f"{r['section']} {r['title']}" for r in lost_bh)),
          "校正后不再显著的（Holm）：" + ("无" if not lost_holm else "；".join(f"{r['section']} {r['title']}" for r in lost_holm))]
    text = "\n".join(L) + "\n"
    (M / "multiplicity.txt").write_text(text, encoding="utf-8")
    (M / "multiplicity.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
