"""Defense figures from the metric JSONs (no recomputation).

  fig1_query_f1.png      query F1 without / with normalization: CCKS2019 test, CCKS2017 zero-shot NER, fine-tuned NER
  fig2_jury_distinct.png blind jury: extra distinct correct drugs in the top 5 (normalized − raw), per dataset
  fig3_llm_tradeoff.png  LLM normalization on the queries it changes: top-1 precision vs distinct correct drugs,
                         plain fallback / LLM first / own drugs first (REPORT 3.9, 3.11)

  python src/make_defense_figs.py   # → outputs/figures/defense/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "outputs" / "metrics"
OUT = ROOT / "outputs" / "figures" / "defense"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RAW, NORM = "#9aa5b1", "#2b6cb0"


def load(name: str) -> dict | None:
    p = M / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig_query_f1() -> None:
    rows = []
    b = load("bootstrap_query_f1")
    for rel in ("使用药物", "实施手术"):
        r = b["relations"][rel]
        rows.append(("CCKS2019", rel, r["f1_raw"], r["f1_norm"], r["bootstrap_by_disease"]["delta_ci95"]))
    ft = load("ccks2017_finetune")
    if ft:
        for key, title in (("zero_shot", "CCKS2017\n直接迁移"), ("finetuned", "CCKS2017\n微调后")):
            for rel in ("使用药物", "实施手术"):
                e = ft["graph"][key][rel]
                rows.append((title, rel, e["f1_raw"], e["f1_norm"], e["ci95"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, rel in zip(axes, ("使用药物", "实施手术")):
        sub = [r for r in rows if r[1] == rel]
        x = range(len(sub))
        ax.bar([i - 0.2 for i in x], [r[2] for r in sub], 0.4, color=RAW, label="不标准化")
        ax.bar([i + 0.2 for i in x], [r[3] for r in sub], 0.4, color=NORM, label="本方法标准化")
        for i, r in enumerate(sub):
            d = r[3] - r[2]
            ax.text(i, max(r[2], r[3]) + 0.02, f"+{d:.3f}\n[{r[4][0]:+.3f}, {r[4][1]:+.3f}]", ha="center", fontsize=8)
        ax.set_xticks(list(x), [r[0] for r in sub])
        ax.set_title(f"按病查{'药' if rel == '使用药物' else '手术'}：查询 F1")
        ax.set_ylim(0, 1.05)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_query_f1.png", dpi=200)
    plt.close(fig)


def fig_jury_distinct() -> None:
    rows = []
    j = load("llm_judge")
    if j:
        for lvl in ("strict", "lenient"):
            c = j["facts"][lvl]["g_norm-g_raw"]
            rows.append(("CCKS2019 NER 预测", lvl, c["delta"], c["ci95"]))
    c17 = load("llm_judge_ccks2017") or {}
    for key, title in (("gold", "CCKS2017 人工标注实体"), ("pred", "CCKS2017 直接迁移 NER"), ("pred_ft", "CCKS2017 微调后 NER")):
        if key in c17:
            for lvl in ("strict", "lenient"):
                c = c17[key][f"all_{lvl}_distinct@5"]
                rows.append((title, lvl, c["delta"], c["ci95"]))
    titles = list(dict.fromkeys(r[0] for r in rows))
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(titles) + 1.4))
    for off, lvl, color, name in ((-0.12, "strict", NORM, "严格（只算治疗用药）"), (0.12, "lenient", "#dd6b20", "宽松（含相关用药）")):
        pts = [(titles.index(r[0]) + off, r[2], r[3]) for r in rows if r[1] == lvl]
        ax.errorbar([p[1] for p in pts], [p[0] for p in pts],
                    xerr=[[p[1] - p[2][0] for p in pts], [p[2][1] - p[1] for p in pts]],
                    fmt="o", color=color, capsize=3, label=name)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(range(len(titles)), titles)
    ax.invert_yaxis()
    ax.set_xlabel("前 5 名里多出的不同正确药（标准化 − 不标准化，95% 区间）")
    ax.set_title("三模型盲评：标准化让按病查药「更全、去重」")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_jury_distinct.png", dpi=200)
    plt.close(fig)


def fig_llm_tradeoff() -> None:
    t, bf = load("llm_judge_targeted"), load("llm_judge_backfill")
    if not t:
        return
    pts = [("大模型兜底", t["g_llm_fb_changed"], "#c53030"), ("大模型优先", t["g_llm_primary_changed"], "#dd6b20")]
    if bf:
        pts += [("兜底 + 本方法的药排前", bf["g_bf_fb_all_vs_g_norm"], NORM),
                ("优先 + 本方法的药排前", bf["g_bf_primary_all_vs_g_norm"], "#2f855a")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, lvl, lt in ((axes[0], "strict", "严格"), (axes[1], "lenient", "宽松")):
        for name, c, color in pts:
            p, d = c[f"{lvl}_P@1"], c[f"{lvl}_distinct@5"]
            ax.errorbar(d["delta"], p["delta"], xerr=[[d["delta"] - d["ci95"][0]], [d["ci95"][1] - d["delta"]]],
                        yerr=[[p["delta"] - p["ci95"][0]], [p["ci95"][1] - p["delta"]]],
                        fmt="o", color=color, capsize=3, label=name)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("前 5 名不同正确药 与本方法的差")
        ax.set_ylabel("排第一的药准确率（P@1）与本方法的差")
        ax.set_title(f"大模型标准化改变答案的查询（{lt}口径）")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_llm_tradeoff.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_query_f1()
    fig_jury_distinct()
    fig_llm_tradeoff()
    print("\n".join(str(p) for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
