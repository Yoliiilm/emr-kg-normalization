"""Bootstrap confidence intervals for G_raw vs G_norm query F1 (vs_gold)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_g_raw_vs_g_norm import QUERY_RELS, build_graphs, query_rows
from ner_data import load_bio_docs
from normalize import configure_normalization

OUT = ROOT / "outputs" / "metrics"
KEEP = ("query", "canonical", "f_raw", "f_norm", "delta")


N_BOOT = 2000
N_PERM = 10000
SEED = 42


def ci95(stats: list[float]) -> list[float]:
    s = sorted(stats)
    return [s[int(0.025 * len(s))], s[int(0.975 * len(s)) - 1]]


def cluster_bootstrap(rows: list[dict], key: str | None, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Paired bootstrap of mean F1; key=None resamples queries, key='canonical' resamples diseases."""
    rng = random.Random(seed)
    if key is None:
        clusters = [[r] for r in rows]
    else:
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r)
        clusters = list(groups.values())
    k = len(clusters)
    raw_m, norm_m, delta_m = [], [], []
    delta_le0 = 0
    for _ in range(n_boot):
        sample = [r for _ in range(k) for r in clusters[rng.randrange(k)]]
        n = len(sample)
        fr = sum(r["f_raw"] for r in sample) / n
        fn = sum(r["f_norm"] for r in sample) / n
        raw_m.append(fr)
        norm_m.append(fn)
        delta_m.append(fn - fr)
        if fn - fr <= 0:
            delta_le0 += 1
    return {
        "n_units": k,
        "g_raw_ci95": ci95(raw_m),
        "g_norm_ci95": ci95(norm_m),
        "delta_ci95": ci95(delta_m),
        "p_one_sided": max(delta_le0, 1) / n_boot,
    }


def sign_flip_test(deltas: list[float], n_perm: int = N_PERM, seed: int = SEED) -> float:
    """Two-sided paired randomization test on per-query ΔF1."""
    rng = random.Random(seed)
    n = len(deltas)
    obs = abs(sum(deltas) / n)
    hits = 0
    for _ in range(n_perm):
        m = abs(sum(d if rng.random() < 0.5 else -d for d in deltas) / n)
        if m >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main() -> None:
    configure_normalization()
    processed = ROOT / "data" / "processed"
    gold_docs = load_bio_docs(processed / "test.bio")
    pred_docs = load_bio_docs(processed / "test_pred.bio")
    g = build_graphs(gold_docs, pred_docs)

    report = {"seed": SEED, "n_boot": N_BOOT, "n_perm": N_PERM, "relations": {}}
    lines = [
        "查询 F1 显著性（G_raw vs G_norm，对照官方标注图 vs_gold_graph，完整归一化系统）",
        "=" * 72,
        f"配对 bootstrap {N_BOOT} 次（按查询 / 按疾病规范名整簇重采样）+ 符号翻转随机化检验 {N_PERM} 次",
        "",
    ]
    for rel in QUERY_RELS:
        rows = [{k: r[k] for k in KEEP} for r in query_rows(g, rel)]
        n = len(rows)
        f_raw = sum(r["f_raw"] for r in rows) / n
        f_norm = sum(r["f_norm"] for r in rows) / n
        deltas = [r["delta"] for r in rows]
        by_query = cluster_bootstrap(rows, None)
        by_disease = cluster_bootstrap(rows, "canonical")
        p_perm = sign_flip_test(deltas)
        n_up = sum(1 for d in deltas if d > 0)
        n_down = sum(1 for d in deltas if d < 0)
        report["relations"][rel] = {
            "n_query": n,
            "f1_raw": f_raw,
            "f1_norm": f_norm,
            "delta": f_norm - f_raw,
            "n_improved": n_up,
            "n_worse": n_down,
            "n_same": n - n_up - n_down,
            "bootstrap_by_query": by_query,
            "bootstrap_by_disease": by_disease,
            "p_sign_flip_two_sided": p_perm,
            "worse_examples": sorted(
                (r for r in rows if r["delta"] < 0), key=lambda r: r["delta"]
            )[:20],
        }
        lines.append(f"【{rel}】 查询数={n}  疾病簇数={by_disease['n_units']}")
        lines.append(f"  F1  g_raw={f_raw:.4f}  g_norm={f_norm:.4f}  Δ={f_norm - f_raw:+.4f}")
        lines.append(f"  逐查询: 变好 {n_up}  变差 {n_down}  不变 {n - n_up - n_down}")
        for title, b in (("按查询重采样", by_query), ("按疾病重采样", by_disease)):
            lines.append(
                f"  {title}: g_raw 95%CI=[{b['g_raw_ci95'][0]:.4f}, {b['g_raw_ci95'][1]:.4f}]  "
                f"g_norm 95%CI=[{b['g_norm_ci95'][0]:.4f}, {b['g_norm_ci95'][1]:.4f}]"
            )
            lines.append(
                f"      Δ 95%CI=[{b['delta_ci95'][0]:+.4f}, {b['delta_ci95'][1]:+.4f}]  "
                f"单侧 p≤{b['p_one_sided']:.4f}"
            )
        lines.append(f"  符号翻转检验（双侧）p={p_perm:.4g}")
        lines.append("")

    lines.append("解读: Δ 的 95%CI 不含 0 且 p<0.05，即可报告归一化带来的提升统计显著。")
    lines.append("p 值下限受重采样次数限制（bootstrap 最小 1/2000，随机化检验最小 1/10001）。")
    text = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bootstrap_query_f1.txt").write_text(text, encoding="utf-8")
    (OUT / "bootstrap_query_f1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(text)


if __name__ == "__main__":
    main()
