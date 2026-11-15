"""Score the manually labeled normalization sample (outputs/metrics/manual_norm_sample.csv)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "metrics"
SAMPLE = OUT / "manual_norm_sample.csv"
DRAFT = OUT / "manual_norm_sample_AI初判.csv"
ICD = ROOT / "data" / "dicts" / "icd10_beijing_v601.json"

LABEL_COL = "判定(1=对/0=错)"
GOLD_COL = "正确标准名(判错时填)"
LINKED = {
    "icd_exact", "icd_variant", "icd_retrieve", "icd_rule", "bert_retrieve", "curated", "chip_cdn",
    "fuzzy_icd", "cdn_sup",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def fmt(k: int, n: int) -> str:
    if n == 0:
        return "n=0"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def icd_names() -> dict[str, list[str]]:
    return json.loads(ICD.read_text(encoding="utf-8"))["name_to_codes"]


def agreement(rows: list[dict], draft: list[dict]) -> tuple[list[str], dict]:
    """Human labels vs the AI pre-annotation: changed rows, raw agreement, Cohen's kappa."""
    ai = {r["编号"]: ((r.get(LABEL_COL) or "").strip(), (r.get(GOLD_COL) or "").strip()) for r in draft}
    changed, both = [], []
    for r in rows:
        y, g = (r.get(LABEL_COL) or "").strip(), (r.get(GOLD_COL) or "").strip()
        ay, ag = ai.get(r["编号"], ("", ""))
        if (y, g) != (ay, ag):
            changed.append(f"  #{r['编号']} {r['原文表面词']}: AI {ay or '空'}/{ag or '-'} → 人工 {y or '空'}/{g or '-'}")
        if y in {"0", "1"} and ay in {"0", "1"}:
            both.append((int(ay), int(y)))
    n = len(both)
    po = sum(a == b for a, b in both) / n if n else 0.0
    pa, ph = (sum(a for a, _ in both) / n, sum(b for _, b in both) / n) if n else (0.0, 0.0)
    pe = pa * ph + (1 - pa) * (1 - ph)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return changed, {"n_both": n, "n_changed": len(changed), "agree": po, "kappa": kappa}


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the labeled normalization sample.")
    ap.add_argument("--sample", type=Path, default=SAMPLE, help="labeled CSV (default: manual_norm_sample.csv)")
    ap.add_argument("--tag", default="manual_norm_acc", help="output name under outputs/metrics/")
    ap.add_argument("--title", default="抽检：归一化准确率（CCKS 预测疾病表面词，分层抽样，CHIP-CDN 口径）")
    ap.add_argument("--find", metavar="关键词", help="只列出包含该关键词的 ICD 标准名（填「正确标准名」时查用），不计分")
    args = ap.parse_args()
    if args.find:
        hits = [(n, c) for n, c in icd_names().items() if args.find in n]
        for name, codes in hits[:40]:
            print(f"{name}\t{','.join(codes)}")
        print(f"（共 {len(hits)} 个{'，只显示前 40 个' if len(hits) > 40 else ''}）")
        return
    sample = args.sample
    rows = read_rows(sample)

    labeled = []
    skipped = 0
    for r in rows:
        v = (r.get(LABEL_COL) or "").strip()
        if v in {"1", "0"}:
            labeled.append({**r, "_y": int(v)})
        else:
            skipped += 1
    if not labeled:
        raise SystemExit(f"{sample} 里还没有填写「{LABEL_COL}」列")

    k = sum(r["_y"] for r in labeled)
    n = len(labeled)
    wk = sum(r["_y"] * int(r["出现次数"] or 1) for r in labeled)
    wn = sum(int(r["出现次数"] or 1) for r in labeled)

    by_src: dict[str, list[int]] = defaultdict(list)
    for r in labeled:
        by_src[r["系统来源"]].append(r["_y"])
    linked = [r["_y"] for r in labeled if r["系统来源"] in LINKED]
    kept = [r["_y"] for r in labeled if r["系统来源"] not in LINKED]

    notes: dict[str, int] = defaultdict(int)
    for r in labeled:
        if r["_y"] == 0:
            notes[(r.get("备注") or "未注明").strip() or "未注明"] += 1

    lines = [
        args.title,
        "=" * 64,
        f"文件: {sample.name}",
        f"样本 {len(rows)} 条；已判定 {n} 条；未判定/NER碎片 {skipped} 条（不计入）",
        "",
        f"总体 Acc            {fmt(k, n)}",
        f"按出现次数加权 Acc  {wk}/{wn} = {wk / wn:.4f}",
        f"已链接部分 Acc      {fmt(sum(linked), len(linked))}   ← 系统确实改了名的条目，衡量误并",
        f"原样保留部分 Acc    {fmt(sum(kept), len(kept))}   ← heuristic，衡量漏并",
        "",
        "按来源:",
    ]
    for src in sorted(by_src):
        ys = by_src[src]
        lines.append(f"  {src:14s} {fmt(sum(ys), len(ys))}")
    lines.append("")
    lines.append("错误备注分布:")
    for note, c in sorted(notes.items(), key=lambda x: -x[1]):
        lines.append(f"  {note}: {c}")
    lines.append("")

    names = icd_names()
    missing = [
        f"  #{r['编号']} {r['原文表面词']}: {p}"
        for r in labeled
        if r["_y"] == 0
        for p in (r.get(GOLD_COL) or "").split("##")
        if p.strip() and p.strip() not in names
    ]
    lines.append(f"「正确标准名」不在 ICD 字典里的: {len(missing)} 处" + ("（请用 --find 查到准确写法后改正）" if missing else ""))
    lines.extend(missing)
    lines.append("")

    agree = None
    if DRAFT.exists() and sample.resolve() != DRAFT.resolve():
        changed, agree = agreement(rows, read_rows(DRAFT))
        lines.append(
            f"与 AI 预标注对比: 改动 {agree['n_changed']} 行；判定一致率 {agree['agree']:.4f}，"
            f"Cohen's kappa {agree['kappa']:.4f}（两边都判了 1/0 的 {agree['n_both']} 行）"
        )
        lines.extend(changed)
        lines.append("")

    lines.append("注: 抽样按来源分层，总体 Acc 近似反映全部 664 个表面词；CI 为 Wilson 区间。")
    lines.append("外部参照: CHIP-CDN_dev 自动评测见 cdn_baselines.txt。")

    text = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.tag}.txt").write_text(text, encoding="utf-8")
    (OUT / f"{args.tag}.json").write_text(
        json.dumps(
            {
                "n_labeled": n,
                "n_skipped": skipped,
                "acc": k / n,
                "acc_freq_weighted": wk / wn,
                "acc_linked": (sum(linked) / len(linked)) if linked else None,
                "acc_kept": (sum(kept) / len(kept)) if kept else None,
                "by_source": {s: {"n": len(v), "acc": sum(v) / len(v)} for s, v in by_src.items()},
                "error_notes": dict(notes),
                "gold_names_not_in_icd": len(missing),
                "vs_ai_draft": agree,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(text)


if __name__ == "__main__":
    main()
