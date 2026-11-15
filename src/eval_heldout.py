"""Held-out check: are the new disease rules and the curated alias table overfit to test?

Both were written while reading CCKS-test and CHIP-CDN-dev error analyses. Two checks on
data that was not inspected for that:

A. CCKS train (gold entities): how often each component fires, vs CCKS test. Query F1 is
   not used here: with gold entities on both sides the score only measures agreement with
   our own normalization of the standard answer.
B. CHIP-CDN train (human gold ICD names): accuracy on the mentions each component changes.
   The CHIP alias table is built from CHIP-CDN train, so it is disabled in every setting.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_g_raw_vs_g_norm import collect_freq
from ner_data import load_bio_docs
from normalize import configure_normalization, coverage_report, get_index

OUT = ROOT / "outputs" / "metrics"
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "dicts" / "raw"
CURATED = ROOT / "data" / "dicts" / "disease_alias_extra.json"
WRONG_EXAMPLES = 25

FULL = {
    "use_chip_cdn": True,
    "use_drug_alias": True,
    "use_surgery_alias": True,
    "disease_rules_only": False,
    "use_split": True,
    "use_context": True,
    "use_new_rules": True,
    "use_curated": True,
}
ABLATE = [
    ("new_rules", "本周新规则", {"use_new_rules": False}),
    ("curated", "人工别名表", {"use_curated": False}),
]


def cache_bert() -> None:
    try:
        from bert_linker import get_linker

        linker = get_linker()
    except (ImportError, FileNotFoundError):
        return
    linker.retrieve = lru_cache(maxsize=None)(linker.retrieve)


def disease_map(texts, cfg: dict) -> dict[str, tuple[str, ...]]:
    configure_normalization(**cfg)
    idx = get_index()
    return {t: tuple(n for n, _ in idx.lookup_disease_multi(t)) for t in texts}


def ccks_part() -> dict:
    out = {}
    for split in ("train", "test"):
        docs = load_bio_docs(PROCESSED / f"{split}.bio")
        freq = collect_freq(docs)
        surfaces = sorted(freq.get("疾病和诊断", {}))
        curated = set(json.loads(CURATED.read_text(encoding="utf-8"))["alias"])
        full = disease_map(surfaces, FULL)
        configure_normalization(**FULL)
        row = {
            "n_docs": len(docs),
            "n_surfaces": len(surfaces),
            "curated_keys_present": len(curated & set(surfaces)),
            "hit_full": coverage_report(freq)["icd_hit_rate"],
            "ablate": {},
        }
        for key, _, off in ABLATE:
            cfg = {**FULL, **off}
            other = disease_map(surfaces, cfg)
            hit = coverage_report(freq)["icd_hit_rate"]
            row["ablate"][key] = {
                "hit_without": hit,
                "changed": sum(full[t] != other[t] for t in surfaces),
            }
        configure_normalization(**FULL)
        out[split] = row
    return out


def chip_part() -> dict:
    base = {**FULL, "use_chip_cdn": False}
    out = {}
    for split in ("train", "dev"):
        data = json.loads((RAW / f"CHIP-CDN_{split}.json").read_text(encoding="utf-8"))
        gold = {}
        for d in data:
            gold.setdefault(d["text"], set()).update(x for x in d["normalized_result"].split("##") if x)
        texts = sorted(gold)
        full = disease_map(texts, base)
        res = {"n": len(texts), "ablate": {}}
        for key, _, off in ABLATE:
            other = disease_map(texts, {**base, **off})
            changed = [t for t in texts if full[t] != other[t]]

            def score(m):
                exact = sum(set(m[t]) == gold[t] for t in changed)
                anyhit = sum(bool(set(m[t]) & gold[t]) for t in changed)
                return exact, anyhit

            e_with, a_with = score(full)
            e_wo, a_wo = score(other)
            wrong = [
                {"text": t, "with": "、".join(full[t]), "without": "、".join(other[t]), "gold": "##".join(sorted(gold[t]))}
                for t in changed
                if not set(full[t]) & gold[t]
            ]
            res["ablate"][key] = {
                "changed": len(changed),
                "exact_with": e_with,
                "exact_without": e_wo,
                "anyhit_with": a_with,
                "anyhit_without": a_wo,
                "wrong_examples": wrong[:WRONG_EXAMPLES],
            }
        out[split] = res
    configure_normalization(**FULL)
    return out


def pct(a: int, n: int) -> str:
    return f"{a}/{n} = {a / n:.3f}" if n else "0/0"


def render(ccks: dict, chip: dict) -> list[str]:
    lines = [
        "独立检验：本周新规则和人工别名表是否过拟合测试集",
        "=" * 72,
        "背景：新规则（Ⅱ型写法 / 括号部位 / X转移→继发 / 部位+癌、腺瘤）和人工别名表（103 条）",
        "是看着 CCKS 测试集和 CHIP-CDN dev 的错误写的；CCKS 训练集和 CHIP-CDN 训练集的错误没看过。",
        "",
        "A. CCKS：组件在没看过的训练集上是否同样起作用（官方标注实体；只看覆盖，不看查询 F1）",
        "   查询 F1 不能用：两边都用标注实体时，标准答案也由本系统归一化，分数只反映和自己的一致程度。",
    ]
    for split, r in ccks.items():
        lines.append(
            f"   【{split}】病历 {r['n_docs']} 篇，疾病表面词 {r['n_surfaces']} 个，完整系统 ICD 命中率 {r['hit_full']:.3f}；"
            f"人工别名表 103 条中出现 {r['curated_keys_present']} 条"
        )
        for key, title, _ in ABLATE:
            a = r["ablate"][key]
            lines.append(
                f"      去掉{title}：命中率 {a['hit_without']:.3f}（完整系统高 {r['hit_full'] - a['hit_without']:+.3f}），"
                f"改动表面词 {a['changed']} 个（{a['changed'] / r['n_surfaces']:.1%}）"
            )
    lines += [
        "",
        "B. CHIP-CDN：组件改动的那些条目，改后是否比改前更对（人工标注 gold；所有设定都关掉 CHIP 同义词表）",
        "   完全正确 = 输出的标准名集合与 gold 完全一致；命中 = 至少一个输出名在 gold 中。",
    ]
    for split, r in chip.items():
        seen = "（设计规则时看过其错误，仅作对照）" if split == "dev" else "（设计时没看过）"
        lines.append(f"   【CHIP-CDN {split}】{r['n']} 条{seen}")
        for key, title, _ in ABLATE:
            a = r["ablate"][key]
            n = a["changed"]
            lines.append(f"      {title}改动 {n} 条：")
            lines.append(f"         完全正确  有它 {pct(a['exact_with'], n)}   没它 {pct(a['exact_without'], n)}")
            lines.append(f"         命中      有它 {pct(a['anyhit_with'], n)}   没它 {pct(a['anyhit_without'], n)}")
    lines += ["", f"C. CHIP-CDN train 上新规则改错的例子（有它仍一个都没命中，最多 {WRONG_EXAMPLES} 条）"]
    for w in chip["train"]["ablate"]["new_rules"]["wrong_examples"]:
        lines.append(f"   {w['text']}  → 有规则: {w['with']} | 无规则: {w['without']} | gold: {w['gold']}")
    lines += [
        "",
        "读法：A 中组件在 train 上的改动比例不低于 test，说明不是只对测试集的具体写法生效；",
        "B 中 train 上「有它」明显高于「没它」，说明它改得对，而不是只改得多。",
    ]
    return lines


def main() -> None:
    cache_bert()
    ccks = ccks_part()
    chip = chip_part()
    lines = render(ccks, chip)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "heldout_train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "heldout_train.json").write_text(
        json.dumps({"ccks": ccks, "chip_cdn": chip}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
