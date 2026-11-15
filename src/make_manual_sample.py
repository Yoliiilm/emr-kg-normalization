"""Draw a stratified sample of predicted disease surfaces for manual normalization labeling."""

from __future__ import annotations

import csv
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ner_data import bio_to_entities, load_bio_docs
from normalize import clean_surface, configure_normalization, get_index, keep_entity, strip_grade

OUT_DIR = ROOT / "outputs" / "metrics"
N_SAMPLE = 200
SEED = 42

COLUMNS = [
    "编号",
    "原文表面词",
    "出现次数",
    "系统输出",
    "ICD编码",
    "判定(1=对/0=错)",
    "正确标准名(判错时填)",
    "备注",
    "参考ICD候选(BERT top3)",
    "系统来源",
]

GUIDE = """人工标注说明（归一化准确率，200 条）
========================================
文件: manual_norm_sample.csv（Excel 直接打开，UTF-8 带 BOM）

每行是一条 NER 预测出的疾病表面词，以及系统给出的标准名。只需填三列：

1. 判定(1=对/0=错)
   1 = 系统输出是这个表面词可接受的标准诊断名
   0 = 系统输出不对
   判定规则:
   - 系统输出与表面词含义一致，且是 ICD 标准写法或临床公认规范写法 → 1
     例: 直肠癌 → 直肠恶性肿瘤 (1)；冠心病 → 冠状动脉粥样硬化性心脏病 (1)
   - 系统把它并到了错误/更宽/更窄/换部位的病名 → 0
     例: 肺癌 → 右肺恶性肿瘤 (0，凭空加了侧别)
   - 系统输出只是原样保留（来源=heuristic）:
       · ICD 里存在明确对应的标准名 → 0（漏并），并填正确标准名
       · 本身不是疾病（窦性心律、血糖、极高危等 NER 误标）或无法对应 → 1，备注写「非疾病」或「无对应」
   - 表面词是 NER 切碎/截断的片段，无法判断 → 留空判定，备注写「NER碎片」（不计入 Acc）

2. 正确标准名(判错时填)
   判 0 时填你认为正确的标准名；可参考「参考ICD候选」列，也可自己写。

3. 备注（可选）
   建议用这几个标签之一，便于统计: 非疾病 / 无对应 / NER碎片 / 侧别 / 部位 / 粒度过粗 / 粒度过细 / 其他

填完保存（保持 CSV 格式），然后运行:
    python src/eval_manual_acc.py
输出 outputs/metrics/manual_norm_acc.txt（总体 Acc、按来源 Acc、按出现次数加权 Acc）。
"""


def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder proportional allocation, at least 1 per non-empty stratum."""
    n = sum(counts.values())
    quotas = {k: total * v / n for k, v in counts.items()}
    alloc = {k: max(1, int(q)) for k, q in quotas.items()}
    rest = total - sum(alloc.values())
    for k in sorted(quotas, key=lambda k: quotas[k] - int(quotas[k]), reverse=True):
        if rest <= 0:
            break
        if alloc[k] < counts[k]:
            alloc[k] += 1
            rest -= 1
    return alloc


def main() -> None:
    configure_normalization(use_chip_cdn=True, use_drug_alias=True, use_surgery_alias=True)
    idx = get_index()

    freq: Counter[str] = Counter()
    for chars, tags in load_bio_docs(ROOT / "data" / "processed" / "test_pred.bio"):
        for e in bio_to_entities(chars, tags):
            if e["label"] == "疾病和诊断" and e["text"].strip() and keep_entity(e["text"], e["label"]):
                freq[e["text"]] += 1

    by_source: dict[str, list[tuple[str, str]]] = {}
    for surface in sorted(freq):
        pred, src = idx.lookup_disease(surface)
        by_source.setdefault(src, []).append((surface, pred))

    alloc = allocate({k: len(v) for k, v in by_source.items()}, N_SAMPLE)
    rng = random.Random(SEED)
    picked: list[tuple[str, str, str]] = []
    for src, items in sorted(by_source.items()):
        for surface, pred in rng.sample(items, min(alloc[src], len(items))):
            picked.append((surface, pred, src))
    rng.shuffle(picked)

    from bert_linker import get_linker

    linker = get_linker()

    def top3(surface: str) -> str:
        q = strip_grade(clean_surface(surface)) or surface
        vec = linker.encode(q)
        scores = (linker.emb @ vec.T).squeeze(1)
        return " | ".join(linker.names[i] for i in scores.topk(3).indices.tolist())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "manual_norm_sample.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i, (surface, pred, src) in enumerate(picked, 1):
            w.writerow(
                [i, surface, freq[surface], pred, idx.preferred_code(pred), "", "", "", top3(surface), src]
            )
    (OUT_DIR / "标注说明.txt").write_text(GUIDE, encoding="utf-8")

    print(f"wrote {path} ({len(picked)} rows)")
    print("strata:", {k: f"{alloc[k]}/{len(v)}" for k, v in sorted(by_source.items())})


if __name__ == "__main__":
    main()
