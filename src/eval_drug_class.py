"""How much of the drug class terms (REPORT 2.10: 27% of CCKS2017 drug mentions) an ATC class lexicon resolves.

For each input: share of drug mentions that are class terms; of those, mapped to one / several ATC
classes or left unmapped (with the most frequent unmapped ones); drug surfaces → nodes and the redundant
mention share (node with ≥2 surfaces) with the lexicon off / on; and the share of mentions whose node
changes at all, which on CCKS2019 should be ~0 (the lexicon must not touch ordinary drug names).

  python src/eval_drug_class.py   # → outputs/metrics/drug_class.{txt,json}
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drug_class import classify, is_class_term
from eval_g_raw_vs_g_norm import collect_freq
from ner_data import load_bio_docs
from normalize import canonical_names, configure_normalization

OUT = ROOT / "outputs" / "metrics"
PROCESSED = ROOT / "data" / "processed"
SETS = (("CCKS2019 测试集 人工标注", "test.bio"), ("CCKS2019 测试集 NER 预测", "test_pred.bio"),
        ("CCKS2017 人工标注", "ccks2017_gold.bio"), ("CCKS2017 NER 预测", "ccks2017_pred.bio"),
        ("CCKS2017 微调后 NER 预测", "ccks2017_pred_ft.bio"))
DRUG = "药物"


def nodes(freq: dict[str, int], on: bool) -> dict:
    configure_normalization(use_drug_class=on)
    canon = {s: canonical_names(s, DRUG) for s in freq}
    configure_normalization()
    by_node: dict[str, set[str]] = defaultdict(set)
    for s, ns in canon.items():
        for n in ns:
            by_node[n].add(s)
    m = sum(freq.values())
    red = sum(c for s, c in freq.items() if any(len(by_node[n]) >= 2 for n in canon[s]))
    return {"canon": canon, "surfaces": len(freq), "nodes": len(by_node), "redundant_share": red / m}


def main() -> None:
    res, L = {}, ["药物类别词对到 ATC 类别（src/drug_class.py）", "=" * 72,
                  "类别词 = 以「药物/药/类/拮抗剂/抑制剂/阻滞剂」结尾或「抗生素」「激素」；组合短语按「、，及和与等」切开，每段都要认得才改写。", ""]
    for title, fname in SETS:
        path = PROCESSED / fname
        if not path.exists():
            continue
        freq = collect_freq(load_bio_docs(path)).get(DRUG, {})
        m = sum(freq.values())
        cls = {s: c for s, c in freq.items() if is_class_term(s)}
        mapped = {s: classify(s) for s in cls}
        one = sum(c for s, c in cls.items() if len(mapped[s]) == 1)
        many = sum(c for s, c in cls.items() if len(mapped[s]) > 1)
        unm = Counter({s: c for s, c in cls.items() if not mapped[s]})
        off, on = nodes(freq, False), nodes(freq, True)
        changed = sum(c for s, c in freq.items() if off["canon"][s] != on["canon"][s])
        by_class = Counter()
        for s, c in cls.items():
            for n in mapped[s]:
                by_class[n] += c
        r = res[title] = {
            "mentions": m, "class_share": sum(cls.values()) / m if m else 0.0,
            "class_mapped_one": one, "class_mapped_many": many, "class_unmapped": sum(unm.values()),
            "changed_share": changed / m if m else 0.0,
            "off": {k: v for k, v in off.items() if k != "canon"}, "on": {k: v for k, v in on.items() if k != "canon"},
            "top_unmapped": unm.most_common(12), "top_classes": by_class.most_common(10)}
        nc = sum(cls.values()) or 1
        L += [f"【{title}】药物提及 {m}，其中类别词 {sum(cls.values())}（{r['class_share']:.0%}）",
              f"  类别词里：对到 1 个 ATC 类 {one}（{one / nc:.0%}），对到多个类 {many}（{many / nc:.0%}），对不上 {r['class_unmapped']}（{r['class_unmapped'] / nc:.0%}）",
              f"  节点改变的药物提及 {changed}（{r['changed_share']:.1%}）",
              f"  写法 → 节点：关 {off['surfaces']} → {off['nodes']}，开 {on['surfaces']} → {on['nodes']}；"
              f"冗余提及 {off['redundant_share']:.0%} → {on['redundant_share']:.0%}",
              "  对上的类别：" + "、".join(f"{n.split(' ', 2)[2]}({c})" for n, c in r["top_classes"]),
              "  对不上的：" + "、".join(f"{s}({c})" for s, c in r["top_unmapped"]), ""]
    text = "\n".join(L) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "drug_class.txt").write_text(text, encoding="utf-8")
    (OUT / "drug_class.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
