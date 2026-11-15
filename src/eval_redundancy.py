"""How much surface-form redundancy each dataset leaves for normalization to remove.

Explains why the gain is large on CCKS2019 and small on CCKS2017: for diseases and drugs,
count surfaces vs canonical nodes, the share of mentions whose node has ≥2 surfaces, and for
drugs the share written as brand names / abbreviations (rewritten by an alias table) and the
share that are drug classes rather than drugs (抗生素, 降压药物), which no normalizer can resolve.
Gold entities only; full system settings.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_cross_dataset import load_ccks2017
from eval_g_raw_vs_g_norm import collect_freq
from ner_data import load_bio_docs
from normalize import canonical_names, configure_normalization, lookup_with_source

OUT = ROOT / "outputs" / "metrics"
CLASS_RE = re.compile(r"(药物|药)$|^(抗生素|中药|西药|头孢|激素)$")


def redundancy(freq: dict[str, int], label: str) -> dict:
    by_node: dict[str, list[str]] = defaultdict(list)
    for s in freq:
        by_node[canonical_names(s, label)[0]].append(s)
    mentions = sum(freq.values())
    redundant = sum(freq[s] for ss in by_node.values() if len(ss) >= 2 for s in ss)
    out = {"surfaces": len(freq), "nodes": len(by_node), "reduction": 1 - len(by_node) / len(freq),
           "mentions": mentions, "redundant_mention_share": redundant / mentions}
    if label == "药物":
        alias = sum(c for s, c in freq.items() if lookup_with_source(s, label)[1] == "drug_alias"
                    and lookup_with_source(s, label)[0] != s.strip())
        cls = Counter({s: c for s, c in freq.items() if CLASS_RE.search(s.strip())})
        out |= {"alias_rewritten_share": alias / mentions, "class_term_share": sum(cls.values()) / mentions,
                "top_class_terms": cls.most_common(8)}
    return out


def main() -> None:
    processed = ROOT / "data" / "processed"
    datasets = {
        "CCKS2019 测试集（379 份）": load_bio_docs(processed / "test.bio"),
        "CCKS2019 全部（1379 份）": [d for n in ("train.bio", "dev.bio", "test.bio") for d in load_bio_docs(processed / n)],
        "CCKS2017（300 份）": load_ccks2017()[0],
    }
    configure_normalization()
    res = {}
    for name, docs in datasets.items():
        freq = collect_freq(docs)
        res[name] = {lab: redundancy(freq.get(lab, {}), lab) for lab in ("疾病和诊断", "药物")}

    L = ["写法冗余度：标准化能消除多少重复写法（人工标注实体，完整系统）", "=" * 64,
         "冗余提及 = 所在规范节点有 ≥2 种写法的提及；商品名/缩写 = 被药物别名表改写；泛称 = 抗生素、降压药物这类非具体药名", ""]
    for name, r in res.items():
        d, m = r["疾病和诊断"], r["药物"]
        L.append(f"【{name}】")
        L.append(f"  疾病  写法 {d['surfaces']} → 节点 {d['nodes']}（减少 {d['reduction']:.0%}）  冗余提及 {d['redundant_mention_share']:.0%}")
        L.append(f"  药物  写法 {m['surfaces']} → 节点 {m['nodes']}（减少 {m['reduction']:.0%}）  冗余提及 {m['redundant_mention_share']:.0%}  "
                 f"商品名/缩写 {m['alias_rewritten_share']:.0%}  泛称 {m['class_term_share']:.0%}")
        L.append("        泛称举例：" + "、".join(f"{s}({c})" for s, c in m["top_class_terms"]))
    text = "\n".join(L) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "redundancy.txt").write_text(text, encoding="utf-8")
    (OUT / "redundancy.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
