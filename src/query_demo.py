"""Compare G_raw vs G_norm neighbors for paper example queries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_g_raw_vs_g_norm import collect_freq, graph_counts, neighbors_by_rel
from ner_data import load_bio_docs
from normalize import build_canonical_map, lookup_with_source, get_index

QUERIES = [
    "直肠癌",
    "直肠腺癌",
    "乙状结肠癌",
    "乙状结肠腺癌",
    "子宫内膜癌",
    "子宫内膜癌IB期（高分化）",
    "宫颈鳞癌IIB期",
    "子宫肌瘤",
    "乙状结肠中分化腺癌",
]
RELS = ("使用药物", "实施手术")


def names_of(nids: set[str]) -> list[str]:
    out = []
    for nid in sorted(nids):
        _lab, name = nid.split("::", 1)
        out.append(name)
    return out


def main() -> None:
    docs = load_bio_docs(ROOT / "data" / "processed" / "test_pred.bio")
    canon = build_canonical_map(collect_freq(docs))
    raw_nodes, raw_rels, _, _ = graph_counts(docs, None)
    norm_nodes, norm_rels, _, _ = graph_counts(docs, canon)
    idx = get_index()
    rows = []
    for q in QUERIES:
        canon_name, source = lookup_with_source(q, "疾病和诊断")
        rec = {
            "query": q,
            "canonical": canon_name,
            "source": source,
            "icd_code": idx.preferred_code(canon_name),
        }
        for rel in RELS:
            raw_nb = neighbors_by_rel(raw_rels, rel)
            norm_nb = neighbors_by_rel(norm_rels, rel)
            raw_id = f"疾病和诊断::{q}"
            norm_id = f"疾病和诊断::{canon_name}"
            rec[rel] = {
                "g_raw": names_of(raw_nb.get(raw_id, set())),
                "g_norm": names_of(norm_nb.get(norm_id, set())),
            }
        rows.append(rec)
    out = ROOT / "outputs" / "metrics" / "query_examples.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for rec in rows:
        print(
            f"{rec['query']} -> {rec['canonical']} [{rec['source']}] {rec['icd_code']}"
        )
        for rel in RELS:
            raw_n = len(rec[rel]["g_raw"])
            norm_n = len(rec[rel]["g_norm"])
            print(f"  {rel}: raw={raw_n} norm={norm_n}")
    print("wrote", out)


if __name__ == "__main__":
    main()
