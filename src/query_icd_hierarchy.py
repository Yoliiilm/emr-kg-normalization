"""Query a normalized graph with exact-to-coarse ICD fallback, without renaming nodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", required=True, help="disease name or surface form")
    parser.add_argument("--bio", type=Path, default=ROOT / "data" / "processed" / "test_pred.bio")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    from eval_g_raw_vs_g_norm import collect_freq, graph_counts
    from icd_hierarchy import query_graph_with_rollup
    from ner_data import load_bio_docs
    from normalize import build_canonical_map, get_index, lookup_with_source

    docs = load_bio_docs(args.bio)
    canon_map = build_canonical_map(collect_freq(docs))
    nodes, relations, _surfaces, _evidence = graph_counts(docs, canon_map)
    graph = {
        "nodes": list(nodes.values()),
        "relations": [
            {"source": source, "type": rel, "target": target, "weight": weight}
            for (source, rel, target), weight in relations.items()
        ],
    }
    canonical, source = lookup_with_source(args.disease, "疾病和诊断")
    indexed_names = get_index().names
    result = query_graph_with_rollup(graph, indexed_names, canonical, args.top_k)
    result["disease_surface"] = args.disease
    result["normalization_source"] = source
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
