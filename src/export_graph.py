"""Build a document-level EMR knowledge graph from BIO files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ner_data import bio_to_entities, load_bio_docs
from normalize import keep_entity

REL_RULES = [
    ("疾病和诊断", "药物", "使用药物"),
    ("疾病和诊断", "手术", "实施手术"),
    ("疾病和诊断", "解剖部位", "发生于"),
    ("疾病和诊断", "影像检查", "进行影像检查"),
    ("疾病和诊断", "实验室检验", "进行实验室检验"),
    ("手术", "解剖部位", "手术部位"),
    # 不再建 药物-治疗->疾病：与 使用药物 同对共现、仅方向相反
]


def node_id(text: str, label: str) -> str:
    return f"{label}::{text}"


def graph_from_docs(docs: list[tuple[list[str], list[str]]], source: str) -> dict:
    nodes = {}
    rel_counts = defaultdict(int)
    doc_mentions = []

    for doc_i, (chars, tags) in enumerate(docs):
        ents = [
            e
            for e in bio_to_entities(chars, tags)
            if e["text"].strip() and keep_entity(e["text"], e["label"])
        ]
        by_type = defaultdict(list)
        for e in ents:
            nid = node_id(e["text"], e["label"])
            if nid not in nodes:
                nodes[nid] = {
                    "id": nid,
                    "name": e["text"],
                    "type": e["label"],
                    "mention_count": 0,
                    "doc_count": 0,
                    "docs": set(),
                }
            nodes[nid]["mention_count"] += 1
            nodes[nid]["docs"].add(doc_i)
            by_type[e["label"]].append(e)

        for src_type, dst_type, rel in REL_RULES:
            srcs = {e["text"] for e in by_type.get(src_type, [])}
            dsts = {e["text"] for e in by_type.get(dst_type, [])}
            for s in srcs:
                for d in dsts:
                    if s == d and src_type == dst_type:
                        continue
                    rel_counts[(node_id(s, src_type), rel, node_id(d, dst_type))] += 1

        doc_mentions.append(
            {
                "doc_id": f"{source}_{doc_i}",
                "n_entities": len(ents),
                "entities": ents,
            }
        )

    for node in nodes.values():
        node["doc_count"] = len(node["docs"])
        node.pop("docs", None)

    relations = [
        {"source": s, "type": r, "target": t, "weight": w, "graph": source}
        for (s, r, t), w in sorted(rel_counts.items(), key=lambda x: -x[1])
    ]
    return {
        "source": source,
        "n_docs": len(docs),
        "nodes": list(nodes.values()),
        "relations": relations,
        "n_nodes": len(nodes),
        "n_relations": len(relations),
    }


def write_csv(graph: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = out_dir / f"nodes_{graph['source']}.csv"
    rels_path = out_dir / f"relations_{graph['source']}.csv"
    with nodes_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id:ID", "name", "type:LABEL", "mention_count:int", "doc_count:int", "icd_code:string"],
        )
        w.writeheader()
        for n in graph["nodes"]:
            w.writerow(
                {
                    "id:ID": n["id"],
                    "name": n["name"],
                    "type:LABEL": n["type"],
                    "mention_count:int": n["mention_count"],
                    "doc_count:int": n["doc_count"],
                    "icd_code:string": n.get("icd_code", ""),
                }
            )
    with rels_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[":START_ID", ":END_ID", ":TYPE", "weight:int"])
        w.writeheader()
        for r in graph["relations"]:
            w.writerow(
                {
                    ":START_ID": r["source"],
                    ":END_ID": r["target"],
                    ":TYPE": r["type"],
                    "weight:int": r["weight"],
                }
            )


def write_cypher(graph: dict, path: Path) -> None:
    lines = [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;",
        "",
    ]
    for n in graph["nodes"]:
        name = n["name"].replace("\\", "\\\\").replace("'", "\\'")
        nid = n["id"].replace("\\", "\\\\").replace("'", "\\'")
        ntype = n["type"].replace("'", "\\'")
        lines.append(
            f"MERGE (n:Entity {{id: '{nid}'}}) "
            f"SET n.name = '{name}', n.type = '{ntype}', "
            f"n.mention_count = {n['mention_count']}, n.doc_count = {n['doc_count']};"
        )
    for r in graph["relations"]:
        s = r["source"].replace("\\", "\\\\").replace("'", "\\'")
        t = r["target"].replace("\\", "\\\\").replace("'", "\\'")
        rel = r["type"]
        lines.append(
            f"MATCH (a:Entity {{id: '{s}'}}), (b:Entity {{id: '{t}'}}) "
            f"MERGE (a)-[r:`{rel}`]->(b) SET r.weight = {r['weight']};"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bio", default="")
    p.add_argument("--source", default="pred")
    return p.parse_args()


def export_one(bio_path: Path, source: str, out_dir: Path) -> dict:
    docs = load_bio_docs(bio_path)
    graph = graph_from_docs(docs, source)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"graph_{source}.json").write_text(
        json.dumps(
            {
                **graph,
                "nodes": graph["nodes"],
                "relations": graph["relations"][:5000],
                "n_relations_full": graph["n_relations"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(graph, out_dir)
    write_cypher(graph, out_dir / f"import_{source}.cypher")
    stats = {
        "source": source,
        "bio": str(bio_path),
        "n_docs": graph["n_docs"],
        "n_nodes": graph["n_nodes"],
        "n_relations": graph["n_relations"],
        "type_counts": {},
    }
    type_counts = defaultdict(int)
    for n in graph["nodes"]:
        type_counts[n["type"]] += 1
    stats["type_counts"] = dict(type_counts)
    (out_dir / f"stats_{source}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[{source}] docs={stats['n_docs']} nodes={stats['n_nodes']} "
        f"rels={stats['n_relations']} types={stats['type_counts']}"
    )
    return graph


def main() -> None:
    args = parse_args()
    processed = ROOT / "data" / "processed"
    out_dir = ROOT / "outputs" / "neo4j"
    if args.bio:
        export_one(Path(args.bio), args.source, out_dir)
        return
    export_one(processed / "test.bio", "gold_test", out_dir)
    pred = processed / "test_pred.bio"
    if pred.exists():
        export_one(pred, "pred_test", out_dir)
    else:
        print("test_pred.bio not found; skip predicted graph")


if __name__ == "__main__":
    main()
