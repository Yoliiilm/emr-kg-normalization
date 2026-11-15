"""Import predicted EMR graph CSV into a running local Neo4j."""

from __future__ import annotations

import argparse
import os
import csv
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "neo4j"
URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
AUTH = (os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"])


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def chunks(seq, n=400):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="pred_test")
    p.add_argument("--graph", default="raw", help="Property n.graph / r.graph")
    p.add_argument("--id-prefix", default="", help="Prefix node ids to keep graphs distinct")
    p.add_argument("--wipe", action="store_true", help="Delete the whole DB first")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    nodes = read_csv(OUT / f"nodes_{args.source}.csv")
    rels = read_csv(OUT / f"relations_{args.source}.csv")
    prefix = args.id_prefix
    print(f"csv nodes={len(nodes)} rels={len(rels)} graph={args.graph} prefix={prefix!r} wipe={args.wipe}")

    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session(database="neo4j") as s:
        if args.wipe:
            s.run("MATCH (n) DETACH DELETE n")
        elif args.graph:
            s.run("MATCH (n {graph: $g}) DETACH DELETE n", g=args.graph)
        s.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")

        n_ok = 0
        for batch in chunks(nodes, 400):
            rows = [
                {
                    "id": prefix + r["id:ID"],
                    "name": r["name"],
                    "type": r["type:LABEL"],
                    "mention_count": int(r["mention_count:int"]),
                    "doc_count": int(r["doc_count:int"]),
                    "icd_code": r.get("icd_code:string", ""),
                    "graph": args.graph,
                }
                for r in batch
            ]
            s.run(
                """
                UNWIND $rows AS row
                MERGE (n:Entity {id: row.id})
                SET n.name = row.name,
                    n.type = row.type,
                    n.graph = row.graph,
                    n.icd_code = row.icd_code,
                    n.mention_count = row.mention_count,
                    n.doc_count = row.doc_count
                FOREACH (_ IN CASE WHEN row.type = '疾病和诊断' THEN [1] ELSE [] END | SET n:`疾病和诊断`)
                FOREACH (_ IN CASE WHEN row.type = '解剖部位' THEN [1] ELSE [] END | SET n:`解剖部位`)
                FOREACH (_ IN CASE WHEN row.type = '药物' THEN [1] ELSE [] END | SET n:`药物`)
                FOREACH (_ IN CASE WHEN row.type = '手术' THEN [1] ELSE [] END | SET n:`手术`)
                FOREACH (_ IN CASE WHEN row.type = '影像检查' THEN [1] ELSE [] END | SET n:`影像检查`)
                FOREACH (_ IN CASE WHEN row.type = '实验室检验' THEN [1] ELSE [] END | SET n:`实验室检验`)
                """,
                rows=rows,
            )
            n_ok += len(rows)
        print(f"imported nodes={n_ok}")

        by_type: dict[str, list] = defaultdict(list)
        for r in rels:
            by_type[r[":TYPE"]].append(
                {
                    "s": prefix + r[":START_ID"],
                    "t": prefix + r[":END_ID"],
                    "w": int(r["weight:int"]),
                }
            )
        r_ok = 0
        for rtype, items in by_type.items():
            for batch in chunks(items, 400):
                s.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (a:Entity {{id: row.s}})
                    MATCH (b:Entity {{id: row.t}})
                    MERGE (a)-[rel:`{rtype}`]->(b)
                    SET rel.weight = row.w, rel.graph = $graph
                    """,
                    rows=batch,
                    graph=args.graph,
                )
                r_ok += len(batch)
            print(f"  rel {rtype}: {len(items)}")
        n_nodes = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        n_rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"DB nodes={n_nodes} rels={n_rels}")

    driver.close()
    print("IMPORT_OK")


if __name__ == "__main__":
    main()
