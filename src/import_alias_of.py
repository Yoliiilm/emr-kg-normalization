"""Import surface->canonical ALIAS_OF edges into Neo4j (from canonical_map.json).

Alias nodes use graph='alias' so a later norm reimport does not wipe them.
Re-run this script after refreshing G_norm.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "neo4j"
URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
AUTH = (os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"])
GRAPH = "alias"
CANON_PREFIX = "n:"
# Prefer disease/drug merges for demo; skip punctuation-only heuristics.
KEEP_TYPES = {"疾病和诊断", "药物", "手术"}
PUNCT_ONLY = re.compile(r"^[\s\W_]+$", re.UNICODE)


def chunks(seq, n=400):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def load_rows() -> list[dict]:
    cm = json.loads((OUT / "canonical_map.json").read_text(encoding="utf-8"))
    rows = []
    for key, meta in cm.items():
        if "::" not in key:
            continue
        etype, surface = key.split("::", 1)
        if etype not in KEEP_TYPES:
            continue
        to = meta.get("to", "")
        targets = [str(t).strip() for t in (to if isinstance(to, list) else [to])]
        surface = surface.strip()
        for target in targets:
            if not surface or not target or surface == target:
                continue
            if PUNCT_ONLY.match(surface) or PUNCT_ONLY.match(target):
                continue
            rows.append(
                {
                    "alias_id": f"a:{etype}::{surface}",
                    "canon_id": f"{CANON_PREFIX}{etype}::{target}",
                    "name": surface,
                    "type": etype,
                    "to": target,
                    "source": meta.get("source", "") if len(targets) == 1 else "split",
                    "icd_code": meta.get("icd_code", "") or "",
                }
            )
    return rows


def main() -> None:
    rows = load_rows()
    print(f"alias candidates={len(rows)}")

    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session(database="neo4j") as s:
        s.run("MATCH (n {graph: $g}) DETACH DELETE n", g=GRAPH)
        s.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")

        linked = 0
        skipped = 0
        for batch in chunks(rows, 300):
            result = s.run(
                """
                UNWIND $rows AS row
                OPTIONAL MATCH (c:Entity {id: row.canon_id})
                WITH row, c
                WHERE c IS NOT NULL
                MERGE (a:Entity {id: row.alias_id})
                SET a:Alias,
                    a.name = row.name,
                    a.type = row.type,
                    a.graph = $graph,
                    a.is_alias = true,
                    a.norm_source = row.source,
                    a.icd_code = row.icd_code,
                    a.mention_count = 0,
                    a.doc_count = 0
                FOREACH (_ IN CASE WHEN row.type = '疾病和诊断' THEN [1] ELSE [] END | SET a:`疾病和诊断`)
                FOREACH (_ IN CASE WHEN row.type = '药物' THEN [1] ELSE [] END | SET a:`药物`)
                FOREACH (_ IN CASE WHEN row.type = '手术' THEN [1] ELSE [] END | SET a:`手术`)
                MERGE (a)-[r:ALIAS_OF]->(c)
                SET r.graph = $graph, r.source = row.source
                RETURN count(*) AS n
                """,
                rows=batch,
                graph=GRAPH,
            )
            linked += result.single()["n"]
        skipped = len(rows) - linked
        n_alias = s.run(
            "MATCH (n:Entity {graph:$g}) WHERE n.is_alias = true RETURN count(n) AS c",
            g=GRAPH,
        ).single()["c"]
        n_rel = s.run(
            "MATCH ()-[r:ALIAS_OF {graph:$g}]->() RETURN count(r) AS c",
            g=GRAPH,
        ).single()["c"]
        print(f"linked={linked} skipped_no_canon={skipped}")
        print(f"DB alias_nodes={n_alias} ALIAS_OF={n_rel}")

        sample = list(
            s.run(
                """
                MATCH (a:Entity {name:'直肠癌', graph:$g})-[r:ALIAS_OF]->(c)
                RETURN a.name AS from, c.name AS to, r.source AS source, c.icd_code AS icd
                """,
                g=GRAPH,
            )
        )
        print("sample 直肠癌:", [dict(x) for x in sample])

    driver.close()
    print("ALIAS_IMPORT_OK")


if __name__ == "__main__":
    main()
