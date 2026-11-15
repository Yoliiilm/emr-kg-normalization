"""Blind items for the LLM jury (improvement 4): is drug Y used to treat disease X?

Queries: disease nodes of G_norm on the test notes (predicted entities, leak-free drug table) with at least
MIN_DRUGS drug neighbours, the N_QUERIES most mentioned. Each query is answered by
  g_raw      the node's most-mentioned raw surface, drugs ranked by co-occurring notes
  g_norm     the normalized node, drugs ranked by co-occurring notes
  g_norm_em  the normalized node, drugs ranked by EM attribution (src/attribution.py, α / iters from
             outputs/metrics/attribution_tune.json)
Every drug that can land in a graph's top K (score ≥ the K-th score) becomes an item, so P@k is an exact
expectation over random tie-breaks. The disease is always shown under the normalized name, so an item does not
reveal which graph produced it.

Calibration items from the external KBs (CMeIE ∪ 寻医问药), for queries the KBs cover: drugs the KBs list for the
disease (positive) and drugs the KBs list only for a query disease of another ICD chapter (negative; may contain
a few true uses, so specificity is a lower bound).

Items are deduplicated on (disease, drug), shuffled and split into batches; sources are kept only in key.json.
Writes outputs/llm_judge/{items.csv, key.json, batch_XX.csv}.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import normalize
from attribution import REL, attribute
from eval_ablation import FULL
from eval_attribution import TUNE_FILE, build, leak_free_table
from eval_external_extra import PROCESSED
from eval_external_kg import DISEASE, disease_classes, neighbours, supported
from eval_query_level import combine_sources, ext_index
from eval_robustness import load_sources
from export_graph import node_id
from ner_data import load_bio_docs
from normalize import configure_normalization

OUT = ROOT / "outputs" / "llm_judge"
N_QUERIES = 100
MIN_DRUGS = 3
K = 5
BATCH = 100
CALIB_PER_QUERY = 2
GRAPHS = ("g_raw", "g_norm", "g_norm_em")


def top_candidates(scores: dict[str, float]) -> list[tuple[str, float]]:
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    if len(ranked) <= K:
        return ranked
    cut = ranked[K - 1][1]
    return [x for x in ranked if x[1] >= cut]


def main() -> None:
    kb = combine_sources(load_sources())
    kb_drugs = ext_index(kb)
    docs = load_bio_docs(PROCESSED / "test_pred.bio")
    g_raw, _ = build(docs, None)
    g_norm, pairs = build(docs, leak_free_table())
    tuned = json.loads(TUNE_FILE.read_text(encoding="utf-8"))
    em = attribute(g_norm, pairs, iters=tuned["iters"], background=True, alpha=tuned["alpha"])
    nb_raw, nb_norm = neighbours(g_raw, REL), neighbours(g_norm, REL)

    queries = []
    for d in sorted((d for d, ms in nb_norm.items() if len(ms) >= MIN_DRUGS), key=lambda d: (-g_norm["mentions"][d], d)):
        raw = [node_id(n, DISEASE) for n in g_norm["names"][d] if node_id(n, DISEASE) in nb_raw]
        if raw:
            queries.append((d, max(raw, key=lambda n: (g_raw["mentions"][n], n))))
        if len(queries) == N_QUERIES:
            break

    items: dict[tuple[str, str], int] = {}

    def item(disease: str, drug: str) -> int:
        return items.setdefault((disease, drug), len(items))

    lists = []
    for d, r in queries:
        disease = d.split("::", 1)[1]
        answers = {"g_raw": (g_raw, r, {m: g_raw["co"][(r, REL, m)] for m in nb_raw[r]}),
                   "g_norm": (g_norm, d, {m: g_norm["co"][(d, REL, m)] for m in nb_norm[d]}),
                   "g_norm_em": (g_norm, d, {m: em[(d, REL, m)] for m in nb_norm[d]})}
        cls = disease_classes(g_norm, d, kb) & kb["subjects"][REL]
        entry = {"node": d, "raw_node": r, "disease": disease, "kb_covered": bool(cls),
                 "kb_drugs": sorted(set().union(*(kb_drugs[c] for c in cls))) if cls else [], "graphs": {}}
        for gn, (g, node, sc) in answers.items():
            entry["graphs"][gn] = [{"item": item(disease, m.split("::", 1)[1]), "score": s,
                                    "kb": supported(g, node, m, REL, kb) if cls else None}
                                   for m, s in top_candidates(sc)]
        lists.append(entry)

    configure_normalization(**FULL)
    idx = normalize.get_index()
    chapter = {e["node"]: (idx.preferred_code(e["disease"]) or "")[:1] for e in lists}
    configure_normalization()
    rng = random.Random(2026)
    covered = [e for e in lists if e["kb_drugs"]]
    calib = []
    for e in covered:
        own = set(e["kb_drugs"])
        for m in rng.sample(e["kb_drugs"], min(CALIB_PER_QUERY, len(own))):
            calib.append({"item": item(e["disease"], m), "kind": "pos", "node": e["node"]})
        if not chapter[e["node"]]:
            continue
        pool = sorted({m for o in covered if chapter[o["node"]] and chapter[o["node"]] != chapter[e["node"]]
                       for m in o["kb_drugs"] if len(m) >= 2 and not any(m in x or x in m for x in own)})
        for m in rng.sample(pool, min(CALIB_PER_QUERY, len(pool))):
            calib.append({"item": item(e["disease"], m), "kind": "neg", "node": e["node"]})

    order = list(items.items())
    rng.shuffle(order)
    ids = {old: f"Q{i:04d}" for i, (_, old) in enumerate(order)}
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob("batch_*.csv"):
        p.unlink()
    rows = [{"id": ids[old], "disease": d, "drug": m} for (d, m), old in order]
    with (OUT / "items.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "disease", "drug"])
        w.writeheader()
        w.writerows(rows)
    for b in range(0, len(rows), BATCH):
        with (OUT / f"batch_{b // BATCH:02d}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "disease", "drug"])
            w.writeheader()
            w.writerows(rows[b:b + BATCH])
    for e in lists:
        for ranked in e["graphs"].values():
            for x in ranked:
                x["item"] = ids[x["item"]]
    for c in calib:
        c["item"] = ids[c["item"]]
    (OUT / "key.json").write_text(json.dumps({"k": K, "graphs": GRAPHS, "queries": lists, "calibration": calib},
                                             ensure_ascii=False, indent=1), encoding="utf-8")
    per_graph = {gn: sum(len(e["graphs"][gn]) for e in lists) for gn in GRAPHS}
    print(f"queries {len(lists)} (kb covered {len(covered)})  unique items {len(rows)}  "
          f"batches {(len(rows) + BATCH - 1) // BATCH}  per graph {per_graph}  calibration "
          f"pos {sum(c['kind'] == 'pos' for c in calib)} neg {sum(c['kind'] == 'neg' for c in calib)}")


if __name__ == "__main__":
    main()
