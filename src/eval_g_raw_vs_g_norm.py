"""Build G_raw vs G_norm and evaluate disease-drug / disease-surgery queries.

Two metrics:
1. alias_completeness — how much G_raw misses because aliases are split.
   Gold is the union of neighbors over a predicted alias cluster (diagnostic;
   G_norm recall is 1.0 by construction).
2. vs_gold_graph — query official NER graph (normalized with the same rules).
   Both G_raw and G_norm can score below 1.0.
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

from export_graph import REL_RULES, node_id, write_csv
from ner_data import bio_to_entities, load_bio_docs
from normalize import (
    build_canonical_map,
    canonical_names,
    configure_normalization,
    coverage_report,
    entity_key,
    get_index,
    histology_without_site,
    keep_entity,
    lookup_with_source,
    needs_context,
    normalization_settings,
)

QUERY_RELS = ("使用药物", "实施手术")


def collect_freq(docs) -> dict[str, dict[str, int]]:
    freq: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for chars, tags in docs:
        for e in bio_to_entities(chars, tags):
            if e["text"].strip() and keep_entity(e["text"], e["label"]):
                freq[e["label"]][e["text"]] += 1
    return freq


def semantic_id(nid: str) -> str:
    label, name = nid.split("::", 1)
    return f"{label}::{entity_key(name, label)}"


def neighbors_by_rel(rel_counts, rel_type: str) -> dict[str, set[str]]:
    out = defaultdict(set)
    for (s, r, t), _w in rel_counts.items():
        if r == rel_type:
            out[s].add(t)
    return out


def graph_counts(docs, canon_map=None, relation_mode: str = "cooccur"):
    """relation_mode: cooccur | pattern | hybrid

    - cooccur: classic same-document entity co-occurrence (Xu-style)
    - pattern: only keep disease-drug/surgery edges with EMR cue evidence
    - hybrid: keep all cooccur edges; pattern-supported edges get +1 weight
    """
    from relation_patterns import score_pair

    nodes = {}
    rel_counts = defaultdict(int)
    surfaces = defaultdict(set)
    evidence_counts = defaultdict(lambda: defaultdict(int))

    def names_of(text, label) -> tuple[str, ...]:
        if canon_map is None:
            return (text,)
        canon = canon_map.get((label, text), text)
        return canon if isinstance(canon, tuple) else (canon,)

    for chars, tags in docs:
        text = "".join(chars)
        ents = [
            e
            for e in bio_to_entities(chars, tags)
            if e["text"].strip() and keep_entity(e["text"], e["label"])
        ]
        mention_names = [names_of(e["text"], e["label"]) for e in ents]
        primary = Counter(
            n
            for e, names in zip(ents, mention_names)
            if e["label"] == "疾病和诊断"
            for n in names
            if n.endswith("恶性肿瘤") and not n.endswith("继发恶性肿瘤")
        )
        if canon_map is not None and primary:
            ctx = primary.most_common(1)[0][0]
            mention_names = [
                (ctx,) if needs_context(e["text"], e["label"], names) else names
                for e, names in zip(ents, mention_names)
            ]

        by_type = defaultdict(list)
        for e, names in zip(ents, mention_names):
            e["nids"] = [node_id(n, e["label"]) for n in names]
            for nid, name in zip(e["nids"], names):
                if nid not in nodes:
                    nodes[nid] = {
                        "id": nid,
                        "name": name,
                        "type": e["label"],
                        "mention_count": 0,
                        "doc_count": 0,
                        "icd_code": get_index().preferred_code(name) if e["label"] == "疾病和诊断" else "",
                        "docs": set(),
                    }
                nodes[nid]["mention_count"] += 1
                nodes[nid]["docs"].add(id(chars))
                surfaces[nid].add(e["text"])
            by_type[e["label"]].append(e)

        for src_type, dst_type, rel in REL_RULES:
            for s_ent in by_type.get(src_type, []):
                for d_ent in by_type.get(dst_type, []):
                    if s_ent["text"] == d_ent["text"] and src_type == dst_type:
                        continue
                    bonus, evidence = score_pair(text, s_ent, d_ent, rel)
                    if relation_mode == "pattern" and rel in QUERY_RELS and evidence != "pattern":
                        continue
                    for s_nid in s_ent["nids"]:
                        for d_nid in d_ent["nids"]:
                            if s_nid == d_nid:
                                continue
                            rel_counts[(s_nid, rel, d_nid)] += 1 + (bonus if relation_mode == "hybrid" else 0)
                            evidence_counts[(s_nid, rel, d_nid)][evidence] += 1

    for node in nodes.values():
        node["doc_count"] = len(node["docs"])
        node.pop("docs", None)
    return nodes, rel_counts, surfaces, evidence_counts


def index_by_semantic(nodes: dict) -> dict[tuple[str, str], str]:
    """Map (type, entity_key) to a node; on collision prefer the node whose name is the key."""
    idx = {}
    for nid, node in nodes.items():
        key = (node["type"], entity_key(node["name"], node["type"]))
        if key not in idx or node["name"] == key[1]:
            idx[key] = nid
    return idx


def prf(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    if not gold:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f


def mean_prf(rows: list[tuple[float, float, float]]) -> dict:
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
    p = sum(x[0] for x in rows) / len(rows)
    r = sum(x[1] for x in rows) / len(rows)
    f = sum(x[2] for x in rows) / len(rows)
    return {"precision": p, "recall": r, "f1": f, "n": len(rows)}


def eval_alias_completeness(rel_raw, rel_norm, surfaces, rel_type: str) -> dict:
    """Gold = union of G_norm neighbors for the predicted cluster."""
    gold_nb = neighbors_by_rel(rel_norm, rel_type)
    raw_nb = neighbors_by_rel(rel_raw, rel_type)
    scores_raw, scores_norm = [], []
    examples = []

    for canon_nid, aliases in surfaces.items():
        if not canon_nid.startswith("疾病和诊断::"):
            continue
        gold = {semantic_id(t) for t in gold_nb.get(canon_nid, set())}
        if not gold:
            continue
        if len(aliases) < 2:
            continue
        for alias in sorted(aliases):
            raw_nid = node_id(alias, "疾病和诊断")
            pred_raw = {semantic_id(t) for t in raw_nb.get(raw_nid, set())}
            pred_norm = gold
            scores_raw.append(prf(pred_raw, gold))
            scores_norm.append(prf(pred_norm, gold))
            if pred_raw != gold and len(examples) < 10:
                examples.append(
                    {
                        "query": alias,
                        "canonical": canon_nid.split("::", 1)[1],
                        "rel": rel_type,
                        "gold_n": len(gold),
                        "raw_n": len(pred_raw),
                        "norm_n": len(pred_norm),
                        "raw_f1": scores_raw[-1][2],
                        "norm_f1": scores_norm[-1][2],
                    }
                )
    return {
        "note": "Gold is G_norm neighbor union; G_norm F1 is 1.0 by design. Use vs_gold_graph for a real score.",
        "g_raw": mean_prf(scores_raw),
        "g_norm": mean_prf(scores_norm),
        "examples": examples,
    }


def query_rows(g: dict, rel_type: str) -> list[dict]:
    """One row per (gold disease cluster, surface alias) with paired raw / norm P-R-F1.

    The judge (semantic_id, gold graph) uses the normalizer active at call time; the
    predicted G_norm node for an alias comes from g['pred_alias_names'], i.e. the
    normalizer that built G_norm (may differ in ablations).
    """
    gold_nb = neighbors_by_rel(g["gold_norm_rels"], rel_type)
    raw_nb = neighbors_by_rel(g["pred_raw_rels"], rel_type)
    norm_nb = neighbors_by_rel(g["pred_norm_rels"], rel_type)
    norm_idx = index_by_semantic(g["pred_norm_nodes"])
    alias_names = g["pred_alias_names"]

    rows = []
    for canon_nid, aliases in g["gold_surfaces"].items():
        if not canon_nid.startswith("疾病和诊断::"):
            continue
        canonical = canon_nid.split("::", 1)[1]
        gold = {semantic_id(t) for t in gold_nb.get(canon_nid, set())}
        if not gold:
            continue
        for alias in sorted(aliases):
            raw_nid = node_id(alias, "疾病和诊断")
            pred_raw = (
                {semantic_id(t) for t in raw_nb.get(raw_nid, set())}
                if raw_nid in g["pred_raw_nodes"]
                else set()
            )

            cands = [
                node_id(n, "疾病和诊断")
                for n in alias_names.get(alias, ())
                if node_id(n, "疾病和诊断") in g["pred_norm_nodes"]
            ]
            same = [c for c in cands if entity_key(g["pred_norm_nodes"][c]["name"], "疾病和诊断") == canonical]
            if same:
                norm_nid = same[0]
            elif g["pred_context"] and histology_without_site(alias):
                norm_nid = norm_idx.get(("疾病和诊断", canonical))
            else:
                norm_nid = cands[0] if cands else None
            pred_norm = {semantic_id(t) for t in norm_nb.get(norm_nid, set())} if norm_nid else set()

            p_raw = prf(pred_raw, gold)
            p_norm = prf(pred_norm, gold)
            rows.append(
                {
                    "query": alias,
                    "canonical": canonical,
                    "rel": rel_type,
                    "gold_n": len(gold),
                    "raw_n": len(pred_raw),
                    "norm_n": len(pred_norm),
                    "p_raw": p_raw,
                    "p_norm": p_norm,
                    "f_raw": p_raw[2],
                    "f_norm": p_norm[2],
                    "delta": p_norm[2] - p_raw[2],
                }
            )
    return rows


def eval_vs_gold(g: dict, rel_type: str) -> dict:
    rows = query_rows(g, rel_type)
    examples = [
        {k: r[k] for k in ("query", "canonical", "rel", "gold_n", "raw_n", "norm_n")}
        | {"raw_f1": r["f_raw"], "norm_f1": r["f_norm"]}
        for r in rows
        if r["f_raw"] != r["f_norm"]
    ][:12]
    return {
        "g_raw": mean_prf([r["p_raw"] for r in rows]),
        "g_norm": mean_prf([r["p_norm"] for r in rows]),
        "examples": examples,
    }


def type_counts(nodes: dict) -> dict[str, int]:
    tc = defaultdict(int)
    for n in nodes.values():
        tc[n["type"]] += 1
    return dict(tc)


def build_graphs(gold_docs, pred_docs, pred_settings: dict | None = None, relation_mode: str = "cooccur") -> dict:
    """Gold graph + judge use the currently active normalizer (the full system as the fixed
    standard answer); G_norm is built with pred_settings when given (ablation rows)."""
    gold_freq = collect_freq(gold_docs)
    pred_freq = collect_freq(pred_docs)
    if "off" != normalization_settings().get("supervised_cdn", "off") or (
        pred_settings and pred_settings.get("supervised_cdn", "off") != "off"
    ):
        from cdn_linker import warm

        warm({t.strip() for f in (gold_freq, pred_freq) for t in f.get("疾病和诊断", {})})
    gold_canon = build_canonical_map(gold_freq)
    gold_raw_nodes, gold_raw_rels, _, _ = graph_counts(gold_docs, None)
    gold_norm_nodes, gold_norm_rels, gold_surfaces, _ = graph_counts(gold_docs, gold_canon)
    pred_raw_nodes, pred_raw_rels, _, _ = graph_counts(pred_docs, None, relation_mode=relation_mode)

    judge_settings = normalization_settings()
    if pred_settings is not None:
        configure_normalization(**pred_settings)
    pred_canon = build_canonical_map(pred_freq)
    pred_norm_nodes, pred_norm_rels, pred_surfaces, pred_ev = graph_counts(
        pred_docs, pred_canon, relation_mode=relation_mode
    )
    gold_aliases = {a for nid, al in gold_surfaces.items() if nid.startswith("疾病和诊断::") for a in al}
    pred_alias_names = {a: canonical_names(a, "疾病和诊断") for a in gold_aliases}
    pred_context = bool(normalization_settings().get("use_context")) and not normalization_settings().get(
        "disease_rules_only"
    )
    coverage = coverage_report(pred_freq)
    if pred_settings is not None:
        configure_normalization(**judge_settings)

    return {
        "gold_canon": gold_canon,
        "gold_raw_nodes": gold_raw_nodes,
        "gold_raw_rels": gold_raw_rels,
        "gold_norm_nodes": gold_norm_nodes,
        "gold_norm_rels": gold_norm_rels,
        "gold_surfaces": gold_surfaces,
        "pred_canon": pred_canon,
        "pred_raw_nodes": pred_raw_nodes,
        "pred_raw_rels": pred_raw_rels,
        "pred_norm_nodes": pred_norm_nodes,
        "pred_norm_rels": pred_norm_rels,
        "pred_surfaces": pred_surfaces,
        "pred_evidence": pred_ev,
        "pred_alias_names": pred_alias_names,
        "pred_context": pred_context,
        "coverage": coverage,
    }


def compute_stats(gold_docs, pred_docs, pred_settings: dict | None = None) -> dict:
    g = build_graphs(gold_docs, pred_docs, pred_settings)
    pred_canon = g["pred_canon"]
    gold_raw_nodes, gold_raw_rels = g["gold_raw_nodes"], g["gold_raw_rels"]
    gold_norm_nodes, gold_norm_rels = g["gold_norm_nodes"], g["gold_norm_rels"]
    pred_raw_nodes, pred_raw_rels = g["pred_raw_nodes"], g["pred_raw_rels"]
    pred_norm_nodes, pred_norm_rels = g["pred_norm_nodes"], g["pred_norm_rels"]
    pred_surfaces = g["pred_surfaces"]

    n_merged = sum(1 for (lab, name), canon in pred_canon.items() if name != canon)
    stats = {
        "g_raw": {
            "n_nodes": len(pred_raw_nodes),
            "n_relations": len(pred_raw_rels),
            "type_counts": type_counts(pred_raw_nodes),
        },
        "g_norm": {
            "n_nodes": len(pred_norm_nodes),
            "n_relations": len(pred_norm_rels),
            "type_counts": type_counts(pred_norm_nodes),
            "n_surfaces_merged": n_merged,
        },
        "g_gold_raw": {
            "n_nodes": len(gold_raw_nodes),
            "n_relations": len(gold_raw_rels),
            "type_counts": type_counts(gold_raw_nodes),
        },
        "g_gold_norm": {
            "n_nodes": len(gold_norm_nodes),
            "n_relations": len(gold_norm_rels),
            "type_counts": type_counts(gold_norm_nodes),
        },
        "dict_coverage": g["coverage"],
        "alias_completeness": {},
        "vs_gold_graph": {},
        "sample_merges": [],
        "pred_canon": pred_canon,
        "pred_norm_nodes": pred_norm_nodes,
        "pred_norm_rels": pred_norm_rels,
    }

    for rel in QUERY_RELS:
        stats["alias_completeness"][rel] = eval_alias_completeness(
            pred_raw_rels, pred_norm_rels, pred_surfaces, rel
        )
        stats["vs_gold_graph"][rel] = eval_vs_gold(g, rel)

    merges = [
        (lab, src, "、".join(dst) if isinstance(dst, tuple) else dst)
        for (lab, src), dst in pred_canon.items()
        if src != dst
    ]
    merges.sort(key=lambda x: (x[0], x[2], x[1]))
    disease_merges = [m for m in merges if m[0] == "疾病和诊断"]
    stats["sample_merges"] = [
        {"type": lab, "from": src, "to": dst} for lab, src, dst in disease_merges[:30]
    ]
    stats["n_disease_merges"] = len(disease_merges)
    return stats


def main() -> None:
    processed = ROOT / "data" / "processed"
    out_dir = ROOT / "outputs" / "neo4j"
    metric_dir = ROOT / "outputs" / "metrics"
    gold_docs = load_bio_docs(processed / "test.bio")
    pred_docs = load_bio_docs(processed / "test_pred.bio")
    stats = compute_stats(gold_docs, pred_docs)
    pred_canon = stats.pop("pred_canon")
    pred_norm_nodes = stats.pop("pred_norm_nodes")
    pred_norm_rels = stats.pop("pred_norm_rels")

    graph_norm = {
        "source": "pred_test_norm",
        "nodes": list(pred_norm_nodes.values()),
        "relations": [
            {"source": s, "type": r, "target": t, "weight": w}
            for (s, r, t), w in sorted(pred_norm_rels.items(), key=lambda x: -x[1])
        ],
        "n_nodes": len(pred_norm_nodes),
        "n_relations": len(pred_norm_rels),
    }
    write_csv(graph_norm, out_dir)
    metric_dir.mkdir(parents=True, exist_ok=True)
    (metric_dir / "g_raw_vs_g_norm.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "canonical_map.json").write_text(
        json.dumps(
            {
                f"{k[0]}::{k[1]}": {
                    "to": list(v) if isinstance(v, tuple) else v,
                    "source": lookup_with_source(k[1], k[0])[1],
                    "icd_code": (
                        get_index().preferred_code(v[0] if isinstance(v, tuple) else v)
                        if k[0] == "疾病和诊断"
                        else ""
                    ),
                }
                for k, v in pred_canon.items()
                if k[1] != v
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
