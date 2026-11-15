"""ICD-10 query-time roll-up helpers; graph node names remain unchanged."""

from __future__ import annotations

import re


def _code_key(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(code).upper().split("+")[0].split("/")[0])


def icd_query_levels(name_to_codes: dict[str, list[str]], name: str) -> list[dict[str, str]]:
    """Return the exact name, then available ICD subcategory and category names."""
    codes = name_to_codes.get(name, [])
    if not codes:
        return [{"level": "exact", "name": name, "code": ""}]

    by_code: dict[str, list[tuple[str, str]]] = {}
    for candidate_name, candidate_codes in name_to_codes.items():
        for candidate_code in candidate_codes:
            by_code.setdefault(_code_key(candidate_code), []).append((candidate_name, candidate_code))

    levels = [{"level": "exact", "name": name, "code": codes[0]}]
    seen = {name}
    for level in ("subcategory", "category"):
        matches = []
        for code in codes:
            key = _code_key(code)
            if len(key) < 3 or not key[:1].isalpha() or not key[1:3].isdigit():
                continue
            prefix = key[:4] if level == "subcategory" and len(key) >= 4 else key[:3]
            if level == "subcategory" and len(prefix) < 4:
                continue
            matches.extend(by_code.get(prefix, []))
        for parent_name, parent_code in matches:
            if parent_name not in seen:
                levels.append({"level": level, "name": parent_name, "code": parent_code})
                seen.add(parent_name)
    return levels


def query_graph_with_rollup(graph: dict, name_to_codes: dict[str, list[str]], name: str, top_k: int = 10) -> dict:
    """Try increasingly broad ICD names until a disease node has drug edges."""
    nodes = {node["id"]: node for node in graph.get("nodes", [])}
    edges_by_source: dict[str, list[dict]] = {}
    for edge in graph.get("relations", []):
        if edge.get("type") == "使用药物":
            edges_by_source.setdefault(edge["source"], []).append(edge)

    tried = icd_query_levels(name_to_codes, name)
    for level in tried:
        node_id = f"疾病和诊断::{level['name']}"
        node = nodes.get(node_id)
        edges = edges_by_source.get(node_id, [])
        if node and edges:
            drugs = []
            for edge in sorted(edges, key=lambda item: (-int(item.get("weight", 0)), item["target"]))[:top_k]:
                drug = nodes.get(edge["target"], {"name": edge["target"].split("::", 1)[-1]})
                drugs.append({"name": drug["name"], "weight": int(edge.get("weight", 0))})
            return {
                "query": name,
                "matched_name": level["name"],
                "matched_code": level["code"],
                "level": level["level"],
                "fallback_depth": next(i for i, tried_level in enumerate(tried) if tried_level == level),
                "drugs": drugs,
            }
    return {"query": name, "matched_name": None, "matched_code": "", "level": None, "fallback_depth": None, "drugs": []}
