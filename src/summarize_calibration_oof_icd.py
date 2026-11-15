"""Summarize ICD-code relationships for every row in a train OOF audit.

ICD relationships are diagnostic only. They do not replace strict string-match
labels or establish semantic correctness.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "metrics" / "experiments" / "source_shrinkage" / "train_oof_audit_neural_source_chip_reject.json"
DEFAULT_OUT = ROOT / "outputs" / "metrics" / "experiments" / "source_shrinkage" / "train_oof_icd_relation_summary.json"
DEFAULT_CSV = ROOT / "outputs" / "metrics" / "experiments" / "source_shrinkage" / "train_oof_icd_relation_all.csv"


def normalized_code(code: str) -> str:
    return "".join(char for char in code.upper() if char.isalnum())


def classify_relation(candidate_name: str | None, gold_names: list[str], name_to_codes: dict[str, list[str]]) -> str:
    if candidate_name is None:
        return "no_candidate"
    if candidate_name in gold_names:
        return "exact_name_match"
    candidate_codes = {normalized_code(code) for code in name_to_codes.get(candidate_name, [])}
    gold_codes = {normalized_code(code) for name in gold_names for code in name_to_codes.get(name, [])}
    candidate_codes.discard("")
    gold_codes.discard("")
    if not candidate_codes or not gold_codes:
        return "missing_icd_code"
    if candidate_codes & gold_codes:
        return "same_icd_code_different_name"
    if any(len(parent) >= 3 and len(child) > len(parent) and child.startswith(parent)
           for parent in candidate_codes for child in gold_codes):
        return "candidate_is_broader_parent"
    if any(len(parent) >= 3 and len(child) > len(parent) and child.startswith(parent)
           for parent in gold_codes for child in candidate_codes):
        return "candidate_is_narrower_child"
    if any(len(left) >= 3 and len(right) >= 3 and left[:3] == right[:3]
           for left in candidate_codes for right in gold_codes):
        return "same_three_character_category"
    return "different_icd_category"


RELATIONS = (
    "exact_name_match",
    "same_icd_code_different_name",
    "candidate_is_broader_parent",
    "candidate_is_narrower_child",
    "same_three_character_category",
    "different_icd_category",
    "missing_icd_code",
    "no_candidate",
)


def audit_rows(records: list[dict], name_to_codes: dict[str, list[str]]) -> list[dict]:
    rows = []
    for record in records:
        first = record["top5"][0] if record["top5"] else None
        candidate = first["name"] if first else None
        gold = record["gold"]
        relation = classify_relation(candidate, gold, name_to_codes)
        rows.append({
            "fold": record["fold"],
            "surface": record["surface"],
            "gold": gold,
            "accepted": bool(record["accepted"]),
            "strict_string_match": bool(record["accepted"] and candidate in gold),
            "candidate": candidate,
            "candidate_source": first["source"] if first else record["best_source"],
            "relation": relation,
            "candidate_icd_codes": sorted(set(name_to_codes.get(candidate, []))) if candidate else [],
            "gold_icd_codes": sorted({code for name in gold for code in name_to_codes.get(name, [])}),
            "confidence": record["confidence"],
            "margin": record["margin"],
            "decision_reason": record["reason"],
        })
    return rows


def relation_counts(rows: list[dict]) -> dict:
    counts = Counter(row["relation"] for row in rows)
    total = len(rows)
    return {
        "mentions": total,
        "by_relation": {
            relation: {"count": counts[relation], "rate": counts[relation] / total if total else 0.0}
            for relation in RELATIONS
        },
    }


def summarize(rows: list[dict]) -> dict:
    accepted = [row for row in rows if row["accepted"]]
    rejected_with_candidate = [row for row in rows if not row["accepted"] and row["candidate"]]
    all_with_candidate = [row for row in rows if row["candidate"]]
    by_fold = defaultdict(list)
    for row in rows:
        by_fold[row["fold"]].append(row)
    return {
        "all_mentions_top1": relation_counts(rows),
        "accepted_predictions": relation_counts(accepted),
        "rejected_top1_candidates_diagnostic_only": relation_counts(rejected_with_candidate),
        "all_available_top1_candidates": relation_counts(all_with_candidate),
        "strict_accepted_precision": sum(row["strict_string_match"] for row in accepted) / len(accepted) if accepted else None,
        "accepted_non_exact_hierarchy_related": sum(
            row["relation"] in {"same_icd_code_different_name", "candidate_is_broader_parent", "candidate_is_narrower_child"}
            for row in accepted
        ),
        "by_fold_accepted_predictions": {
            str(fold): relation_counts([row for row in fold_rows if row["accepted"]])
            for fold, fold_rows in sorted(by_fold.items())
        },
        "by_fold_rejected_top1_candidates": {
            str(fold): relation_counts([row for row in fold_rows if not row["accepted"] and row["candidate"]])
            for fold, fold_rows in sorted(by_fold.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--icd", type=Path, default=ROOT / "data" / "dicts" / "icd10_beijing_v601.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    audit = json.loads(args.input.read_text(encoding="utf-8"))
    icd = json.loads(args.icd.read_text(encoding="utf-8"))
    rows = audit_rows(audit["records"], icd["name_to_codes"])
    result = {
        "protocol": audit["protocol"],
        "input": str(args.input),
        "icd_dictionary": str(args.icd),
        "rows": len(rows),
        "interpretation": "ICD-code relationships are diagnostic only; only exact normalized-name match counts as correct in this report.",
        "summary": summarize(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    columns = ["fold", "surface", "gold", "accepted", "strict_string_match", "candidate", "candidate_source",
               "relation", "candidate_icd_codes", "gold_icd_codes", "confidence", "margin", "decision_reason"]
    with args.csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row,
                "gold": " || ".join(row["gold"]),
                "candidate_icd_codes": " || ".join(row["candidate_icd_codes"]),
                "gold_icd_codes": " || ".join(row["gold_icd_codes"]),
            })
    print(json.dumps({"out": str(args.out), "csv": str(args.csv), "summary": result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
