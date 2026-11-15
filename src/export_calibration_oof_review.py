"""Export prioritized manual-review rows from a train OOF audit JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "metrics" / "experiments" / "source_shrinkage" / "train_oof_audit_neural_source_chip_reject.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "metrics" / "experiments" / "source_shrinkage" / "train_oof_priority_review.csv"


def normalized_code(code: str) -> str:
    return "".join(char for char in code.upper() if char.isalnum())


def icd_relation(candidate_codes: list[str], gold_codes: list[str]) -> str:
    if not candidate_codes or not gold_codes:
        return "missing_icd_code"
    candidate = [normalized_code(code) for code in candidate_codes]
    gold = [normalized_code(code) for code in gold_codes]
    candidate = [code for code in candidate if code]
    gold = [code for code in gold if code]
    if not candidate or not gold:
        return "missing_icd_code"
    if set(candidate) & set(gold):
        return "same_icd_code"
    if any(len(parent) >= 3 and child.startswith(parent) and len(child) > len(parent)
           for parent in candidate for child in gold):
        return "candidate_is_broader_parent"
    if any(len(parent) >= 3 and child.startswith(parent) and len(child) > len(parent)
           for parent in gold for child in candidate):
        return "candidate_is_narrower_child"
    if any(left[:3] == right[:3] for left in candidate for right in gold):
        return "same_three_character_category"
    return "different_icd_category"


def priority_rows(audit: dict, icd_names: dict[str, list[str]]) -> list[dict]:
    thresholds = {str(fold["fold"]): fold["threshold"] for fold in audit["folds"]}
    rows = []
    for record in audit["records"]:
        first = record["top5"][0] if record["top5"] else None
        top1_matches = bool(first and first["name"] in record["gold"])
        if record["accepted"] and not record["correct"]:
            bucket = "accepted_false_positive"
        elif not record["accepted"] and top1_matches and record["confidence"] >= 0.65:
            bucket = "rejected_high_score_top1_matches_gold"
        else:
            continue
        gold_names = [name for name in record["gold"] if icd_names.get(name)]
        gold_codes = sorted({code for name in gold_names for code in icd_names[name]})
        candidate_codes = icd_names.get(first["name"], []) if first else []
        rows.append({
            "review_bucket": bucket,
            "fold": record["fold"],
            "fold_threshold": thresholds[str(record["fold"])],
            "surface": record["surface"],
            "gold": " || ".join(record["gold"]),
            "candidate": first["name"] if first else "",
            "candidate_source": first["source"] if first else record["best_source"],
            "candidate_icd_codes": " || ".join(candidate_codes),
            "gold_icd_names": " || ".join(gold_names),
            "gold_icd_codes": " || ".join(gold_codes),
            "icd_relation_to_gold": icd_relation(candidate_codes, gold_codes),
            "confidence": record["confidence"],
            "margin": record["margin"],
            "decision_reason": record["reason"],
            "accepted": record["accepted"],
            "top1_matches_gold": top1_matches,
            "top5": json.dumps(record["top5"], ensure_ascii=False),
            "manual_semantic_equivalent": "",
            "manual_granularity_correct": "",
            "manual_scope_correct": "",
            "manual_final_label": "",
            "manual_notes": "",
        })
    return sorted(rows, key=lambda row: (row["review_bucket"], row["fold"], row["surface"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="allow replacing the specified generated CSV")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        parser.error(f"output exists; choose another --out or pass --force: {args.out}")

    audit = json.loads(args.input.read_text(encoding="utf-8"))
    icd_path = ROOT / "data" / "dicts" / "icd10_beijing_v601.json"
    icd_names = json.loads(icd_path.read_text(encoding="utf-8"))["name_to_codes"]
    rows = priority_rows(audit, icd_names)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "review_bucket", "fold", "fold_threshold", "surface", "gold", "candidate",
        "candidate_source", "candidate_icd_codes", "gold_icd_names", "gold_icd_codes",
        "icd_relation_to_gold", "confidence", "margin", "decision_reason", "accepted",
        "top1_matches_gold", "top5", "manual_semantic_equivalent",
        "manual_granularity_correct", "manual_scope_correct", "manual_final_label",
        "manual_notes",
    ]
    with args.out.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    icd_counts = {}
    for row in rows:
        counts[row["review_bucket"]] = counts.get(row["review_bucket"], 0) + 1
        key = f"{row['review_bucket']}|{row['icd_relation_to_gold']}"
        icd_counts[key] = icd_counts.get(key, 0) + 1
    print(json.dumps({
        "output": str(args.out),
        "rows": len(rows),
        "by_bucket": counts,
        "by_review_bucket_and_icd_relation": icd_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
