"""Score a completed CCKS2017 blind annotation sheet against its sealed predictions."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "outputs" / "metrics" / "experiments" / "independent_gold"


def norm(value: str) -> str:
    return re.sub(r"\s+", "", (value or "").strip())


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if not n:
        return None
    labels = set(a) | set(b)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else (1.0 if observed == 1 else 0.0)


def rate(rows: list[dict], weighted: bool) -> dict:
    weight = lambda r: int(r["mention_count"]) if weighted else 1
    accepted = [r for r in rows if r["prediction"]["accepted"]]
    judged = [r for r in accepted if r["gold_name"]]
    name_correct = [r for r in judged if norm(r["prediction"]["normalized_name"]) == norm(r["gold_name"])]
    code_judged = [r for r in accepted if r["gold_code"]]
    code_correct = [r for r in code_judged if norm(r["prediction"].get("icd10_code", "")) == norm(r["gold_code"])]
    total_w = sum(weight(r) for r in rows)
    accepted_w = sum(weight(r) for r in accepted)
    judged_w = sum(weight(r) for r in judged)
    correct_w = sum(weight(r) for r in name_correct)
    code_w = sum(weight(r) for r in code_judged)
    code_correct_w = sum(weight(r) for r in code_correct)
    return {
        "n_units": len(rows),
        "n_mentions": total_w,
        "accepted_units": len(accepted),
        "accepted_mentions": accepted_w,
        "accepted_coverage": accepted_w / total_w if total_w else None,
        "accepted_precision_name_evaluable": correct_w / judged_w if judged_w else None,
        "accepted_precision_name_conservative": correct_w / accepted_w if accepted_w else None,
        "accepted_evaluable_units": len(judged),
        "accepted_uncertain_or_blank_units": len(accepted) - len(judged),
        "accepted_precision_icd_evaluable": code_correct_w / code_w if code_w else None,
        "accepted_icd_evaluable_units": len(code_judged),
    }


def bootstrap_ci(rows: list[dict], weighted: bool, replicates: int, seed: int) -> dict:
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        m = rate(sample, weighted)
        p, conservative, c = (m["accepted_precision_name_evaluable"],
                              m["accepted_precision_name_conservative"], m["accepted_coverage"])
        if p is not None and conservative is not None and c is not None:
            draws.append((p, conservative, c))
    if not draws:
        return {"accepted_precision_name": None, "accepted_coverage": None}
    def interval(index: int) -> list[float]:
        values = sorted(pair[index] for pair in draws)
        return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]
    return {"accepted_precision_name_evaluable": interval(0),
            "accepted_precision_name_conservative": interval(1),
            "accepted_coverage": interval(2), "replicates": len(draws), "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_DIR / "ccks2017_blind_annotation.csv")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_DIR / "ccks2017_sealed_predictions.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR / "ccks2017_independent_gold_eval.json")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["predictions"]
    with args.annotations.open(encoding="utf-8-sig", newline="") as stream:
        sheet = list(csv.DictReader(stream))
    incomplete = [r["blind_id"] for r in sheet if not r.get("adjudicated_certainty")]
    if incomplete:
        raise SystemExit(f"Adjudication is incomplete for {len(incomplete)} rows; refusing to score.")
    rows = [{
        "blind_id": r["blind_id"],
        "mention_count": r["mention_count"],
        "gold_name": r.get("adjudicated_standard_name", ""),
        "gold_code": r.get("adjudicated_icd10_code", ""),
        "prediction": predictions[r["disease_surface"]],
    } for r in sheet]
    name_agreement = [norm(r["A_standard_name"]) for r in sheet]
    name_agreement_b = [norm(r["B_standard_name"]) for r in sheet]
    code_agreement = [norm(r["A_icd10_code"]) for r in sheet]
    code_agreement_b = [norm(r["B_icd10_code"]) for r in sheet]
    result = {
        "n_unique_surfaces": len(rows),
        "n_mentions": sum(int(r["mention_count"]) for r in rows),
        "decision_support": "Uncertain adjudications are excluded from name-precision denominator and counted separately; report their count alongside precision.",
        "surface_macro": rate(rows, False),
        "mention_weighted": rate(rows, True),
        "surface_macro_bootstrap_95ci": bootstrap_ci(rows, False, args.bootstrap, 17),
        "mention_weighted_bootstrap_95ci": bootstrap_ci(rows, True, args.bootstrap, 18),
        "annotator_agreement": {
            "name_exact_agreement": sum(a == b for a, b in zip(name_agreement, name_agreement_b)) / len(rows),
            "name_cohen_kappa": cohen_kappa(name_agreement, name_agreement_b),
            "icd_exact_agreement": sum(a == b for a, b in zip(code_agreement, code_agreement_b)) / len(rows),
            "icd_cohen_kappa": cohen_kappa(code_agreement, code_agreement_b),
        },
        "limitations": ["CCKS2017 has only 236 distinct disease surfaces; this is a limited external sample.",
                        "ICD code equality is reported separately and does not substitute for exact standardized-name correctness."],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
