"""Evaluate the frozen surgery normalizer on the local CHIP-2019 BEN vocabulary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from normalize import configure_normalization, lookup_with_source, surgery_core

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "dicts" / "raw" / "手术标注词2500.tsv"
DEFAULT_OUT = ROOT / "outputs" / "metrics" / "experiments" / "surgery_ben" / "complete_eval_20261026.json"
ACCEPTED_SOURCES = {"surgery_alias", "surgery_norm"}


def strict_key(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).strip()


def relaxed_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", strict_key(text), flags=re.UNICODE).casefold()


def read_ben(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError(f"Expected exactly three TSV columns in every row: {path}")
    records = []
    for raw, code, gold in rows:
        raw, code, gold = raw.strip(), code.strip(), gold.strip()
        if not raw or not code or not gold:
            raise ValueError(f"Blank value in BEN row: {raw!r}, {code!r}, {gold!r}")
        records.append({"raw": raw, "code": code, "gold": gold})
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(rows: list[dict[str, str]], prediction_key: str, codes_by_name: dict[str, set[str]]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0, "name_exact": None, "code_accuracy": None}
    name_correct = 0
    code_correct = 0
    for row in rows:
        pred = row[prediction_key]
        name_correct += strict_key(pred) == strict_key(row["gold"])
        code_correct += row["code"] in codes_by_name.get(relaxed_key(pred), set())
    return {
        "n": n,
        "name_exact": name_correct / n,
        "code_accuracy": code_correct / n,
        "name_exact_count": name_correct,
        "code_correct_count": code_correct,
    }


def paired_bootstrap_delta(rows: list[dict[str, str]], codes_by_name: dict[str, set[str]], *, seed: int = 20261026, n_boot: int = 5000) -> dict:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["code"]].append(row)
    code_groups = list(groups.values())

    def correct(row: dict[str, str], key: str) -> bool:
        return row["code"] in codes_by_name.get(relaxed_key(row[key]), set())

    wins = sum(correct(row, "prediction") and not correct(row, "raw") for row in rows)
    losses = sum(correct(row, "raw") and not correct(row, "prediction") for row in rows)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        sampled_groups = [rng.choice(code_groups) for _ in code_groups]
        sampled = [row for group in sampled_groups for row in group]
        if sampled:
            delta = sum(correct(row, "prediction") - correct(row, "raw") for row in sampled) / len(sampled)
            deltas.append(delta)
    deltas.sort()
    return {
        "unit": "ICD-9 code cluster; all aliases of a sampled code stay together",
        "seed": seed,
        "replicates": len(deltas),
        "paired_accuracy_delta": (sum(correct(row, "prediction") - correct(row, "raw") for row in rows) / len(rows)) if rows else None,
        "bootstrap_95_ci": [deltas[int(0.025 * len(deltas))], deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]] if deltas else [None, None],
        "discordant_pairs_normalizer_wins": wins,
        "discordant_pairs_normalizer_losses": losses,
    }


def candidate_retrieval(records: list[dict[str, str]]) -> dict:
    """Fixed character TF-IDF over the supplied standard-name candidate vocabulary."""
    candidate_codes: dict[str, set[str]] = defaultdict(set)
    for row in records:
        candidate_codes[row["gold"]].add(row["code"])
    candidates = sorted(candidate_codes)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=False)
    candidate_vectors = vectorizer.fit_transform(candidates)
    query_vectors = vectorizer.transform([row["raw"] for row in records])
    scores = (query_vectors @ candidate_vectors.T).toarray()
    top = np.argsort(-scores, axis=1, kind="stable")[:, : min(5, len(candidates))]
    hits = {k: 0 for k in (1, 3, 5)}
    for row, ranked in zip(records, top):
        for k in hits:
            if any(row["code"] in candidate_codes[candidates[idx]] for idx in ranked[:k]):
                hits[k] += 1
    return {
        "method": "scikit-learn character TF-IDF; analyzer=char, ngram_range=(2,4), fixed without tuning",
        "candidate_vocabulary": "unique standardized names in the BEN file (1,089); no query-to-label training",
        "n_candidates": len(candidates),
        "n_queries": len(records),
        "hits_at_1": hits[1] / len(records),
        "hits_at_3": hits[3] / len(records),
        "hits_at_5": hits[5] / len(records),
        "counts": {f"hits_at_{k}": v for k, v in hits.items()},
    }


def evaluate(records: list[dict[str, str]], alias_path: Path, data_path: Path) -> dict:
    configure_normalization()
    from normalize import get_index

    index = get_index()
    codes_by_name: dict[str, set[str]] = defaultdict(set)
    for row in records:
        codes_by_name[relaxed_key(row["gold"])].add(row["code"])

    aliases = index.surgery_alias
    alias_keys = {strict_key(key) for key in aliases}
    alias_core_keys = {strict_key(surgery_core(key)) for key in aliases}
    evaluated = []
    for row in records:
        prediction, source = lookup_with_source(row["raw"], "手术")
        evaluated.append({
            **row,
            "prediction": prediction,
            "source": source,
            "table_key_overlap": strict_key(row["raw"]) in alias_keys or strict_key(surgery_core(row["raw"])) in alias_core_keys,
        })

    accepted = [row for row in evaluated if row["source"] in ACCEPTED_SOURCES]
    overlap_excluded = [row for row in evaluated if not row["table_key_overlap"]]
    source_metrics = {}
    for source, subset in sorted((key, [row for row in evaluated if row["source"] == key]) for key in {row["source"] for row in evaluated}):
        source_metrics[source] = metric(subset, "prediction", codes_by_name)

    by_code = defaultdict(list)
    for row in evaluated:
        by_code[row["code"]].append(row)
    code_group_accuracy = [metric(rows, "prediction", codes_by_name)["code_accuracy"] for rows in by_code.values()]
    overall = metric(evaluated, "prediction", codes_by_name)
    raw_baseline = metric(evaluated, "raw", codes_by_name)
    exact_accepted = metric(accepted, "prediction", codes_by_name)

    return {
        "experiment": "frozen surgery normalization on CHIP-2019 BEN auxiliary vocabulary",
        "run_date": date(2026, 10, 26).isoformat(),
        "scope_warning": "Auxiliary surgery-name normalization evaluation only; it is not disease mention normalization evidence or record-level ICD coding evidence.",
        "inputs": {
            "dataset": str(data_path.relative_to(ROOT)),
            "dataset_sha256": sha256(data_path),
            "frozen_surgery_alias": str(alias_path.relative_to(ROOT)),
            "frozen_surgery_alias_sha256": sha256(alias_path),
            "n_rows": len(records),
            "n_unique_surfaces": len({row["raw"] for row in records}),
            "n_icd9_codes": len({row["code"] for row in records}),
            "n_standard_names": len({row["gold"] for row in records}),
        },
        "protocol": {
            "system": "current frozen normalize.lookup_with_source(..., '手术')",
            "baseline": "identity: retain raw operation name",
            "name_match": "NFKC plus whitespace removal; punctuation is retained",
            "code_match": "prediction must match a BEN canonical label after punctuation/whitespace normalization, then its gold ICD-9 code is checked",
            "accepted_sources": sorted(ACCEPTED_SOURCES),
            "no_threshold_tuning": True,
            "outputs_do_not_modify_main_model": True,
        },
        "results": {
            "identity_baseline": raw_baseline,
            "frozen_normalizer_all_rows": overall,
            "frozen_normalizer_accepted_only": {
                **exact_accepted,
                "coverage": len(accepted) / len(evaluated) if evaluated else None,
                "precision": exact_accepted["code_accuracy"],
            },
            "source_breakdown": source_metrics,
            "candidate_retrieval_baseline": candidate_retrieval(records),
            "macro_code_accuracy": sum(code_group_accuracy) / len(code_group_accuracy) if code_group_accuracy else None,
            "exact_surface_alias_overlap_rows": sum(row["table_key_overlap"] for row in evaluated),
            "overlap_excluded_sensitivity": metric(overlap_excluded, "prediction", codes_by_name),
            "overlap_excluded_n": len(overlap_excluded),
            "paired_bootstrap_vs_identity": paired_bootstrap_delta(evaluated, codes_by_name),
        },
        "limitations": [
            "The BEN file has no declared train/dev/test split; it is treated as a fixed auxiliary benchmark, not used to tune aliases or rules.",
            "The normalizer's surgery aliases were curated for CCKS2019. Exact overlap with those table keys is reported, but the overlap-excluded slice does not rule out broader source contamination.",
            "This benchmark measures surgery mention-to-ICD-9 concept normalization, not diagnosis coding from full clinical records.",
            "The local project copy's acquisition and reuse terms should be checked against the official CBLUE/CHIP data terms before publication or redistribution.",
        ],
    }


def render_text(result: dict) -> str:
    res = result["results"]
    lines = [
        result["experiment"],
        f"Rows: {result['inputs']['n_rows']}; unique ICD-9 codes: {result['inputs']['n_icd9_codes']}; standard names: {result['inputs']['n_standard_names']}",
        f"Identity code accuracy: {res['identity_baseline']['code_accuracy']:.4f} ({res['identity_baseline']['code_correct_count']}/{res['identity_baseline']['n']})",
        f"Frozen normalizer code accuracy: {res['frozen_normalizer_all_rows']['code_accuracy']:.4f} ({res['frozen_normalizer_all_rows']['code_correct_count']}/{res['frozen_normalizer_all_rows']['n']})",
        f"Character TF-IDF candidate retrieval: Hits@1={res['candidate_retrieval_baseline']['hits_at_1']:.4f}, Hits@3={res['candidate_retrieval_baseline']['hits_at_3']:.4f}, Hits@5={res['candidate_retrieval_baseline']['hits_at_5']:.4f}",
        f"Accepted coverage: {res['frozen_normalizer_accepted_only']['coverage']:.4f}; accepted precision: {res['frozen_normalizer_accepted_only']['precision']:.4f} (n={res['frozen_normalizer_accepted_only']['n']})",
        f"Macro ICD-9 accuracy: {res['macro_code_accuracy']:.4f}",
        f"Alias-table exact-overlap rows: {res['exact_surface_alias_overlap_rows']}; overlap-excluded code accuracy: {res['overlap_excluded_sensitivity']['code_accuracy']:.4f} (n={res['overlap_excluded_n']})",
        "Source breakdown:",
    ]
    for source, values in res["source_breakdown"].items():
        lines.append(f"  {source}: n={values['n']}, code_accuracy={values['code_accuracy']:.4f}")
    lines.extend(["", "Limitations:", *(f"- {item}" for item in result["limitations"])])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--alias", type=Path, default=ROOT / "data" / "dicts" / "surgery_alias.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".txt").exists():
        raise SystemExit(f"Refusing to overwrite experiment output: {args.output}")
    records = read_ben(args.data)
    result = evaluate(records, args.alias, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".txt").write_text(render_text(result), encoding="utf-8")
    print(render_text(result))
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
