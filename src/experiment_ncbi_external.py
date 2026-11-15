"""Train-only lexical baselines on the public NCBI Disease Corpus.

This is an English external benchmark, not an evaluation of the frozen Chinese
ICD normalization system. The development split selects the cosine threshold;
the official test split is used once for final reporting.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "outputs" / "external_data" / "ncbi_disease_corpus"
DEFAULT_OUT = ROOT / "outputs" / "metrics" / "experiments" / "external_validation" / "ncbi_lexical_baseline.json"


def normalize_surface(value: str) -> str:
    return re.sub(r"[^\w]+", "", value.casefold(), flags=re.UNICODE)


def read_annotations(path: Path) -> list[dict]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split("\t")
        if len(fields) != 6:
            continue
        pmid, start, end, mention, annotation_type, concept_id = fields
        try:
            int(start)
            int(end)
        except ValueError:
            continue
        rows.append({"pmid": pmid, "mention": mention, "type": annotation_type, "gold": concept_id,
                     "line": line_no})
    return rows


def build_aliases(rows: list[dict]) -> tuple[dict[str, Counter], dict[str, str]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        key = normalize_surface(row["mention"])
        if key:
            counts[key][row["gold"]] += 1
    aliases = {key: counter for key, counter in counts.items()}
    labels = {key: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
              for key, counter in aliases.items()}
    return aliases, labels


def score_rows(rows: list[dict], model, aliases: dict[str, Counter], labels: dict[str, str]) -> list[dict]:
    keys = [normalize_surface(row["mention"]) for row in rows]
    vectorizer, alias_keys, matrix = model
    query_matrix = vectorizer.transform([row["mention"] for row in rows])
    similarities = query_matrix @ matrix.T
    results = []
    for index, row in enumerate(rows):
        key = keys[index]
        if key in labels:
            exact_label = labels[key]
        else:
            exact_label = None
        scores = similarities.getrow(index)
        if scores.nnz:
            max_score = float(scores.data.max())
            tied_indices = [i for i, value in zip(scores.indices, scores.data)
                            if math.isclose(float(value), max_score, rel_tol=0.0, abs_tol=1e-12)]
            tied_labels = Counter(labels[alias_keys[i]] for i in tied_indices)
            # Resolve score ties by how often each concept occurred in train.
            winner = sorted(tied_labels, key=lambda lab: (
                -sum(aliases.get(alias_keys[i], Counter()).get(lab, 0) for i in tied_indices), lab
            ))[0]
        else:
            max_score, winner = 0.0, None
        results.append({**row, "exact_prediction": exact_label, "retrieval_prediction": winner,
                        "similarity": max_score})
    return results


def choose_threshold(dev_rows: list[dict], target_precision: float) -> dict:
    thresholds = sorted({row["similarity"] for row in dev_rows}, reverse=True)
    eligible = []
    for threshold in thresholds:
        accepted = [row for row in dev_rows if row["similarity"] >= threshold]
        precision = sum(row["retrieval_prediction"] == row["gold"] for row in accepted) / len(accepted)
        if precision >= target_precision:
            eligible.append((len(accepted), threshold, precision))
    if not eligible:
        return {"threshold": None, "accepted": 0, "precision": None, "target_precision": target_precision}
    accepted, threshold, precision = max(eligible, key=lambda item: (item[0], item[1]))
    return {"threshold": threshold, "accepted": accepted, "precision": precision,
            "target_precision": target_precision}


def metrics(rows: list[dict], prediction_key: str, threshold: float | None = None) -> dict:
    accepted = [row for row in rows if row[prediction_key] is not None and
                (threshold is None or row["similarity"] >= threshold)]
    correct = sum(row[prediction_key] == row["gold"] for row in accepted)
    return {"n_mentions": len(rows), "n_documents": len({row["pmid"] for row in rows}),
            "accepted": len(accepted), "coverage": len(accepted) / len(rows) if rows else 0.0,
            "accepted_precision": correct / len(accepted) if accepted else None,
            "end_to_end_accuracy": correct / len(rows) if rows else 0.0}


def document_bootstrap(rows: list[dict], prediction_key: str, threshold: float | None,
                       replicates: int, seed: int) -> dict:
    docs: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        docs[row["pmid"]].append(row)
    doc_ids = sorted(docs)
    if not doc_ids or replicates <= 0:
        return {"replicates": 0}
    rng = random.Random(seed)
    precision_values, coverage_values, accuracy_values = [], [], []
    for _ in range(replicates):
        sample = [row for _ in doc_ids for doc in [rng.choice(doc_ids)] for row in docs[doc]]
        result = metrics(sample, prediction_key, threshold)
        precision_values.append(result["accepted_precision"] or 0.0)
        coverage_values.append(result["coverage"])
        accuracy_values.append(result["end_to_end_accuracy"])

    def ci(values: list[float]) -> list[float]:
        values.sort()
        return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]

    return {"unit": "PMID/document", "replicates": replicates, "seed": seed,
            "accepted_precision_95ci": ci(precision_values), "coverage_95ci": ci(coverage_values),
            "end_to_end_accuracy_95ci": ci(accuracy_values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    train = read_annotations(args.data_dir / "NCBItrainset_corpus.txt")
    dev = read_annotations(args.data_dir / "NCBIdevelopset_corpus.txt")
    test = read_annotations(args.data_dir / "NCBItestset_corpus.txt")
    aliases, labels = build_aliases(train)
    alias_keys = sorted(labels)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True, sublinear_tf=True)
    alias_matrix = vectorizer.fit_transform(alias_keys)
    model = (vectorizer, alias_keys, alias_matrix)
    train_counts = {key: counter for key, counter in aliases.items()}
    dev_scored = score_rows(dev, model, train_counts, labels)
    test_scored = score_rows(test, model, train_counts, labels)
    dev_specific = [row for row in dev_scored if row["type"] == "SpecificDisease"]
    threshold = choose_threshold(dev_specific, args.target_precision)
    cutoff = threshold["threshold"]

    result = {
        "dataset": "NCBI Disease Corpus",
        "source": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/",
        "license": "Official public-domain notice: freely available for public use and reproduction; cite corpus authors.",
        "citation": ["Dogan RI, Leaman R, Lu Z. An improved corpus of disease mentions in PubMed citations. BioNLP 2012.",
                     "Dogan RI, Leaman R, Lu Z. NCBI Disease Corpus: A Resource for Disease Name Recognition and Normalization. 2014."],
        "scope": "English external lexical baseline only; not a score for the frozen Chinese ICD normalization system.",
        "protocol": {"train": {"mentions": len(train), "documents": len({r['pmid'] for r in train})},
                     "development": {"mentions": len(dev), "documents": len({r['pmid'] for r in dev})},
                     "test": {"mentions": len(test), "documents": len({r['pmid'] for r in test})},
                     "training_aliases": len(aliases), "threshold_selection": "SpecificDisease only: maximum dev coverage at target accepted precision",
                     "target_precision": args.target_precision, "test_used_for_selection": False},
        "dev_threshold_selection": threshold,
        "test": {
            "train_alias_exact": metrics(test_scored, "exact_prediction"),
            "char_tfidf_nearest_alias_at_any_score": metrics(test_scored, "retrieval_prediction"),
            "char_tfidf_thresholded": metrics(test_scored, "retrieval_prediction", cutoff) if cutoff is not None else
                {"n_mentions": len(test_scored), "accepted": 0, "coverage": 0.0,
                 "accepted_precision": None, "end_to_end_accuracy": 0.0},
            "char_tfidf_thresholded_document_bootstrap_95ci": document_bootstrap(
                test_scored, "retrieval_prediction", cutoff, args.bootstrap, seed=20261025
            ) if cutoff is not None else {"replicates": 0},
            "by_annotation_type": {},
        },
        "limitations": ["This benchmark is English and uses MeSH/OMIM IDs, unlike the frozen Chinese ICD system.",
                        "The lexical methods are deliberately simple baselines, not the proposed Chinese system.",
                        "Repeated mentions from one PMID are clustered for bootstrap intervals."],
    }
    for annotation_type in sorted({row["type"] for row in test_scored}):
        subset = [row for row in test_scored if row["type"] == annotation_type]
        result["test"]["by_annotation_type"][annotation_type] = {
            "n_mentions": len(subset),
            "train_alias_exact": metrics(subset, "exact_prediction"),
            "char_tfidf_thresholded": metrics(subset, "retrieval_prediction", cutoff) if cutoff is not None else None,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
