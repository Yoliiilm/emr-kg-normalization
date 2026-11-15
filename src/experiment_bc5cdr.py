"""Train-only lexical disease normalization baseline on public BC5CDR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from experiment_ncbi_external import (
    build_aliases,
    choose_threshold,
    document_bootstrap,
    metrics,
    read_annotations,
    score_rows,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "outputs" / "external_data" / "bc5cdr" / "extracted" / "CDR_Data" / "CDR.Corpus.v010516"
DEFAULT_OUT = ROOT / "outputs" / "metrics" / "experiments" / "external_validation" / "bc5cdr_lexical_baseline.json"


def read_disease_rows(path: Path) -> list[dict]:
    return [row for row in read_annotations(path) if row["type"] == "Disease" and row["gold"] != "-1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    split_files = {
        "train": "CDR_TrainingSet.PubTator.txt",
        "dev": "CDR_DevelopmentSet.PubTator.txt",
        "test": "CDR_TestSet.PubTator.txt",
    }
    splits = {name: read_disease_rows(args.data_dir / filename) for name, filename in split_files.items()}
    aliases, labels = build_aliases(splits["train"])
    alias_keys = sorted(labels)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True, sublinear_tf=True)
    alias_matrix = vectorizer.fit_transform(alias_keys)
    model = (vectorizer, alias_keys, alias_matrix)
    dev_scored = score_rows(splits["dev"], model, aliases, labels)
    test_scored = score_rows(splits["test"], model, aliases, labels)
    threshold = choose_threshold(dev_scored, args.target_precision)
    cutoff = threshold["threshold"]

    test_result = {
        "train_alias_exact": metrics(test_scored, "exact_prediction"),
        "char_tfidf_nearest_alias_at_any_score": metrics(test_scored, "retrieval_prediction"),
        "char_tfidf_thresholded": metrics(test_scored, "retrieval_prediction", cutoff) if cutoff is not None else {
            "n_mentions": len(test_scored), "n_documents": len({r['pmid'] for r in test_scored}),
            "accepted": 0, "coverage": 0.0, "accepted_precision": None, "end_to_end_accuracy": 0.0,
        },
        "char_tfidf_thresholded_document_bootstrap_95ci": document_bootstrap(
            test_scored, "retrieval_prediction", cutoff, args.bootstrap, seed=20261026
        ) if cutoff is not None else {"replicates": 0},
    }
    result = {
        "dataset": "BioCreative V Chemical Disease Relation corpus (BC5CDR)",
        "source": "https://huggingface.co/datasets/bigbio/bc5cdr",
        "upstream_source": "http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/",
        "license": "NCBI corpus README states Public Domain Notice; public use and reproduction permitted.",
        "scope": "English disease-mention lexical baseline; not an evaluation of the frozen Chinese ICD system.",
        "protocol": {
            "split_counts": {name: {"mentions": len(rows), "documents": len({r['pmid'] for r in rows})}
                             for name, rows in splits.items()},
            "included_annotations": "Disease mentions with a non-missing MeSH ID; chemical annotations and ID -1 excluded.",
            "training_aliases": len(aliases),
            "threshold_selection": "development disease mentions only: maximum coverage at target accepted precision",
            "target_precision": args.target_precision,
            "test_used_for_selection": False,
        },
        "dev_threshold_selection": threshold,
        "test": test_result,
        "limitations": [
            "The corpus is English PubMed text annotated with MeSH IDs, not Chinese ICD clinical notes.",
            "The compared lexical methods are external baselines, not the frozen Chinese normalization system.",
            "These are PubMed mention annotations, not patient-level treatment or chart-query labels.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
