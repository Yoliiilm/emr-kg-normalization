"""Train-only lexical disease normalization baseline on MedMentions (T047)."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from experiment_ncbi_external import build_aliases, choose_threshold, document_bootstrap, metrics, score_rows

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "outputs" / "external_data" / "medmentions"
DEFAULT_OUT = ROOT / "outputs" / "metrics" / "experiments" / "external_validation" / "medmentions_t047_lexical_baseline.json"
SPLITS = {
    "train": "corpus_pubtator_pmids_trng.txt",
    "dev": "corpus_pubtator_pmids_dev.txt",
    "test": "corpus_pubtator_pmids_test.txt",
}


def parse_disease_annotation(fields: list[str]) -> dict | None:
    if len(fields) != 6 or "T047" not in fields[4].split(","):
        return None
    pmid, start, end, mention, _, concept_id = fields
    try:
        int(start)
        int(end)
    except ValueError:
        return None
    concept_id = concept_id.strip()
    if not concept_id or concept_id == "-1":
        return None
    return {"pmid": pmid, "mention": mention, "type": "T047", "gold": concept_id}


def load_splits(data_dir: Path) -> tuple[dict[str, list[dict]], dict[str, int]]:
    split_ids = {name: set((data_dir / file).read_text(encoding="utf-8").split())
                 for name, file in SPLITS.items()}
    rows = {name: [] for name in SPLITS}
    with gzip.open(data_dir / "corpus_pubtator.txt.gz", "rt", encoding="utf-8") as stream:
        for raw in stream:
            row = parse_disease_annotation(raw.rstrip("\r\n").split("\t"))
            if row is None:
                continue
            for name, ids in split_ids.items():
                if row["pmid"] in ids:
                    rows[name].append(row)
                    break
    return rows, {name: len(ids) for name, ids in split_ids.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    splits, document_counts = load_splits(args.data_dir)
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
            test_scored, "retrieval_prediction", cutoff, args.bootstrap, seed=20261027
        ) if cutoff is not None else {"replicates": 0},
    }
    result = {
        "dataset": "MedMentions full corpus; semantic type T047 (Disease or Syndrome)",
        "source": "https://github.com/chanzuckerberg/MedMentions",
        "dataset_card": "https://huggingface.co/datasets/bigbio/medmentions",
        "license_status": "BigBio dataset card declares CC0-1.0; original source repository has no explicit license. Local-only experiment authorized by user; no redistribution.",
        "scope": "English UMLS-CUI disease-mention lexical baseline; not an evaluation of the frozen Chinese ICD system.",
        "protocol": {
            "document_counts": document_counts,
            "mention_counts": {name: len(rows) for name, rows in splits.items()},
            "unique_concepts_by_split": {name: len({row['gold'] for row in rows}) for name, rows in splits.items()},
            "semantic_type": "T047 only",
            "training_aliases": len(aliases),
            "threshold_selection": "development T047 mentions only: maximum coverage at target accepted precision",
            "target_precision": args.target_precision,
            "test_used_for_selection": False,
        },
        "dev_threshold_selection": threshold,
        "test": test_result,
        "limitations": [
            "English PubMed mentions and UMLS CUIs are not Chinese clinical notes or ICD-10 labels.",
            "Only T047 is evaluated; the broader MedMentions annotation contains many other semantic types.",
            "The upstream source repository has no explicit license despite the BigBio CC0 dataset-card declaration.",
            "This is a local lexical-baseline experiment, not an evaluation of the frozen Chinese system.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
