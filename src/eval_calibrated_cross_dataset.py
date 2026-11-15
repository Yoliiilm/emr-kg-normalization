"""Transfer the CHIP-CDN-fitted abstaining normalizer to CCKS2017.

CCKS2017 has disease spans but no canonical-name gold. This experiment therefore
reports acceptance/ICD coverage and independent external-KG edge support, not
normalization precision.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from calibrated_normalize import (
    CandidateGenerator,
    ConfidenceCalibrator,
    install_alias_rows,
    load_chip_rows,
    split_calibration_rows,
)
from eval_ablation import FULL
from eval_cross_dataset import compare, load_ccks2017, norm_graph
from eval_external_kg import doc_level_graph, load_cmeie, load_medicalkg
from eval_g_raw_vs_g_norm import collect_freq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "outputs" / "models" / "experiments" / "source_shrinkage" / "normalization_calibrator_neural_source_chip_reject.json"
DEFAULT_OUT = ROOT / "outputs" / "metrics" / "experiments" / "cross_dataset_calibration" / "ccks2017.json"


def calibrated_mapping(docs, model_path: Path, retriever: str, include_neural: bool) -> tuple[dict, dict]:
    from normalize import canonical_names, configure_normalization, get_index

    settings = {
        **FULL,
        "use_chip_cdn": False,
        "retriever": retriever,
        "calibration_mode": "off",
    }
    configure_normalization(**settings)
    train_rows = load_chip_rows(ROOT / "data" / "dicts" / "raw" / "CHIP-CDN_train.json")
    fit_rows, _calibration_rows = split_calibration_rows(train_rows)
    install_alias_rows(fit_rows)
    generator = CandidateGenerator(include_neural=include_neural, retriever=retriever)
    calibrator = ConfidenceCalibrator.load(model_path)
    freq = collect_freq(docs)
    mapping = {}
    disease_mentions = sum(freq.get("疾病和诊断", {}).values())
    accepted_mentions = accepted_surfaces = accepted_icd_mentions = 0
    sources = Counter()
    for label, surfaces in freq.items():
        for surface, count in surfaces.items():
            if label != "疾病和诊断":
                names = canonical_names(surface, label)
                mapping[(label, surface)] = names[0] if len(names) == 1 else names
                continue
            decision = calibrator.decide(surface, generator.generate(surface))
            mapping[(label, surface)] = decision.name if decision.accepted else surface
            sources[decision.source] += count
            if decision.accepted:
                accepted_surfaces += 1
                accepted_mentions += count
                if decision.name in get_index().names:
                    accepted_icd_mentions += count

    summary = {
        "unique_disease_surfaces": len(freq.get("疾病和诊断", {})),
        "disease_mentions": disease_mentions,
        "accepted_unique_surfaces": accepted_surfaces,
        "accepted_mentions": accepted_mentions,
        "surface_coverage": accepted_surfaces / max(1, len(freq.get("疾病和诊断", {}))),
        "mention_coverage": accepted_mentions / max(1, disease_mentions),
        "accepted_names_with_icd": accepted_icd_mentions,
        "accepted_icd_hit_rate": accepted_icd_mentions / max(1, accepted_mentions),
        "mention_weighted_decision_sources": dict(sources),
        "strict_precision": None,
        "precision_note": "CCKS2017 provides disease spans but no canonical-name gold; precision is not measurable.",
    }
    return mapping, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--retriever", choices=("roberta", "sapbert"), default="roberta")
    parser.add_argument("--no-neural", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    docs, load_stats = load_ccks2017()
    raw = doc_level_graph(docs, None)
    legacy = norm_graph(docs, FULL)
    calibrated_map, acceptance = calibrated_mapping(docs, args.model, args.retriever, not args.no_neural)
    calibrated = doc_level_graph(docs, calibrated_map)
    sources = {
        "cmeie": load_cmeie(ROOT / "data" / "dicts" / "raw" / "CMeIE"),
        "medicalkg": load_medicalkg(ROOT / "data" / "dicts" / "raw" / "medicalkg" / "medical.json"),
    }
    external = {}
    for source_name, source in sources.items():
        external[source_name] = {
            "legacy_full": compare(raw, legacy, source),
            "calibrated_abstain": compare(raw, calibrated, source),
        }
    result = {
        "protocol": "External transfer from CHIP-CDN train-fit calibrator to CCKS2017 annotated training corpus; no dev used.",
        "model": str(args.model),
        "retriever": args.retriever,
        "neural_candidates": not args.no_neural,
        "records": len(docs),
        "load_stats": load_stats,
        "acceptance": acceptance,
        "external_kg_edge_support": external,
        "limitations": [
            "CCKS2017 does not include canonical disease names, so strict normalization precision/recall cannot be computed.",
            "External-KG support is corroboration, not clinical correctness or treatment recommendation.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
