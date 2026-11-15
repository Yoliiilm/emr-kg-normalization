"""Prepare blinded CCKS2017 disease-normalization annotation files.

The annotator CSV never contains system outputs. Predictions are written to a
separate sealed-key file and must not be shown before adjudication is locked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ccks2017" / "data_origin"
OUT = ROOT / "outputs" / "metrics" / "experiments" / "independent_gold"
SECTIONS = ("一般项目", "病史特点", "诊疗经过", "出院情况")
DEFAULT_MODEL = ROOT / "outputs" / "models" / "experiments" / "source_shrinkage" / "normalization_calibrator_sapbert_source_chip_reject.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def context_for(text: str, start: int, end: int, radius: int = 100) -> str:
    left, right = max(0, start - radius), min(len(text), end + radius)
    boundaries = "。！？；\n"
    while left < start and text[left] not in boundaries:
        left += 1
    while right > end and text[right - 1] not in boundaries:
        right -= 1
    return text[left:right].strip()


def collect_mentions() -> tuple[dict[str, list[dict]], list[Path]]:
    mentions: dict[str, list[dict]] = defaultdict(list)
    files = []
    for section in SECTIONS:
        folder = RAW / section
        for ann in sorted(folder.glob(f"{section}-*.txt")):
            if ann.name.endswith("original.txt"):
                continue
            files.append(ann)
            original = folder / f"{ann.name}original.txt"
            files.append(original)
            text = original.read_text(encoding="utf-8-sig").strip()
            record_id = f"{section}-{ann.stem.split('-')[1]}"
            for line in ann.read_text(encoding="utf-8-sig").splitlines():
                parts = line.split("\t")
                if len(parts) != 4 or parts[3].strip() != "疾病和诊断":
                    continue
                surface, start, end = parts[0], int(parts[1]), int(parts[2]) + 1
                if text[start:end] != surface:
                    continue
                mentions[surface].append({
                    "record_id": record_id,
                    "context": context_for(text, start, end),
                })
    return dict(mentions), files


def annotation_rows(mentions: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for surface in sorted(mentions):
        contexts = list(dict.fromkeys(item["context"] for item in mentions[surface]))[:3]
        digest = hashlib.sha1(surface.encode("utf-8")).hexdigest()[:10]
        row = {
            "blind_id": f"D{digest}",
            "disease_surface": surface,
            "mention_count": len(mentions[surface]),
        }
        for i in range(3):
            row[f"context_{i + 1}"] = contexts[i] if i < len(contexts) else ""
        for annotator in ("A", "B", "adjudicated"):
            row[f"{annotator}_standard_name"] = ""
            row[f"{annotator}_icd10_code"] = ""
            row[f"{annotator}_certainty"] = ""
            row[f"{annotator}_notes"] = ""
        rows.append(row)
    return rows


def predict(mentions: dict[str, list[dict]], model_path: Path, retriever: str, neural: bool) -> dict:
    from calibrated_normalize import (
        CandidateGenerator,
        ConfidenceCalibrator,
        install_alias_rows,
        load_chip_rows,
        split_calibration_rows,
    )
    from eval_ablation import FULL
    import normalize

    normalize.configure_normalization(**{**FULL, "use_chip_cdn": False}, retriever=retriever, calibration_mode="off")
    train = load_chip_rows(ROOT / "data" / "dicts" / "raw" / "CHIP-CDN_train.json")
    fit_rows, _ = split_calibration_rows(train)
    install_alias_rows(fit_rows)
    generator = CandidateGenerator(include_neural=neural, retriever=retriever)
    calibrator = ConfidenceCalibrator.load(model_path)
    out = {}
    for surface in sorted(mentions):
        decision = calibrator.decide(surface, generator.generate(surface))
        predicted_name = decision.name if decision.accepted else surface
        out[surface] = {
            "accepted": decision.accepted,
            "normalized_name": predicted_name,
            "icd10_code": normalize.icd_code_for(predicted_name, "疾病和诊断"),
            "source": decision.source,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "candidate_list": list(decision.candidates),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--retriever", choices=("roberta", "sapbert"), default="sapbert")
    parser.add_argument("--no-neural", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()

    mentions, files = collect_mentions()
    rows = annotation_rows(mentions)
    predictions = predict(mentions, args.model, args.retriever, not args.no_neural)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blind_path = args.out_dir / "ccks2017_blind_annotation.csv"
    sealed_path = args.out_dir / "ccks2017_sealed_predictions.json"
    with blind_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sealed = {
        "protocol": "Predictions are sealed until two independent annotations and adjudication are locked.",
        "model": str(args.model),
        "retriever": args.retriever,
        "neural_candidates": not args.no_neural,
        "n_unique_surfaces": len(mentions),
        "n_mentions": sum(map(len, mentions.values())),
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(files))},
        "model_sha256": sha256(args.model),
        "predictions": predictions,
    }
    sealed_path.write_text(json.dumps(sealed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "unique_disease_surfaces": len(mentions),
        "disease_mentions": sum(map(len, mentions.values())),
        "blind_annotation_csv": str(blind_path),
        "sealed_predictions": str(sealed_path),
        "annotators": "Two independent annotators; adjudicate disagreements before unsealing predictions.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
