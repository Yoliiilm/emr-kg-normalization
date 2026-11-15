"""Create or verify a SHA-256 snapshot of the frozen normalization system."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs" / "metrics" / "experiments" / "system_freeze" / "frozen_system_manifest.json"

FROZEN_PATHS = (
    "src/normalize.py",
    "src/clinical_gates.py",
    "src/calibrated_normalize.py",
    "src/fit_calibrator.py",
    "src/eval_calibrator.py",
    "src/check_calibration_folds.py",
    "src/eval_cdn_baselines.py",
    "data/dicts/icd10_beijing_v601.json",
    "data/dicts/icd_roberta.pt",
    "data/dicts/icd_sapbert.pt",
    "data/dicts/disease_alias_extra.json",
    "data/dicts/drug_alias.json",
    "data/dicts/drug_alias_public.json",
    "data/dicts/surgery_alias.json",
    "data/dicts/chip_cdn_alias.json",
    "data/models/sapbert_cdn/config.json",
    "data/models/sapbert_cdn/model.safetensors",
    "data/models/sapbert_cdn/tokenizer.json",
    "data/models/sapbert_cdn/tokenizer_config.json",
    "data/dicts/raw/CHIP-CDN_train.json",
    "data/dicts/raw/CHIP-CDN_dev.json",
    "data/dicts/raw/CHIP-CDN_test.json",
    "data/processed/train.bio",
    "data/processed/dev.bio",
    "data/processed/test.bio",
    "data/processed/test_pred.bio",
    "outputs/models/experiments/source_shrinkage/normalization_calibrator_sapbert_source_chip_reject.json",
)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def current_snapshot() -> dict[str, str]:
    snapshot = {}
    missing = []
    for relative in FROZEN_PATHS:
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
        else:
            snapshot[relative] = digest(path)
    if missing:
        raise SystemExit("Missing frozen inputs: " + ", ".join(missing))
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true", help="create a manifest; refuses to overwrite")
    action.add_argument("--verify", action="store_true", help="compare files against the frozen manifest")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest_path = args.manifest

    if args.create:
        if manifest_path.exists():
            raise SystemExit(f"Refusing to overwrite existing freeze manifest: {manifest_path}")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "freeze_date": date(2026, 10, 25).isoformat(),
            "status": "frozen; no rule, alias, model, threshold, or source-policy changes after this snapshot",
            "formal_normalization_benchmark": "CHIP-CDN dev; fixed held-out benchmark, historically inspected before freeze",
            "calibration_policy": {
                "train_only": True,
                "source_model_mode": "source",
                "source_shrinkage_strength": 0,
                "min_source_positives": 20,
                "min_source_negatives": 50,
                "rejected_sources": ["chip"],
                "threshold": 0.76,
                "min_margin": 0.12,
            },
            "warning": "data/dicts/chip_cdn_alias.json contains train/dev-derived aliases and must not be enabled for the formal CHIP-CDN dev evaluation; eval_calibrator defaults to train-fit-only aliases.",
            "files": current_snapshot(),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"created": str(manifest_path), "files": len(manifest["files"])}, ensure_ascii=False))
        return

    if not manifest_path.is_file():
        raise SystemExit(f"Freeze manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = manifest["files"]
    current = current_snapshot()
    changed = [path for path, value in frozen.items() if current.get(path) != value]
    added = sorted(set(current) - set(frozen))
    removed = sorted(set(frozen) - set(current))
    result = {"verified": not (changed or added or removed), "files_checked": len(frozen),
              "changed": changed, "added": added, "removed": removed}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
