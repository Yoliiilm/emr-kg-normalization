"""Run convert -> train NER -> export graph -> try Neo4j import."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from notify import run_and_notify

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PYTHON = sys.executable


def run_step(title: str, script: str, extra: list[str] | None = None) -> None:
    cmd = [PYTHON, str(SRC / script), *(extra or [])]
    print("\n" + "=" * 60)
    print(title)
    print(" ".join(cmd))
    print("=" * 60, flush=True)
    code = run_and_notify(cmd)
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def main() -> None:
    processed = ROOT / "data" / "processed"
    if not (processed / "train.bio").exists():
        run_step("1/4 Convert raw JSON to BIO", "convert_to_bio.py")
    else:
        print("BIO files exist, skip convert")

    run_step("2/4 Train RoBERTa-BiLSTM-CRF", "train_ner.py")
    run_step("3/4 Export knowledge graph files", "export_graph.py")
    run_step("4/4 Try Neo4j import (optional)", "neo4j_import.py")

    summary_path = ROOT / "outputs" / "metrics" / "summary.json"
    if summary_path.exists():
        print("\n===== NER SUMMARY =====")
        print(summary_path.read_text(encoding="utf-8"))
    print("Pipeline finished. Artifacts:")
    print("  models :", ROOT / "outputs" / "models")
    print("  metrics:", ROOT / "outputs" / "metrics")
    print("  graph  :", ROOT / "outputs" / "neo4j")


if __name__ == "__main__":
    main()
