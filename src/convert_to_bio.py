"""Convert CCKS2019 / Yidu-S4K subtask1 JSONL into character-level BIO files."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

LABELS = [
    "疾病和诊断",
    "解剖部位",
    "药物",
    "手术",
    "影像检查",
    "实验室检验",
]

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
TRAIN_FILES = [
    RAW_DIR / "subtask1_training_part1.txt",
    RAW_DIR / "subtask1_training_part2.txt",
]
TEST_FILE = RAW_DIR / "subtask1_test_set_with_answer.json"
DEV_RATIO = 0.1
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no} is not valid JSON") from exc
    return records


def resolve_overlaps(entities: list[dict]) -> tuple[list[dict], int]:
    """Keep longer spans first; skip any span that collides with an accepted one."""
    cleaned = []
    for ent in entities:
        start = int(ent["start_pos"])
        end = int(ent["end_pos"])
        label = ent["label_type"]
        if end <= start:
            continue
        cleaned.append({"start_pos": start, "end_pos": end, "label_type": label})

    cleaned.sort(key=lambda e: (-(e["end_pos"] - e["start_pos"]), e["start_pos"]))
    kept = []
    occupied = set()
    skipped = 0
    for ent in cleaned:
        span = range(ent["start_pos"], ent["end_pos"])
        if any(i in occupied for i in span):
            skipped += 1
            continue
        occupied.update(span)
        kept.append(ent)
    kept.sort(key=lambda e: e["start_pos"])
    return kept, skipped


def to_bio(text: str, entities: list[dict]) -> tuple[list[tuple[str, str]], int]:
    tags = ["O"] * len(text)
    kept, skipped = resolve_overlaps(entities)
    for ent in kept:
        start, end, label = ent["start_pos"], ent["end_pos"], ent["label_type"]
        if start < 0 or end > len(text) or label not in LABELS:
            continue
        tags[start] = f"B-{label}"
        for i in range(start + 1, end):
            tags[i] = f"I-{label}"
    tokens = [(ch.replace("\n", " ").replace("\t", " "), tag) for ch, tag in zip(text, tags)]
    return tokens, skipped


def write_bio(path: Path, docs: list[list[tuple[str, str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            for ch, tag in doc:
                f.write(f"{ch} {tag}\n")
            if i < len(docs) - 1:
                f.write("\n")


def count_entities(docs: list[list[tuple[str, str]]]) -> Counter:
    counts = Counter()
    for doc in docs:
        for _, tag in doc:
            if tag.startswith("B-"):
                counts[tag[2:]] += 1
    return counts


def convert_records(records: list[dict]) -> tuple[list[list[tuple[str, str]]], dict]:
    docs = []
    skipped_overlap = 0
    n_gold = 0
    oob = 0
    for rec in records:
        text = rec["originalText"]
        ents = rec.get("entities") or []
        n_gold += len(ents)
        for ent in ents:
            if int(ent["end_pos"]) > len(text) or int(ent["start_pos"]) < 0:
                oob += 1
        tokens, skipped = to_bio(text, ents)
        skipped_overlap += skipped
        docs.append(tokens)
    stats = {
        "n_docs": len(docs),
        "n_gold_entities": n_gold,
        "n_overlap_dropped": skipped_overlap,
        "n_oob": oob,
        "n_chars": sum(len(d) for d in docs),
        "entity_counts": dict(count_entities(docs)),
    }
    return docs, stats


def main() -> None:
    train_all = []
    for path in TRAIN_FILES:
        train_all.extend(load_jsonl(path))
    test_records = load_jsonl(TEST_FILE)

    rng = random.Random(SEED)
    indices = list(range(len(train_all)))
    rng.shuffle(indices)
    n_dev = max(1, int(len(train_all) * DEV_RATIO))
    dev_idx = set(indices[:n_dev])
    train_records = [train_all[i] for i in range(len(train_all)) if i not in dev_idx]
    dev_records = [train_all[i] for i in range(len(train_all)) if i in dev_idx]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": train_records,
        "dev": dev_records,
        "test": test_records,
    }

    all_stats = {}
    for name, records in splits.items():
        docs, stats = convert_records(records)
        write_bio(OUT_DIR / f"{name}.bio", docs)
        all_stats[name] = stats

    label_list = ["O"] + [f"{p}-{lab}" for lab in LABELS for p in ("B", "I")]
    (OUT_DIR / "label_list.txt").write_text("\n".join(label_list) + "\n", encoding="utf-8")
    (OUT_DIR / "stats.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Keep a few human-readable examples for checking.
    examples = []
    for rec in train_records[:3]:
        text = rec["originalText"]
        snippets = []
        for ent in rec.get("entities") or []:
            s, e = int(ent["start_pos"]), int(ent["end_pos"])
            snippets.append(
                {
                    "span": text[s:e],
                    "label": ent["label_type"],
                    "start": s,
                    "end": e,
                }
            )
        examples.append({"text": text[:120], "entities": snippets[:8]})
    (OUT_DIR / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Wrote BIO files to", OUT_DIR)
    for name, stats in all_stats.items():
        print(
            f"{name:5s} docs={stats['n_docs']:4d}  "
            f"entities={stats['n_gold_entities']:5d}  "
            f"overlap_dropped={stats['n_overlap_dropped']:3d}  "
            f"oob={stats['n_oob']}"
        )
        print("      ", stats["entity_counts"])


if __name__ == "__main__":
    main()
