"""Build LLM normalization items: every disease mention the graph experiments will ask about,
plus all of CHIP-CDN dev, each with SapBERT-retrieved ICD candidates.

The LLM picks the correct candidate(s) or 0 (keep as is); src/llm_norm_score.py turns the
answers into outputs/llm_norm/map.json for normalize.llm_names. Candidates keep every answer a
valid ICD-10 Beijing name. Compound mentions also get candidates for each split part.

Writes outputs/llm_norm/key.json, items.csv and batch_XX.csv (id, text, candidates).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import normalize
from bert_linker import get_linker
from eval_g_raw_vs_g_norm import build_graphs
from ner_data import load_bio_docs
from normalize import WEAK_SPLIT_RE, clean_surface, configure_normalization, light_clean, strip_grade

PROCESSED = ROOT / "data" / "processed"
CDN_DEV = ROOT / "data" / "dicts" / "raw" / "CHIP-CDN_dev.json"
OUT = ROOT / "outputs" / "llm_norm"
GRAPH_SETS = (("test.bio", "test_pred.bio"), ("ccks2017_gold.bio", "ccks2017_pred.bio"))
TOP_WHOLE, TOP_PART, MAX_CANDS, BATCH = 15, 5, 20, 200


def graph_mentions() -> set[str]:
    normalize.LLM_MISSING.clear()
    for gold_name, pred_name in GRAPH_SETS:
        gold, pred = load_bio_docs(PROCESSED / gold_name), load_bio_docs(PROCESSED / pred_name)
        for mode in ("primary", "fallback"):
            configure_normalization()
            build_graphs(gold, pred, pred_settings={"llm_cdn": mode})
    configure_normalization()
    return {t for t in normalize.LLM_MISSING if t.strip()}


def cdn_texts() -> list[str]:
    rows = json.loads(CDN_DEV.read_text(encoding="utf-8"))
    return [t for t in (str(r.get("text", "")).lstrip("\ufeff").strip() for r in rows) if t]


@torch.no_grad()
def topk(linker, query: str, k: int) -> list[str]:
    if len(query) < 2:
        return []
    scores = (linker.emb @ linker.encode(query).T).squeeze(1)
    return [linker.names[i] for i in torch.topk(scores, k).indices.tolist()]


def candidates(linker, text: str) -> list[str]:
    q = strip_grade(clean_surface(text)) or text
    out = topk(linker, q, TOP_WHOLE)
    parts = [p for p in WEAK_SPLIT_RE.split(light_clean(text)) if len(p) >= 2]
    if len(parts) >= 2:
        for p in parts:
            out += topk(linker, strip_grade(clean_surface(p)) or p, TOP_PART)
    return list(dict.fromkeys(out))[:MAX_CANDS]


def main() -> None:
    graph = graph_mentions()
    cdn = cdn_texts()
    texts = list(dict.fromkeys(sorted(graph) + cdn))
    print(f"graph mentions {len(graph)}, CHIP-CDN dev {len(cdn)}, unique {len(texts)}", flush=True)

    linker = get_linker("sapbert")
    items = []
    for i, t in enumerate(texts):
        items.append({"id": f"N{i:04d}", "text": t, "cands": candidates(linker, t),
                      "graph": t in graph, "cdn": t in set(cdn)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "key.json").write_text(json.dumps(items, ensure_ascii=False, indent=0), encoding="utf-8")

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text", "candidates"])
            for r in rows:
                w.writerow([r["id"], r["text"], " | ".join(f"{j}={c}" for j, c in enumerate(r["cands"], 1))])

    write(OUT / "items.csv", items)
    for b in range(0, len(items), BATCH):
        write(OUT / f"batch_{b // BATCH:02d}.csv", items[b : b + BATCH])
    print(f"wrote {len(items)} items in {(len(items) + BATCH - 1) // BATCH} batches to {OUT}")


if __name__ == "__main__":
    main()
