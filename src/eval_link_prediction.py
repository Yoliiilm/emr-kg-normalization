"""Record-held-out TransE comparison for raw and normalized disease-drug graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_g_raw_vs_g_norm import collect_freq, graph_counts
from export_graph import node_id
from ner_data import load_bio_docs
from normalize import build_canonical_map

DEFAULT_OUT = ROOT / "outputs" / "metrics" / "experiments" / "link_prediction" / "paired_train_only" / "transe.json"


def make_transe(n_entities: int, dim: int):
    import torch
    from torch import nn

    class TransE(nn.Module):
        def __init__(self):
            super().__init__()
            self.entities = nn.Embedding(n_entities, dim)
            self.relation = nn.Embedding(1, dim)
            nn.init.uniform_(self.entities.weight, -6 / dim**0.5, 6 / dim**0.5)
            nn.init.uniform_(self.relation.weight, -6 / dim**0.5, 6 / dim**0.5)
            self.normalize_()

        @torch.no_grad()
        def normalize_(self):
            self.entities.weight.copy_(nn.functional.normalize(self.entities.weight, p=2, dim=1))
            self.relation.weight.copy_(nn.functional.normalize(self.relation.weight, p=2, dim=1))

        def score(self, heads: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
            relation = self.relation.weight[0]
            return torch.linalg.vector_norm(self.entities(heads) + relation - self.entities(tails), ord=1, dim=-1)

    return TransE()


def graph_pairs(docs, canon_map=None) -> set[tuple[str, str]]:
    _nodes, relations, _surfaces, _evidence = graph_counts(docs, canon_map)
    return {(source, target) for (source, rel, target), _weight in relations.items() if rel == "使用药物"}


def map_disease_drug_pair(pair: tuple[str, str], canon_map: dict) -> tuple[str, str] | None:
    """Map a raw disease-drug pair through a train-only map; skip split diagnoses."""
    disease_id, drug_id = pair
    disease = disease_id.split("::", 1)[1]
    drug = drug_id.split("::", 1)[1]
    mapped_disease = canon_map.get(("疾病和诊断", disease), disease)
    mapped_drug = canon_map.get(("药物", drug), drug)
    if isinstance(mapped_disease, tuple) or isinstance(mapped_drug, tuple):
        return None
    return (
        node_id(mapped_disease, "疾病和诊断"),
        node_id(mapped_drug, "药物"),
    )


def paired_test_edges(
    raw_train: set[tuple[str, str]],
    raw_heldout: set[tuple[str, str]],
    norm_train: set[tuple[str, str]],
    raw_entities: set[str],
    norm_entities: set[str],
    canon_map: dict,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Return one-to-one raw/norm test edges eligible in both graph vocabularies."""
    eligible = []
    for raw_pair in sorted(raw_heldout - raw_train):
        norm_pair = map_disease_drug_pair(raw_pair, canon_map)
        if norm_pair is None or norm_pair in norm_train:
            continue
        if not set(raw_pair) <= raw_entities or not set(norm_pair) <= norm_entities:
            continue
        eligible.append((raw_pair, norm_pair))

    # Several raw aliases may collapse to one normalized edge. Keep a stable
    # one-to-one paired sample so both metrics use exactly the same test count.
    paired = {}
    for raw_pair, norm_pair in eligible:
        paired.setdefault(norm_pair, raw_pair)
    return set(paired.values()), set(paired)


def evaluate_degree(test_pairs, entity_ids, train_pairs, all_test_pairs) -> dict:
    """Rank candidate drugs by training-graph degree as a non-embedding baseline."""
    train_by_head: dict[int, set[int]] = defaultdict(set)
    tail_degree: dict[int, int] = defaultdict(int)
    test_by_head: dict[int, set[int]] = defaultdict(set)
    for head, tail in train_pairs:
        train_by_head[entity_ids[head]].add(entity_ids[tail])
        tail_degree[entity_ids[tail]] += 1
    for head, tail in all_test_pairs:
        test_by_head[entity_ids[head]].add(entity_ids[tail])
    drug_ids = sorted({entity_ids[t] for _h, t in train_pairs})
    ranks = []
    for head_name, tail_name in sorted(test_pairs):
        head, target = entity_ids[head_name], entity_ids[tail_name]
        candidates = [
            drug for drug in drug_ids
            if drug not in train_by_head[head]
            and drug not in (test_by_head[head] - {target})
        ]
        if target not in candidates:
            candidates.append(target)
        ordered = sorted(candidates, key=lambda drug: (-tail_degree[drug], drug))
        ranks.append(1 + ordered.index(target))
    return {
        "n_test_edges": len(ranks),
        "mrr": sum(1 / rank for rank in ranks) / len(ranks) if ranks else 0.0,
        "hits_at_1": sum(rank <= 1 for rank in ranks) / len(ranks) if ranks else 0.0,
        "hits_at_3": sum(rank <= 3 for rank in ranks) / len(ranks) if ranks else 0.0,
        "hits_at_10": sum(rank <= 10 for rank in ranks) / len(ranks) if ranks else 0.0,
        "mean_filtered_rank": sum(ranks) / len(ranks) if ranks else 0.0,
    }


def train_transe(train_pairs, entity_ids, *, dim, epochs, batch_size, margin, seed):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    model = make_transe(len(entity_ids), dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    pairs = [(entity_ids[h], entity_ids[t]) for h, t in sorted(train_pairs)]
    drug_ids = sorted({t for _h, t in pairs})
    known = set(pairs)
    if not pairs or not drug_ids:
        raise ValueError("training split has no disease-drug edges")
    positives = torch.tensor(pairs, dtype=torch.long)
    for _epoch in range(epochs):
        order = torch.randperm(len(pairs))
        for start in range(0, len(pairs), batch_size):
            batch = positives[order[start : start + batch_size]]
            negative_tails = []
            for head, _tail in batch.tolist():
                candidate = random.choice(drug_ids)
                attempts = 0
                while (head, candidate) in known and attempts < 20:
                    candidate = random.choice(drug_ids)
                    attempts += 1
                negative_tails.append(candidate)
            heads, tails = batch[:, 0], batch[:, 1]
            neg_tails = torch.tensor(negative_tails, dtype=torch.long)
            loss = torch.relu(margin + model.score(heads, tails) - model.score(heads, neg_tails)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.normalize_()
    return model


def evaluate(model, test_pairs, entity_ids, train_pairs, all_test_pairs, seed: int) -> dict:
    import torch

    rng = random.Random(seed)
    drug_ids = sorted({entity_ids[t] for _h, t in train_pairs})
    train_by_head: dict[int, set[int]] = defaultdict(set)
    for head, tail in train_pairs:
        train_by_head[entity_ids[head]].add(entity_ids[tail])
    test_by_head: dict[int, set[int]] = defaultdict(set)
    for head, tail in all_test_pairs:
        test_by_head[entity_ids[head]].add(entity_ids[tail])
    ranks, auc_wins, auc_total = [], 0, 0
    with torch.no_grad():
        for head_name, tail_name in sorted(test_pairs):
            head, target = entity_ids[head_name], entity_ids[tail_name]
            candidates = [d for d in drug_ids if d not in train_by_head[head] and d not in (test_by_head[head] - {target})]
            if target not in candidates:
                candidates.append(target)
            candidate_tensor = torch.tensor(candidates, dtype=torch.long)
            head_tensor = torch.full_like(candidate_tensor, head)
            scores = model.score(head_tensor, candidate_tensor)
            target_score = model.score(torch.tensor([head]), torch.tensor([target]))[0]
            ranks.append(1 + int((scores < target_score).sum().item()))

            negatives = [d for d in candidates if d != target]
            if len(negatives) > 100:
                negatives = rng.sample(negatives, 100)
            if negatives:
                neg_tensor = torch.tensor(negatives, dtype=torch.long)
                neg_scores = model.score(torch.full_like(neg_tensor, head), neg_tensor)
                auc_wins += int((neg_scores > target_score).sum().item())
                auc_total += len(negatives)
    if not ranks:
        return {"n_test_edges": 0, "mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "hits_at_10": 0.0, "sampled_auc": None}
    return {
        "n_test_edges": len(ranks),
        "mrr": sum(1 / rank for rank in ranks) / len(ranks),
        "hits_at_1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "hits_at_3": sum(rank <= 3 for rank in ranks) / len(ranks),
        "hits_at_10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "sampled_auc": auc_wins / auc_total if auc_total else None,
        "mean_filtered_rank": sum(ranks) / len(ranks),
    }


def run_graph(train_docs, heldout_docs, canon_map, normalized: bool, args, paired_edges) -> dict:
    mapping = canon_map if normalized else None
    train_pairs = graph_pairs(train_docs, mapping)
    heldout_pairs = graph_pairs(heldout_docs, mapping)
    entity_ids = {name: index for index, name in enumerate(sorted({x for pair in train_pairs for x in pair}))}
    eligible = paired_edges["g_norm"] if normalized else paired_edges["g_raw"]
    if not eligible:
        raise ValueError("no common held-out disease-drug edges are eligible in both graphs")
    model = train_transe(
        train_pairs,
        entity_ids,
        dim=args.dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        margin=args.margin,
        seed=args.seed,
    )
    metrics = evaluate(model, eligible, entity_ids, train_pairs, eligible, args.seed + 1)
    degree_metrics = evaluate_degree(eligible, entity_ids, train_pairs, eligible)
    return {
        "train_edges": len(train_pairs),
        "heldout_edges": len(heldout_pairs),
        "eligible_unseen_edges": len(eligible),
        "disease_entities": sum(name.startswith("疾病和诊断::") for name in entity_ids),
        "drug_entities": sum(name.startswith("药物::") for name in entity_ids),
        "metrics": metrics,
        "degree_baseline": degree_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "train.bio")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    import torch

    if not 0 < args.holdout_fraction < 1:
        parser.error("--holdout-fraction must be between 0 and 1")
    docs = load_bio_docs(args.data)
    order = list(range(len(docs)))
    random.Random(args.seed).shuffle(order)
    n_heldout = max(1, min(len(docs) - 1, round(len(docs) * args.holdout_fraction)))
    heldout_indices = set(order[:n_heldout])
    train_docs = [doc for i, doc in enumerate(docs) if i not in heldout_indices]
    heldout_docs = [doc for i, doc in enumerate(docs) if i in heldout_indices]
    # Canonicalization is fitted on training records only; held-out surfaces
    # unseen during training fall back to their raw spelling.
    canon_map = build_canonical_map(collect_freq(train_docs))
    raw_train = graph_pairs(train_docs)
    raw_heldout = graph_pairs(heldout_docs)
    norm_train = graph_pairs(train_docs, canon_map)
    raw_entities = {name for pair in raw_train for name in pair}
    norm_entities = {name for pair in norm_train for name in pair}
    paired_raw, paired_norm = paired_test_edges(
        raw_train, raw_heldout, norm_train, raw_entities, norm_entities, canon_map
    )
    paired_edges = {"g_raw": paired_raw, "g_norm": paired_norm}
    result = {
        "method": "record-held-out TransE; train-only canonical map; one-to-one paired unseen disease-drug edges eligible in both graph vocabularies; filtered tail ranking",
        "data": str(args.data),
        "n_records": len(docs),
        "n_train_records": len(train_docs),
        "n_heldout_records": len(heldout_docs),
        "holdout_fraction": args.holdout_fraction,
        "seed": args.seed,
        "hyperparameters": {"dim": args.dim, "epochs": args.epochs, "batch_size": args.batch_size, "margin": args.margin},
        "paired_test_edges": len(paired_raw),
        "paired_test_edges_sha256": hashlib.sha256(
            json.dumps(
                [(raw, map_disease_drug_pair(raw, canon_map)) for raw in sorted(paired_raw)],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "excluded_unpaired_raw_edges": len(raw_heldout - paired_raw),
        "excluded_unpaired_norm_edges": len(graph_pairs(heldout_docs, canon_map) - paired_norm),
        "canonical_map_source": "training records only",
        "g_raw": run_graph(train_docs, heldout_docs, canon_map, False, args, paired_edges),
        "g_norm": run_graph(train_docs, heldout_docs, canon_map, True, args, paired_edges),
    }
    result["delta_norm_minus_raw"] = {
        metric: result["g_norm"]["metrics"][metric] - result["g_raw"]["metrics"][metric]
        for metric in ("mrr", "hits_at_1", "hits_at_3", "hits_at_10", "sampled_auc")
        if result["g_norm"]["metrics"][metric] is not None and result["g_raw"]["metrics"][metric] is not None
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
