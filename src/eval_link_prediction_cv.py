"""Repeated record-level cross-validation for disease-drug link prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_link_prediction import (
    DEFAULT_OUT,
    evaluate,
    graph_pairs,
    map_disease_drug_pair,
    paired_test_edges,
    train_transe,
)
from eval_g_raw_vs_g_norm import collect_freq
from ner_data import load_bio_docs
from normalize import build_canonical_map

DEFAULT_DIR = DEFAULT_OUT.parent / "repeated_record_cv"
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
METRICS = ("mrr", "hits_at_1", "hits_at_3", "hits_at_10")


def fold_indices(n: int, folds: int, seed: int) -> list[set[int]]:
    if folds < 2 or folds > n:
        raise ValueError("folds must be between 2 and the number of records")
    order = list(range(n))
    random.Random(seed).shuffle(order)
    result = [set() for _ in range(folds)]
    for index, record_index in enumerate(order):
        result[index % folds].add(record_index)
    return result


def make_distmult(n_entities: int, dim: int):
    import torch
    from torch import nn

    class DistMult(nn.Module):
        def __init__(self):
            super().__init__()
            self.entities = nn.Embedding(n_entities, dim)
            self.relation = nn.Embedding(1, dim)
            nn.init.xavier_uniform_(self.entities.weight)
            nn.init.xavier_uniform_(self.relation.weight)

        def score(self, heads, tails):
            return (self.entities(heads) * self.relation.weight[0] * self.entities(tails)).sum(dim=-1)

    return DistMult()


def train_distmult(train_pairs, entity_ids, *, dim: int, epochs: int, batch_size: int, seed: int):
    import torch
    import torch.nn.functional as F

    random.seed(seed)
    torch.manual_seed(seed)
    model = make_distmult(len(entity_ids), dim)
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
            negatives = []
            for head, _tail in batch.tolist():
                candidate = random.choice(drug_ids)
                attempts = 0
                while (head, candidate) in known and attempts < 50:
                    candidate = random.choice(drug_ids)
                    attempts += 1
                negatives.append(candidate)
            heads, tails = batch[:, 0], batch[:, 1]
            neg_tails = torch.tensor(negatives, dtype=torch.long)
            positive_logits = model.score(heads, tails)
            negative_logits = model.score(heads, neg_tails)
            loss = F.softplus(-positive_logits).mean() + F.softplus(negative_logits).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def collaborative_scores(train_pairs: set[tuple[str, str]], method: str) -> dict[tuple[str, str], float]:
    """Score disease-drug links by propagating drug evidence across similar diseases."""
    disease_drugs: dict[str, set[str]] = defaultdict(set)
    drug_diseases: dict[str, set[str]] = defaultdict(set)
    for disease, drug in train_pairs:
        disease_drugs[disease].add(drug)
        drug_diseases[drug].add(disease)
    result = {}
    for disease, drugs in disease_drugs.items():
        for other, other_drugs in disease_drugs.items():
            if disease == other:
                continue
            shared = drugs & other_drugs
            if not shared:
                continue
            if method == "common_neighbors":
                similarity = float(len(shared))
            elif method == "jaccard_disease":
                similarity = len(shared) / len(drugs | other_drugs)
            elif method == "cosine_disease":
                similarity = len(shared) / math.sqrt(len(drugs) * len(other_drugs))
            elif method == "resource_allocation":
                similarity = sum(1 / len(drug_diseases[drug]) for drug in shared)
            else:
                raise ValueError(f"unknown collaborative method: {method}")
            for drug in other_drugs:
                if drug not in drugs:
                    result[(disease, drug)] = result.get((disease, drug), 0.0) + similarity
    return result


def ranking_metrics(ranks: list[int]) -> dict:
    n = len(ranks)
    return {
        "n_test_edges": n,
        "mrr": sum(1 / rank for rank in ranks) / n if n else 0.0,
        "hits_at_1": sum(rank <= 1 for rank in ranks) / n if n else 0.0,
        "hits_at_3": sum(rank <= 3 for rank in ranks) / n if n else 0.0,
        "hits_at_10": sum(rank <= 10 for rank in ranks) / n if n else 0.0,
    }


def evaluate_scores(test_pairs, train_pairs, all_test_pairs, score_fn, candidate_drugs=None) -> dict:
    train_by_head: dict[str, set[str]] = defaultdict(set)
    drugs = sorted(candidate_drugs if candidate_drugs is not None else {drug for _disease, drug in train_pairs})
    test_by_head: dict[str, set[str]] = defaultdict(set)
    for disease, drug in train_pairs:
        train_by_head[disease].add(drug)
    for disease, drug in all_test_pairs:
        test_by_head[disease].add(drug)
    ranks = []
    for disease, target in sorted(test_pairs):
        candidates = [drug for drug in drugs if drug not in train_by_head[disease] and drug not in (test_by_head[disease] - {target})]
        if target not in candidates:
            candidates.append(target)
        ordered = sorted(candidates, key=lambda drug: (-score_fn(disease, drug), drug))
        ranks.append(1 + ordered.index(target))
    return ranking_metrics(ranks)


def evaluate_degree(test_pairs, train_pairs, all_test_pairs, candidate_drugs=None) -> dict:
    degree = defaultdict(int)
    for _disease, drug in train_pairs:
        degree[drug] += 1
    return evaluate_scores(test_pairs, train_pairs, all_test_pairs, lambda _d, drug: degree[drug], candidate_drugs)


def evaluate_collaborative(test_pairs, train_pairs, all_test_pairs, method, candidate_drugs=None) -> dict:
    scores = collaborative_scores(train_pairs, method)
    return evaluate_scores(test_pairs, train_pairs, all_test_pairs, lambda disease, drug: scores.get((disease, drug), 0.0), candidate_drugs)


def run_fold(docs, test_indices, seed, args) -> dict:
    import torch

    train_docs = [doc for i, doc in enumerate(docs) if i not in test_indices]
    heldout_docs = [doc for i, doc in enumerate(docs) if i in test_indices]
    canon_map = build_canonical_map(collect_freq(train_docs))
    raw_train, raw_heldout = graph_pairs(train_docs), graph_pairs(heldout_docs)
    norm_train = graph_pairs(train_docs, canon_map)
    raw_entities = {name for pair in raw_train for name in pair}
    norm_entities = {name for pair in norm_train for name in pair}
    raw_test, norm_test = paired_test_edges(raw_train, raw_heldout, norm_train, raw_entities, norm_entities, canon_map)
    paired = {"g_raw": raw_test, "g_norm": norm_test}
    raw_drugs = {drug for _disease, drug in raw_train}
    norm_drugs = {drug for _disease, drug in norm_train}
    common_drugs = raw_drugs & norm_drugs
    common_raw_test = set()
    common_norm_test = set()
    for raw_pair in raw_test:
        norm_pair = map_disease_drug_pair(raw_pair, canon_map)
        if norm_pair and raw_pair[1] == norm_pair[1] and raw_pair[1] in common_drugs:
            common_raw_test.add(raw_pair)
            common_norm_test.add(norm_pair)
    common_test = {"g_raw": common_raw_test, "g_norm": common_norm_test}
    outputs = {}
    for side in ("g_raw", "g_norm"):
        train_pairs = raw_train if side == "g_raw" else norm_train
        test_pairs = paired[side]
        common_test_pairs = common_test[side]
        names = {name for pair in train_pairs for name in pair}
        entity_ids = {name: index for index, name in enumerate(sorted(names))}
        methods = {"drug_degree": evaluate_degree(test_pairs, train_pairs, test_pairs)}
        common_metrics = {"drug_degree": evaluate_degree(common_test_pairs, train_pairs, common_test_pairs, common_drugs)}
        for method in ("common_neighbors", "jaccard_disease", "cosine_disease", "resource_allocation"):
            methods[method] = evaluate_collaborative(test_pairs, train_pairs, test_pairs, method)
            common_metrics[method] = evaluate_collaborative(common_test_pairs, train_pairs, common_test_pairs, method, common_drugs)
        transe = train_transe(train_pairs, entity_ids, dim=args.dim, epochs=args.epochs, batch_size=args.batch_size, margin=1.0, seed=seed)
        methods["transe"] = evaluate(transe, test_pairs, entity_ids, train_pairs, test_pairs, seed + 1)
        common_metrics["transe"] = evaluate_scores(
            common_test_pairs,
            train_pairs,
            common_test_pairs,
            lambda disease, drug: -float(transe.score(
                torch.tensor([entity_ids[disease]]), torch.tensor([entity_ids[drug]])
            )[0].item()),
            common_drugs,
        )
        distmult = train_distmult(train_pairs, entity_ids, dim=args.dim, epochs=args.epochs, batch_size=args.batch_size, seed=seed + 2)
        methods["distmult"] = evaluate(distmult, test_pairs, entity_ids, train_pairs, test_pairs, seed + 3)
        common_metrics["distmult"] = evaluate_scores(
            common_test_pairs,
            train_pairs,
            common_test_pairs,
            lambda disease, drug: float(distmult.score(
                torch.tensor([entity_ids[disease]]), torch.tensor([entity_ids[drug]])
            )[0].item()),
            common_drugs,
        )
        for name in methods:
            methods[name]["common_drug_candidates"] = common_metrics[name]
        outputs[side] = {"train_edges": len(train_pairs), "test_edges": len(test_pairs), "common_drug_test_edges": len(common_test_pairs), "disease_entities": sum(n.startswith("疾病和诊断::") for n in names), "drug_entities": sum(n.startswith("药物::") for n in names), "common_drug_candidates": len(common_drugs), "methods": methods}
    edge_pairs = [(raw, map_disease_drug_pair(raw, canon_map)) for raw in sorted(raw_test)]
    return {
        "seed": seed,
        "n_train_records": len(train_docs),
        "n_test_records": len(heldout_docs),
        "paired_test_edges": len(raw_test),
        "paired_edges_sha256": hashlib.sha256(json.dumps(edge_pairs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "g_raw": outputs["g_raw"],
        "g_norm": outputs["g_norm"],
    }


def mean_ci(values: list[float]) -> dict:
    mean = statistics.mean(values)
    if len(values) < 2:
        return {"mean": mean, "t_approx_95ci": None}
    critical = T_975.get(len(values) - 1, 1.96)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "t_approx_95ci": [mean - half, mean + half]}


def summarize(repetitions: list[dict]) -> dict:
    model_names = sorted({name for rep in repetitions for fold in rep["folds"] for name in fold["g_raw"]["methods"]})
    by_method = {}
    for method in model_names:
        by_method[method] = {}
        for metric in METRICS:
            deltas = []
            raw_values, norm_values = [], []
            for rep in repetitions:
                raw_num = norm_num = 0
                raw_sum = norm_sum = 0.0
                for fold in rep["folds"]:
                    n_raw = fold["g_raw"]["methods"][method]["n_test_edges"]
                    n_norm = fold["g_norm"]["methods"][method]["n_test_edges"]
                    if n_raw != n_norm:
                        raise ValueError("raw/norm paired test counts differ")
                    raw_sum += fold["g_raw"]["methods"][method][metric] * n_raw
                    norm_sum += fold["g_norm"]["methods"][method][metric] * n_norm
                    raw_num += n_raw
                    norm_num += n_norm
                raw_values.append(raw_sum / raw_num if raw_num else 0.0)
                norm_values.append(norm_sum / norm_num if norm_num else 0.0)
                deltas.append(norm_values[-1] - raw_values[-1])
            by_method[method][metric] = {
                "raw_by_repeat": raw_values,
                "normalized_by_repeat": norm_values,
                "delta_by_repeat": deltas,
                "delta_summary": mean_ci(deltas),
                "repeats_improved": sum(delta > 0 for delta in deltas),
            }
    return {
        "protocol": "Repeated shuffled record-level K-fold CV; each fold fits canonical map and link predictors from training records only; identical one-to-one raw/norm unseen edges; report all graph candidates and a strict shared-drug-candidate subset; compare degree, disease-neighbor propagation and two embedding models.",
        "n_repeats": len(repetitions),
        "n_folds": len(repetitions[0]["folds"]) if repetitions else 0,
        "paired_test_edges_per_repeat": [sum(f["paired_test_edges"] for f in rep["folds"]) for rep in repetitions],
        "by_method": by_method,
        "shared_drug_candidate_by_method": summarize_shared_candidates(repetitions),
        "repetitions": repetitions,
        "caution": "Edges are disease-drug co-occurrences extracted from records, not verified clinical treatment links. Repeated CV estimates split variability, not independent clinical significance; raw and normalized candidate vocabularies differ after aliases merge. This is exploratory evidence only.",
    }


def summarize_shared_candidates(repetitions: list[dict]) -> dict:
    methods = sorted({name for rep in repetitions for fold in rep["folds"] for name in fold["g_raw"]["methods"]})
    summary = {}
    for method in methods:
        summary[method] = {}
        for metric in METRICS:
            raw_values, norm_values = [], []
            for rep in repetitions:
                raw_sum = norm_sum = 0.0
                count = 0
                for fold in rep["folds"]:
                    raw = fold["g_raw"]["methods"][method]["common_drug_candidates"]
                    norm = fold["g_norm"]["methods"][method]["common_drug_candidates"]
                    if raw["n_test_edges"] != norm["n_test_edges"]:
                        raise ValueError("shared-candidate test edges are not paired")
                    raw_sum += raw[metric] * raw["n_test_edges"]
                    norm_sum += norm[metric] * norm["n_test_edges"]
                    count += raw["n_test_edges"]
                raw_values.append(raw_sum / count if count else 0.0)
                norm_values.append(norm_sum / count if count else 0.0)
            deltas = [norm - raw for raw, norm in zip(raw_values, norm_values)]
            summary[method][metric] = {
                "raw_by_repeat": raw_values,
                "normalized_by_repeat": norm_values,
                "delta_by_repeat": deltas,
                "delta_summary": mean_ci(deltas),
                "repeats_improved": sum(delta > 0 for delta in deltas),
            }
    return summary


def render_text(report: dict) -> str:
    lines = [
        "Disease-drug link prediction: repeated record-level cross-validation",
        "=" * 76,
        f"records={report['n_records']}; folds={report['hyperparameters']['folds']}; repeats={report['hyperparameters']['repeats']};",
        f"paired held-out edges per repeat={report['paired_test_edges_per_repeat']}",
        "Intervals are descriptive t-approximation across repeated partitions, not independent clinical inference.",
        "",
        "All graph-native drug candidates",
        f"{'method':24s} {'MRR raw->norm':25s} {'delta (95% approx CI)':36s} {'H@10 delta':>12s}",
    ]
    for method, values in report["by_method"].items():
        mrr = values["mrr"]
        h10 = values["hits_at_10"]
        low, high = mrr["delta_summary"]["t_approx_95ci"] or (float("nan"), float("nan"))
        lines.append(
            f"{method:24s} {statistics.mean(mrr['raw_by_repeat']):.4f}->{statistics.mean(mrr['normalized_by_repeat']):.4f}"
            f"{'':8s} {mrr['delta_summary']['mean']:+.4f} [{low:+.4f}, {high:+.4f}]"
            f"{'':5s} {h10['delta_summary']['mean']:+.4f}"
        )
    lines += ["", "Shared identical drug-candidate set", f"{'method':24s} {'MRR raw->norm':25s} {'delta (95% approx CI)':36s} {'H@10 delta':>12s}"]
    for method, values in report["shared_drug_candidate_by_method"].items():
        mrr = values["mrr"]
        h10 = values["hits_at_10"]
        low, high = mrr["delta_summary"]["t_approx_95ci"] or (float("nan"), float("nan"))
        lines.append(
            f"{method:24s} {statistics.mean(mrr['raw_by_repeat']):.4f}->{statistics.mean(mrr['normalized_by_repeat']):.4f}"
            f"{'':8s} {mrr['delta_summary']['mean']:+.4f} [{low:+.4f}, {high:+.4f}]"
            f"{'':5s} {h10['delta_summary']['mean']:+.4f}"
        )
    lines += [
        "",
        "Caution: edges are disease-drug co-occurrences extracted from records, not clinically verified treatment links.",
        "See JSON for per-fold counts, per-repeat metrics, candidate vocabularies, and test-edge hashes.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "train.bio")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=5200)
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR / "expanded_cv.json")
    args = parser.parse_args()
    import torch

    docs = load_bio_docs(args.data)
    repetitions = []
    for repeat in range(args.repeats):
        repeat_seed = args.seed + repeat
        folds = fold_indices(len(docs), args.folds, repeat_seed)
        fold_results = []
        for fold_id, test_indices in enumerate(folds):
            fold_seed = repeat_seed * 100 + fold_id
            result = run_fold(docs, test_indices, fold_seed, args)
            result["fold"] = fold_id
            fold_results.append(result)
            print(f"repeat={repeat + 1}/{args.repeats} fold={fold_id + 1}/{args.folds} paired_edges={result['paired_test_edges']}", flush=True)
        repetitions.append({"repeat": repeat, "seed": repeat_seed, "folds": fold_results})
    report = summarize(repetitions)
    report.update({"data": str(args.data), "n_records": len(docs), "hyperparameters": {"folds": args.folds, "repeats": args.repeats, "dim": args.dim, "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed}})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    text_out = args.out.with_suffix(".txt")
    text_out.write_text(render_text(report), encoding="utf-8")
    print(json.dumps({"paired_test_edges_per_repeat": report["paired_test_edges_per_repeat"], "by_method": report["by_method"], "out": str(args.out)}, ensure_ascii=False, indent=2))
    print(f"wrote {text_out}")


if __name__ == "__main__":
    main()
