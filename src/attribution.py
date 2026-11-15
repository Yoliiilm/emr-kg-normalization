"""Drug → disease attribution inside each note (de-confounding co-occurrence edges).

Co-occurrence links every drug in a note to every disease in it, so comorbidities collect the drugs of
the main diagnosis (高血压 → 格列美脲, 糖尿病 → 紫杉醇). Here each (note, drug) occurrence is split over
the note's diseases in proportion to how strongly each disease "prescribes" that drug:

    r(d | m, note) = θ(d, m) / Σ_{d' ∈ note ∪ {∅}} θ(d', m)                (E-step)
    θ(d, m)        = (Σ_notes r(d | m, note) + α·p(m)) / (df(d) + α)       (M-step)

∅ is a background cause present in every note (supportive drugs that belong to no listed disease);
p(m) = share of notes with drug m, so α shrinks rarely seen diseases toward the background rate.
The attributed weight of an edge is a(d, m) = Σ_notes r(d | m, note), an expected count on the same
scale as the co-occurrence count. iters=1 with lift-initialised θ is a one-shot disproportionality
allocation. With α > 0 this is a deterministic counterpart of the Monte Carlo EM of Li et al.
(AMIA Jt Summits 2019), which samples one indication per drug in proportion to shrunk (MGPS) scores.
"""

from __future__ import annotations

from collections import defaultdict

from eval_external_kg import REL_PREDICATE
from eval_g_raw_vs_g_norm import graph_counts

REL = "使用药物"
BACKGROUND = "∅"


def doc_level_graph_with_pairs(docs, canon) -> tuple[dict, list[tuple[list[str], list[str]]]]:
    """eval_external_kg.doc_level_graph plus, per note, its diseases and drugs on 使用药物 edges."""
    df, mentions, co, cue = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    names = defaultdict(set)
    pairs = []
    for doc in docs:
        nodes, rels, surfaces, evidence = graph_counts([doc], canon)
        for nid, node in nodes.items():
            df[nid] += 1
            mentions[nid] += node["mention_count"]
            names[nid].add(node["name"])
            names[nid].update(surfaces.get(nid, ()))
        dis, drugs = set(), set()
        for key in rels:
            if key[1] in REL_PREDICATE:
                co[key] += 1
                if evidence[key].get("pattern", 0) > 0:
                    cue[key] += 1
                if key[1] == REL:
                    dis.add(key[0])
                    drugs.add(key[2])
        if dis and drugs:
            pairs.append((sorted(dis), sorted(drugs)))
    g = {"n_docs": len(docs), "df": df, "mentions": mentions, "names": names, "co": co, "cue": cue}
    return g, pairs


def attribute(g: dict, pairs, iters: int = 30, background: bool = True, init: str = "count",
              alpha: float = 0.0, group: dict[str, str] | None = None) -> dict:
    """Return {(d, REL, m): attributed weight} for every co-occurrence edge.

    group maps a disease to its hierarchy group (e.g. ICD category). Diseases of one group in the same
    note do not compete for a drug: the group enters the normaliser once (its strongest member's θ),
    and each member keeps r = θ(d, m) / z, so 糖尿病 and 2型糖尿病 can both take a diabetes drug.
    """
    n = g["n_docs"]
    df_m = defaultdict(int)
    for _, drugs in pairs:
        for m in drugs:
            df_m[m] += 1
    theta = {}
    for (d, r, m), c in g["co"].items():
        if r == REL:
            theta[(d, m)] = c * n / (g["df"][d] * g["df"][m]) if init == "lift" else c / g["df"][d]
    if background:
        for m, c in df_m.items():
            theta[(BACKGROUND, m)] = c / n
    acc = {}
    for _ in range(iters):
        acc = defaultdict(float)
        for dis, drugs in pairs:
            causes = dis + [BACKGROUND] if background else dis
            for m in drugs:
                w = [theta.get((d, m), 0.0) for d in causes]
                if group:
                    best = defaultdict(float)
                    for d, x in zip(causes, w):
                        key = group.get(d, d)
                        best[key] = max(best[key], x)
                    z = sum(best.values())
                else:
                    z = sum(w)
                if z <= 0:
                    continue
                for d, x in zip(causes, w):
                    if x:
                        acc[(d, m)] += x / z
        theta = {(d, m): a / n if d == BACKGROUND else (a + alpha * df_m[m] / n) / (g["df"][d] + alpha)
                 for (d, m), a in acc.items()}
    return {(d, REL, m): acc.get((d, m), 0.0) for (d, r, m) in g["co"] if r == REL}


def comorbidity_load(pairs) -> dict[str, float]:
    """Mean number of other diseases in the notes where a disease has drug edges."""
    tot, cnt = defaultdict(int), defaultdict(int)
    for dis, _ in pairs:
        for d in dis:
            tot[d] += len(dis) - 1
            cnt[d] += 1
    return {d: tot[d] / cnt[d] for d in cnt}
