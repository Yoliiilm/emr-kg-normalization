"""Graph-feedback check of normalization merges (improvement 3).

A rewritten mention s (surface ≠ canonical name) is merged into node G. The notes where s occurs carry a
context profile P_s (the other nodes in those notes: drugs, surgeries, sites, tests, other diagnoses). If the
merge is right, P_s should look like the profile of G's other surfaces. Each node g of the same type gets a
smoothed context distribution, and s is scored by the average log-lift of its contexts under it:

    score(s, g) = Σ_x P_s(x) · log[ p_g(x) / p_bg(x) ] / Σ_x P_s(x),   p_g(x) = (n_g(x) + β·p_bg(x)) / (N_g + β)

For the own node, n_g leaves s out. The merge is vetoed when s has enough notes and its own node ranks low
among all nodes of that type (rank percentile < tau), i.e. the contexts of s are better explained by many
other nodes than by the one it was merged into. A vetoed surface keeps a node of its own.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from eval_external_kg import drug_key, ext_key
from eval_g_raw_vs_g_norm import graph_counts
from export_graph import node_id

LABELS = ("疾病和诊断", "药物")
# Which node types count as context: "all" = everything in the note; "treat" = the drug–disease view
# (a diagnosis is described by the drugs given with it, a drug by the diagnoses it is given with).
CTX_MODES = {"all": None,
             "treat": {"疾病和诊断": {"药物"}, "药物": {"疾病和诊断"}},
             "treat_surg": {"疾病和诊断": {"药物", "手术"}, "药物": {"疾病和诊断"}}}


def surface_profiles(docs, canon: dict, labels=LABELS, ctx_mode: str = "all") -> dict:
    """Context profiles of every (label, surface) that the canonical map sends to a single node."""
    allowed = CTX_MODES[ctx_mode]
    prof = defaultdict(Counter)
    ndoc = Counter()
    member = {}
    bg = Counter()
    for doc in docs:
        nodes, _, surfaces, _ = graph_counts([doc], canon)
        ids = list(nodes)
        bg.update(ids)
        for nid in ids:
            label = nodes[nid]["type"]
            if label not in labels:
                continue
            ctx = [x for x in ids if x != nid and (allowed is None or nodes[x]["type"] in allowed[label])]
            for s in surfaces[nid]:
                target = canon.get((label, s), s)
                if target != nodes[nid]["name"]:
                    continue
                key = (label, s)
                prof[key].update(ctx)
                ndoc[key] += 1
                member[key] = nid
    return {"prof": prof, "ndoc": ndoc, "member": member, "bg": bg, "n_docs": len(docs), "allowed": allowed}


def merge_scores(p: dict, beta: float = 5.0, min_group_ctx: int = 20) -> dict[tuple[str, str], dict]:
    """For each rewritten surface whose node has other surfaces: own-node score and its rank percentile."""
    prof, member, bg, allowed = p["prof"], p["member"], p["bg"], p.get("allowed")
    groups = defaultdict(Counter)
    surfaces_of = defaultdict(list)
    for key, nid in member.items():
        groups[nid].update(prof[key])
        surfaces_of[nid].append(key)
    out = {}
    for label in LABELS:
        ctx_bg = {x: c for x, c in bg.items() if allowed is None or x.split("::", 1)[0] in allowed[label]}
        total = sum(ctx_bg.values())
        pbg = {x: c / total for x, c in ctx_bg.items()}
        vocab = {x: i for i, x in enumerate(pbg)}
        gids = [g for g in groups if g.startswith(label + "::") and sum(groups[g].values()) >= min_group_ctx]
        if not gids:
            continue
        rows, cols, vals = [], [], []
        base = np.empty(len(gids))
        for i, g in enumerate(gids):
            n_g = sum(groups[g].values())
            base[i] = math.log(beta / (n_g + beta))
            for x, c in groups[g].items():
                rows.append(i)
                cols.append(vocab[x])
                vals.append(math.log((c + beta * pbg[x]) / (beta * pbg[x])))
        mat = csr_matrix((vals, (rows, cols)), shape=(len(gids), len(vocab)))
        gpos = {g: i for i, g in enumerate(gids)}
        for key, nid in member.items():
            if key[0] != label or node_id(key[1], label) == nid or len(surfaces_of[nid]) < 2:
                continue
            ps = prof[key]
            n_s = sum(ps.values())
            if not n_s:
                continue
            own = groups[nid] - ps
            n_own = sum(own.values())
            if n_own == 0:
                continue
            own_score = sum(c * math.log((own[x] + beta * pbg[x]) / (n_own + beta) / pbg[x]) for x, c in ps.items()) / n_s
            vec = np.zeros(len(vocab))
            for x, c in ps.items():
                vec[vocab[x]] = c / n_s
            others = base + mat @ vec
            if nid in gpos:
                others = np.delete(others, gpos[nid])
            pct = (float((others < own_score).sum()) + 0.5 * float((others == own_score).sum())) / max(len(others), 1)
            out[key] = {"node": nid, "score": own_score, "pct": pct, "ndoc": p["ndoc"][key]}
    return out


def kb_merge_scores(p: dict, ext: dict, label: str = "疾病和诊断") -> dict[tuple[str, str], dict]:
    """Check a disease merge against an external disease→drug list instead of note context.

    The drugs given with surface s in the notes are matched against the drugs the KB lists for every node
    (via all of the node's names); a match on drug m weighs idf(m) = log(#KB-covered nodes / #nodes listing m).
    The own node is described without s itself. Only surfaces with note drugs and a KB-covered own node are scored.
    """
    idx = defaultdict(set)
    for c, o in ext["pairs"]["使用药物"]:
        idx[c].add(o)
    names = defaultdict(set)
    for (lb, s), nid in p["member"].items():
        if lb == label:
            names[nid].add(s)

    def kb(nid, skip=None):
        ns = ({nid.split("::", 1)[1]} | names[nid]) - {skip}
        return set().union(*(idx.get(ext["uf"].find(ext_key(n)), set()) for n in ns))

    node_kb = {nid: d for nid in names if (d := kb(nid))}
    df = Counter(m for d in node_kb.values() for m in d)
    idf = {m: math.log(len(node_kb) / c) for m, c in df.items()}
    out = {}
    for (lb, s), nid in p["member"].items():
        if lb != label or node_id(s, lb) == nid or len(names[nid]) < 2:
            continue
        drugs = {drug_key(x.split("::", 1)[1]) for x in p["prof"][(lb, s)] if x.startswith("药物::")}
        own_kb = kb(nid, skip=s)
        if not drugs or not own_kb:
            continue
        own = sum(idf.get(m, 0.0) for m in drugs & own_kb)
        others = [sum(idf.get(m, 0.0) for m in drugs & d) for g, d in node_kb.items() if g != nid]
        pct = (sum(o < own for o in others) + 0.5 * sum(o == own for o in others)) / max(len(others), 1)
        out[(lb, s)] = {"node": nid, "score": own, "pct": pct, "ndoc": p["ndoc"][(lb, s)]}
    return out


def apply_feedback(canon: dict, scores: dict, tau: float, min_docs: int) -> tuple[dict, list]:
    """Return the canonical map with vetoed merges undone, and the vetoed surfaces."""
    out = dict(canon)
    vetoed = []
    for key, s in scores.items():
        if s["ndoc"] >= min_docs and s["pct"] < tau:
            out[key] = key[1]
            vetoed.append((key, s))
    return out, vetoed
