"""Validate EMR graph edges against external knowledge sources and compare edge rankings.

Sources:
  cmeie      CMeIE (CBLUE) 药物治疗 / 手术治疗 triples; 同义词 triples merge disease names.
             data/dicts/raw/CMeIE/CMeIE_{train,dev}.jsonl
  medicalkg  寻医问药 disease KB (liuhuanyong/QASystemOnMedicalKG), common_drug + recommand_drug.
             Drug side only: it has no specific surgery relation.
             data/dicts/raw/medicalkg/medical.json

Test 1 (edge support): what share of disease->drug / disease->surgery edges in G_raw and G_norm
is backed by an external triple. Matching is string matching after width/case/space cleanup
(drug names additionally drop dosage forms and salt prefixes, applied identically to both sides);
our normalizer is never applied to the external source, so the judge is independent of the system.
A G_norm node is matched through its canonical name and all surface forms merged into it.

Test 2 (edge ranking, after Rotmensch et al. 2017): for each covered disease, rank its
neighbours by co-occurrence count, lift, discounted PMI (Pantel & Lin 2002), or treatment-cue
count, and score precision@k / MAP against external support. Ties are broken randomly and averaged.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_g_raw_vs_g_norm import build_graphs, graph_counts
from ner_data import load_bio_docs
from normalize import configure_normalization

REL_PREDICATE = {"使用药物": "药物治疗", "实施手术": "手术治疗"}
REL_TARGET_TYPE = {"使用药物": "药物", "实施手术": "手术"}
DISEASE = "疾病和诊断"
KS = (1, 3, 5)
N_TIE_SHUFFLES = 200
N_BOOT = 2000


def ext_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(ch for ch in s if not ch.isspace()).strip("，,。.;；:：、")


SALT_PREFIXES = ("盐酸", "硫酸", "马来酸", "甲磺酸", "枸橼酸", "磷酸", "醋酸", "酒石酸", "富马酸",
                 "琥珀酸", "苯磺酸", "氢溴酸", "乳酸", "葡萄糖酸", "门冬氨酸", "甲苯磺酸", "二盐酸")
DOSAGE_SUFFIXES = sorted((
    "缓释片", "控释片", "肠溶片", "分散片", "咀嚼片", "泡腾片", "薄膜衣片", "口崩片", "含片", "片",
    "缓释胶囊", "肠溶胶囊", "软胶囊", "胶囊", "胶丸", "注射液", "注射剂", "粉针剂", "冻干粉", "输液",
    "口服溶液", "口服液", "溶液", "颗粒", "干混悬剂", "混悬液", "混悬剂", "滴丸", "丸", "散",
    "乳膏", "软膏", "膏", "凝胶", "栓", "滴眼液", "气雾剂", "喷雾剂", "粉吸入剂", "吸入剂", "粉雾剂",
    "雾化液", "糖浆", "酊", "贴剂", "贴", "植入剂", "乳",
), key=len, reverse=True)


def drug_key(s: str) -> str:
    s = ext_key(s)
    for open_, close in (("(", ")"), ("（", "）"), ("[", "]")):
        while open_ in s and close in s and s.index(open_) < s.index(close):
            s = s[: s.index(open_)] + s[s.index(close) + 1:]
    if s.startswith("注射用"):
        s = s[3:]
    for p in SALT_PREFIXES:
        if s.startswith(p) and len(s) - len(p) >= 2:
            s = s[len(p):]
            break
    changed = True
    while changed:
        changed = False
        for suf in DOSAGE_SUFFIXES:
            if s.endswith(suf) and len(s) - len(suf) >= 2:
                s = s[: -len(suf)]
                changed = True
                break
    return s


def obj_key(rel: str, s: str) -> str:
    return drug_key(s) if rel == "使用药物" else ext_key(s)


# ---------------------------------------------------------------- external KG


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def load_cmeie(cmeie_dir: Path) -> dict:
    files = sorted(p for p in cmeie_dir.glob("*.json*") if "schema" not in p.name and "test" not in p.name)
    if not files:
        raise SystemExit(f"No CMeIE files in {cmeie_dir}. Put CMeIE_train.jsonl / CMeIE_dev.jsonl there.")
    triples = []
    for p in files:
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for spo in rec.get("spo_list", []):
                obj = spo["object"]["@value"] if isinstance(spo["object"], dict) else spo["object"]
                triples.append((spo["subject"], spo["predicate"], obj))

    uf = UnionFind()
    for s, p, o in triples:
        if p == "同义词":
            uf.union(ext_key(s), ext_key(o))

    pairs = {rel: set() for rel in REL_PREDICATE}
    subjects = {rel: set() for rel in REL_PREDICATE}
    raw_counts = {rel: 0 for rel in REL_PREDICATE}
    for s, p, o in triples:
        for rel, pred in REL_PREDICATE.items():
            if p == pred:
                cls = uf.find(ext_key(s))
                pairs[rel].add((cls, obj_key(rel, o)))
                subjects[rel].add(cls)
                raw_counts[rel] += 1
    return {
        "name": "CMeIE",
        "files": [p.name for p in files],
        "n_triples": len(triples),
        "n_synonym": sum(1 for _, p, _ in triples if p == "同义词"),
        "uf": uf,
        "pairs": pairs,
        "subjects": subjects,
        "raw_counts": raw_counts,
    }


def load_medicalkg(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing {path}.")
    uf = UnionFind()
    pairs = {rel: set() for rel in REL_PREDICATE}
    subjects = {rel: set() for rel in REL_PREDICATE}
    raw_counts = {rel: 0 for rel in REL_PREDICATE}
    n_records = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        n_records += 1
        cls = uf.find(ext_key(rec["name"]))
        for drug in set(rec.get("common_drug") or []) | set(rec.get("recommand_drug") or []):
            key = drug_key(drug)
            if len(key) < 2:
                continue
            pairs["使用药物"].add((cls, key))
            subjects["使用药物"].add(cls)
            raw_counts["使用药物"] += 1
    return {
        "name": "寻医问药 medical.json",
        "files": [path.name],
        "n_triples": n_records,
        "n_synonym": 0,
        "uf": uf,
        "pairs": pairs,
        "subjects": subjects,
        "raw_counts": raw_counts,
    }


# ---------------------------------------------------------------- EMR graphs with doc-level counts


def doc_level_graph(pred_docs, canon) -> dict:
    """Per-document pass so that edge counts are document counts (needed for lift / PMI)."""
    df = defaultdict(int)
    mentions = defaultdict(int)
    names = defaultdict(set)
    co = defaultdict(int)
    cue = defaultdict(int)
    for doc in pred_docs:
        nodes, rels, surfaces, evidence = graph_counts([doc], canon)
        for nid, node in nodes.items():
            df[nid] += 1
            mentions[nid] += node["mention_count"]
            names[nid].add(node["name"])
            names[nid].update(surfaces.get(nid, ()))
        for key in rels:
            if key[1] in REL_PREDICATE:
                co[key] += 1
                if evidence[key].get("pattern", 0) > 0:
                    cue[key] += 1
    return {"n_docs": len(pred_docs), "df": df, "mentions": mentions, "names": names, "co": co, "cue": cue}


def disease_classes(g: dict, nid: str, ext: dict) -> set[str]:
    return {ext["uf"].find(ext_key(n)) for n in g["names"][nid]}


def target_keys(g: dict, nid: str, rel: str) -> set[str]:
    return {obj_key(rel, n) for n in g["names"][nid]}


def supported(g: dict, d: str, m: str, rel: str, ext: dict) -> bool:
    pairs = ext["pairs"][rel]
    return any((c, k) in pairs for c in disease_classes(g, d, ext) for k in target_keys(g, m, rel))


def neighbours(g: dict, rel: str) -> dict[str, list[str]]:
    out = defaultdict(list)
    for (s, r, t) in g["co"]:
        if r == rel:
            out[s].append(t)
    return out


# ---------------------------------------------------------------- test 1


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def edge_support(g: dict, rel: str, ext: dict) -> dict:
    nb = neighbours(g, rel)
    diseases = [nid for nid in g["df"] if nid.startswith(DISEASE + "::")]
    covered = [d for d in diseases if disease_classes(g, d, ext) & ext["subjects"][rel]]
    total_mentions = sum(g["mentions"][d] for d in diseases)
    covered_mentions = sum(g["mentions"][d] for d in covered)

    target_type = REL_TARGET_TYPE[rel] + "::"
    vocab = set()
    for nid in g["df"]:
        if nid.startswith(target_type):
            vocab |= target_keys(g, nid, rel)

    n_edges = n_sup = w_edges = w_sup = 0
    findable = found = 0
    per_disease = []
    for d in covered:
        edges = nb.get(d, [])
        sup = [m for m in edges if supported(g, d, m, rel, ext)]
        n_edges += len(edges)
        n_sup += len(sup)
        w_edges += sum(g["co"][(d, rel, m)] for m in edges)
        w_sup += sum(g["co"][(d, rel, m)] for m in sup)
        classes = disease_classes(g, d, ext)
        ext_objs = {o for (c, o) in ext["pairs"][rel] if c in classes and o in vocab}
        hit = {o for o in ext_objs if any(o in target_keys(g, m, rel) for m in edges)}
        findable += len(ext_objs)
        found += len(hit)
        per_disease.append({"disease": d.split("::", 1)[1], "edges": len(edges), "supported": len(sup),
                            "ext_findable": len(ext_objs), "ext_found": len(hit)})

    covered_set = set(covered)
    uncovered = sorted((d for d in diseases if d not in covered_set), key=lambda d: -g["mentions"][d])
    return {
        "top_uncovered": [(d.split("::", 1)[1], g["mentions"][d]) for d in uncovered[:15]],
        "n_disease_nodes": len(diseases),
        "n_covered": len(covered),
        "covered_mention_share": covered_mentions / total_mentions if total_mentions else 0.0,
        "n_edges": n_edges,
        "n_supported": n_sup,
        "support_rate": n_sup / n_edges if n_edges else 0.0,
        "support_rate_ci95": wilson(n_sup, n_edges),
        "support_rate_doc_weighted": w_sup / w_edges if w_edges else 0.0,
        "ext_findable": findable,
        "ext_found": found,
        "ext_recall": found / findable if findable else 0.0,
        "ext_recall_ci95": wilson(found, findable),
        "per_disease": sorted(per_disease, key=lambda r: -r["edges"]),
    }


def paired_support(graphs: dict, rel: str, ext: dict) -> dict:
    """Support rate of G_raw vs G_norm restricted to external diseases matched by both graphs.

    Nodes and external disease names are linked into connected components so that every node falls
    in exactly one cluster; the difference is bootstrapped over these clusters.
    """
    subjects = ext["subjects"][rel]
    node_classes = {}
    for gname, g in graphs.items():
        for nid in g["df"]:
            if nid.startswith(DISEASE + "::"):
                cls = disease_classes(g, nid, ext) & subjects
                if cls:
                    node_classes[(gname, nid)] = cls
    by_graph = {gname: set().union(*(c for (gn, _), c in node_classes.items() if gn == gname) or [set()])
                for gname in graphs}
    common = by_graph["g_raw"] & by_graph["g_norm"]

    uf = UnionFind()
    kept = []
    for (gname, nid), cls in node_classes.items():
        shared = cls & common
        if not shared:
            continue
        kept.append((gname, nid))
        for c in shared:
            uf.union(f"{gname}|{nid}", f"cls|{c}")

    clusters = defaultdict(lambda: {"g_raw": [0, 0], "g_norm": [0, 0]})
    for gname, nid in kept:
        g = graphs[gname]
        edges = [t for (s, r, t) in g["co"] if s == nid and r == rel]
        sup = sum(1 for m in edges if supported(g, nid, m, rel, ext))
        cell = clusters[uf.find(f"{gname}|{nid}")][gname]
        cell[0] += len(edges)
        cell[1] += sup

    rows = list(clusters.values())

    def rates(sample):
        out = {}
        for gname in ("g_raw", "g_norm"):
            e = sum(r[gname][0] for r in sample)
            s = sum(r[gname][1] for r in sample)
            out[gname] = s / e if e else 0.0
        return out

    point = rates(rows)
    totals = {gname: (sum(r[gname][0] for r in rows), sum(r[gname][1] for r in rows)) for gname in graphs}
    rng = random.Random(11)
    diffs = []
    for _ in range(N_BOOT):
        sample = [rows[rng.randrange(len(rows))] for _ in rows] if rows else []
        r = rates(sample)
        diffs.append(r["g_norm"] - r["g_raw"])
    diffs.sort()
    p_le0 = sum(1 for d in diffs if d <= 0) / N_BOOT
    return {
        "n_common_ext_diseases": len(common),
        "n_clusters": len(rows),
        "g_raw": {"edges": totals["g_raw"][0], "supported": totals["g_raw"][1], "rate": point["g_raw"],
                  "ci95": wilson(totals["g_raw"][1], totals["g_raw"][0])},
        "g_norm": {"edges": totals["g_norm"][0], "supported": totals["g_norm"][1], "rate": point["g_norm"],
                   "ci95": wilson(totals["g_norm"][1], totals["g_norm"][0])},
        "delta": point["g_norm"] - point["g_raw"],
        "delta_ci95": (diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT) - 1]) if rows else (0.0, 0.0),
        "boot_share_delta_le_0": p_le0,
    }


# ---------------------------------------------------------------- test 2


def scorers(g: dict, rel: str):
    n = g["n_docs"]

    def count(d, m):
        return g["co"][(d, rel, m)]

    def lift(d, m):
        return g["co"][(d, rel, m)] * n / (g["df"][d] * g["df"][m])

    def pmi_discounted(d, m):
        c, fd, fm = g["co"][(d, rel, m)], g["df"][d], g["df"][m]
        discount = (c / (c + 1)) * (min(fd, fm) / (min(fd, fm) + 1))
        return math.log(c * n / (fd * fm)) * discount

    def cue(d, m):
        return g["cue"][(d, rel, m)]

    return {"count": count, "lift": lift, "pmi_discounted": pmi_discounted, "cue": cue}


def rank_metrics(labels_sorted: list[bool]) -> dict[str, float]:
    out = {}
    for k in KS:
        top = labels_sorted[: min(k, len(labels_sorted))]
        out[f"P@{k}"] = sum(top) / len(top)
    hits, precs = 0, []
    for i, lab in enumerate(labels_sorted, 1):
        if lab:
            hits += 1
            precs.append(hits / i)
    out["AP"] = sum(precs) / max(sum(labels_sorted), 1)
    return out


def ranking_eval(g: dict, rel: str, ext: dict, min_candidates: int = 2) -> dict:
    nb = neighbours(g, rel)
    fns = scorers(g, rel)
    rng = random.Random(42)
    rows = []
    for d, cands in nb.items():
        if not (disease_classes(g, d, ext) & ext["subjects"][rel]) or len(cands) < min_candidates:
            continue
        labels = {m: supported(g, d, m, rel, ext) for m in cands}
        n_pos = sum(labels.values())
        if n_pos == 0:
            continue
        row = {"disease": d.split("::", 1)[1], "n_cand": len(cands), "n_pos": n_pos}
        for name, fn in fns.items():
            scores = {m: fn(d, m) for m in cands}
            acc = defaultdict(float)
            for _ in range(N_TIE_SHUFFLES):
                order = sorted(cands, key=lambda m: (-scores[m], rng.random()))
                for k, v in rank_metrics([labels[m] for m in order]).items():
                    acc[k] += v / N_TIE_SHUFFLES
            row[name] = dict(acc)
        base = n_pos / len(cands)
        row["random"] = {**{f"P@{k}": base for k in KS}, "AP": float("nan")}
        rows.append(row)

    methods = list(fns) + ["random"]
    metric_names = [f"P@{k}" for k in KS] + ["AP"]
    summary = {m: {k: (sum(r[m][k] for r in rows) / len(rows) if rows else 0.0) for k in metric_names}
               for m in methods}

    boot = {}
    brng = random.Random(7)
    for m in ("lift", "pmi_discounted", "cue"):
        boot[m] = {}
        for k in ("P@3", "AP"):
            diffs = [r[m][k] - r["count"][k] for r in rows]
            if not diffs:
                continue
            means = []
            for _ in range(N_BOOT):
                sample = [diffs[brng.randrange(len(diffs))] for _ in diffs]
                means.append(sum(sample) / len(sample))
            means.sort()
            boot[m][k] = {"delta": sum(diffs) / len(diffs),
                          "ci95": (means[int(0.025 * N_BOOT)], means[int(0.975 * N_BOOT) - 1])}
    return {"n_diseases": len(rows), "summary": summary, "vs_count": boot, "rows": rows}


# ---------------------------------------------------------------- report


def fmt_ci(ci) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def report(ext: dict, results: dict) -> str:
    lines = [
        f"外部知识库检验（{ext['name']}；判定不经过本系统的标准化）",
        "=" * 64,
        f"文件: {', '.join(ext['files'])}  记录/三元组 {ext['n_triples']}  同义词 {ext['n_synonym']}",
        f"疾病-药物 {ext['raw_counts']['使用药物']} 条 / 疾病 {len(ext['subjects']['使用药物'])} 个；"
        f"疾病-手术 {ext['raw_counts']['实施手术']} 条 / 疾病 {len(ext['subjects']['实施手术'])} 个",
        "匹配: 全半角/大小写/空白统一后精确匹配；药名两侧同样去掉剂型、「注射用」和成盐前缀；",
        "      外部库有「同义词」关系时用它合并疾病名；G_norm 节点用规范名 + 合并进来的所有原始写法去匹配。",
        "",
        "检验一：边被外部知识印证的比例（只统计外部库覆盖到的疾病）",
    ]
    for rel in REL_PREDICATE:
        lines.append(f"【{rel}】")
        if rel not in results:
            lines.append("  外部库没有这类关系，跳过")
            continue
        for gname in ("g_raw", "g_norm"):
            r = results[rel][gname]["support"]
            lines.append(
                f"  {gname:6s} 覆盖疾病节点 {r['n_covered']}/{r['n_disease_nodes']}"
                f"（提及占比 {r['covered_mention_share']:.3f}）  "
                f"边 {r['n_edges']}  被印证 {r['n_supported']}  "
                f"印证率 {r['support_rate']:.3f} {fmt_ci(r['support_rate_ci95'])}  "
                f"按文档加权 {r['support_rate_doc_weighted']:.3f}  "
                f"外部关系召回 {r['ext_found']}/{r['ext_findable']}={r['ext_recall']:.3f}"
            )
        pr = results[rel]["paired"]
        lines.append(
            f"  配对（两图都能对上的同一批外部疾病 {pr['n_common_ext_diseases']} 个，{pr['n_clusters']} 个簇）："
            f"g_raw {pr['g_raw']['supported']}/{pr['g_raw']['edges']}={pr['g_raw']['rate']:.3f}  "
            f"g_norm {pr['g_norm']['supported']}/{pr['g_norm']['edges']}={pr['g_norm']['rate']:.3f}  "
            f"差 {pr['delta']:+.3f} {fmt_ci(pr['delta_ci95'])}（按簇 bootstrap；差≤0 的比例 {pr['boot_share_delta_le_0']:.3f}）"
        )
    lines += ["", "检验二：边排序 precision@k（G_norm；只评有 ≥1 条被印证边、候选 ≥2 的疾病；并列随机打破取平均）"]
    for rel in results:
        rk = results[rel]["g_norm"]["ranking"]
        lines.append(f"【{rel}】疾病数 {rk['n_diseases']}")
        lines.append(f"  {'排序方式':16s}" + "".join(f"{k:>8s}" for k in ("P@1", "P@3", "P@5", "MAP")))
        for m, v in rk["summary"].items():
            ap = "     n/a" if m == "random" else f"{v['AP']:8.3f}"
            lines.append(f"  {m:16s}{v['P@1']:8.3f}{v['P@3']:8.3f}{v['P@5']:8.3f}{ap}")
        for m, b in rk["vs_count"].items():
            parts = [f"{k} Δ={x['delta']:+.3f} {fmt_ci(x['ci95'])}" for k, x in b.items()]
            lines.append(f"  {m} vs count（按疾病 bootstrap {N_BOOT} 次）: " + "；".join(parts))
    lines += ["", "诊断：G_norm 中提及最多但外部库未覆盖的疾病（检查匹配是否漏掉）"]
    for rel in results:
        top = results[rel]["g_norm"]["support"]["top_uncovered"]
        lines.append(f"【{rel}】" + "、".join(f"{n}({c})" for n, c in top))
    lines += [
        "",
        "说明:",
        "- 外部知识库不完整，没被印证不等于错，印证率是精确率的下界（同 Rotmensch 2017）。",
        "- count = 共现文档数；lift = c·N/(df_d·df_m)；pmi_discounted = PMI × Pantel-Lin 折扣；"
        "cue = 带医嘱/手术线索的共现文档数；random = 候选中被印证边的比例。",
    ]
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("cmeie", "medicalkg", "all"), default="all")
    p.add_argument("--cmeie-dir", default=str(ROOT / "data" / "dicts" / "raw" / "CMeIE"))
    p.add_argument("--medicalkg", default=str(ROOT / "data" / "dicts" / "raw" / "medicalkg" / "medical.json"))
    p.add_argument("--out-dir", default=str(ROOT / "outputs" / "metrics"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sources = {}
    if args.source in ("cmeie", "all"):
        sources["cmeie"] = load_cmeie(Path(args.cmeie_dir))
    if args.source in ("medicalkg", "all"):
        sources["medicalkg"] = load_medicalkg(Path(args.medicalkg))
    configure_normalization()
    processed = ROOT / "data" / "processed"
    gold_docs = load_bio_docs(processed / "test.bio")
    pred_docs = load_bio_docs(processed / "test_pred.bio")
    built = build_graphs(gold_docs, pred_docs)
    graphs = {"g_raw": doc_level_graph(pred_docs, None), "g_norm": doc_level_graph(pred_docs, built["pred_canon"])}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for src_name, ext in sources.items():
        results = {}
        for rel in REL_PREDICATE:
            if not ext["subjects"][rel]:
                continue
            results[rel] = {gname: {"support": edge_support(g, rel, ext), "ranking": ranking_eval(g, rel, ext)}
                            for gname, g in graphs.items()}
            results[rel]["paired"] = paired_support(graphs, rel, ext)
        text = report(ext, results)
        (out / f"external_kg_{src_name}.txt").write_text(text, encoding="utf-8")
        dump = {"source": ext["name"], "files": ext["files"], "n_triples": ext["n_triples"], "results": results}
        (out / f"external_kg_{src_name}.json").write_text(json.dumps(dump, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
