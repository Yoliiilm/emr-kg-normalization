"""Cross-dataset check of the normalizer on CCKS2017 CNER (a different provider than CCKS2019/Yidu).

Data: data/raw/ccks2017/data_origin/<section>/<section>-<id>.txt(original.txt), 300 training records
(the public test set is not included). The four sections of one record are joined into one document.
Gold entities are used throughout, so NER errors do not enter.

Label mapping: 疾病和诊断 → 疾病和诊断, 身体部位 → 解剖部位, 治疗 → 手术 if it looks like a procedure
(SURGERY_RE) else 药物; symptoms and tests are dropped.

Reports
  1. overlap with CCKS2019 text (to rule out duplicated records),
  2. ICD hit rate of disease surfaces with the full system / without this project's new rules /
     without the curated alias table, next to the CCKS2019 figures,
  3. external support of disease->drug edges, G_raw vs G_norm (full, and disease normalization only),
  4. a random sample of changed disease surfaces for manual checking (CSV).
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_ablation import FULL
from eval_external_extra import compare, norm_graph
from eval_external_kg import doc_level_graph, load_cmeie, load_medicalkg
from eval_g_raw_vs_g_norm import collect_freq
from ner_data import load_bio_docs
from normalize import configure_normalization, coverage_report, lookup_with_source

DATA = ROOT / "data" / "raw" / "ccks2017" / "data_origin"
SECTIONS = ("一般项目", "病史特点", "诊疗经过", "出院情况")
LABEL_MAP = {"疾病和诊断": "疾病和诊断", "身体部位": "解剖部位"}
SURGERY_RE = re.compile(r"术|切除|置换|固定|吻合|造瘘|移植|穿刺|引流|缝合|成形|介入|支架|PCI|镜下|复位|结扎|清创|截肢|植入")
OUT = ROOT / "outputs" / "metrics"


def load_ccks2017() -> tuple[list, dict]:
    stats = Counter()
    by_id: dict[int, list[tuple[str, list]]] = {}
    for sec in SECTIONS:
        for ann in (DATA / sec).glob(f"{sec}-*.txt"):
            if ann.name.endswith("original.txt"):
                continue
            rid = int(ann.stem.split("-")[1])
            text = (DATA / sec / f"{ann.name}original.txt").read_text(encoding="utf-8-sig").strip()
            ents = []
            for line in ann.read_text(encoding="utf-8-sig").splitlines():
                parts = line.split("\t")
                if len(parts) != 4:
                    continue
                name, s, e, typ = parts[0], int(parts[1]), int(parts[2]), parts[3].strip()
                if text[s : e + 1] != name:
                    stats["offset_mismatch"] += 1
                    continue
                if typ == "治疗":
                    label = "手术" if SURGERY_RE.search(name) else "药物"
                    stats[f"治疗→{label}"] += 1
                else:
                    label = LABEL_MAP.get(typ)
                if label:
                    ents.append((s, e, label))
                    stats[label] += 1
            by_id.setdefault(rid, []).append((text, ents))
    docs = []
    for rid in sorted(by_id):
        chars, tags = [], []
        for text, ents in by_id[rid]:
            t = ["O"] * len(text)
            for s, e, label in sorted(ents):
                if any(x != "O" for x in t[s : e + 1]):
                    stats["overlap_dropped"] += 1
                    continue
                t[s] = f"B-{label}"
                for i in range(s + 1, e + 1):
                    t[i] = f"I-{label}"
            if chars:
                chars.append("。")
                tags.append("O")
            chars.extend(text)
            tags.extend(t)
        docs.append((chars, tags))
    stats["n_docs"] = len(docs)
    return docs, dict(stats)


def overlap_with_ccks2019(docs) -> dict:
    processed = ROOT / "data" / "processed"
    k = 15
    shingles = set()
    for name in ("train.bio", "dev.bio", "test.bio"):
        for chars, _ in load_bio_docs(processed / name):
            s = "".join(chars)
            shingles.update(s[i : i + k] for i in range(0, max(len(s) - k, 0) + 1))
    shares = []
    for chars, _ in docs:
        s = "".join(chars)
        grams = [s[i : i + k] for i in range(0, max(len(s) - k, 0) + 1)]
        shares.append(sum(g in shingles for g in grams) / len(grams) if grams else 0.0)
    return {"shingle_len": k, "mean_share": sum(shares) / len(shares),
            "n_docs_over_50pct": sum(x > 0.5 for x in shares), "max_share": max(shares)}


def hit_rates(freq) -> dict:
    out = {}
    for key, extra in (("full", {}), ("no_new_rules", {"use_new_rules": False}), ("no_curated", {"use_curated": False})):
        configure_normalization(**FULL, **extra)
        cov = coverage_report(freq)
        out[key] = {"icd_hit_rate": cov["icd_hit_rate"], "source_counts": cov["source_counts"],
                    "drug_alias_hit_rate": cov["drug"]["alias_hit_rate"]}
    configure_normalization()
    return out


def manual_sample(freq, n: int = 60) -> list[dict]:
    configure_normalization(**FULL)
    rows = []
    for surface, cnt in freq.get("疾病和诊断", {}).items():
        canon, source = lookup_with_source(surface, "疾病和诊断")
        if source != "heuristic" and canon != surface:
            rows.append({"surface": surface, "canonical": canon, "source": source, "mentions": cnt})
    configure_normalization()
    random.Random(2017).shuffle(rows)
    return rows[:n], len(rows)


def main() -> None:
    docs, load_stats = load_ccks2017()
    freq = collect_freq(docs)
    print("loaded", load_stats)
    overlap = overlap_with_ccks2019(docs)
    print("overlap", overlap)
    rates = hit_rates(freq)
    print("hit rates", {k: round(v["icd_hit_rate"], 4) for k, v in rates.items()})

    sources = {"cmeie": load_cmeie(ROOT / "data" / "dicts" / "raw" / "CMeIE"),
               "medicalkg": load_medicalkg(ROOT / "data" / "dicts" / "raw" / "medicalkg" / "medical.json")}
    raw = doc_level_graph(docs, None)
    ext = {}
    for key, settings in (("full", FULL), ("disease_only", {**FULL, "use_drug_alias": False, "use_surgery_alias": False})):
        g = norm_graph(docs, settings)
        ext[key] = {s: compare(raw, g, src) for s, src in sources.items()}

    sample, n_changed = manual_sample(freq)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "ccks2017_link_sample.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["surface", "canonical", "source", "mentions", "判断(对/错/不确定)"])
        w.writeheader()
        for r in sample:
            w.writerow(r | {"判断(对/错/不确定)": ""})

    n_dis = len(freq.get("疾病和诊断", {}))
    n_drug = len(freq.get("药物", {}))
    L = ["跨数据集检验：CCKS2017 CNER 训练集（极目云健康，非医渡云；人工标注实体）", "=" * 72,
         f"病历 {load_stats['n_docs']} 份；疾病提及 {load_stats.get('疾病和诊断', 0)}（不同写法 {n_dis}）；"
         f"「治疗」拆为 药物 {load_stats.get('治疗→药物', 0)} / 手术 {load_stats.get('治疗→手术', 0)}（药物写法 {n_drug}）；"
         f"偏移不符丢弃 {load_stats.get('offset_mismatch', 0)}，重叠丢弃 {load_stats.get('overlap_dropped', 0)}",
         f"与 CCKS2019 文本重合：{overlap['shingle_len']} 字片段平均重合 {overlap['mean_share']:.3f}，"
         f"重合超过一半的病历 {overlap['n_docs_over_50pct']} 份（最大 {overlap['max_share']:.3f}）",
         "", "1. 疾病写法链接到 ICD 的比例（CCKS2019 对照：测试集 0.648，训练集 0.612）"]
    for key, title in (("full", "完整系统"), ("no_new_rules", "去掉新规则"), ("no_curated", "去掉人工别名表")):
        r = rates[key]
        L.append(f"  {title:10s} ICD 命中率 {r['icd_hit_rate']:.3f}")
    L.append(f"  完整系统来源分布 {rates['full']['source_counts']}")
    L.append(f"  药物别名命中率 {rates['full']['drug_alias_hit_rate']:.3f}（CCKS2019 测试集 0.506）")
    L += ["", "2. 外部知识库印证率（疾病→药物；配对差按外部疾病簇 bootstrap）"]
    for s, st in (("cmeie", "CMeIE"), ("medicalkg", "寻医问药")):
        g_raw = ext["full"][s]["g_raw"]
        parts = [f"G_raw {g_raw['n_supported']}/{g_raw['n_edges']}={g_raw['support_rate']:.3f}"]
        for key, kt in (("disease_only", "只疾病"), ("full", "完整")):
            r = ext[key][s]
            p = r["paired"]
            parts.append(f"{kt} {r['g_norm']['n_supported']}/{r['g_norm']['n_edges']}={r['g_norm']['support_rate']:.3f} "
                         f"差 {p['delta']:+.3f} [{p['delta_ci95'][0]:+.3f}, {p['delta_ci95'][1]:+.3f}]")
        L.append(f"  {st:6s} " + "  ".join(parts))
        L.append(f"         覆盖疾病提及 {ext['full'][s]['g_raw']['covered_mention_share']:.3f} → "
                 f"{ext['full'][s]['g_norm']['covered_mention_share']:.3f}；外部关系召回 "
                 f"{ext['full'][s]['g_raw']['ext_recall']:.3f} → {ext['full'][s]['g_norm']['ext_recall']:.3f}")
    L += ["", f"3. 人工核对样本：被改写的疾病写法共 {n_changed} 个，随机抽 {len(sample)} 个 → outputs/metrics/ccks2017_link_sample.csv"]
    text = "\n".join(L) + "\n"
    (OUT / "cross_dataset_ccks2017.txt").write_text(text, encoding="utf-8")
    (OUT / "cross_dataset_ccks2017.json").write_text(json.dumps(
        {"load": load_stats, "overlap": overlap, "hit_rates": rates, "external": ext, "n_changed": n_changed},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
